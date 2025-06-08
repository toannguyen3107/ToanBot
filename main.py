# telegram_kali_bot/main.py

import os
import logging
import json
import time
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import CommandHandler, Application, MessageHandler, filters

# Import TranslationService class (không import instance)
from cogs.translate import TranslationService

# Import cogs.commands module để có thể gán các instance LangChain vào đó
import cogs.commands 

# Import LangChain components để khởi tạo RAG chain
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# --- Cấu hình Logging ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Tải biến môi trường từ .env
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Được sử dụng cho RAG (embeddings và ChatOpenAI)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # Được sử dụng cho TranslationService (Gemini)

# Kiểm tra sự tồn tại của các token/API keys
if not TELEGRAM_BOT_TOKEN:
    logger.critical("TELEGRAM_BOT_TOKEN not found in .env file. Exiting.")
    exit(1)
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not found in .env file. RAG feature may be limited or unavailable.")
if not GOOGLE_API_KEY:
    logger.warning("GOOGLE_API_KEY not found in .env file. Translation feature may be limited or unavailable.")

# --- Cấu hình và khởi tạo RAG Chain ---
DATA_FILE = "data/kali_tools_data.json"
CHROMA_DB_DIR = "./chroma_db"

def load_and_prepare_data(filepath: str) -> list[Document]:
    """Tải dữ liệu công cụ đã scrape từ JSON và chuyển đổi sang định dạng LangChain Document."""
    logger.info(f"[{time.strftime('%H:%M:%S')}] Loading data for RAG from {filepath}...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Error: Data file not found at {filepath}.")
        logger.error("Please run 'python scripts/scrape_kali_tools.py' first to generate the data.")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {filepath}: {e}")
        logger.error("Please ensure the JSON file is correctly formatted.")
        return []

    documents = []
    for item in raw_data:
        tool_name = item.get('name', 'N/A')
        main_description = item.get('main_description', 'No detailed description.')
        how_to_install = item.get('how_to_install', 'Installation command not found.')
        tool_url = item.get('url', '')
        
        content_parts = [
            f"Tool Name: {tool_name}",
            f"Description: {main_description}",
            f"How to Install: {how_to_install}"
        ]
        
        if 'commands' in item and item['commands']:
            commands_str = "Commands and Usage Examples:\n"
            for cmd in item['commands']:
                sub_command = cmd.get('sub_command', 'N/A')
                usage_example = cmd.get('usage_example', 'No usage example.')
                if usage_example.strip():
                    commands_str += f"- {sub_command}: {usage_example}\n"
                else:
                    commands_str += f"- {sub_command}: Run `{sub_command} --help` or `man {sub_command}` for usage.\n"

            content_parts.append(commands_str)
        
        full_content = "\n\n".join(content_parts).strip()

        documents.append(
            Document(
                page_content=full_content,
                metadata={
                    "tool": tool_name,
                    "category": item.get("category", "Unknown"),
                    "url": tool_url
                }
            )
        )
    logger.info(f"[{time.strftime('%H:%M:%S')}] Loaded {len(documents)} documents for RAG.")
    return documents

def initialize_rag_chain_instance():
    """Khởi tạo các thành phần của RAG chain."""
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY is not set. Cannot initialize RAG chain.")
        return None

    documents = load_and_prepare_data(DATA_FILE)
    if not documents:
        logger.error("RAG Initialization failed: No documents available for RAG.")
        return None

    # Khởi tạo Embeddings với API key
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY) 
    
    logger.info(f"[{time.strftime('%H:%M:%S')}] Initializing/Loading Chroma DB at {CHROMA_DB_DIR}...")
    try:
        # Cố gắng tải Chroma DB đã tồn tại
        vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
        # Kiểm tra nếu DB không rỗng, nếu rỗng thì tạo mới
        if vectorstore._collection.count() == 0:
            logger.warning(f"[{time.strftime('%H:%M:%S')}] Existing Chroma DB is empty. Re-adding documents.")
            vectorstore.add_documents(documents)
            logger.info(f"[{time.strftime('%H:%M:%S')}] Documents added to existing Chroma DB.")
        else:
            logger.info(f"[{time.strftime('%H:%M:%S')}] Loaded existing Chroma DB with {vectorstore._collection.count()} documents.")

    except Exception as e:
        logger.warning(f"[{time.strftime('%H:%M:%S')}] Could not load existing Chroma DB: {e}. Creating new one.")
        vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=CHROMA_DB_DIR)
        logger.info(f"[{time.strftime('%H:%M:%S')}] Created new Chroma DB from documents.")
    
    # Tạo Retriever: tìm kiếm tài liệu liên quan nhất
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Khởi tạo LLM cho RAG với API key
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, openai_api_key=OPENAI_API_KEY) 

    # Định nghĩa Prompt Template cho RAG
    rag_prompt = ChatPromptTemplate.from_template("""
    Bạn là một chuyên gia pentesting trợ giúp. 
    Dựa vào các thông tin công cụ Kali Linux sau đây, hãy gợi ý các công cụ phù hợp và cung cấp các lệnh mẫu để thực hiện tác vụ pentest của người dùng.
    Nếu thông tin từ 'Ngữ cảnh công cụ' không đủ hoặc không liên quan trực tiếp, hãy sử dụng kiến thức chung của bạn về Kali Linux và pentesting để đưa ra gợi ý hợp lý và thực tế.
    Luôn tập trung vào việc đưa ra các lệnh thực tế, ngắn gọn và hữu ích.
    
    Ngữ cảnh công cụ:
    {context}
    
    Câu hỏi của người dùng: {question}
    """)

    # Xây dựng chuỗi RAG
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    
    logger.info(f"[{time.strftime('%H:%M:%S')}] RAG chain initialized successfully.")
    return rag_chain

# --- Main function của Bot Telegram ---
def main() -> None:
    # --- Khởi tạo các dịch vụ (RAG và Translation) ---
    logger.info(f"[{time.strftime('%H:%M:%S')}] Starting service initialization...")
    
    # Khởi tạo RAG chain
    rag_chain_instance = initialize_rag_chain_instance()
    # Gán instance RAG chain vào biến toàn cục trong cogs.commands
    cogs.commands.rag_chain = rag_chain_instance
    
    if cogs.commands.rag_chain is None:
        logger.critical("Failed to initialize RAG chain. RAG feature will be unavailable.")
        # Bạn có thể chọn exit(1) ở đây nếu RAG là tính năng bắt buộc.
        # Ở đây tôi sẽ để bot tiếp tục chạy nhưng tính năng RAG sẽ không hoạt động.

    # Khởi tạo TranslationService
    translation_service_instance = TranslationService(GOOGLE_API_KEY)
    # Gán instance TranslationService vào biến toàn cục trong cogs.commands
    cogs.commands.translation_service_instance = translation_service_instance
    
    if translation_service_instance.llm is None:
        logger.warning("TranslationService LLM could not be initialized. Translation feature will be unavailable.")


    # --- Cấu hình Telegram Bot ---
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Đăng ký các Command Handlers
    application.add_handler(CommandHandler("start", cogs.commands.start_command))
    application.add_handler(CommandHandler("hello", cogs.commands.hello_command))
    application.add_handler(CommandHandler("ping", cogs.commands.ping_command))
    application.add_handler(CommandHandler("translate", cogs.commands.translate_command))
    application.add_handler(CommandHandler("help", cogs.commands.help_command))
    application.add_handler(CommandHandler("ask_kali", cogs.commands.ask_kali_command))

    # Đăng ký Message Handler cho các tin nhắn không phải là lệnh
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, cogs.commands.echo_message))

    logger.info("Bot đang bắt đầu Long Polling...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)
    logger.info("Bot đã dừng Long Polling.")

if __name__ == '__main__':
    main()