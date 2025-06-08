# telegram_kali_bot/cogs/kali_rag.py

import logging
import json
import os
import time
import shutil # Import shutil for rmtree

from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Cấu hình đường dẫn dữ liệu
DATA_FILE = "data/kali_tools_data.json"
CHROMA_DB_DIR = "./chroma_db"

class KaliRAGService:
    def __init__(self, google_api_key: str):
        self.rag_chain = None
        self.google_api_key = google_api_key

        if self.google_api_key:
            try:
                self._initialize_rag_chain()
                logger.info("KaliRAGService initialized successfully using Google Generative AI.")
            except Exception as e:
                logger.error(f"Failed to initialize KaliRAGService: {e}", exc_info=True)
                self.rag_chain = None
        else:
            logger.warning("GOOGLE_API_KEY not provided. RAG feature will be unavailable.")

    def _load_and_prepare_data(self, filepath: str) -> list[Document]:
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

    def _initialize_rag_chain(self):
        """Khởi tạo các thành phần của RAG chain."""
        documents = self._load_and_prepare_data(DATA_FILE)
        if not documents:
            logger.error("RAG Initialization failed: No documents available for RAG.")
            return None

        embeddings = GoogleGenerativeAIEmbeddings(google_api_key=self.google_api_key, model="models/embedding-001") 
        
        logger.info(f"[{time.strftime('%H:%M:%S')}] Initializing/Loading Chroma DB at {CHROMA_DB_DIR}...")
        os.makedirs(CHROMA_DB_DIR, exist_ok=True) # Ensure Chroma DB directory exists
        try:
            vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
            if vectorstore._collection.count() == 0:
                logger.warning(f"[{time.strftime('%H:%M:%S')}] Existing Chroma DB is empty or incompatible. Re-adding documents.")
                vectorstore.add_documents(documents)
                logger.info(f"[{time.strftime('%H:%M:%S')}] Documents added to existing Chroma DB.")
            else:
                logger.info(f"[{time.strftime('%H:%M:%S')}] Loaded existing Chroma DB with {vectorstore._collection.count()} documents.")

        except Exception as e:
            logger.warning(f"[{time.strftime('%H:%M:%S')}] Could not load existing Chroma DB: {e}. Attempting to recreate.")
            if os.path.exists(CHROMA_DB_DIR):
                shutil.rmtree(CHROMA_DB_DIR)
                os.makedirs(CHROMA_DB_DIR, exist_ok=True)
                logger.warning(f"[{time.strftime('%H:%M:%S')}] Cleared old Chroma DB for recreation.")

            vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=CHROMA_DB_DIR)
            logger.info(f"[{time.strftime('%H:%M:%S')}] Created new Chroma DB from documents.")
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b", temperature=0.3, google_api_key=self.google_api_key) 

        # ĐIỀU CHỈNH PROMPT ĐỂ TRÁNH LỖI PHÂN TÍCH CÚ PHÁP CỦA LANGCHAIN
        # Không dùng bất kỳ ký tự Markdown nào trong phần hướng dẫn format.
        rag_prompt = ChatPromptTemplate.from_template("""
        Bạn là một chuyên gia pentesting trợ giúp. 
        Dựa vào các thông tin công cụ Kali Linux sau đây, hãy gợi ý các công cụ phù hợp và cung cấp các lệnh mẫu để thực hiện tác vụ pentest của người dùng.
        Nếu thông tin từ 'Ngữ cảnh công cụ' không đủ hoặc không liên quan trực tiếp, hãy sử dụng kiến thức chung của bạn về Kali Linux và pentesting để đưa ra gợi ý hợp lý và thực tế.
        
        **QUAN TRỌNG**: Định dạng câu trả lời của bạn bằng *MarkdownV2 syntax*.
        - Để hiển thị các khối mã (code blocks) và lệnh, hãy bắt đầu và kết thúc bằng ba dấu gạch chéo ngược trên một dòng riêng. Ví dụ:
          ```
          nmap -sV example.com
          ```
        - KHÔNG sử dụng các định dạng bôi đậm hoặc in nghiêng thông thường.
        - Tránh sử dụng các ký tự đặc biệt như dấu gạch chéo ngược, dấu sao, dấu gạch dưới, dấu ngoặc vuông, dấu ngoặc đơn, dấu ngã, dấu huyền, dấu lớn hơn, dấu thăng, dấu cộng, dấu trừ, dấu bằng, dấu sổ thẳng, dấu ngoặc nhọn, dấu chấm, dấu chấm than trong văn bản thông thường nếu chúng không phải là một phần của cú pháp Markdown hợp lệ.

        Ngữ cảnh công cụ:
        {context}
        
        Câu hỏi của người dùng: {question}
        """)

        self.rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | rag_prompt
            | llm
            | StrOutputParser()
        )
        
        logger.info(f"[{time.strftime('%H:%M:%S')}] RAG chain in KaliRAGService initialized.")

    async def ask_question(self, query: str) -> str:
        """Invokes the RAG chain with a given query."""
        if self.rag_chain is None:
            return "Tính năng gợi ý công cụ Kali hiện không khả dụng. Vui lòng kiểm tra cấu hình bot."
        
        try:
            response = await self.rag_chain.ainvoke(query)
            return response # Trả về response thô, _escape_markdown_v2 sẽ xử lý sau
        except Exception as e:
            logger.error(f"Error during RAG chain execution: {e}", exc_info=True)
            return f"Đã xảy ra lỗi khi tìm kiếm gợi ý: {e}"