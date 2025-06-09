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

# Helper function to escape MarkdownV2, used for internal error messages
# This is a simplified version or you can import the one from commands.py if structured as a utility module
def _escape_markdown_v2_internal(text: str) -> str:
    """Escapes special characters for Telegram's MarkdownV2 parse_mode for internal use."""
    # Simplified list for basic safety, actual _escape_markdown_v2 is more comprehensive
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    return re.sub(r'([%s])' % re.escape(escape_chars), r'\\\1', text)


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
                    if usage_example.strip() and usage_example != 'No usage example.':
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
            self.rag_chain = None 
            return

        embeddings = GoogleGenerativeAIEmbeddings(google_api_key=self.google_api_key, model="models/embedding-001") 
        
        logger.info(f"[{time.strftime('%H:%M:%S')}] Initializing/Loading Chroma DB at {CHROMA_DB_DIR}...")
        os.makedirs(CHROMA_DB_DIR, exist_ok=True) 
        
        vectorstore = None
        try:
            vectorstore_candidate = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
            if vectorstore_candidate._collection.count() > 0:
                logger.info(f"[{time.strftime('%H:%M:%S')}] Loaded existing Chroma DB with {vectorstore_candidate._collection.count()} documents.")
                vectorstore = vectorstore_candidate
            else:
                logger.warning(f"[{time.strftime('%H:%M:%S')}] Existing Chroma DB at {CHROMA_DB_DIR} is empty. Re-populating.")
                if os.path.exists(CHROMA_DB_DIR): # Should exist due to makedirs, but double check
                    shutil.rmtree(CHROMA_DB_DIR)
                os.makedirs(CHROMA_DB_DIR, exist_ok=True)
                vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=CHROMA_DB_DIR)
                logger.info(f"[{time.strftime('%H:%M:%S')}] Re-created and populated Chroma DB with {len(documents)} documents.")

        except Exception as e: 
            logger.warning(f"[{time.strftime('%H:%M:%S')}] Could not load or use existing Chroma DB (error: {e}). Attempting to create a new one.")
            if os.path.exists(CHROMA_DB_DIR):
                try:
                    shutil.rmtree(CHROMA_DB_DIR) 
                except Exception as e_rm:
                    logger.error(f"Failed to remove existing Chroma DB directory {CHROMA_DB_DIR}: {e_rm}")
            os.makedirs(CHROMA_DB_DIR, exist_ok=True) 
            vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=CHROMA_DB_DIR)
            logger.info(f"[{time.strftime('%H:%M:%S')}] Created new Chroma DB with {len(documents)} documents.")
        
        if vectorstore is None:
            logger.error("Failed to initialize or create Chroma vector store. RAG will be unavailable.")
            self.rag_chain = None
            return

        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3, google_api_key=self.google_api_key) 

        rag_prompt = ChatPromptTemplate.from_template("""
        Bạn là một chuyên gia pentesting trợ giúp, cung cấp câu trả lời bằng tiếng Việt.
        Dựa vào các thông tin công cụ Kali Linux sau đây ('Ngữ cảnh công cụ'), hãy gợi ý các công cụ phù hợp và cung cấp các lệnh mẫu để thực hiện tác vụ pentest của người dùng.
        Nếu thông tin từ 'Ngữ cảnh công cụ' không đủ hoặc không liên quan trực tiếp, hãy sử dụng kiến thức chung của bạn về Kali Linux và pentesting để đưa ra gợi ý hợp lý và thực tế.
        
        **QUAN TRỌNG**: Định dạng câu trả lời của bạn bằng cú pháp *MarkdownV2 của Telegram*.
        - **Khối mã (Code Blocks)**: Để hiển thị các lệnh, cấu hình, hoặc ví dụ mã, hãy sử dụng khối mã. Bắt đầu bằng ba dấu backtick (```) theo sau là gợi ý ngôn ngữ (ví dụ: ```bash hoặc ```text hoặc ```json) trên một dòng, tiếp theo là mã của bạn, và kết thúc bằng ba dấu backtick (```) trên một dòng riêng.
          Ví dụ cho lệnh:
          ```bash
          nmap -sV -p 80,443 example.com
          ```
          Ví dụ cho văn bản/output mẫu:
          ```text
          Some tool output here...
          ```
        - **Nhấn mạnh (Bold/Italics)**: Sử dụng dấu sao cho *văn bản đậm* (`*text*`) và dấu gạch dưới cho _văn bản nghiêng_ (`_text_`) một cách tiết chế, chỉ khi thực sự cần làm nổi bật một thuật ngữ hoặc khái niệm quan trọng. Tránh lạm dụng.
        - **Ký tự đặc biệt**: Telegram MarkdownV2 sử dụng các ký tự đặc biệt: `_*[]()~`>#+-=|{}.!`. Nếu bạn cần hiển thị các ký tự này theo nghĩa đen (không phải là một phần của định dạng Markdown), chúng phải được thoát bằng dấu gạch chéo ngược (`\\`). Ví dụ, để hiển thị `example.com`, bạn sẽ viết `example\\.com`. Hãy cố gắng tạo ra MarkdownV2 hợp lệ và tự thoát các ký tự cần thiết trong văn bản thường.
        - **Danh sách (Lists)**: Nếu bạn muốn tạo danh sách, hãy sử dụng gạch đầu dòng (ví dụ: `\\- Mục 1`, `\\* Mục A`) hoặc số theo sau là dấu chấm (`1\\. Bước một`). Đảm bảo có khoảng trắng sau ký hiệu danh sách.
        - **Liên kết (Links)**: Sử dụng định dạng `[văn bản hiển thị](URL)`. Ví dụ: `[Trang chủ Kali](https://www.kali.org/)`. Tránh URL trần.

        Ngữ cảnh công cụ:
        {context}
        
        Câu hỏi của người dùng: {question}

        Câu trả lời (tiếng Việt, định dạng MarkdownV2):
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
            # This message will be sent by the command handler, which should handle Markdown escaping if needed.
            # However, it's good practice for service methods to return safe strings.
            return _escape_markdown_v2_internal(
                "Tính năng gợi ý công cụ Kali hiện không khả dụng. "
                "Vui lòng kiểm tra cấu hình bot hoặc thông báo cho admin."
            )
        
        try:
            # Response from LLM is expected to be MarkdownV2 formatted
            response = await self.rag_chain.ainvoke(query)
            return response 
        except Exception as e:
            logger.error(f"Error during RAG chain execution: {e}", exc_info=True)
            error_detail = str(e)[:150] # Truncate long error messages
            return _escape_markdown_v2_internal(
                f"Đã xảy ra lỗi khi tìm kiếm gợi ý. Vui lòng thử lại sau. Lỗi: {error_detail}"
            )