# telegram_kali_bot/cogs/kali_rag.py

import logging
import json
import os
import time
import shutil
import re
from langchain_chroma import Chroma

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

DATA_FILE = "data/kali_tools_data.json"
CHROMA_DB_DIR = "./chroma_db" # Sẽ được giải quyết thành /app/chroma_db do WORKDIR

def _escape_markdown_v2_internal(text: str) -> str:
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    return re.sub(r'([%s])' % re.escape(escape_chars), r'\\\1', text)

class KaliRAGService:
    def __init__(self, google_api_key: str):
        self.rag_chain = None
        self.google_api_key = google_api_key
        if self.google_api_key:
            try:
                self._initialize_rag_chain()
                # Kiểm tra sau khi _initialize_rag_chain chạy
                if self.rag_chain:
                    logger.info("KaliRAGService initialized successfully with RAG chain.")
                else:
                    logger.error("KaliRAGService initialization attempted, but RAG chain is still None.")
            except Exception as e:
                logger.error(f"Unhandled exception during KaliRAGService __init__: {e}", exc_info=True)
                self.rag_chain = None
        else:
            logger.warning("GOOGLE_API_KEY not provided. RAG feature will be unavailable.")

    def _load_and_prepare_data(self, filepath: str) -> list[Document]:
        # (Giữ nguyên hàm này)
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
        documents = self._load_and_prepare_data(DATA_FILE)
        if not documents:
            logger.error("RAG Initialization failed: No documents available for RAG.")
            self.rag_chain = None
            return

        embeddings = GoogleGenerativeAIEmbeddings(google_api_key=self.google_api_key, model="models/embedding-001")
        
        logger.info(f"[{time.strftime('%H:%M:%S')}] Preparing Chroma DB at {CHROMA_DB_DIR}...")
        
        # Đảm bảo thư mục CHROMA_DB_DIR được mount từ volume và có thể ghi
        # Docker compose file đã xử lý việc này.
        # os.makedirs(CHROMA_DB_DIR, exist_ok=True) # Không cần thiết nếu volume được mount đúng

        vectorstore = None
        force_recreate_db = False

        if os.path.exists(CHROMA_DB_DIR):
            try:
                logger.info(f"[{time.strftime('%H:%M:%S')}] Attempting to load existing Chroma DB from {CHROMA_DB_DIR}.")
                import chromadb
                client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
                collection_name = "kali_rag_collection"

                try:
                    collection = client.get_collection(name=collection_name)
                    if collection.count() == 0:
                        logger.warning(f"[{time.strftime('%H:%M:%S')}] Existing Chroma collection '{collection_name}' is empty. Will re-populate.")
                        vectorstore = Chroma(
                            client=client,
                            collection_name=collection_name,
                            embedding_function=embeddings,
                            persist_directory=CHROMA_DB_DIR
                        )
                        logger.info(f"[{time.strftime('%H:%M:%S')}] Adding {len(documents)} documents to empty collection.")
                        vectorstore.add_documents(documents)
                        logger.info(f"[{time.strftime('%H:%M:%S')}] Documents added to existing empty collection.")
                    else:
                        logger.info(f"[{time.strftime('%H:%M:%S')}] Loaded existing Chroma collection '{collection_name}' with {collection.count()} documents.")
                        vectorstore = Chroma(
                            client=client, # Sử dụng client hiện có
                            collection_name=collection_name,
                            embedding_function=embeddings,
                            persist_directory=CHROMA_DB_DIR
                        )
                except Exception as get_coll_err: # Ví dụ: collection không tồn tại
                    logger.warning(f"[{time.strftime('%H:%M:%S')}] Could not get collection '{collection_name}' (error: {get_coll_err}). Will try to create new vectorstore.")
                    # Collection không tồn tại, sẽ tạo mới với from_documents
                    force_recreate_db = False # Không cần force, from_documents sẽ tạo nếu chưa có
                    pass # Để cho phép tạo mới ở dưới

            except Exception as load_err:
                logger.warning(f"[{time.strftime('%H:%M:%S')}] Error loading or checking existing Chroma DB: {load_err}. Forcing recreation.")
                force_recreate_db = True
        else: # Thư mục CHROMA_DB_DIR không tồn tại
            logger.info(f"[{time.strftime('%H:%M:%S')}] Chroma DB directory {CHROMA_DB_DIR} not found. Will create new.")
            force_recreate_db = False # Không cần force, from_documents sẽ tạo

        if force_recreate_db:
            logger.warning(f"[{time.strftime('%H:%M:%S')}] Forcing recreation of Chroma DB at {CHROMA_DB_DIR}.")
            if os.path.exists(CHROMA_DB_DIR):
                try:
                    shutil.rmtree(CHROMA_DB_DIR) # Xóa NẾU thực sự cần tạo lại từ đầu
                    logger.info(f"[{time.strftime('%H:%M:%S')}] Successfully removed old Chroma DB directory.")
                except Exception as e_rm:
                    logger.error(f"[{time.strftime('%H:%M:%S')}] Failed to remove old Chroma DB directory {CHROMA_DB_DIR}: {e_rm}. This might cause issues.", exc_info=True)
                    # Nếu không xóa được, có thể lỗi sẽ lặp lại
            
            # os.makedirs(CHROMA_DB_DIR, exist_ok=True) # from_documents sẽ tự tạo nếu persist_directory được cung cấp

        # Tạo vectorstore (mới hoặc load từ thư mục đã xử lý)
        if vectorstore is None: # Chỉ tạo nếu chưa được load ở trên
            try:
                logger.info(f"[{time.strftime('%H:%M:%S')}] Creating/Populating Chroma vectorstore at {CHROMA_DB_DIR}.")
                vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=embeddings,
                    persist_directory=CHROMA_DB_DIR,
                    collection_name="kali_rag_collection" # Nên đặt tên collection rõ ràng
                )
                logger.info(f"[{time.strftime('%H:%M:%S')}] Chroma vectorstore created/populated with {len(documents)} documents.")
            except Exception as create_err:
                logger.error(f"[{time.strftime('%H:%M:%S')}] Failed to create/populate Chroma vectorstore: {create_err}", exc_info=True)
                self.rag_chain = None
                return

        if vectorstore is None:
            logger.error(f"[{time.strftime('%H:%M:%S')}] Vectorstore is still None after all attempts. RAG will be unavailable.")
            self.rag_chain = None
            return

        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3, google_api_key=self.google_api_key)
        
        rag_prompt = ChatPromptTemplate.from_template("""
        Bạn là một chuyên gia pentesting trợ giúp, cung cấp câu trả lời bằng tiếng Việt.
        Dựa vào các thông tin công cụ Kali Linux sau đây ('Ngữ cảnh công cụ'), hãy gợi ý các công cụ phù hợp và cung cấp các lệnh mẫu để thực hiện tác vụ pentest của người dùng.
        Nếu thông tin từ 'Ngữ cảnh công具' không đủ hoặc không liên quan trực tiếp, hãy sử dụng kiến thức chung của bạn về Kali Linux và pentesting để đưa ra gợi ý hợp lý và thực tế. # <--- Lỗi typo "công cụ具"
        
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
        - **Ký tự đặc biệt**: Telegram MarkdownV2 sử dụng các ký tự đặc biệt: `_*[]()~`>#+-=|{{}}.!`. Nếu bạn cần hiển thị các ký tự này theo nghĩa đen (không phải là một phần của định dạng Markdown), chúng phải được thoát bằng dấu gạch chéo ngược (`\\`). Ví dụ, để hiển thị `example.com`, bạn sẽ viết `example\\.com`. Hãy cố gắng tạo ra MarkdownV2 hợp lệ và tự thoát các ký tự cần thiết trong văn bản thường.
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
        logger.info(f"[{time.strftime('%H:%M:%S')}] RAG chain in KaliRAGService initialized successfully.")


    async def ask_question(self, query: str) -> str:
        # (Giữ nguyên hàm này)
        if self.rag_chain is None:
            return _escape_markdown_v2_internal(
                "Tính năng gợi ý công cụ Kali hiện không khả dụng. "
                "Vui lòng kiểm tra cấu hình bot hoặc thông báo cho admin."
            )
        
        try:
            response = await self.rag_chain.ainvoke(query)
            return response 
        except Exception as e:
            logger.error(f"Error during RAG chain execution: {e}", exc_info=True)
            error_detail = str(e)[:150] 
            return _escape_markdown_v2_internal(
                f"Đã xảy ra lỗi khi tìm kiếm gợi ý. Vui lòng thử lại sau. Lỗi: {error_detail}"
            )