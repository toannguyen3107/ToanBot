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
import html # Để escape HTML

logger = logging.getLogger(__name__)

DATA_FILE = "data/kali_tools_data.json"
CHROMA_DB_DIR = "./chroma_db"

def _escape_html_internal(text: str) -> str:
    return html.escape(str(text))

class KaliRAGService:
    def __init__(self, google_api_key: str):
        self.rag_chain = None
        self.google_api_key = google_api_key
        if self.google_api_key:
            try:
                self._initialize_rag_chain()
                if self.rag_chain:
                    logger.info("KaliRAGService initialized successfully with RAG chain.")
                else:
                    logger.error("KaliRAGService initialization attempted, but RAG chain is still None. RAG feature will be unavailable.")
            except Exception as e:
                logger.error(f"Unhandled exception during KaliRAGService __init__: {e}", exc_info=True)
                self.rag_chain = None
        else:
            logger.warning("GOOGLE_API_KEY not provided. RAG feature will be unavailable.")

    def _load_and_prepare_data(self, filepath: str) -> list[Document]:
        # ... (Giữ nguyên hàm này) ...
        logger.info(f"[{time.strftime('%H:%M:%S')}] Loading data for RAG from {filepath}...")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        except FileNotFoundError:
            logger.error(f"Error: Data file not found at {filepath}.")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {filepath}: {e}")
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
                for cmd_item in item['commands']:
                    sub_command = cmd_item.get('sub_command', 'N/A')
                    usage_example = cmd_item.get('usage_example', 'No usage example.')
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
        # ... (Phần load documents và khởi tạo embeddings, vectorstore giữ nguyên) ...
        documents = self._load_and_prepare_data(DATA_FILE)
        if not documents:
            logger.error("RAG Initialization failed: No documents available for RAG.")
            self.rag_chain = None
            return

        embeddings = GoogleGenerativeAIEmbeddings(google_api_key=self.google_api_key, model="models/embedding-001")
        logger.info(f"[{time.strftime('%H:%M:%S')}] Preparing Chroma DB at {CHROMA_DB_DIR}...")
        
        vectorstore = None
        force_recreate_db = False
        collection_name = "kali_rag_collection"

        if os.path.exists(CHROMA_DB_DIR):
            try:
                logger.info(f"[{time.strftime('%H:%M:%S')}] Attempting to load or check existing Chroma DB from {CHROMA_DB_DIR}.")
                import chromadb
                client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
                try:
                    collection = client.get_collection(name=collection_name)
                    if collection.count() == 0:
                        logger.warning(f"[{time.strftime('%H:%M:%S')}] Existing Chroma collection '{collection_name}' is empty. Will re-populate.")
                        vectorstore = Chroma(client=client, collection_name=collection_name, embedding_function=embeddings, persist_directory=CHROMA_DB_DIR)
                        vectorstore.add_documents(documents)
                        logger.info(f"[{time.strftime('%H:%M:%S')}] Documents added to empty collection '{collection_name}'.")
                    else:
                        logger.info(f"[{time.strftime('%H:%M:%S')}] Loaded existing Chroma collection '{collection_name}' with {collection.count()} documents.")
                        vectorstore = Chroma(client=client, collection_name=collection_name, embedding_function=embeddings, persist_directory=CHROMA_DB_DIR)
                except Exception: 
                    logger.warning(f"[{time.strftime('%H:%M:%S')}] Collection '{collection_name}' not found or error accessing. Will attempt to create new vectorstore from documents.")
                    pass 
            except Exception as load_err:
                logger.warning(f"[{time.strftime('%H:%M:%S')}] Error while trying to load/check existing Chroma DB: {load_err}. Forcing recreation.")
                force_recreate_db = True
        else:
            logger.info(f"[{time.strftime('%H:%M:%S')}] Chroma DB directory {CHROMA_DB_DIR} not found. Will create new.")

        if force_recreate_db:
            logger.warning(f"[{time.strftime('%H:%M:%S')}] Forcing recreation of Chroma DB at {CHROMA_DB_DIR}.")
            if os.path.exists(CHROMA_DB_DIR):
                try:
                    shutil.rmtree(CHROMA_DB_DIR)
                    logger.info(f"[{time.strftime('%H:%M:%S')}] Successfully removed old Chroma DB directory.")
                except Exception as e_rm:
                    logger.error(f"[{time.strftime('%H:%M:%S')}] Failed to remove old Chroma DB: {e_rm}.", exc_info=True)

        if vectorstore is None: 
            try:
                logger.info(f"[{time.strftime('%H:%M:%S')}] Creating/Populating Chroma vectorstore at {CHROMA_DB_DIR} for collection '{collection_name}'.")
                vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=embeddings,
                    persist_directory=CHROMA_DB_DIR,
                    collection_name=collection_name 
                )
                logger.info(f"[{time.strftime('%H:%M:%S')}] Chroma vectorstore created/populated for '{collection_name}'.")
            except Exception as create_err:
                logger.error(f"[{time.strftime('%H:%M:%S')}] Failed to create/populate Chroma vectorstore: {create_err}", exc_info=True)
                self.rag_chain = None
                return

        if vectorstore is None:
            logger.error(f"[{time.strftime('%H:%M:%S')}] Vectorstore is still None. RAG will be unavailable.")
            self.rag_chain = None
            return

        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3, google_api_key=self.google_api_key)
        
        # CẬP NHẬT PROMPT HTML THEO CÁC THẺ TELEGRAM HỖ TRỢ
        html_template_string = """Bạn là một chuyên gia pentesting trợ giúp, cung cấp câu trả lời bằng tiếng Việt.
Dựa vào các thông tin công cụ Kali Linux sau đây ('Ngữ cảnh công cụ'), hãy gợi ý các công cụ phù hợp và cung cấp các lệnh mẫu để thực hiện tác vụ pentest của người dùng.
Nếu thông tin từ 'Ngữ cảnh công cụ' không đủ hoặc không liên quan trực tiếp, hãy sử dụng kiến thức chung của bạn về Kali Linux và pentesting để đưa ra gợi ý hợp lý và thực tế. Và lưu ý phải note rằng: "ĐÂY LÀ THÔNG TIN ĐƯỢC GENERATE TỪ LLM (Gemini), không phải từ cơ sở dữ liệu thực tế.".

**QUAN TRỌNG**: Định dạng câu trả lời của bạn bằng cú pháp **HTML** của Telegram. Chỉ sử dụng các thẻ HTML sau, không được thêm bất kỳ thẻ nào khác kể cả `<p>`, `html`, `body`, `head`, `br` và không cần để tag comment `<!-- ....>`. Hay nói cách khác chỉ gồm text và các thẻ HTML sau:
`<b>`, `<strong>` (cho chữ đậm)
`<i>`, `<em>` (cho chữ nghiêng)
`<u>`, `<ins>` (cho chữ gạch chân)
`<s>`, `<strike>`, `<del>` (cho chữ gạch ngang)
`<span class="tg-spoiler">` hoặc `<tg-spoiler>` (cho spoiler/văn bản bị ẩn)
`<a href="URL">text</a>` (cho siêu liên kết)
`<code>text</code>` (cho mã inline)
`<pre>text</pre>` (cho khối mã, có thể lồng `<code>` bên trong: `<pre><code>...</code></pre>`)

- **Khối mã (Code Blocks)**: Sử dụng thẻ `<pre><code>...</code></pre>` để hiển thị các lệnh hoặc ví dụ mã. Bên trong `<code>`, các ký tự `<`, `>`, `&` phải được escape thành `<`, `>`, `&`.
  Ví dụ cho lệnh:
  <pre><code>nmap -sV -p 80,443 example.com</code></pre>
- **Nhấn mạnh**: Sử dụng `<b>text</b>` cho đậm, `<i>text</i>` cho nghiêng, `<u>text</u>` cho gạch chân, `<s>text</s>` cho gạch ngang.
- **Ký tự đặc biệt HTML**: Trong văn bản thông thường (ngoài thẻ `<code>` được đặt trong `<pre>`), các ký tự `<`, `>`, `&` phải được escape thành `<`, `>`, `&`.
- **Danh sách (Lists)**: Vì thẻ `<ul>` và `<li>` không được hỗ trợ chính thức, hãy tạo danh sách bằng cách sử dụng dấu gạch đầu dòng (ví dụ: `-` hoặc `•`) hoặc số, theo sau là văn bản và ký tự <br> để xuống dòng cho mỗi mục.
  Ví dụ tạo danh sách:
  - Mục 1
  - Mục 2
  Hoặc:
  1. Bước một
  2. Bước hai
- **Liên kết (Links)**: Sử dụng `<a href="URL">văn bản hiển thị</a>`.

Ngữ cảnh công cụ:
{context}

Câu hỏi của người dùng: {question}

Câu trả lời (tiếng Việt, định dạng HTML hợp lệ theo các thẻ đã liệt kê):
"""
        rag_prompt = ChatPromptTemplate.from_template(html_template_string)

        self.rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | rag_prompt
            | llm
            | StrOutputParser()
        )
        logger.info(f"[{time.strftime('%H:%M:%S')}] RAG chain in KaliRAGService initialized successfully (HTML mode - restricted tags).")

    async def ask_question(self, query: str) -> str:
        if self.rag_chain is None:
            return _escape_html_internal(
                "Tính năng gợi ý công cụ Kali hiện không khả dụng. "
                "Vui lòng kiểm tra cấu hình bot hoặc thông báo cho admin."
            )
        try:
            response = await self.rag_chain.ainvoke(query)
            return response 
        except Exception as e:
            logger.error(f"Error during RAG chain execution for query '{query}': {e}", exc_info=True)
            error_detail = str(e)[:150] 
            return _escape_html_internal(
                f"Đã xảy ra lỗi khi tìm kiếm gợi ý. Vui lòng thử lại sau. Lỗi: {error_detail}"
            )