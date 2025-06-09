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
import html 

logger = logging.getLogger(__name__)

DATA_FILE = "data/kali_tools_data.json"
CHROMA_DB_DIR = "./chroma_db"

def _escape_html_internal(text: str) -> str:
    return html.escape(str(text))

class KaliRAGService:
    def __init__(self, google_api_key: str):
        self.rag_chain_phase1 = None
        self.llm_chain_phase2 = None
        self.retriever = None
        self.llm = None
        self.google_api_key = google_api_key

        if self.google_api_key:
            try:
                self._initialize_chains() 
                if self.rag_chain_phase1 and self.llm_chain_phase2:
                    logger.info("KaliRAGService initialized successfully with Phase 1 & 2 chains.")
                else:
                    logger.error("KaliRAGService initialization failed for one or both chains. RAG feature may be partially or fully unavailable.")
            except Exception as e:
                logger.error(f"Unhandled exception during KaliRAGService __init__: {e}", exc_info=True)
        else:
            logger.warning("GOOGLE_API_KEY not provided. RAG feature will be unavailable.")

    def _load_and_prepare_data(self, filepath: str) -> list[Document]:
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

    def _initialize_chains(self):
        documents = self._load_and_prepare_data(DATA_FILE)
        if not documents:
            logger.error("RAG Initialization failed: No documents available for RAG.")
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
                return

        if vectorstore is None:
            logger.error(f"[{time.strftime('%H:%M:%S')}] Vectorstore is still None. RAG will be unavailable.")
            return

        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3, google_api_key=self.google_api_key)
        
        html_template_phase1 = """Bạn là một trợ lý tìm kiếm thông tin.
Nhiệm vụ của bạn là trả lời câu hỏi của người dùng DỰA HOÀN TOÀN vào 'Ngữ cảnh công cụ' được cung cấp.
- Nếu 'Ngữ cảnh công cụ' chứa thông tin đủ để trả lời câu hỏi, hãy định dạng câu trả lời bằng HTML của Telegram theo các quy tắc sau:
    - Chỉ sử dụng các thẻ: <b>, <strong>, <i>, <em>, <u>, <ins>, <s>, <strike>, <del>, <span class="tg-spoiler">, <tg-spoiler>, <a href="URL">, <code>, <pre>.
    - KHÔNG dùng: <p>, <div>, <ul>, <li>, <br>, <html>, <head>, <body>.
    - Escape các ký tự HTML đặc biệt (&, <, >) trong văn bản và trong <code> (nếu không nằm trong <pre>).
    - Dùng \n cho xuống dòng, \n\n cho đoạn văn.
- Nếu 'Ngữ cảnh công cụ' KHÔNG chứa thông tin liên quan hoặc KHÔNG đủ để trả lời, hoặc bạn không chắc chắn, hãy trả về CHÍNH XÁC chuỗi: [NO_CONTEXT_DATA_FOUND]
- KHÔNG thêm bất kỳ thông tin nào khác ngoài chuỗi đó nếu không có context. KHÔNG giải thích, KHÔNG xin lỗi.

Ngữ cảnh công cụ:
{context}

Câu hỏi của người dùng: {question}

Câu trả lời (HTML hoặc chuỗi [NO_CONTEXT_DATA_FOUND]):
"""
        prompt_phase1 = ChatPromptTemplate.from_template(html_template_phase1)
        
        self.rag_chain_phase1 = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt_phase1
            | self.llm
            | StrOutputParser()
        )
        logger.info("RAG Chain Phase 1 initialized.")

        html_template_phase2 = """Bạn là một chuyên gia pentesting trợ giúp, cung cấp câu trả lời bằng tiếng Việt dựa trên kiến thức chung của bạn.
Hãy trả lời câu hỏi của người dùng một cách chi tiết và hữu ích.

**YÊU CẦU ĐỊNH DẠNG HTML NGHIÊM NGẶT CHO TELEGRAM:**
1.  **Chỉ sử dụng các thẻ HTML sau**: `<b>` (hoặc `<strong>`), `<i>` (hoặc `<em>`), `<u>` (hoặc `<ins>`), `<s>` (hoặc `<strike>`, `<del>`), `<span class="tg-spoiler">` (hoặc `<tg-spoiler>`), `<a href="URL">`, `<code>`, `<pre>`.
2.  **TUYỆT ĐỐI KHÔNG SỬ DỤNG**: các thẻ như `<html>`, `<head>`, `<body>`, `<p>`, `<div>`, `<ul>`, `<li>`, `<br>`, hoặc bất kỳ thẻ HTML nào khác không được liệt kê ở mục 1.
3.  **Không bao gồm các comment HTML** (`<!-- ... -->`).
4.  Toàn bộ phản hồi phải là một đoạn HTML hợp lệ, chỉ chứa văn bản và các thẻ được phép.
5.  **Escape ký tự HTML**: Trong văn bản thông thường (bên ngoài `<code>` trong `<pre>`), các ký tự `<`, `>`, `&` **BẮT BUỘC** phải được escape thành `<`, `>`, `&`. Bên trong `<code>` (khi nằm trong `<pre>`), việc escape các ký tự này cũng được khuyến khích để đảm bảo an toàn.
6.  **Khối mã**: Sử dụng `<pre><code>...</code></pre>`. Ví dụ: <pre><code>nmap -sV example.com</code></pre>
7.  **Mã inline**: Sử dụng `<code>tên_lệnh</code>`.
8.  **Danh sách**: Dùng dấu gạch đầu dòng (`- ` hoặc `• `) hoặc số (`1. `) ở đầu mỗi mục, sau đó là văn bản. Kết thúc mỗi mục bằng một ký tự xuống dòng (`\n`).
    Ví dụ:
    - Mục một
    - Mục hai
9.  **Đoạn văn**: Tách các đoạn văn bằng một dòng trống (hai ký tự `\n\n`).
10. **BẮT BUỘC bao gồm ghi chú sau ở cuối câu trả lời của bạn, định dạng bằng thẻ <i>**: "<i>ĐÂY LÀ THÔNG TIN ĐƯỢC GENERATE TỪ LLM (Gemini), không phải từ cơ sở dữ liệu thực tế, vui lòng kiểm chứng thông tin.</i>"

Câu hỏi của người dùng: {question}

Câu trả lời (tiếng Việt, định dạng HTML hợp lệ theo các hướng dẫn, và có ghi chú ở cuối):
"""
        prompt_phase2 = ChatPromptTemplate.from_template(html_template_phase2)
        
        self.llm_chain_phase2 = (
            prompt_phase2 
            | self.llm
            | StrOutputParser()
        )
        logger.info("LLM Chain Phase 2 initialized.")


    async def ask_question(self, query: str) -> str:
        no_context_marker = "[NO_CONTEXT_DATA_FOUND]"

        if not self.rag_chain_phase1:
            return _escape_html_internal("Lỗi: RAG Chain Pha 1 chưa được khởi tạo.")

        logger.info(f"Phase 1 RAG: Querying for '{query}'")
        response_phase1 = await self.rag_chain_phase1.ainvoke(query)
        response_phase1 = response_phase1.strip()
        
        # SỬA LỖI F-STRING Ở ĐÂY
        log_response_preview = response_phase1[:200].replace('\n', ' ')
        logger.info(f"Phase 1 RAG: Response: '{log_response_preview}...'")

        if response_phase1 != no_context_marker:
            logger.info("Phase 1 RAG: Answer found in context.")
            return response_phase1
        else:
            logger.info("Phase 1 RAG: No context found. Proceeding to Phase 2 (LLM only).")
            if not self.llm_chain_phase2:
                return _escape_html_internal("Lỗi: LLM Chain Pha 2 chưa được khởi tạo.")
            
            response_phase2 = await self.llm_chain_phase2.ainvoke({"question": query})
            return response_phase2.strip()