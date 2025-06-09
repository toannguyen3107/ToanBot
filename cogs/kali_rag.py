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
            vectorstore = None 

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
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2, google_api_key=self.google_api_key) # Slightly lower temp
        
        html_template_phase1 = """Bạn là một trợ lý tìm kiếm thông tin.
Nhiệm vụ của bạn là trả lời câu hỏi của người dùng DỰA HOÀN TOÀN vào 'Ngữ cảnh công cụ' được cung cấp.
- Nếu 'Ngữ cảnh công cụ' chứa thông tin đủ để trả lời câu hỏi, hãy định dạng câu trả lời bằng HTML của Telegram theo các quy tắc sau:
    - Chỉ sử dụng các thẻ: <b>, <strong>, <i>, <em>, <u>, <ins>, <s>, <strike>, <del>, <span class="tg-spoiler">, <tg-spoiler>, <a href="URL">, <code>, <pre>.
    - KHÔNG dùng: <p>, <div>, <ul>, <li>, <br>, <html>, <head>, <body>.
    - QUAN TRỌNG: Bên trong thẻ <code> và <pre>, các ký tự '<', '>', '&' PHẢI được escape thành '<', '>', '&'. Ví dụ: <pre><code>ls -l & echo \"Done\"</code></pre> hoặc <code><command></code>.
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
Nhiệm vụ của bạn là trả lời câu hỏi của người dùng một cách chi tiết và hữu ích.

**YÊU CẦU ĐỊNH DẠNG HTML NGHIÊM NGẶT CHO TELEGRAM:**
1.  **TOÀN BỘ PHẢN HỒI PHẢI LÀ HTML.** Không được có văn bản thuần túy không nằm trong thẻ nào (trừ khi đó là nội dung của một thẻ).
2.  **Chỉ sử dụng các thẻ HTML sau**: `<b>` (hoặc `<strong>`), `<i>` (hoặc `<em>`), `<u>` (hoặc `<ins>`), `<s>` (hoặc `<strike>`, `<del>`), `<span class="tg-spoiler">` (hoặc `<tg-spoiler>`), `<a href="URL">`, `<code>`, `<pre>`.
3.  **TUYỆT ĐỐI KHÔNG SỬ DỤNG**: các thẻ như `<html>`, `<head>`, `<body>`, `<p>`, `<div>`, `<ul>`, `<li>`, `<br>`, hoặc bất kỳ thẻ HTML nào khác không được liệt kê ở mục 2.
4.  **Không bao gồm các comment HTML** (`<!-- ... -->`).
5.  **QUAN TRỌNG - Escape ký tự HTML**:
    - **Bên trong nội dung của thẻ `<code>` và thẻ `<pre>` (bao gồm cả `<code>` bên trong `<pre>`)**: Bất kỳ ký tự nào là `<` PHẢI được thay thế bằng `<`, ký tự `>` PHẢI được thay thế bằng `>`, và ký tự `&` PHẢI được thay thế bằng `&`. Điều này CỰC KỲ QUAN TRỌNG để các lệnh mẫu và placeholder như `<target>` hoặc `<ip_address>` được hiển thị chính xác dưới dạng văn bản, không phải là thẻ HTML.
      Ví dụ đúng: `<code>nmap -p 1-100 <target_ip></code>`
      Ví dụ đúng: `<pre><code>ping -c 4 <hostname>\n# Lệnh trên ping 4 lần tới <hostname> & hiển thị kết quả.</code></pre>`
      Ví dụ SAI: `<code>nmap -p 1-100 <target_ip></code>` (sẽ gây lỗi)
    - **Trong văn bản thông thường (bên ngoài `<code>` và `<pre>`)**: Các ký tự `<`, `>`, `&` cũng PHẢI được escape tương ứng thành `<`, `>`, `&`.
6.  **Khối mã**: Sử dụng `<pre><code>...nội dung mã đã được escape THEO QUY TẮC Ở MỤC 5...</code></pre>`. Nội dung bên trong `<pre><code>` sẽ giữ nguyên định dạng (preserve whitespace and newlines).
    Ví dụ: <pre><code>nmap -sV <target_ip>\n# Quét phiên bản dịch vụ cho <target_ip></code></pre>
7.  **Mã inline**: Sử dụng `<code>tên_lệnh_đã_escape_THEO_MỤC_5</code>`. Ví dụ: Sử dụng lệnh `<code>nmap</code>`. Nếu có placeholder: `<code><tool_name> --help</code>`.
8.  **Tạo danh sách hoặc mục**:
    - Vì thẻ `<ul>` và `<li>` không được phép, hãy tạo danh sách bằng cách bắt đầu mỗi mục bằng một ký tự như `• ` (hoặc `- `, `* `) hoặc số (`1. `), theo sau là văn bản.
    - Mỗi mục danh sách nên là một dòng riêng, kết thúc bằng `\n`.
    - Các thẻ định dạng như `<b>`, `<i>`, `<code>` có thể được sử dụng bên trong văn bản của mục (nội dung của `<code>` vẫn phải tuân theo quy tắc escape ở mục 5).
    Ví dụ:
    • Mục <b>quan trọng</b> đầu tiên với lệnh `<code>apt update && apt upgrade</code>`.\n
    • Mục thứ hai với <i>chi tiết</i>.\n
9.  **Đoạn văn và xuống dòng**:
    - Tách các đoạn văn bằng một dòng trống (hai ký tự `\n\n`).
    - Sử dụng một ký tự `\n` để xuống dòng đơn.
10. **Ví dụ về phản hồi HTML hoàn chỉnh (TUÂN THỦ MỌI QUY TẮC TRÊN)**:
    Giả sử câu hỏi là "cách dùng nmap quét cổng". Phản hồi có thể là:
    <b>Nmap (Network Mapper)</b> là một công cụ mạnh mẽ để quét mạng.\n\nĐể quét các cổng phổ biến trên một mục tiêu, bạn có thể dùng:\n<pre><code>nmap -sV <địa_chỉ_target></code></pre>\nTrong đó:\n• <code>-sV</code>: Dùng để phát hiện phiên bản dịch vụ.\n• <code><địa_chỉ_target></code>: Là mục tiêu của bạn (IP hoặc hostname đã được escape).\n\nBạn có thể tìm hiểu thêm tại <a href="https://nmap.org">trang chủ Nmap</a>.\n\n<i>ĐÂY LÀ THÔNG TIN ĐƯỢC GENERATE TỪ LLM (Gemini), không phải từ cơ sở dữ liệu thực tế, vui lòng kiểm chứng thông tin.</i>

11. **BẮT BUỘC bao gồm ghi chú sau ở cuối câu trả lời của bạn, định dạng chính xác bằng thẻ <i> như ví dụ trên**: "<i>ĐÂY LÀ THÔNG TIN ĐƯỢC GENERATE TỪ LLM (Gemini), không phải từ cơ sở dữ liệu thực tế, vui lòng kiểm chứng thông tin.</i>"

Câu hỏi của người dùng: {question}

Câu trả lời (TUYỆT ĐỐI LÀ HTML tiếng Việt, tuân thủ MỌI quy tắc trên, đặc biệt là quy tắc ESCAPE ký tự trong `<code>` và `<pre>`, và có ghi chú ở cuối):
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
            logger.error("RAG Chain Phase 1 is not initialized in ask_question.")
            return _escape_html_internal("Lỗi: RAG Chain Pha 1 chưa được khởi tạo.")

        logger.info(f"Phase 1 RAG: Querying for '{_escape_html_internal(query)}'")
        response_phase1 = await self.rag_chain_phase1.ainvoke(query)
        response_phase1 = response_phase1.strip()
        
        log_response_preview_p1 = response_phase1.replace('\n', ' ')[:200]
        logger.info(f"Phase 1 RAG: Response: '{log_response_preview_p1}...'") 

        if response_phase1 != no_context_marker:
            logger.info("Phase 1 RAG: Answer found in context.")
            return response_phase1
        else:
            logger.info("Phase 1 RAG: No context found. Proceeding to Phase 2 (LLM only).")
            if not self.llm_chain_phase2:
                logger.error("LLM Chain Phase 2 is not initialized in ask_question.")
                return _escape_html_internal("Lỗi: LLM Chain Pha 2 chưa được khởi tạo.")
            
            logger.info(f"Phase 2 LLM: Querying for '{_escape_html_internal(query)}'") 
            response_phase2 = await self.llm_chain_phase2.ainvoke({"question": query})
            response_phase2_stripped = response_phase2.strip()
            
            log_response_preview_p2 = response_phase2_stripped.replace('\n', ' ')[:300]
            logger.info(f"Phase 2 LLM: Raw Response: '{log_response_preview_p2}...'") 
            
            return response_phase2_stripped