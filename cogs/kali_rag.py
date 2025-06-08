# telegram_kali_bot/cogs/kali_rag.py

import logging
import json
import os
import time
import shutil # Import shutil for rmtree
from bs4 import BeautifulSoup
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

        # ĐIỀU CHỈNH PROMPT ĐỂ YÊU CẦU HTML RẤT CƠ BẢN VÀ KHÔNG CHỨA CÁC TAG DOCUMENT-LEVEL
        rag_prompt = ChatPromptTemplate.from_template("""
        Bạn là một chuyên gia pentesting trợ giúp. 
        Dựa vào các thông tin công cụ Kali Linux sau đây, hãy gợi ý các công cụ phù hợp và cung cấp các lệnh mẫu để thực hiện tác vụ pentest của người dùng.
        Nếu thông tin từ 'Ngữ cảnh công cụ' không đủ hoặc không liên quan trực tiếp, hãy sử dụng kiến thức chung của bạn về Kali Linux và pentesting để đưa ra gợi ý hợp lý và thực tế.
        
        **QUAN TRỌNG**: Định dạng câu trả lời của bạn bằng các thẻ **HTML NỘI DUNG** mà Telegram Bot API hỗ trợ.
        - **KHÔNG** bao gồm bất kỳ tag document-level nào như `<!DOCTYPE html>`, `<html>`, `<head>`, `<body>`.
        - Sử dụng thẻ `<pre><code>` để bọc các lệnh và ví dụ code.
        - Sử dụng thẻ `<b>` để bôi đậm các tên công cụ hoặc từ khóa quan trọng.
        - Các ký tự đặc biệt của HTML như `<`, `>`, `&` trong văn bản bình thường (không phải trong code) phải được thoát thành `<`, `>`, `&`.
        - Đảm bảo toàn bộ phản hồi là một đoạn HTML hợp lệ và đơn giản, không có các tag không được Telegram hỗ trợ.
        
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
            # THÊM LỚP BẢO VỆ: Loại bỏ các thẻ HTML không hợp lệ nếu LLM vẫn tạo ra
            # Sử dụng BeautifulSoup để làm sạch HTML không hợp lệ mà Telegram không hỗ trợ
            cleaned_response = _strip_unsupported_html_tags(response)
            return cleaned_response
        except Exception as e:
            logger.error(f"Error during RAG chain execution: {e}", exc_info=True)
            return f"Đã xảy ra lỗi khi tìm kiếm gợi ý: {e}"

# --- Hàm trợ giúp để loại bỏ các thẻ HTML không được Telegram hỗ trợ ---
# (Thêm vào cuối file kali_rag.py)
def _strip_unsupported_html_tags(html_string: str) -> str:
    """
    Strips HTML tags that are not supported by Telegram Bot API for HTML parse mode.
    Supported tags: <b>, <i>, <u>, <s>, <code>, <pre>, <a href="...">, <tg-spoiler>
    """
    supported_tags = ['b', 'i', 'u', 's', 'code', 'pre', 'a', 'tg-spoiler']
    
    soup = BeautifulSoup(html_string, 'html.parser')
    
    for tag in soup.find_all(True): # Iterate over all tags
        if tag.name not in supported_tags:
            tag.unwrap() # Remove the tag, but keep its content
        elif tag.name == 'a' and not tag.get('href'): # Ensure <a> tags have href
            tag.unwrap()
        elif tag.name == 'pre' and not tag.find('code'): # Ensure <pre> contains <code>
            tag.unwrap() # Or handle differently based on desired behavior
        elif tag.name == 'code' and not tag.find_parent('pre'): # Ensure <code> is inside <pre> for block
            # For inline code, you might want to convert ` to backticks or escape it.
            # For now, if code is not in pre, just unwrap it.
            tag.unwrap()


    # After unwrapping unsupported tags, convert back to string
    # Replace non-breaking spaces with regular spaces as they can sometimes cause issues
    cleaned_html = str(soup).replace('\xa0', ' ')
    
    # Remove DOCTYPE, html, head, body tags if they somehow get generated
    cleaned_html = re.sub(r'<!DOCTYPE html[^>]*>', '', cleaned_html, flags=re.IGNORECASE).strip()
    cleaned_html = re.sub(r'<\/?html[^>]*>', '', cleaned_html, flags=re.IGNORECASE).strip()
    cleaned_html = re.sub(r'<\/?head[^>]*>', '', cleaned_html, flags=re.IGNORECASE).strip()
    cleaned_html = re.sub(r'<\/?body[^>]*>', '', cleaned_html, flags=re.IGNORECASE).strip()
    
    return cleaned_html