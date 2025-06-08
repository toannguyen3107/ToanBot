# telegram_kali_bot/cogs/commands.py

import logging
from telegram import Update
from telegram.ext import ContextTypes

# Import TranslationService và KaliRAGService classes
from cogs.translate import TranslationService
from cogs.kali_rag import KaliRAGService # MỚI

logger = logging.getLogger(__name__)

# Khai báo các biến toàn cục để giữ instance của các Service
# Các biến này sẽ được gán giá trị từ main.py khi bot khởi động.
translation_service_instance: TranslationService = None
kali_rag_service_instance: KaliRAGService = None # MỚI

# --- Các hàm xử lý lệnh Telegram ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_name = update.effective_user.first_name
    await update.message.reply_text(f"Chào mừng {user_name} đến với bot hổ trợ công việc!")

async def hello_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Chào!")

async def ping_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Pong!")

async def translate_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text_to_translate = ' '.join(context.args)
    if not text_to_translate:
        await update.message.reply_text("Bạn cần cung cấp văn bản để thông dịch. Ví dụ: /translate Xin chào thế giới")
        return

    if translation_service_instance is None or translation_service_instance.llm is None:
        await update.message.reply_text("Tính năng thông dịch hiện không khả dụng. Vui lòng kiểm tra cấu hình bot (GOOGLE_API_KEY).")
        logger.warning("TranslationService instance not initialized or LLM is None for translate_command.")
        return

    await update.message.reply_text("Đang thông dịch, vui lòng chờ...")
    
    try:
        translated_text = await translation_service_instance.translate_text(text_to_translate)
        response_message_html = f"Kết quả thông dịch:\n\n<pre>{translated_text}</pre>"
        await update.message.reply_html(response_message_html)
    except Exception as e:
        logger.error(f"Lỗi khi thực hiện thông dịch: {e}", exc_info=True)
        await update.message.reply_text("Đã xảy ra lỗi khi thông dịch văn bản của bạn. Vui lòng thử lại.")

async def ask_kali_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles requests for Kali tool recommendations using RAG."""
    if not context.args:
        await update.message.reply_text("Vui lòng cung cấp câu hỏi. Ví dụ: /ask_kali cách sử dụng nmap để quét port.")
        return

    query = " ".join(context.args)
    await update.message.reply_text(f"Đang tìm kiếm gợi ý cho: '{query}'...")

    # Kiểm tra xem kali_rag_service_instance đã được khởi tạo và có sẵn RAG chain không
    if kali_rag_service_instance is None or kali_rag_service_instance.rag_chain is None:
        await update.message.reply_text("Bot RAG chưa được khởi tạo hoặc không khả dụng. Vui lòng thử lại sau hoặc thông báo cho admin.")
        logger.error("KaliRAGService instance not initialized or RAG chain is None for ask_kali_command.")
        return

    try:
        # Gọi phương thức ask_question từ instance của KaliRAGService
        response = await kali_rag_service_instance.ask_question(query) 
        await update.message.reply_text(response)
    except Exception as e:
        logger.error(f"Lỗi khi gọi Kali RAG service: {e}", exc_info=True)
        await update.message.reply_text("Đã xảy ra lỗi khi xử lý yêu cầu của bạn. Vui lòng thử lại.")

async def echo_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Xử lý các tin nhắn không phải là lệnh. Gợi ý sử dụng lệnh."""
    if update.message.text and (update.message.text.startswith('/ask_kali') or update.message.text.startswith('/translate')):
        return
        
    await update.message.reply_text(
        f"Tôi là bot dịch thuật và gợi ý lệnh pentest. "
        f"Vui lòng sử dụng:\n"
        f"  /translate <văn bản của bạn> để dịch.\n"
        f"  /ask_kali <câu hỏi của bạn> để hỏi về công cụ Kali.\n"
        f"  Hoặc /help để biết thêm."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = (
        "Xin chao! Tôi là bot hỗ trợ pentest.\n"
        "Dưới đây là các lệnh bạn có thể sử dụng:\n\n"
        "/start - Bắt đầu lại cuộc trò chuyện và nhận lời chào mừng.\n"
        "/hello - Lời chào thân thiện.\n"
        "/ping - Kiểm tra xem bot có đang hoạt động không.\n"
        "/translate <văn bản> - Dịch văn bản của bạn (tiếng Việt sang Anh, hoặc sửa ngữ pháp tiếng Anh).\n"
        "/ask_kali <câu hỏi> - Gợi ý công cụ Kali Linux và lệnh pentest.\n"
        "/help - Hiển thị hướng dẫn sử dụng bot.\n\n"
        "Hãy gõ / và chọn lệnh, hoặc gõ trực tiếp lệnh bạn muốn!"
    )
    await update.message.reply_text(help_text)