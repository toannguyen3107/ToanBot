# telegram_kali_bot/cogs/commands.py

import logging
from telegram import Update, error as telegram_error 
from telegram.ext import ContextTypes
from telegram.constants import ParseMode
import re 
import html # Để escape HTML

from cogs.translate import TranslationService
from cogs.kali_rag import KaliRAGService

logger = logging.getLogger(__name__)

translation_service_instance: TranslationService = None
kali_rag_service_instance: KaliRAGService = None

# Hàm trợ giúp để escape ký tự đặc biệt cho HTML
def _escape_html(text: str, escape_quotes: bool = True) -> str:
    """Escapes special characters for Telegram's HTML parse_mode."""
    # Loại bỏ các thẻ <p> và thay </p> bằng xuống dòng
    text = re.sub(r'<p\s*>', '', text)
    text = re.sub(r'</p\s*>', '\n', text)
    return html.escape(str(text), quote=escape_quotes)



async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_name = update.effective_user.first_name
    await update.message.reply_text(
        f"Chào mừng <b>{_escape_html(user_name)}</b> đến với bot hổ trợ công việc!", 
        parse_mode=ParseMode.HTML
    )

async def hello_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Chào!", parse_mode=ParseMode.HTML) # HTML đơn giản

async def ping_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Pong!", parse_mode=ParseMode.HTML) # HTML đơn giản

async def translate_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text_to_translate = ' '.join(context.args)
    if not text_to_translate:
        example_command_html = f"<code>/translate Xin chào thế giới</code>"
        await update.message.reply_text(
            f"Bạn cần cung cấp văn bản để thông dịch. Ví dụ: {example_command_html}", 
            parse_mode=ParseMode.HTML
        )
        return

    if translation_service_instance is None or translation_service_instance.llm is None:
        await update.message.reply_text(
            _escape_html("Tính năng thông dịch hiện không khả dụng. Vui lòng kiểm tra cấu hình bot (GOOGLE_API_KEY)."),
            parse_mode=ParseMode.HTML
        )
        logger.warning("TranslationService instance not initialized or LLM is None for translate_command.")
        return

    await update.message.reply_text(
        _escape_html("Đang thông dịch, vui lòng chờ..."), 
        parse_mode=ParseMode.HTML
    )
    
    try:
        translated_text = await translation_service_instance.translate_text(text_to_translate)
        # Kết quả dịch đã an toàn để hiển thị trong <pre>
        response_message_html = f"Kết quả thông dịch:\n\n<pre>{_escape_html(translated_text)}</pre>"
        await update.message.reply_html(response_message_html) 
    except Exception as e:
        logger.error(f"Lỗi khi thực hiện thông dịch: {e}", exc_info=True)
        await update.message.reply_text(
            _escape_html("Đã xảy ra lỗi khi thông dịch văn bản của bạn. Vui lòng thử lại."),
            parse_mode=ParseMode.HTML
        )

async def ask_kali_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        example_command_html = f"<code>/ask_kali cách sử dụng nmap để quét port.</code>"
        await update.message.reply_text(
            f"Vui lòng cung cấp câu hỏi. Ví dụ: {example_command_html}", 
            parse_mode=ParseMode.HTML
        )
        return

    query = " ".join(context.args)
    await update.message.reply_text(
        f"Đang tìm kiếm gợi ý cho: <i>{_escape_html(query)}</i>...", 
        parse_mode=ParseMode.HTML # Đã đổi sang HTML
    )

    if kali_rag_service_instance is None or kali_rag_service_instance.rag_chain is None:
        await update.message.reply_text(
            _escape_html("Bot RAG chưa được khởi tạo hoặc không khả dụng. Vui lòng thử lại sau hoặc thông báo cho admin."),
            parse_mode=ParseMode.HTML
        )
        logger.error("KaliRAGService instance not initialized or RAG chain is None for ask_kali_command.")
        return

    response_html = "" 
    try:
        # LLM giờ sẽ trả về HTML
        response_html = await kali_rag_service_instance.ask_question(query)
        logger.info(f"LLM Raw HTML Response for query '{_escape_html(query)}':\n---\n{response_html}\n---")
        if "<br>" in response_html:
            response_html = response_html.replace("<br>", "\n")
        await update.message.reply_text(response_html, parse_mode=ParseMode.HTML) 
        
    except telegram_error.BadRequest as e_tg_bad:
        logger.error(
            f"Telegram BadRequest sending LLM HTML response. Query: '{_escape_html(query)}'. LLM Response was:\n---\n{response_html}\n---\nError: {e_tg_bad}", 
            exc_info=True
        )
        # Nếu LLM vẫn tạo HTML không hợp lệ, thông báo lỗi chung
        user_error_message = _escape_html(f"Đã xảy ra lỗi khi hiển thị kết quả từ AI do vấn đề định dạng HTML.\nChi tiết kỹ thuật: {str(response_html)[:100]}")
        await update.message.reply_text(user_error_message, parse_mode=ParseMode.HTML)

    except Exception as e:
        logger.error(f"Lỗi không xác định khi gọi Kali RAG service for query '{_escape_html(query)}': {e}", exc_info=True)
        error_detail = str(e)[:100] 
        user_error_message = _escape_html(f"Đã xảy ra lỗi khi xử lý yêu cầu của bạn. Vui lòng thử lại.\nChi tiết: {error_detail}")
        await update.message.reply_text(user_error_message, parse_mode=ParseMode.HTML)

async def echo_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message.text and update.message.text.startswith('/'): 
        return 
        
    echo_reply_text_html = (
        "Tôi là bot dịch thuật và gợi ý lệnh pentest.\n"
        "Vui lòng sử dụng:\n"
        "  • <code>/translate &lt;văn bản của bạn&gt;</code> để dịch.\n"
        "  • <code>/ask_kali &lt;câu hỏi của bạn&gt;</code> để hỏi về công cụ Kali.\n"
        "  • Hoặc <code>/help</code> để biết thêm."
    )
    await update.message.reply_text(echo_reply_text_html, parse_mode=ParseMode.HTML)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text_html = """Xin chào! Tôi là bot hỗ trợ pentest.
Dưới đây là các lệnh bạn có thể sử dụng:

<b>Lệnh chung:</b>
  • <code>/start</code> - Bắt đầu lại cuộc trò chuyện và nhận lời chào mừng.
  • <code>/hello</code> - Lời chào thân thiện.
  • <code>/ping</code> - Kiểm tra xem bot có đang hoạt động không.
  • <code>/help</code> - Hiển thị hướng dẫn sử dụng bot.

<b>Chức năng chính:</b>
  • <code>/translate &lt;văn bản&gt;</code> - Dịch văn bản của bạn (Việt-Anh, Anh-Việt, hoặc sửa ngữ pháp tiếng Anh).
     <i>Ví dụ: <code>/translate hello world</code></i>
  • <code>/ask_kali &lt;câu hỏi&gt;</code> - Gợi ý công cụ Kali Linux và lệnh pentest dựa trên mô tả của bạn.
     <i>Ví dụ: <code>/ask_kali làm sao để quét cổng UDP bằng nmap</code></i>

Hãy gõ <code>/</code> và chọn lệnh từ danh sách gợi ý, hoặc gõ trực tiếp lệnh bạn muốn!
"""
    await update.message.reply_text(help_text_html, parse_mode=ParseMode.HTML)