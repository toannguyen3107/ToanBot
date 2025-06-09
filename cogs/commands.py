# telegram_kali_bot/cogs/commands.py
import logging
from telegram import Update, error as telegram_error 
from telegram.ext import ContextTypes
from telegram.constants import ParseMode
import re 
import html 
import bleach
from cogs.translate import TranslationService
from cogs.kali_rag import KaliRAGService

logger = logging.getLogger(__name__)

translation_service_instance: TranslationService = None
kali_rag_service_instance: KaliRAGService = None

def _escape_html(text: str, escape_quotes: bool = True) -> str:
    return html.escape(str(text), quote=escape_quotes)

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_name = update.effective_user.first_name
    await update.message.reply_text(
        f"Chào mừng <b>{_escape_html(user_name)}</b> đến với bot hổ trợ công việc!", 
        parse_mode=ParseMode.HTML
    )

async def hello_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Chào!", parse_mode=ParseMode.HTML)

async def ping_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Pong!", parse_mode=ParseMode.HTML)

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
        example_command_html = f"<code>/ask_kali <câu hỏi của bạn></code>" # Đã sửa placeholder
        await update.message.reply_text(
            f"Vui lòng cung cấp câu hỏi. Ví dụ: {example_command_html}", 
            parse_mode=ParseMode.HTML
        )
        return

    query = " ".join(context.args)
    await update.message.reply_text(
        f"Đang tìm kiếm gợi ý cho: <i>{_escape_html(query)}</i>...", 
        parse_mode=ParseMode.HTML
    )

    # SỬA LỖI KIỂM TRA Ở ĐÂY
    if kali_rag_service_instance is None or \
       kali_rag_service_instance.rag_chain_phase1 is None or \
       kali_rag_service_instance.llm_chain_phase2 is None:
        await update.message.reply_text(
            _escape_html("Bot RAG chưa được khởi tạo đúng cách hoặc không khả dụng. Vui lòng thử lại sau hoặc thông báo cho admin."),
            parse_mode=ParseMode.HTML
        )
        logger.error("KaliRAGService instance or its chains (Phase 1/2) are not initialized for ask_kali_command.")
        return

    raw_response_html = "" 
    cleaned_html = ""
    try:
        raw_response_html = await kali_rag_service_instance.ask_question(query)
        raw_response_html = raw_response_html.strip()
        logger.info(f"LLM Raw HTML (before bleaching) for query '{_escape_html(query)}':\n---\n{raw_response_html}\n---")

        ALLOWED_TAGS = ['b', 'strong', 'i', 'em', 'u', 'ins', 's', 'strike', 'del', 
                        'span', 'tg-spoiler', 'a', 'code', 'pre']
        ALLOWED_ATTRIBUTES = {
            'a': ['href'],
            'span': ['class'], 
            'tg-spoiler': [] 
        }
        
        cleaned_html = bleach.clean(raw_response_html,
                                    tags=ALLOWED_TAGS,
                                    attributes=ALLOWED_ATTRIBUTES,
                                    strip=True, 
                                    strip_comments=True)
        
        cleaned_html = re.sub(r'<br\s*/?>', '\n', cleaned_html, flags=re.IGNORECASE)
        cleaned_html = re.sub(r'<p\s*[^>]*>', '', cleaned_html, flags=re.IGNORECASE)
        cleaned_html = re.sub(r'</p\s*>', '\n\n', cleaned_html, flags=re.IGNORECASE)
        cleaned_html = cleaned_html.strip()

        if cleaned_html != raw_response_html: # Chỉ log nếu có sự thay đổi
             logger.info(f"LLM HTML Response (after bleaching and cleaning) for query '{_escape_html(query)}':\n---\n{cleaned_html}\n---")
        
        await update.message.reply_text(cleaned_html, parse_mode=ParseMode.HTML) 
        
    except telegram_error.BadRequest as e_tg_bad:
        logger.error(
            f"Telegram BadRequest sending LLM HTML response. Query: '{_escape_html(query)}'. "
            f"Raw HTML was:\n---\n{raw_response_html}\n---\nCleaned HTML was:\n---\n{cleaned_html}\n---\nError: {e_tg_bad}", 
            exc_info=True
        )
        try:
            plain_text_from_cleaned = re.sub(r'<[^>]+>', '', cleaned_html if cleaned_html else raw_response_html)
            plain_text_from_cleaned = html.unescape(plain_text_from_cleaned).strip()

            if plain_text_from_cleaned:
                await update.message.reply_text(
                    f"Lỗi hiển thị định dạng HTML từ AI. Nội dung thuần:\n{_escape_html(plain_text_from_cleaned)}",
                    parse_mode=ParseMode.HTML
                )
            else:
                await update.message.reply_text(
                     _escape_html(f"Đã xảy ra lỗi khi hiển thị kết quả từ AI. Chi tiết kỹ thuật: {str(e_tg_bad)[:80]}"),
                     parse_mode=ParseMode.HTML
                )
        except Exception as e_final_fallback:
            logger.error(f"Error sending plain text fallback: {e_final_fallback}", exc_info=True)
            await update.message.reply_text(
                _escape_html("Đã có lỗi nghiêm trọng khi hiển thị phản hồi từ AI. Vui lòng thử lại sau."),
                parse_mode=ParseMode.HTML
            )

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
        "  • <code>/translate STMT</code> để dịch.\n"
        "  • <code>/ask_kali QUESTION</code> để hỏi về công cụ Kali.\n"
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
  • <code>/translate STMT</code> - Dịch văn bản của bạn (Việt-Anh, Anh-Việt, hoặc sửa ngữ pháp tiếng Anh).
     <i>Ví dụ: <code>/translate hello world</code></i>
  • <code>/ask_kali QUESTION</code> - Gợi ý công cụ Kali Linux và lệnh pentest dựa trên mô tả của bạn.
     <i>Ví dụ: <code>/ask_kali làm sao để quét cổng UDP bằng nmap</code></i>

Hãy gõ <code>/</code> và chọn lệnh từ danh sách gợi ý, hoặc gõ trực tiếp lệnh bạn muốn!
"""
    await update.message.reply_text(help_text_html, parse_mode=ParseMode.HTML)