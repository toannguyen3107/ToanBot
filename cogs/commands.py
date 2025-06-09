# telegram_kali_bot/cogs/commands.py

import logging
from telegram import Update, error as telegram_error # Import telegram_error for specific exception handling
from telegram.ext import ContextTypes
from telegram.constants import ParseMode
import re 

from cogs.translate import TranslationService
from cogs.kali_rag import KaliRAGService

logger = logging.getLogger(__name__)

translation_service_instance: TranslationService = None
kali_rag_service_instance: KaliRAGService = None

def _escape_markdown_v2(text: str) -> str:
    """
    Escapes special characters for Telegram's MarkdownV2 parse_mode,
    avoiding escape inside triple backtick code blocks.
    """
    # THÊM KÝ TỰ '(' VÀO DANH SÁCH KÝ TỰ ĐẶC BIỆT CẦN ESCAPE
    code_block_pattern = r'```(?:[a-zA-Z0-9_]+)?\n(.*?)\n```'
    special_chars = r'_*[]()~`>#+-=|{}.!'  # Đã thêm '(' vào đây
    
    code_blocks = []
    def replace_code_block(match):
        code_blocks.append(match.group(0))
        return f"__CODE_BLOCK_PLACEHOLDER_{len(code_blocks) - 1}__"

    text_with_placeholders = re.sub(code_block_pattern, replace_code_block, text, flags=re.DOTALL)
    escaped_text = text_with_placeholders.replace('\\', '\\\\')
    # SỬA LẠI REGEX ĐỂ ESCAPE TẤT CẢ KÝ TỰ ĐẶC BIỆT
    escaped_text = re.sub(r'([%s])' % re.escape(special_chars), r'\\\1', escaped_text)

    for i, code_block in enumerate(code_blocks):
        escaped_text = escaped_text.replace(f"__CODE_BLOCK_PLACEHOLDER_{i}__", code_block)
    return escaped_text

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_name = update.effective_user.first_name
    await update.message.reply_text(
        f"Chào mừng {_escape_markdown_v2(user_name)} đến với bot hổ trợ công việc\\!", 
        parse_mode=ParseMode.MARKDOWN_V2
    )

async def hello_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Chào\\!", parse_mode=ParseMode.MARKDOWN_V2)

async def ping_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Pong\\!", parse_mode=ParseMode.MARKDOWN_V2)

async def translate_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text_to_translate = ' '.join(context.args)
    if not text_to_translate:
        example_command = _escape_markdown_v2("/translate Xin chào thế giới")
        await update.message.reply_text(
            f"Bạn cần cung cấp văn bản để thông dịch\\. Ví dụ: {example_command}", 
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return

    if translation_service_instance is None or translation_service_instance.llm is None:
        await update.message.reply_text(
            _escape_markdown_v2("Tính năng thông dịch hiện không khả dụng. Vui lòng kiểm tra cấu hình bot (GOOGLE_API_KEY)."),
            parse_mode=ParseMode.MARKDOWN_V2
        )
        logger.warning("TranslationService instance not initialized or LLM is None for translate_command.")
        return

    await update.message.reply_text(
        _escape_markdown_v2("Đang thông dịch, vui lòng chờ..."), 
        parse_mode=ParseMode.MARKDOWN_V2
    )
    
    try:
        translated_text = await translation_service_instance.translate_text(text_to_translate)
        response_message_html = f"Kết quả thông dịch:\n\n<pre>{translated_text}</pre>"
        await update.message.reply_html(response_message_html) 
    except Exception as e:
        logger.error(f"Lỗi khi thực hiện thông dịch: {e}", exc_info=True)
        await update.message.reply_text(
            _escape_markdown_v2("Đã xảy ra lỗi khi thông dịch văn bản của bạn. Vui lòng thử lại."),
            parse_mode=ParseMode.MARKDOWN_V2
        )

async def ask_kali_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        example_command = _escape_markdown_v2("/ask_kali cách sử dụng nmap để quét port.")
        await update.message.reply_text(
            f"Vui lòng cung cấp câu hỏi\\. Ví dụ: {example_command}", 
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return

    query = " ".join(context.args)
    escaped_query_display = _escape_markdown_v2(query)
    await update.message.reply_text(
        f"Đang tìm kiếm gợi ý cho: '{escaped_query_display}'\\.\\.\\.", 
        parse_mode=ParseMode.MARKDOWN_V2
    )

    if kali_rag_service_instance is None or kali_rag_service_instance.rag_chain is None:
        await update.message.reply_text(
            _escape_markdown_v2("Bot RAG chưa được khởi tạo hoặc không khả dụng. Vui lòng thử lại sau hoặc thông báo cho admin."),
            parse_mode=ParseMode.MARKDOWN_V2
        )
        logger.error("KaliRAGService instance not initialized or RAG chain is None for ask_kali_command.")
        return

    response_markdown = "" # Initialize to ensure it's defined for the except block
    try:
        response_markdown = await kali_rag_service_instance.ask_question(query)
        logger.info(f"LLM Raw Response for query '{query}':\n---\n{response_markdown}\n---")
        await update.message.reply_text(response_markdown, parse_mode=ParseMode.MARKDOWN_V2) 
        
    except telegram_error.BadRequest as e_tg_bad:
        logger.error(
            f"Telegram BadRequest sending LLM response. Query: '{query}'. LLM Response was:\n---\n{response_markdown}\n---\nError: {e_tg_bad}", 
            exc_info=True
        )
        user_error_message_key = "Đã xảy ra lỗi khi hiển thị kết quả từ AI do vấn đề định dạng. Vui lòng thử lại hoặc báo cáo lỗi này."
        if "Can't parse entities" in str(e_tg_bad):
            # Lỗi cụ thể này thường do ký tự đặc biệt không được escape đúng
             user_error_message_key = f"Phản hồi từ AI có chứa định dạng không hợp lệ cho Telegram ({str(e_tg_bad)[:60]})."
        
        # Fallback: cố gắng gửi phiên bản đã được escape hoàn toàn
        logger.warning(f"Attempting to send LLM response with full MarkdownV2 escaping as a fallback for: {query}")
        try:
            escaped_response_fallback = _escape_markdown_v2(response_markdown)
            await update.message.reply_text(
                f"{_escape_markdown_v2(user_error_message_key)}\n\n"
                f"Nội dung (đã xử lý định dạng):\n{escaped_response_fallback}",
                parse_mode=ParseMode.MARKDOWN_V2
            )
        except Exception as e_fallback:
            logger.error(f"Error sending fallback escaped message: {e_fallback}", exc_info=True)
            await update.message.reply_text(
                _escape_markdown_v2("Đã có lỗi nghiêm trọng khi xử lý và hiển thị phản hồi từ AI. Vui lòng báo cáo lỗi này."),
                parse_mode=ParseMode.MARKDOWN_V2
            )

    except Exception as e:
        logger.error(f"Lỗi không xác định khi gọi Kali RAG service for query '{query}': {e}", exc_info=True)
        error_detail = str(e)[:100] 
        user_error_message = _escape_markdown_v2(f"Đã xảy ra lỗi khi xử lý yêu cầu của bạn. Vui lòng thử lại.\nChi tiết: {error_detail}")
        await update.message.reply_text(user_error_message, parse_mode=ParseMode.MARKDOWN_V2)

async def echo_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message.text and update.message.text.startswith('/'): 
        return 
        
    echo_reply_text_md = (
        "Tôi là bot dịch thuật và gợi ý lệnh pentest\\. \n"
        "Vui lòng sử dụng:\n"
        "  • `/translate <văn bản của bạn>` để dịch\\.\n"
        "  • `/ask_kali <câu hỏi của bạn>` để hỏi về công cụ Kali\\.\n"
        "  • Hoặc `/help` để biết thêm\\."
    )
    await update.message.reply_text(echo_reply_text_md, parse_mode=ParseMode.MARKDOWN_V2)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text_markdown_v2 = (
        "Xin chào\\! Tôi là bot hỗ trợ pentest\\.\n"
        "Dưới đây là các lệnh bạn có thể sử dụng:\n\n"
        "*Lệnh chung:*\n"
        "  • `/start` \\- Bắt đầu lại cuộc trò chuyện và nhận lời chào mừng\\.\n"
        "  • `/hello` \\- Lời chào thân thiện\\.\n"
        "  • `/ping` \\- Kiểm tra xem bot có đang hoạt động không\\.\n"
        "  • `/help` \\- Hiển thị hướng dẫn sử dụng bot\\.\n\n"
        "*Chức năng chính:*\n"
        "  • `/translate <văn bản>` \\- Dịch văn bản của bạn \\(Việt\\-Anh, Anh\\-Việt, hoặc sửa ngữ pháp tiếng Anh\\)\\.\n"
        "     _Ví dụ: `/translate hello world`_\n"
        "  • `/ask_kali <câu hỏi>` \\- Gợi ý công cụ Kali Linux và lệnh pentest dựa trên mô tả của bạn\\.\n"
        "     _Ví dụ: `/ask_kali làm sao để quét cổng UDP bằng nmap`_\n\n"
        "Hãy gõ `/` và chọn lệnh từ danh sách gợi ý, hoặc gõ trực tiếp lệnh bạn muốn\\!"
    )
    await update.message.reply_text(help_text_markdown_v2, parse_mode=ParseMode.MARKDOWN_V2)