# telegram_kali_bot/cogs/commands.py

import logging
from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ParseMode
import re # Import re module

# Import TranslationService và KaliRAGService classes
from cogs.translate import TranslationService
from cogs.kali_rag import KaliRAGService

logger = logging.getLogger(__name__)

# Khai báo các biến toàn cục để giữ instance của các Service
translation_service_instance: TranslationService = None
kali_rag_service_instance: KaliRAGService = None

# --- Hàm trợ giúp để thoát ký tự cho MarkdownV2 ---
def _escape_markdown_v2(text: str) -> str:
    """
    Escapes special characters for Telegram's MarkdownV2 parse_mode.
    This function strictly escapes all characters that can be interpreted
    as MarkdownV2 formatting, ensuring plain text is sent without parsing errors.
    It intelligently avoids escaping characters inside triple backtick code blocks.
    """
    # Regex to find triple backtick code blocks
    # Allows optional language hint (e.g., ```python, ```json, ```bash, ```text)
    code_block_pattern = r'```(?:[a-zA-Z0-9_]+)?\n(.*?)\n```'
    
    # List of special characters that need escaping in MarkdownV2 outside of code blocks
    # Reference: https://core.telegram.org/bots/api#markdownv2-style
    special_chars = r'_*[]()~`>#+-=|{}.!' # Added . and ! as they are special in some contexts
    
    # Store code blocks and replace them with placeholders
    code_blocks = []
    def replace_code_block(match):
        code_blocks.append(match.group(0))
        return f"__CODE_BLOCK_{len(code_blocks) - 1}__"

    text_with_placeholders = re.sub(code_block_pattern, replace_code_block, text, flags=re.DOTALL)
    
    # Escape special characters in the text outside of code blocks
    # First, escape backslashes themselves
    escaped_text_parts = text_with_placeholders.replace('\\', '\\\\')
    
    # Then escape other special characters
    # Use a lambda function to add a backslash before each matched character
    escaped_text_parts = re.sub(r'([%s])' % re.escape(special_chars), r'\\\1', escaped_text_parts)

    # Restore code blocks
    for i, code_block in enumerate(code_blocks):
        escaped_text_parts = escaped_text_parts.replace(f"__CODE_BLOCK_{i}__", code_block)

    return escaped_text_parts

# --- Các hàm xử lý lệnh Telegram ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_name = update.effective_user.first_name
    # For simple messages like this, MarkdownV2 is not strictly necessary unless you plan to add formatting.
    # If using plain text, ParseMode is not needed, or explicitly use None.
    # For consistency if other messages use MarkdownV2, escape user_name if it could contain special chars.
    await update.message.reply_text(f"Chào mừng {_escape_markdown_v2(user_name)} đến với bot hổ trợ công việc\\!", parse_mode=ParseMode.MARKDOWN_V2)

async def hello_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Chào\\!", parse_mode=ParseMode.MARKDOWN_V2) # Escape '!'

async def ping_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Pong\\!", parse_mode=ParseMode.MARKDOWN_V2) # Escape '!'

async def translate_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text_to_translate = ' '.join(context.args)
    if not text_to_translate:
        # Escape the example command to be safe
        example_command = _escape_markdown_v2("/translate Xin chào thế giới")
        await update.message.reply_text(f"Bạn cần cung cấp văn bản để thông dịch\\. Ví dụ: {example_command}", parse_mode=ParseMode.MARKDOWN_V2)
        return

    if translation_service_instance is None or translation_service_instance.llm is None:
        await update.message.reply_text(
            _escape_markdown_v2("Tính năng thông dịch hiện không khả dụng. Vui lòng kiểm tra cấu hình bot (GOOGLE_API_KEY)."),
            parse_mode=ParseMode.MARKDOWN_V2
        )
        logger.warning("TranslationService instance not initialized or LLM is None for translate_command.")
        return

    await update.message.reply_text(_escape_markdown_v2("Đang thông dịch, vui lòng chờ..."), parse_mode=ParseMode.MARKDOWN_V2)
    
    try:
        translated_text = await translation_service_instance.translate_text(text_to_translate)
        # Using HTML <pre> tag for preformatted text is a good choice for translations as it preserves spacing
        # and avoids Markdown parsing issues for the translated content itself.
        response_message_html = f"Kết quả thông dịch:\n\n<pre>{translated_text}</pre>" # translated_text should be HTML escaped by the library if it's not already.
        await update.message.reply_html(response_message_html) 
    except Exception as e:
        logger.error(f"Lỗi khi thực hiện thông dịch: {e}", exc_info=True)
        await update.message.reply_text(
            _escape_markdown_v2("Đã xảy ra lỗi khi thông dịch văn bản của bạn. Vui lòng thử lại."),
            parse_mode=ParseMode.MARKDOWN_V2
        )

async def ask_kali_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles requests for Kali tool recommendations using RAG."""
    if not context.args:
        example_command = _escape_markdown_v2("/ask_kali cách sử dụng nmap để quét port.")
        await update.message.reply_text(f"Vui lòng cung cấp câu hỏi\\. Ví dụ: {example_command}", parse_mode=ParseMode.MARKDOWN_V2)
        return

    query = " ".join(context.args)
    escaped_query_display = _escape_markdown_v2(query)
    # SỬA LỖI Ở ĐÂY: thoát các dấu chấm trong "..."
    await update.message.reply_text(f"Đang tìm kiếm gợi ý cho: '{escaped_query_display}'\\.\\.\\.", parse_mode=ParseMode.MARKDOWN_V2)

    if kali_rag_service_instance is None or kali_rag_service_instance.rag_chain is None:
        await update.message.reply_text(
            _escape_markdown_v2("Bot RAG chưa được khởi tạo hoặc không khả dụng. Vui lòng thử lại sau hoặc thông báo cho admin."),
            parse_mode=ParseMode.MARKDOWN_V2
        )
        logger.error("KaliRAGService instance not initialized or RAG chain is None for ask_kali_command.")
        return

    try:
        response_markdown = await kali_rag_service_instance.ask_question(query)
        await update.message.reply_text(response_markdown, parse_mode=ParseMode.MARKDOWN_V2) 
        
    except Exception as e:
        logger.error(f"Lỗi khi gọi Kali RAG service: {e}", exc_info=True)
        error_detail = str(e)[:100] 
        user_error_message = _escape_markdown_v2(f"Đã xảy ra lỗi khi xử lý yêu cầu của bạn. Vui lòng thử lại.\nChi tiết: {error_detail}")
        await update.message.reply_text(user_error_message, parse_mode=ParseMode.MARKDOWN_V2)


async def echo_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Xử lý các tin nhắn không phải là lệnh. Gợi ý sử dụng lệnh."""
    if update.message.text:
        if update.message.text.startswith('/'): 
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