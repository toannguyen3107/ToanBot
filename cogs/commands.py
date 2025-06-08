# telegram_kali_bot/cogs/commands.py

import logging
from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ParseMode
import re # Import regex module

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
    This function specifically handles characters that are NOT inside code blocks
    (which are expected to be handled by the LLM with triple backticks).
    It ensures that characters like '.', '(', ')', etc., in regular text are escaped.
    """
    # Characters that must be escaped in MarkdownV2 if not part of a formatting entity:
    # `_ * [ ] ( ) ~ ` > # + - = | { } . !`
    
    # First, escape the backslash itself to prevent double escaping or issues.
    text = text.replace('\\', '\\\\')

    # Use a regex to replace other special characters, but be careful with code blocks.
    # This regex attempts to find and escape special characters that are *not*
    # part of MarkdownV2 code blocks (```) or inline code (`).
    
    # This is a common and robust pattern to escape MarkdownV2,
    # assuming the LLM correctly generates triple backticks for code blocks.
    # It will escape: _, *, [, ], (, ), ~, `, >, #, +, -, =, |, {, }, ., !
    
    # A simplified version that is generally safe and addresses common errors:
    # Escape all characters in the list that are NOT part of ```` blocks.
    # This requires more complex logic.
    # For now, let's use a simpler, more direct approach for common offenders.

    # Strict escaping as per PTB documentation:
    # https://github.com/python-telegram-bot/python-telegram-bot/wiki/Working-with-Messages#escaping-v2-style
    # This will escape *all* occurrences, even if the LLM tried to bold/italicize.
    # That's why we told the LLM NOT to use those.

    escape_chars = r'_*[]()~`>#+-=|{}.!'
    # Use re.sub with a replacement function to conditionally escape.
    # This pattern will match a triple backtick code block or an inline code block.
    # Everything else will be processed for escaping.
    
    # This regex is a simplified approach to find and escape.
    # If the LLM generates proper ```` blocks, content inside should be safe.
    # We focus on escaping the *plain text* that comes out of the LLM.
    
    # The actual implementation of strict escaping for MarkdownV2 is usually complex
    # if you want to preserve *all* valid Markdown.
    # For LLM output where we only expect ```` for code, and plain text otherwise:
    
    # The simplest way to apply the strict escaping for *all* characters
    # (assuming LLM will NOT produce `*` for bold etc. as per new prompt instructions):
    
    text = re.sub(r'([_*\[\]()~`>#+\-=|{}.!])', r'\\\1', text)
    return text

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
        # reply_html đã ngụ ý ParseMode.HTML, và TranslationService đã đảm bảo output HTML
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

    if kali_rag_service_instance is None or kali_rag_service_instance.rag_chain is None:
        await update.message.reply_text("Bot RAG chưa được khởi tạo hoặc không khả dụng. Vui lòng thử lại sau hoặc thông báo cho admin.")
        logger.error("KaliRAGService instance not initialized or RAG chain is None for ask_kali_command.")
        return

    try:
        response = await kali_rag_service_instance.ask_question(query)
        # Áp dụng hàm thoát ký tự cho phản hồi từ RAG chain
        # Sau đó gửi với ParseMode.MARKDOWN_V2
        escaped_response = _escape_markdown_v2(response)
        await update.message.reply_text(escaped_response, parse_mode=ParseMode.MARKDOWN_V2) 
        
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
        "\\*/start\\* \\- Bắt đầu lại cuộc trò chuyện và nhận lời chào mừng\\.\n" # Escaped for MarkdownV2
        "\\*/hello\\* \\- Lời chào thân thiện\\.\n" # Escaped for MarkdownV2
        "\\*/ping\\* \\- Kiểm tra xem bot có đang hoạt động không\\.\n" # Escaped for MarkdownV2
        "\\*/translate \\<văn bản\\>* \\- Dịch văn bản của bạn \\(tiếng Việt sang Anh, hoặc sửa ngữ pháp tiếng Anh\\)\\.\n" # Escaped for MarkdownV2
        "\\*/ask\\_kali \\<câu hỏi\\>* \\- Gợi ý công cụ Kali Linux và lệnh pentest\\.\n" # Escaped for MarkdownV2
        "\\*/help\\* \\- Hiển thị hướng dẫn sử dụng bot\\.\n\n" # Escaped for MarkdownV2
        "Hãy gõ / và chọn lệnh, hoặc gõ trực tiếp lệnh bạn muốn\\!" # Escaped for MarkdownV2
    )
    # help_text đã được hardcode và cần được tự thoát ký tự.
    # Trong trường hợp này, vì help_text là chuỗi tĩnh, bạn phải tự thêm '\\' vào các ký tự đặc biệt.
    # Hoặc áp dụng _escape_markdown_v2(help_text) nếu chuỗi này có thể thay đổi động.
    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN_V2)