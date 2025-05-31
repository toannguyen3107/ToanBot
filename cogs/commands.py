from cogs.translate import TranslationService
import os
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import logging

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    logger.warning("GOOGLE_API_KEY not found in environment variables. Translation feature will be disabled.")
translation_service = TranslationService(GOOGLE_API_KEY)
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

    await update.message.reply_text("Đang thông dịch, vui lòng chờ...")
    
    translated_text = await translation_service.translate_text(text_to_translate)
    response_message_html = f"Kết quả thông dịch:\n<pre>{translated_text}</pre>"
    await update.message.reply_html(response_message_html)
async def echo_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Tôi là bot dịch thuật. Vui lòng sử dụng lệnh /translate <văn bản của bạn> để dịch.")
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = (
        "Xin chao! Tôi là Toan. Dưới đây là các lệnh tôi có thể làm:\n\n"
        "/start - Bắt đầu lại cuộc trò chuyện và nhận lời chào mừng.\n"
        "/hello - Toàn trả lời chào.\n"
        "/ping - Kiểm tra xem bot có đang hoạt động không.\n"
        "/translate <văn bản> - Dịch văn bản của bạn (tiếng Việt sang Anh, hoặc sửa ngữ pháp tiếng Anh).\n"
        "/help - Hiển thị hướng dẫn sử dụng bot.\n\n"
        "Hãy gõ / và chọn lệnh, hoặc gõ trực tiếp lệnh bạn muốn!"
    )
    await update.message.reply_text(help_text)