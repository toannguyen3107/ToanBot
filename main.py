import os
import logging
from dotenv import load_dotenv

# Import các thư viện Telegram Bot
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from cogs.translate import TranslationService
# Langchain và Google GenAI


# --- Cấu hình Logging ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Load Biến Môi Trường ---
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN not found in environment variables.")
if not GOOGLE_API_KEY:
    logger.warning("GOOGLE_API_KEY not found in environment variables. Translation feature will be disabled.")
# Khởi tạo TranslationService toàn cục
translation_service = TranslationService(GOOGLE_API_KEY)
# --- Hàm xử lý lệnh /start ---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_name = update.effective_user.first_name
    await update.message.reply_text(f"Chào mừng {user_name} đến với bot dịch thuật! "
                                    f"Tôi có thể giúp bạn dịch văn bản.\n"
                                    f"Gõ /translate <văn bản cần dịch>.")
# --- Hàm xử lý lệnh /hello ---
async def hello_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Chào!")
# -- Hàm xử lý lệnh /ping ---
async def ping_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Pong!")
# --- Hàm xử lý lệnh /translate ---
async def translate_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text_to_translate = ' '.join(context.args) # Lấy tất cả các từ sau lệnh /translate

    if not text_to_translate:
        await update.message.reply_text("Bạn cần cung cấp văn bản để thông dịch. Ví dụ: /translate Xin chào thế giới")
        return

    await update.message.reply_text("Đang thông dịch, vui lòng chờ...")
    
    translated_text = await translation_service.translate_text(text_to_translate)
    await update.message.reply_text(f"Kết quả thông dịch:\n{translated_text}")
# --- Hàm xử lý tin nhắn văn bản không phải lệnh ---
async def echo_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Tôi là bot dịch thuật. Vui lòng sử dụng lệnh /translate <văn bản của bạn> để dịch.")
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = (
        "Chào bạn! Tôi là bot dịch thuật. Dưới đây là các lệnh tôi có thể làm:\n\n"
        "/start - Bắt đầu lại cuộc trò chuyện và nhận lời chào mừng.\n"
        "/hello - Bot trả lời chào.\n"
        "/ping - Kiểm tra xem bot có đang hoạt động không.\n"
        "/translate <văn bản> - Dịch văn bản của bạn (tiếng Việt sang Anh, hoặc sửa ngữ pháp tiếng Anh).\n\n"
        "/help - Hiển thị hướng dẫn sử dụng bot.\n\n"
        "Hãy gõ / và chọn lệnh, hoặc gõ trực tiếp lệnh bạn muốn!"
    )
    await update.message.reply_text(help_text)
# Trong hàm main(), đăng ký handler cho /help:
def main() -> None:
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("hello", hello_command))
    application.add_handler(CommandHandler("ping", ping_command))
    application.add_handler(CommandHandler("translate", translate_command))
    application.add_handler(CommandHandler("help", help_command)) # THÊM DÒNG NÀY

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo_message))

    logger.info("Bot đang bắt đầu Long Polling...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)
    logger.info("Bot đã dừng Long Polling.")

if __name__ == '__main__':
    main()