# telegram_kali_bot/main.py

import os
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import CommandHandler, Application, MessageHandler, filters

# Import TranslationService và KaliRAGService classes
from cogs.translate import TranslationService
from cogs.kali_rag import KaliRAGService

# Import cogs.commands module để gán các instance service vào các biến toàn cục
import cogs.commands 

# --- Cấu hình Logging ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Tải biến môi trường từ .env
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
# THAY ĐỔI: Xóa OPENAI_API_KEY nếu không còn sử dụng trực tiếp trong main.py
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # Được sử dụng cho cả RAG (Gemini) và Translation (Gemini)

# Kiểm tra sự tồn tại của các token/API keys
if not TELEGRAM_BOT_TOKEN:
    logger.critical("TELEGRAM_BOT_TOKEN not found in .env file. Exiting.")
    exit(1)
# THAY ĐỔI: Kiểm tra GOOGLE_API_KEY cho cả hai tính năng
if not GOOGLE_API_KEY:
    logger.critical("GOOGLE_API_KEY not found in .env file. Both RAG and Translation features will be unavailable. Exiting.")
    exit(1)


# --- Main function của Bot Telegram ---
def main() -> None:
    # --- Khởi tạo các dịch vụ (Translation và Kali RAG) ---
    logger.info("Starting service initialization...")
    
    # Khởi tạo TranslationService
    # Gán instance TranslationService vào biến toàn cục trong cogs.commands
    cogs.commands.translation_service_instance = TranslationService(GOOGLE_API_KEY)
    if cogs.commands.translation_service_instance.llm is None:
        logger.warning("TranslationService LLM could not be initialized. Translation feature will be unavailable.")

    # Khởi tạo KaliRAGService
    # Gán instance KaliRAGService vào biến toàn cục trong cogs.commands
    cogs.commands.kali_rag_service_instance = KaliRAGService(GOOGLE_API_KEY) # THAY ĐỔI: Truyền GOOGLE_API_KEY
    if cogs.commands.kali_rag_service_instance.rag_chain is None:
        logger.critical("Failed to initialize KaliRAGService. RAG feature will be unavailable.")
        # Bạn có thể chọn exit(1) ở đây nếu RAG là tính năng bắt buộc.
        # Ở đây tôi sẽ để bot tiếp tục chạy nhưng tính năng RAG sẽ không hoạt động.


    # --- Cấu hình Telegram Bot ---
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Đăng ký các Command Handlers từ cogs.commands
    application.add_handler(CommandHandler("start", cogs.commands.start_command))
    application.add_handler(CommandHandler("hello", cogs.commands.hello_command))
    application.add_handler(CommandHandler("ping", cogs.commands.ping_command))
    application.add_handler(CommandHandler("translate", cogs.commands.translate_command))
    application.add_handler(CommandHandler("help", cogs.commands.help_command))
    application.add_handler(CommandHandler("ask_kali", cogs.commands.ask_kali_command))

    # Đăng ký Message Handler cho các tin nhắn không phải là lệnh
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, cogs.commands.echo_message))

    logger.info("Bot đang bắt đầu Long Polling...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)
    logger.info("Bot đã dừng Long Polling.")

if __name__ == '__main__':
    main()