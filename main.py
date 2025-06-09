# telegram_kali_bot/main.py

import os
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import CommandHandler, Application, MessageHandler, filters

from cogs.translate import TranslationService
from cogs.kali_rag import KaliRAGService

import cogs.commands 

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # Được sử dụng cho cả RAG (Gemini) và Translation (Gemini)

if not TELEGRAM_BOT_TOKEN:
    logger.critical("TELEGRAM_BOT_TOKEN not found in .env file. Exiting.")
    exit(1)
if not GOOGLE_API_KEY:
    logger.critical("GOOGLE_API_KEY not found in .env file. Both RAG and Translation features will be unavailable. Exiting.")
    exit(1)


def main() -> None:
    logger.info("Starting service initialization...")
    
    cogs.commands.translation_service_instance = TranslationService(GOOGLE_API_KEY)
    if cogs.commands.translation_service_instance.llm is None:
        logger.warning("TranslationService LLM could not be initialized. Translation feature will be unavailable.")
    cogs.commands.kali_rag_service_instance = KaliRAGService(GOOGLE_API_KEY)
    if cogs.commands.kali_rag_service_instance.rag_chain is None:
        logger.critical("Failed to initialize KaliRAGService. RAG feature will be unavailable.")
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", cogs.commands.start_command))
    application.add_handler(CommandHandler("hello", cogs.commands.hello_command))
    application.add_handler(CommandHandler("ping", cogs.commands.ping_command))
    application.add_handler(CommandHandler("translate", cogs.commands.translate_command))
    application.add_handler(CommandHandler("help", cogs.commands.help_command))
    application.add_handler(CommandHandler("ask_kali", cogs.commands.ask_kali_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, cogs.commands.echo_message))

    logger.info("Bot đang bắt đầu Long Polling...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)
    logger.info("Bot đã dừng Long Polling.")

if __name__ == '__main__':
    main()