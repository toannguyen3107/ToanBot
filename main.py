import os
import asyncio
import logging
from dotenv import load_dotenv

# Import các thư viện Telegram Bot
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Langchain và Google GenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# --- Cấu hình Logging ---
# Luôn đặt logging sớm nhất có thể
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

# --- Logic Dịch Thuật (từ TranslationCog của bạn, đã sửa đổi) ---
class TranslationService:
    def __init__(self, google_api_key: str):
        self.llm = None
        self.chain = None
        
        if google_api_key:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash-8b",
                    temperature=0.5,
                    google_api_key=google_api_key
                )
                logger.info("Gemini LLM initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini LLM: {e}")
                self.llm = None
        else:
            logger.warning("GOOGLE_API_KEY not provided. Translation feature will be unavailable.")

        if self.llm:
            self.response_schemas = [
                ResponseSchema(name="input", description="Câu văn ban đầu"),
                ResponseSchema(name="output", description="Câu văn đã được thông dịch")
            ]
            self.output_parser = StructuredOutputParser.from_response_schemas(self.response_schemas)
            self.format_instructions = self.output_parser.get_format_instructions()

            self.prompt = PromptTemplate(
                input_variables=["val"],
                template="""Bạn là một chuyên gia trong ngôn ngữ anh. Bạn sẽ đóng vai trò thông dịch cho tôi bất kể là tiếng việt hay tiếng anh. Bạn sẽ:
                1. Chỉnh sửa lại ngữ pháp, nếu là tiếng việt hãy viết sang tiếng anh. Nếu là tiếng anh hãy làm nó đúng ngữ pháp.
                2. Làm cho câu văn dễ hiểu.
                3. Nếu là tiếng anh trộn với việt hãy chuyển nó hết thành tiếng anh và dùng tiếp các rules trên.
                4. Trong đoạn văn sẽ có thể đề cập tới các param hay trích dẫn trong dấu ngoặc kép hãy giữ nguyên chúng.
                5. Output trả về chỉ trả lại đoạn text đã thông dịch không trả bất kỳ giải thích gì khác.

                {format_instructions}

                Bên dưới là câu văn cần thông dịch: {val}
                """,
                partial_variables={"format_instructions": self.format_instructions}
            )

            try:
                self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
                logger.info("Langchain LLMChain initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Langchain LLMChain: {e}")
                self.chain = None
        else:
            self.chain = None

    async def translate_text(self, text: str) -> str:
        if not self.chain:
            return "Tính năng thông dịch hiện không khả dụng. Vui lòng kiểm tra cấu hình bot."

        loop = asyncio.get_event_loop()
        try:
            raw_output = await loop.run_in_executor(
                None,  # Use default executor
                self.chain.run,
                text
            )
            parsed_output = self.output_parser.parse(raw_output)
            return parsed_output.get('output', 'Không thể phân tích kết quả thông dịch.')
        except Exception as e:
            logger.error(f"Error during translation chain execution or parsing: {e}")
            return f"Đã xảy ra lỗi khi thông dịch: {e}"

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

# --- Hàm xử lý lệnh /ping ---
async def ping_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Telegram Bot API không cung cấp độ trễ mạng trực tiếp như Discord
    # Có thể trả lời một tin nhắn Ping đơn giản
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
    # Có thể làm gì đó với tin nhắn không phải lệnh, ví dụ: nhắc nhở
    await update.message.reply_text("Tôi là bot dịch thuật. Vui lòng sử dụng lệnh /translate <văn bản của bạn> để dịch.")


# --- Hàm chính để chạy bot ---
def main() -> None:
    # Tạo Application builder
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Đăng ký các hàm xử lý lệnh
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("hello", hello_command))
    application.add_handler(CommandHandler("ping", ping_command))
    application.add_handler(CommandHandler("translate", translate_command))

    # Đăng ký hàm xử lý tất cả tin nhắn văn bản không phải lệnh
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo_message))

    # Bắt đầu chế độ Long Polling
    logger.info("Bot đang bắt đầu Long Polling...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)
    logger.info("Bot đã dừng Long Polling.")


if __name__ == '__main__':
    main()