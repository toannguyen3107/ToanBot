from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import asyncio

import logging
# --- Cấu hình Logging ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

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
                None,
                self.chain.run,
                text
            )
            parsed_output = self.output_parser.parse(raw_output)
            return parsed_output.get('output', 'Không thể phân tích kết quả thông dịch.')
        except Exception as e:
            logger.error(f"Error during translation chain execution or parsing: {e}")
            return f"Đã xảy ra lỗi khi thông dịch: {e}"
