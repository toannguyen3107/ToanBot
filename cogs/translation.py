import discord
from discord.ext import commands
import discord.app_commands
import os
from dotenv import load_dotenv # load_dotenv cũng có thể dùng trong cog nếu cần biến riêng
import asyncio # Cần cho run_in_executor

# Import Langchain và Google GenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
# Không cần IPython.display trên môi trường bot


class TranslationCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.google_api_key = os.getenv("GOOGLE_API_KEY") # Lấy API Key từ env

        if not self.google_api_key:
            print("WARNING: GOOGLE_API_KEY not found in environment variables. Translation feature will not work.")
            self.llm = None # Khởi tạo None nếu thiếu key
        else:
            # Khởi tạo LLM instance (chỉ 1 lần)
            self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b", temperature=0.5, google_api_key=self.google_api_key)

        # Định nghĩa Response Schemas cho output parser
        self.response_schemas = [
            ResponseSchema(name="input", description="Câu văn ban đầu"),
            ResponseSchema(name="output", description="Câu văn đã được thông dịch")
        ]
        self.output_parser = StructuredOutputParser.from_response_schemas(self.response_schemas)
        self.format_instructions = self.output_parser.get_format_instructions()

        # Định nghĩa Prompt Template
        self.prompt = PromptTemplate(
            input_variables=["val"],
            template="""Bạn là một chuyên gia trong ngôn ngữ anh. Ban sẽ đóng vai trò thông dịch cho tôi bất kể là tiếng việt hay tiếng anh. Bạn sẽ:
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

        # Tạo LLMChain (chỉ 1 lần)
        if self.llm: # Chỉ tạo chain nếu LLM được khởi tạo thành công
             self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
        else:
             self.chain = None


    # --- Helper function để chạy sync code trong executor ---
    # Cần thiết vì chain.run là synchronous và không nên chạy trực tiếp trong async Discord event loop
    async def _run_translation_chain(self, text: str) -> str | None:
        """Runs the translation chain in a separate thread."""
        if not self.chain:
            return "Lỗi: Tính năng thông dịch chưa được cấu hình (thiếu API Key?)."

        # Lấy event loop hiện tại
        loop = asyncio.get_event_loop()
        # Chạy chain.run trong một thread pool executor
        try:
            raw_output = await loop.run_in_executor(
                None, # Sử dụng default executor
                self.chain.run, # Hàm muốn chạy
                text # Tham số cho hàm
            )
            # Parse output
            parsed_output = self.output_parser.parse(raw_output)
            return parsed_output.get('output', 'Không thể phân tích kết quả thông dịch.')
        except Exception as e:
            print(f"Lỗi khi chạy hoặc phân tích chain thông dịch: {e}")
            return f"Lỗi khi thông dịch: {e}"


    # --- Lệnh Slash cho thông dịch ---
    @app_commands.command(name="translate", description="Thông dịch câu văn sử dụng AI (Tiếng Việt/Anh).")
    @app_commands.describe(text="Câu văn bạn muốn thông dịch.")
    async def slash_translate(self, interaction: discord.Interaction, text: str):
        """Thông dịch câu văn sử dụng AI."""
        if not self.llm or not self.chain:
             await interaction.response.send_message("Tính năng thông dịch hiện không khả dụng. Vui lòng kiểm tra cấu hình bot.", ephemeral=True)
             return

        # Acknowledge the interaction immediately as translation might take time
        await interaction.response.defer(ephemeral=False) # ephemeral=True nếu chỉ muốn người dùng thấy

        # Chạy logic thông dịch
        translated_text = await self._run_translation_chain(text)

        # Gửi kết quả
        await interaction.followup.send(translated_text)


# --- Setup function for the cog ---
async def setup(bot: commands.Bot):
    await bot.add_cog(TranslationCog(bot))
    print("Cog Translation đã được thiết lập.")