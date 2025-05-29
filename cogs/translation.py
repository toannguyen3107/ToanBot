import discord
from discord.ext import commands
from discord import app_commands # Cần import này trong mỗi file cog có lệnh slash
import os
from dotenv import load_dotenv
import asyncio

# Import Langchain và Google GenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema


class TranslationCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.google_api_key = os.getenv("GOOGLE_API_KEY")

        if not self.google_api_key:
            print("WARNING: GOOGLE_API_KEY not found in environment variables. Translation feature will not work.")
            self.llm = None
        else:
            self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b", temperature=0.5, google_api_key=self.google_api_key)

        self.response_schemas = [
            ResponseSchema(name="input", description="Câu văn ban đầu"),
            ResponseSchema(name="output", description="Câu văn đã được thông dịch")
        ]
        self.output_parser = StructuredOutputParser.from_response_schemas(self.response_schemas)
        self.format_instructions = self.output_parser.get_format_instructions()

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

        if self.llm:
             # self.chain = LLMChain(llm=self.llm, prompt=self.prompt) # Vẫn dùng LLMChain để phù hợp với chain.run
             self.chain = LLMChain(llm=self.llm, prompt=self.prompt) # GIỮ LẠI CÁI NÀY VÌ self.chain.run
        else:
             self.chain = None


    async def _run_translation_chain(self, text: str) -> str | None:
        """Runs the translation chain in a separate thread."""
        if not self.chain:
            return "Lỗi: Tính năng thông dịch chưa được cấu hình (thiếu API Key?)."

        loop = asyncio.get_event_loop()
        try:
            raw_output = await loop.run_in_executor(
                None,
                self.chain.run, # Cần .run nếu dùng LLMChain
                text
            )
            parsed_output = self.output_parser.parse(raw_output)
            return parsed_output.get('output', 'Không thể phân tích kết quả thông dịch.')
        except Exception as e:
            print(f"Lỗi khi chạy hoặc phân tích chain thông dịch: {e}")
            return f"Lỗi khi thông dịch: {e}"


    @app_commands.command(name="translate", description="Thông dịch câu văn sử dụng AI (Tiếng Việt/Anh).")
    @app_commands.describe(text="Câu văn bạn muốn thông dịch.")
    async def slash_translate(self, interaction: discord.Interaction, text: str):
        """Thông dịch câu văn sử dụng AI."""
        print(f"Received translation request: {text}")
        if not self.llm or not self.chain:
             await interaction.response.send_message("Tính năng thông dịch hiện không khả dụng. Vui lòng kiểm tra cấu hình bot.", ephemeral=True)
             return

        await interaction.response.defer(ephemeral=False)

        translated_text = await self._run_translation_chain(text)

        await interaction.followup.send(translated_text)


# --- Setup function for the cog ---
async def setup(bot: commands.Bot):
    await bot.add_cog(TranslationCog(bot))
    print("Cog Translation đã được thiết lập.")