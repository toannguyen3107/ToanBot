import discord
from discord.ext import commands
from discord import app_commands
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
        
        # Khởi tạo LLM và chain
        self.llm = None
        self.chain = None
        self.output_parser = None
        
        self._initialize_llm_chain()

    def _initialize_llm_chain(self):
        """Khởi tạo LLM và các chain xử lý"""
        if not self.google_api_key:
            print("WARNING: GOOGLE_API_KEY not found in environment variables. Translation feature will not work.")
            return
        
        try:
            # Khởi tạo LLM
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash", 
                temperature=0.5, 
                google_api_key=self.google_api_key
            )
            
            # Định nghĩa output schema
            self.response_schemas = [
                ResponseSchema(name="input", description="Câu văn ban đầu"),
                ResponseSchema(name="output", description="Câu văn đã được thông dịch")
            ]
            self.output_parser = StructuredOutputParser.from_response_schemas(self.response_schemas)
            format_instructions = self.output_parser.get_format_instructions()

            # Tạo prompt template
            self.prompt = PromptTemplate(
                input_variables=["text"],
                template="""Bạn là một chuyên gia trong ngôn ngữ anh. Bạn sẽ đóng vai trò thông dịch cho tôi bất kể là tiếng việt hay tiếng anh. Bạn sẽ:
                1. Chỉnh sửa lại ngữ pháp, nếu là tiếng việt hãy viết sang tiếng anh. Nếu là tiếng anh hãy làm nó đúng ngữ pháp.
                2. Làm cho câu văn dễ hiểu.
                3. Nếu là tiếng anh trộn với việt hãy chuyển nó hết thành tiếng anh và dùng tiếp các rules trên.
                4. Trong đoạn văn sẽ có thể đề cập tới các param hay trích dẫn trong dấu ngoặc kép hãy giữ nguyên chúng.
                5. Output trả về chỉ trả lại đoạn text đã thông dịch không trả bất kỳ giải thích gì khác.

                {format_instructions}

                Bên dưới là câu văn cần thông dịch: {text}
                """,
                partial_variables={"format_instructions": format_instructions}
            )

            # Khởi tạo chain
            self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
            
        except Exception as e:
            print(f"Lỗi khi khởi tạo LLM hoặc chain: {e}")
            self.llm = None
            self.chain = None

    async def _run_translation_chain(self, text: str) -> str:
        """Thực hiện dịch văn bản"""
        if not self.chain:
            return "⚠️ Tính năng thông dịch hiện không khả dụng. Vui lòng liên hệ quản trị viên."
        
        try:
            # Chạy chain
            raw_output = await asyncio.to_thread(self.chain.run, text=text)
            
            # Phân tích kết quả
            if not raw_output:
                return "❌ Không nhận được phản hồi từ dịch vụ thông dịch."
                
            parsed_output = self.output_parser.parse(raw_output)
            return parsed_output.get('output', '❌ Không thể phân tích kết quả thông dịch.')
            
        except Exception as e:
            print(f"Lỗi khi thực hiện thông dịch: {e}")
            return f"❌ Đã xảy ra lỗi khi thông dịch: {str(e)}"

    @app_commands.command(name="translate", description="Thông dịch câu văn sử dụng AI (Tiếng Việt/Anh).")
    @app_commands.describe(text="Câu văn bạn muốn thông dịch.")
    async def slash_translate(self, interaction: discord.Interaction, text: str):
        """Xử lý lệnh slash translate"""
        if not text or len(text.strip()) == 0:
            await interaction.response.send_message("⚠️ Vui lòng nhập văn bản cần dịch.", ephemeral=True)
            return
            
        if not self.chain:
            await interaction.response.send_message(
                "⚠️ Tính năng thông dịch hiện không khả dụng. Vui lòng liên hệ quản trị viên.",
                ephemeral=True
            )
            return

        # Phản hồi defer để tránh timeout
        await interaction.response.defer(ephemeral=False)
        
        try:
            # Thực hiện dịch
            translated_text = await self._run_translation_chain(text)
            
            # Kiểm tra độ dài tin nhắn
            if len(translated_text) > 2000:
                translated_text = translated_text[:1997] + "..."
                
            # Gửi kết quả
            await interaction.followup.send(f"**Kết quả thông dịch:**\n{translated_text}")
            
        except Exception as e:
            print(f"Lỗi trong quá trình xử lý lệnh: {e}")
            await interaction.followup.send(
                "❌ Đã xảy ra lỗi khi xử lý yêu cầu của bạn. Vui lòng thử lại sau.",
                ephemeral=True
            )


async def setup(bot: commands.Bot):
    """Thiết lập cog"""
    await bot.add_cog(TranslationCog(bot))
    print("✅ Cog Translation đã được thiết lập.")