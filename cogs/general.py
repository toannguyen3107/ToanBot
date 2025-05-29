import discord
from discord.ext import commands
from discord import app_commands # Cần import này trong mỗi file cog có lệnh slash

class GeneralCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    # Lắng nghe sự kiện on_ready riêng cho cog (Tùy chọn, có thể dùng on_ready global)
    # @commands.Cog.listener()
    # async def on_ready(self):
    #     print(f'Cog General đã sẵn sàng!')

    # --- Lệnh Prefix ---
    @commands.command(name='hello', help='Trả lời bằng "Chào!" (Lệnh prefix)')
    async def hello(self, ctx: commands.Context):
        """Trả lời bằng 'Chào!' (Lệnh prefix)"""
        await ctx.send('Chào!')

    @commands.command(name='ping', help='Trả lời Pong! và độ trễ (Lệnh prefix)')
    async def ping(self, ctx: commands.Context):
        """Trả lời bằng 'Pong!' và độ trễ (Lệnh prefix)"""
        await ctx.send(f'Pong! {round(self.bot.latency * 1000)}ms')

    # --- Lệnh Slash ---
    @app_commands.command(name="hello", description="Bot trả lời chào! (Lệnh slash)")
    async def slash_hello(self, interaction: discord.Interaction):
        """Bot trả lời chào! (Lệnh slash)"""
        await interaction.response.send_message("Chào!")

    @app_commands.command(name="ping", description="Trả lời Pong! và độ trễ (Lệnh slash)")
    async def slash_ping(self, interaction: discord.Interaction):
        """Trả lời Pong! và độ trễ (Lệnh slash)"""
        await interaction.response.send_message(f"Pong! {round(self.bot.latency * 1000)}ms")


# --- Setup function for the cog ---
# Hàm này cần thiết để main.py có thể load cog
async def setup(bot: commands.Bot):
    await bot.add_cog(GeneralCog(bot))
    print("Cog General đã được thiết lập.")