
import discord
from discord.ext import commands
import discord.app_commands
import os
from dotenv import load_dotenv
import asyncio

load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN")
TEST_GUILD_ID = os.getenv("TEST_GUILD_ID")

if not TOKEN:
    raise ValueError("DISCORD_TOKEN not found in environment variables. Please set it in your .env file.")
if not TEST_GUILD_ID:
     print("WARNING: TEST_GUILD_ID not found in .env. Slash commands will sync globally (takes time).")

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='!', intents=intents)

async def load_cogs():
    """Loads all extensions (cogs) from the cogs directory."""
    for filename in os.listdir("./cogs"):
        if filename.endswith(".py") and filename != "__init__.py":
            cog_name = filename[:-3]
            try:
                await bot.load_extension(f"cogs.{cog_name}")
                print(f"Đã tải cog: {cog_name}")
            except Exception as e:
                print(f"Không thể tải cog {cog_name}: {e}")



@bot.event
async def on_ready():
    print(f'Bot đã đăng nhập với tên: {bot.user}')
    print('------')

    await load_cogs()

    try:
        if TEST_GUILD_ID:
            guild = discord.Object(id=int(TEST_GUILD_ID))
            synced = await bot.tree.sync(guild=guild)
            print(f"Đã đồng bộ {len(synced)} lệnh slash tới guild test {TEST_GUILD_ID}")
        else:
            synced = await bot.tree.sync()
            print(f"Đã đồng bộ {len(synced)} lệnh slash toàn cầu")

    except Exception as e:
        print(f"Lỗi khi đồng bộ lệnh slash: {e}")

    print('Bot đã sẵn sàng và các cogs đã được tải!')
    print('------')


bot.run(TOKEN)