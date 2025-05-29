
import discord
from discord.ext import commands
import os
from dotenv import load_dotenv
import asyncio

# Load environment variables
load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN")
TEST_GUILD_ID = os.getenv("TEST_GUILD_ID") # Optional for testing slash sync

if not TOKEN:
    raise ValueError("DISCORD_TOKEN not found in environment variables. Please set it in your .env file.")
if not TEST_GUILD_ID:
     print("WARNING: TEST_GUILD_ID not found in .env. Slash commands will sync globally (takes time).")

# Setup Intents (giữ lại message_content nếu muốn dùng cả lệnh prefix)
intents = discord.Intents.default()
intents.message_content = True
# intents.guilds = True # uncomment if your cogs need Guild information directly

bot = commands.Bot(command_prefix='!', intents=intents)

# --- Async function to load cogs ---
async def load_cogs():
    """Loads all extensions (cogs) from the cogs directory."""
    # Duyệt qua tất cả các file .py trong thư mục cogs
    for filename in os.listdir("./cogs"):
        if filename.endswith(".py") and filename != "__init__.py":
            # Bỏ ".py" để lấy tên module (ví dụ: general, translation)
            cog_name = filename[:-3]
            try:
                # Load cog. Format là 'ten_thu_muc.ten_file'
                await bot.load_extension(f"cogs.{cog_name}")
                print(f"Đã tải cog: {cog_name}")
            except Exception as e:
                print(f"Không thể tải cog {cog_name}: {e}")


# --- Events ---

@bot.event
async def on_ready():
    print(f'Bot đã đăng nhập với tên: {bot.user}')
    print('------')

    # Load cogs trước khi đồng bộ lệnh slash
    await load_cogs()

    # --- Đồng bộ lệnh Slash ---
    try:
        if TEST_GUILD_ID:
            # Đồng bộ lệnh tới một guild cụ thể (Test Guild)
            guild = discord.Object(id=int(TEST_GUILD_ID))
            synced = await bot.tree.sync(guild=guild)
            print(f"Đã đồng bộ {len(synced)} lệnh slash tới guild test {TEST_GUILD_ID}")
        else:
            # Đồng bộ lệnh toàn cầu
            synced = await bot.tree.sync()
            print(f"Đã đồng bộ {len(synced)} lệnh slash toàn cầu")

    except Exception as e:
        print(f"Lỗi khi đồng bộ lệnh slash: {e}")

    print('Bot đã sẵn sàng và các cogs đã được tải!')
    print('------')


# Bạn có thể thêm các lệnh hoặc sự kiện global khác ở đây nếu cần,
# nhưng thường nên đặt chúng vào các cogs.

# --- Chạy Bot ---
# Không cần thay đổi dòng này
bot.run(TOKEN)