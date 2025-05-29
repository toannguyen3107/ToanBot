import discord
from discord.ext import commands
import discord.app_commands 
import os
from dotenv import load_dotenv
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

@bot.event
async def on_ready():
    print(f'Bot đã đăng nhập với tên: {bot.user}')
    print('------')
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
    print('Bot đã sẵn sàng!')
    print('------')

@bot.command()
async def hello(ctx):
    """Trả lời bằng 'Chào!' (Lệnh prefix)"""
    await ctx.send('Chào!')

@bot.command()
async def ping(ctx):
    """Trả lời bằng 'Pong!' và độ trễ (Lệnh prefix)"""
    await ctx.send(f'Pong! {round(bot.latency * 1000)}ms')
@bot.tree.command(name="hello", description="Bot trả lời chào!")

async def slash_hello(interaction: discord.Interaction):
    """Bot trả lời chào! (Lệnh slash)"""
    await interaction.response.send_message("Chào!")

@bot.tree.command(name="ping", description="Trả lời Pong! và độ trễ")
async def slash_ping(interaction: discord.Interaction):
    """Trả lời Pong! và độ trễ (Lệnh slash)"""
    await interaction.response.send_message(f"Pong! {round(interaction.client.latency * 1000)}ms")
    
bot.run(TOKEN)