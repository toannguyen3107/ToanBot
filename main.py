import discord
from discord.ext import commands 
import os
from dotenv import load_dotenv
load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
if not TOKEN:
    raise ValueError("DISCORD_TOKEN not found in environment variables. Please set it in your .env file.")

intents = discord.Intents.default()
# Bật các intent mà bạn đã cấu hình ở Bước 1
intents.message_content = True

# Tạo instance của bot
bot = commands.Bot(command_prefix='!', intents=intents)


@bot.event
async def on_ready():
    print(f'Bot đã đăng nhập với tên: {bot.user}')
    print('------')


@bot.command()
async def hello(ctx):
    """Trả lời bằng 'Chào!'"""
    await ctx.send('Chào!')


@bot.command()
async def ping(ctx):
    """Trả lời bằng 'Pong!' và độ trễ"""
    await ctx.send(f'Pong! {round(bot.latency * 1000)}ms')


bot.run(TOKEN)