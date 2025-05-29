import discord
from discord.ext import commands
import discord.app_commands
import os
from dotenv import load_dotenv
import asyncio
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bot.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN")
TEST_GUILD_ID = os.getenv("TEST_GUILD_ID")

# Validate environment variables
if not TOKEN:
    logger.error("DISCORD_TOKEN not found in environment variables. Please set it in your .env file.")
    raise ValueError("Missing DISCORD_TOKEN in environment variables")

if not TEST_GUILD_ID:
    logger.warning("TEST_GUILD_ID not found in .env. Slash commands will sync globally (may take up to 1 hour).")

# Configure bot intents
intents = discord.Intents.default()
intents.message_content = True

class MyBot(commands.Bot):
    def __init__(self):
        super().__init__(
            command_prefix='!',
            intents=intents,
            help_command=None,  # Disable default help command
            activity=discord.Game(name="Dịch thuật AI")
        )

    async def setup_hook(self):
        """Load cogs and sync commands"""
        await self.load_cogs()
        await self.sync_commands()

    async def load_cogs(self):
        """Load all cogs from the cogs directory"""
        loaded_cogs = []
        for filename in os.listdir("./cogs"):
            if filename.endswith(".py") and filename != "__init__.py":
                cog_name = filename[:-3]
                try:
                    await self.load_extension(f"cogs.{cog_name}")
                    loaded_cogs.append(cog_name)
                    logger.info(f"Successfully loaded cog: {cog_name}")
                except Exception as e:
                    logger.error(f"Failed to load cog {cog_name}: {str(e)}")
        
        if not loaded_cogs:
            logger.warning("No cogs were loaded. Check your cogs directory.")

    async def sync_commands(self):
        """Sync application commands"""
        try:
            if TEST_GUILD_ID:
                guild = discord.Object(id=int(TEST_GUILD_ID))
                synced = await self.tree.sync(guild=guild)
                logger.info(f"Synced {len(synced)} commands to test guild {TEST_GUILD_ID}")
            else:
                synced = await self.tree.sync()
                logger.info(f"Synced {len(synced)} commands globally")
        except Exception as e:
            logger.error(f"Error syncing commands: {str(e)}")

bot = MyBot()

@bot.event
async def on_ready():
    """Called when the bot is ready"""
    logger.info(f'Logged in as {bot.user} (ID: {bot.user.id})')
    logger.info('------')

@bot.event
async def on_command_error(ctx, error):
    """Handle command errors"""
    if isinstance(error, commands.CommandNotFound):
        return
    logger.error(f"Command error: {str(error)}", exc_info=error)
    await ctx.send(f"❌ Đã xảy ra lỗi: {str(error)}")

if __name__ == "__main__":
    try:
        bot.run(TOKEN)
    except discord.LoginFailure:
        logger.error("Invalid DISCORD_TOKEN. Please check your .env file")
    except KeyboardInterrupt:
        logger.info("Bot shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")