# main.py
import os
import asyncio
import logging
import discord
from discord.ext import commands

from ai_engine import AIEngine
from webhook import create_app

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")

# ---- ENV VARIABLES ----
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DISCORD_CHANNEL_ID = int(os.getenv("DISCORD_CHANNEL_ID", "0"))

# ---- DISCORD SETUP ----
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# ---- AI ENGINE ----
ai = AIEngine(bot=bot, channel_id=DISCORD_CHANNEL_ID)

# ---- DISCORD READY EVENT ----
@bot.event
async def on_ready():
    log.info(f"Bot logged in as {bot.user} (id:{bot.user.id})")
    asyncio.create_task(ai._worker())
    log.info("AI queue worker started successfully")


# ---- BOT COMMAND ----
@bot.command()
async def ping(ctx):
    await ctx.send("Pong! Bot is online.")


# ---- FLASK / WEBHOOK SERVER ----
flask_app = create_app(ai)

def start_web_server():
    from threading import Thread
    import waitress

    def run():
        port = int(os.environ.get("PORT", "5000"))  # <-- FIXED
        log.info(f"Starting webhook server on port {port}")
        waitress.serve(
            flask_app,
            host="0.0.0.0",
            port=port
        )

    Thread(target=run, daemon=True).start()


# ---- START EVERYTHING ----
def main():
    start_web_server()
    bot.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()
