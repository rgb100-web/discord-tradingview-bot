# main.py
import os
import asyncio
import logging
import discord
from discord.ext import commands

from ai_engine import AIEngine           # AI processing engine
from webhook import create_app           # Flask webhook server

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")

# ---- ENV VARIABLES ----
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
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
    # Start AI queue worker only after event loop is alive
    asyncio.create_task(ai._worker())
    log.info("AI queue worker started successfully")

# ---- BASIC BOT COMMAND ----
@bot.command()
async def ping(ctx):
    await ctx.send("Pong! Bot is online.")

# ---- START WEBHOOK SERVER ----
# create Flask app and run it on a background thread
flask_app = create_app(ai)

def start_web_server():
    from threading import Thread
    import waitress                     # Production WSGI server for Render

    def run():
        waitress.serve(
            flask_app,
            host="0.0.0.0",
            port=int(os.environ.get("PORT", 10000))
        )

    Thread(target=run, daemon=True).start()
    log.info("Webhook web server started")

# ---- START EVERYTHING ----
def main():
    # Start webhook endpoint first
    start_web_server()

    # Start Discord bot — stays alive forever
    bot.run(DISCORD_TOKEN)

if __name__ == "__main__":
    main()
