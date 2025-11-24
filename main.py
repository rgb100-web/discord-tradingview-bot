ai = AIEngine(bot=bot, channel_id=DISCORD_CHANNEL_ID)

@bot.event
async def on_ready():
    print(f"Bot logged in as {bot.user} (id:{bot.user.id})")
    asyncio.create_task(ai._worker())   # <-- start queue here

import os
import asyncio
import logging
from webhook import create_app
from ai_engine import AIEngine
from discord.ext import commands
import discord

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")

DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
DISCORD_CHANNEL_ID = int(os.getenv("DISCORD_CHANNEL_ID", "0"))

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

ai = AIEngine(bot=bot, channel_id=DISCORD_CHANNEL_ID)

app = create_app(ai)  # <-- FIXED


@bot.event
async def on_ready():
    log.info(f"Bot logged in as {bot.user} (id:{bot.user.id})")

@bot.command(name="ping")
async def ping(ctx):
    await ctx.send("Pong! Bot is online.")

def run_bot_and_server():
    # Run flask server in an executor, discord bot in main loop
    import threading
    def run_flask():
        # use port from env (Railway sets PORT)
        port = int(os.getenv("PORT", "5000"))
        app.run(host="0.0.0.0", port=port)

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    # Start discord bot (this blocks)
    bot.run(DISCORD_TOKEN)

if __name__ == "__main__":
    run_bot_and_server()

