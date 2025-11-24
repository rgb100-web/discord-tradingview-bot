import asyncio
import logging
import os
from queue import Queue

import openai

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ai_engine")


class AIEngine:
    def __init__(self, bot, channel_id):
        self.bot = bot
        self.channel_id = int(channel_id)
        self.queue = Queue()
        openai.api_key = os.getenv("OPENAI_API_KEY")

    # Called by webhook
    def enqueue_alert(self, alert_message: str):
        log.info(f"AIEngine: queued alert -> {alert_message}")
        self.queue.put(alert_message)

    async def _worker(self):
        """Background worker that processes queued alerts and sends to Discord"""
        log.info("AIEngine worker started")
        await self._wait_for_bot_ready()

        while True:
            if self.queue.empty():
                await asyncio.sleep(0.5)
                continue

            raw_alert = self.queue.get()
            log.info(f"AIEngine: processing alert -> {raw_alert}")

            # Format / enrich message with AI (optional)
            final_msg = await self._process_with_openai(raw_alert)

            channel = self.bot.get_channel(self.channel_id)
            if channel is None:
                log.error(f"AIEngine ERROR — Channel not found: {self.channel_id}")
                continue

            try:
                log.info(f"AIEngine sending to channel {self.channel_id}: {final_msg}")
                await channel.send(final_msg)
            except Exception as e:
                log.error(f"AIEngine ERROR sending to Discord: {e}")

    async def _wait_for_bot_ready(self):
        """Ensure Discord bot is logged in before sending"""
        while not self.bot.is_ready():
            await asyncio.sleep(1)
        log.info("AIEngine confirmed bot ready")

    async def _process_with_openai(self, raw_alert):
        """Optional AI enhancement. If OpenAI errors, message still goes through."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            log.warning("OPENAI_API_KEY missing — sending raw message without AI")
            return raw_alert

        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Turn the alert into a clean trading signal message."},
                    {"role": "user", "content": raw_alert}
                ]
            )
            ai_text = response.choices[0].message.content.strip()
            return ai_text
        except Exception as e:
            log.error(f"OpenAI API ERROR — sending raw message instead: {e}")
            return raw_alert
