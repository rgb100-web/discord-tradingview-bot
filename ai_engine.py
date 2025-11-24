# ai_engine.py
import os
import asyncio
import logging
import json
import aiohttp

log = logging.getLogger("ai_engine")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FAST_MODEL = os.getenv("FAST_MODEL", "gpt-4.1-mini")
DEEP_MODEL = os.getenv("DEEP_MODEL", "gpt-4.1")


class AIEngine:
    """
    Handles AI evaluation pipeline:
    - Webhook pushes payload via enqueue()
    - Worker pulls payload from queue asynchronously
    - Fast model filters for A+ setups
    - Deep model assigns probability, entry, stops, targets, and notes
    - Sends message to Discord
    """

    def __init__(self, bot=None, channel_id=None):
        self.bot = bot
        self.channel_id = int(channel_id) if channel_id else None
        self.queue = asyncio.Queue()


    def enqueue(self, payload: dict):
        """Queue webhook payload safely from external (Flask) thread."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        loop.call_soon_threadsafe(self.queue.put_nowait, payload)


    async def _worker(self):
        """Background AI queue worker — runs forever."""
        log.info("AIEngine worker started")
        while True:
            payload = await self.queue.get()
            try:
                decision = await self.evaluate_payload(payload)
                await self.post_to_discord(decision)
            except Exception as e:
                log.exception(f"❌ AIEngine worker error: {e}")
            finally:
                self.queue.task_done()


    async def evaluate_payload(self, payload: dict) -> dict:
        """
        Fast filter → Deep reasoning → final structured JSON dict.
        """

        # ---- FALLBACK MODE if no OpenAI key ----
        if not OPENAI_API_KEY:
            log.warning("⚠ OPENAI_API_KEY missing — using stub response")
            price = payload.get("price", 0)
            atr = payload.get("atr", 1)
            return {
                "symbol": payload.get("symbol", "?"),
                "action": "suggest",
                "side": "long",
                "probability": 65,
                "entry": price,
                "stop": round(price - atr * 1.5, 2),
                "targets": [round(price + atr * 4.5, 2)],
                "notes": "Stub output — add OpenAI API key for real analysis."
            }

        # ---- FAST FILTER ----
        fast_prompt = f"""
Analyze this market snapshot. Decide if this trade setup should be evaluated further.
Return ONLY JSON: {{"pass": true/false, "reason": ""}}

Payload:
{json.dumps(payload)}
"""
        fast_result = await self._call_openai(FAST_MODEL, fast_prompt)

        try:
            fast_json = json.loads(fast_result)
            if not fast_json.get("pass", True):
                return {
                    "symbol": payload.get("symbol"),
                    "action": "reject",
                    "reason": fast_json.get("reason")
                }
        except Exception:
            log.warning("⚠ Fast model returned non-JSON — proceeding anyway")

        # -
