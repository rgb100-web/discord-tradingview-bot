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

        # ---- DEEP ANALYSIS ----
        deep_prompt = f"""
You are a professional trading analyst.

Evaluate the trade:
- Trend alignment
- Divergences
- Volume confirmation
- A+ setups only
- Risk/reward >= 1:3
- Clean entry, stop, and 1–2 targets
- Probability of success (0-100)

Return ONLY JSON:
{{
  "symbol": "",
  "action": "suggest" or "reject",
  "side": "long" or "short",
  "probability": 0-100,
  "entry": float,
  "stop": float,
  "targets": [float, float?],
  "notes": ""
}}

Payload:
{json.dumps(payload)}
"""
        deep_result = await self._call_openai(DEEP_MODEL, deep_prompt)

        try:
            final_json = json.loads(deep_result)
            return final_json
        except Exception:
            log.warning("⚠ Deep model returned non-JSON — using fallback")
            price = payload.get("price", 0)
            atr = payload.get("atr", 1)

            return {
                "symbol": payload.get("symbol", "?"),
                "action": "suggest",
                "side": "long",
                "probability": 60,
                "entry": price,
                "stop": round(price - atr * 1.5, 2),
                "targets": [round(price + atr * 4.5, 2)],
                "notes": "Fallback used — model returned non-JSON"
            }


    async def _call_openai(self, model: str, prompt: str) -> str:
        """Send a chat completion request to OpenAI."""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are an ultra-precise trading signal engine."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.05,
            "max_tokens": 800
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=body) as resp:
                r = await resp.json()
                return r["choices"][0]["message"]["content"]


    async def post_to_discord(self, result: dict):
        """Send formatted AI output to configured Discord channel."""
        if not self.bot or not self.channel_id:
            log.warning("⚠ No bot or channel configured")
            print(result)
            return

        channel = self.bot.get_channel(self.channel_id)
        if channel is None:
            try:
                channel = await self.bot.fetch_channel(self.channel_id)
            except Exception:
                log.exception("❌ Failed to fetch channel")
                return

        msg = (
            f"📈 **AI Signal — {result.get('symbol','?')}**\n\n"
            f"**Action:** {result.get('action')}\n"
            f"**Side:** {result.get('side')}\n"
            f"**Probability:** {result.get('probability')}%\n"
            f"**Entry:** {result.get('entry')}\n"
            f"**Stop:** {result.get('stop')}\n"
            f"**Targets:** {result.get('targets')}\n\n"
            f"{result.get('notes')}"
        )

        log.info("📤 Sending to Discord:")
        log.info(msg)

        await channel.send(msg)
