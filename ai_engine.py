# ai_engine.py
import os
import asyncio
import logging
import json
import aiohttp

log = logging.getLogger("ai_engine")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FAST_MODEL = os.getenv("FAST_MODEL", "gpt-4.1-mini")    # Fast filter model
DEEP_MODEL = os.getenv("DEEP_MODEL", "gpt-4.1")         # Final reasoning model

class AIEngine:
    """
    Handles AI evaluation pipeline:
    1) Payload queued from webhook
    2) Fast model determines pass/fail
    3) Deep model produces probability, entry, stop, targets, notes
    4) Sends formatted results to Discord channel
    """

    def __init__(self, bot=None, channel_id=None):
        self.bot = bot
        self.channel_id = int(channel_id) if channel_id else None
        self.queue = asyncio.Queue()

        # Start the worker loop inside Discord's event loop
        loop = asyncio.get_event_loop()
        loop.create_task(self._worker())

    # 🔥 SAFE QUEUEING FIX (prevents "no event loop in thread" error)
    def enqueue(self, payload: dict):
        """Called by webhook thread to safely queue payload to async loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        loop.call_soon_threadsafe(self.queue.put_nowait, payload)

    async def _worker(self):
        """Processes payloads one at a time."""
        while True:
            payload = await self.queue.get()
            try:
                decision = await self.evaluate_payload(payload)
                await self.post_to_discord(decision)
            except Exception as e:
                log.exception(f"❌ AIEngine processing error: {e}")
            finally:
                self.queue.task_done()

    async def evaluate_payload(self, payload: dict) -> dict:
        """
        Fast model (filter) → Deep model (analysis)
        Returns a final structured dict.
        """

        # ⏳ If no API key → output stub so alerts still test cleanly
        if not OPENAI_API_KEY:
            return {
                "symbol": payload.get("symbol"),
                "action": "suggest",
                "side": "long",
                "probability": 67,
                "entry": payload.get("price"),
                "stop": round(payload.get("price", 0) - payload.get("atr", 1) * 1.5, 2),
                "targets": [round(payload.get("price", 0) + payload.get("atr", 1) * 4.5, 2)],
                "notes": "AI key missing — using stub response."
            }

        # ----------- Fast Filter (GPT-4.1-mini) -----------
        fast_prompt = f"""
Analyze the following market snapshot. Decide if it is worth evaluating further.
Return ONLY JSON:
{{"pass": true/false, "reason": ""}}

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
                    "reason": fast_json.get("reason", "Fast filter rejected setup.")
                }
        except Exception:
            pass  # continue to deep reasoning if json parsing fails

        # ----------- Deep Reasoning (GPT-4.1) -----------
        deep_prompt = f"""
You are an elite trading analyst. Evaluate the trade using:
- probability of success
- risk/reward >= 1:3
- trend, momentum, volume, volatility, divergences
- clean entry, stop, 1–2 targets

Return ONLY JSON with fields:
{{
  "symbol": "",
  "action": "suggest" or "reject",
  "side": "long" or "short",
  "probability": 0–100,
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
            log.warning("⚠ Deep model returned non-JSON — falling back")
            return {
                "symbol": payload.get("symbol"),
                "action": "suggest",
                "side": "long",
                "probability": 60,
                "entry": payload.get("price"),
                "stop": round(payload.get("price", 0) - payload.get("atr", 1) * 1.5, 2),
                "targets": [round(payload.get("price", 0) + payload.get("atr", 1) * 4.5, 2)],
                "notes": "Fallback — model returned non-JSON format."
            }

    async def _call_openai(self, model: str, prompt: str) -> str:
        """OpenAI chat completions request → returns raw content string."""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a precise trading analysis engine."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 800
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=body) as r:
                response = await r.json()
                return response["choices"][0]["message"]["content"]

    async def post_to_discord(self, result: dict):
        """Send formatted AI recommendation to the configured Discord channel."""
        if not self.bot or not self.channel_id:
            log.warning("⚠ No bot or channel configured — printing fallback")
            print(result)
            return

        channel = self.bot.get_channel(self.channel_id)
        if channel is None:
            try:
                channel = await self.bot.fetch_channel(self.channel_id)
            except Exception:
                log.exception("❌ Cannot fetch Discord channel")
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
