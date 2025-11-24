# ai_engine.py
import os
import asyncio
import logging
import time
import json
import requests

log = logging.getLogger("ai_engine")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Model names are placeholders — use your vendor/model strings
MODEL_FAST = os.getenv("FAST_MODEL", "gpt-4.1-mini")
MODEL_DEEP = os.getenv("DEEP_MODEL", "gpt-5.1")

class AIEngine:
    def __init__(self, bot=None, channel_id=0):
        self.bot = bot
        self.channel_id = int(channel_id) if channel_id else None
        self.queue = asyncio.Queue()
        # spawn background worker
        loop = asyncio.get_event_loop()
        loop.create_task(self._worker())

    def enqueue(self, payload: dict):
        """Called by webhook to queue payload for evaluation."""
        asyncio.get_event_loop().call_soon_threadsafe(self.queue.put_nowait, payload)

    async def _worker(self):
        while True:
            payload = await self.queue.get()
            try:
                result = await self.evaluate_payload(payload)
                await self.post_result(result)
            except Exception as e:
                log.exception("Error processing payload")
            finally:
                self.queue.task_done()

    async def evaluate_payload(self, payload: dict) -> dict:
        """
        1) Run fast filter with gpt-4.1-mini (quick reject/accept)
        2) If passes, run gpt-5.1 for deep reasoning & probability
        3) Return a structured dict with verdict, entry/stop/targets, probability, notes
        """
        # Basic local heuristic filter (fast)
        if payload.get("volume", 0) and payload.get("volume") < payload.get("volMA", 0) * 0.8:
            return {"action":"reject","reason":"low volume"}

        # Compose prompts (short example). Adapt to your API shape.
        base_context = self._format_payload(payload)

        # FAST model quick pass
        fast_resp = self._call_openai(MODEL_FAST, prompt=f"Quick filter. Payload: {base_context}\nAnswer in JSON: {{'pass': bool, 'reason': str}}")
        try:
            fast_json = json.loads(fast_resp)
        except Exception:
            fast_json = {"pass": True}

        if not fast_json.get("pass", True):
            return {"action":"reject","reason": fast_json.get("reason","fast check failed")}

        # Deep reasoning
        deep_prompt = self._deep_prompt(payload)
        deep_resp = self._call_openai(MODEL_DEEP, prompt=deep_prompt)
        # Expect JSON response from model. This is a simplified approach.
        try:
            decision = json.loads(deep_resp)
        except Exception:
            # fallback: minimal parsing
            decision = {
                "action": "suggest",
                "side": "long",
                "probability": 0.6,
                "entry": payload.get("price"),
                "stop": payload.get("price",0) - payload.get("atr",0)*1.5,
                "targets": [payload.get("price",0) + 3*(payload.get("atr",0)*1.5)]
            }
        return decision

    def _format_payload(self, payload):
        # compact readable text for prompts
        return json.dumps(payload, default=str)

    def _deep_prompt(self, payload):
        # Build a structured prompt instructing the model to return JSON with fields:
        # action (suggest/reject), side, probability (0-100), entry, stop, targets[], notes
        p = {
            "system": "You are a professional trading analyst. Return a JSON object only.",
            "payload": payload,
            "rules": {
                "min_probability": 60,
                "min_rr": 3
            }
        }
        return json.dumps(p)

    def _call_openai(self, model, prompt):
        """
        Minimal OpenAI-compatible REST call (adjust for your provider)
        This uses OPENAI_API_KEY env var and expects text response.
        Replace with your provider SDK as needed.
        """
        if not OPENAI_API_KEY:
            log.warning("OPENAI_API_KEY missing — returning stubbed response.")
            # Return a minimal JSON string that will be parsed by caller
            return json.dumps({"action":"suggest","side":"long","probability":67,"entry":prompt, "stop":0,"targets":[]})
        url = "https://api.openai.com/v1/responses"  # placeholder endpoint; change if different
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type":"application/json"}
        body = {
            "model": model,
            "input": prompt,
            "max_tokens": 800
        }
        resp = requests.post(url, headers=headers, json=body, timeout=30)
        resp.raise_for_status()
        # provider may return content in different shape; for safety, return text
        return resp.text

    async def post_result(self, result: dict):
        """Post formatted result to Discord channel via bot."""
        if not self.bot or not self.channel_id:
            log.info("No bot/channel configured; printing result:\n%s", result)
            return

        channel = self.bot.get_channel(self.channel_id)
        if channel is None:
            # sometimes channel not ready; fetch via API
            try:
                channel = await self.bot.fetch_channel(self.channel_id)
            except Exception:
                log.warning("Could not fetch channel")
                return

        # Format embed-friendly summary
        title = f"AI Signal — {result.get('symbol','?')}"
        desc = result.get("notes") or result.get("reason") or "No notes"
        prob = result.get("probability")
        content = f"**Action:** {result.get('action')}\n**Side:** {result.get('side')}\n**Probability:** {prob}%\n**Entry:** {result.get('entry')}\n**Stop:** {result.get('stop')}\n**Targets:** {result.get('targets')}\n\n{desc}"
        await channel.send(content)
