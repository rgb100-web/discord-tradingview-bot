import asyncio
import logging
import os
from queue import Queue
from typing import Any, Dict, Optional

import discord

try:
    # New OpenAI client (1.x)
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None  # library not installed, we’ll fall back

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ai_engine")


class AIEngine:
    """
    AI engine that:
    - Receives alerts from webhook (as dict payloads)
    - Parses direction, entry, target, stop, probability, timeframe
    - Optionally adds AI summary (if OPENAI_API_KEY is set)
    - Sends nicely formatted embeds into Discord,
      using a per-ticker thread (e.g., "QQQ Alerts").
    """

    def __init__(self, bot: discord.Client, channel_id: int):
        self.bot = bot
        self.channel_id = int(channel_id)
        self.queue: Queue[Dict[str, Any]] = Queue()
        self.threads: dict[str, int] = {}  # ticker -> thread_id

        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and AsyncOpenAI is not None:
            self.openai_client: Optional[AsyncOpenAI] = AsyncOpenAI(api_key=api_key)
            log.info("AIEngine: OpenAI client initialized")
        else:
            self.openai_client = None
            log.info("AIEngine: OpenAI client NOT initialized (no key or library)")

    # Called from webhook (sync context)
    def enqueue_alert(self, payload: Dict[str, Any]) -> None:
        log.info(f"AIEngine: queued alert payload -> {payload}")
        self.queue.put(payload)

    async def _worker(self):
        """Background worker: processes queued alerts, sends messages."""
        log.info("AIEngine worker started")
        await self._wait_for_bot_ready()
        log.info("AIEngine confirmed bot ready")

        while True:
            if self.queue.empty():
                await asyncio.sleep(0.25)
                continue

            payload = self.queue.get()
            try:
                await self._handle_alert(payload)
            except Exception:
                log.exception("AIEngine: error while handling alert")

    async def _wait_for_bot_ready(self):
        """Ensure bot is logged in before trying to send messages."""
        while not self.bot.is_ready():
            await asyncio.sleep(1)

    async def _handle_alert(self, payload: Dict[str, Any]):
        parsed = self._parse_payload(payload)
        log.info(f"AIEngine: parsed alert -> {parsed}")

        ai_summary = await self._generate_ai_summary(parsed)

        channel = self.bot.get_channel(self.channel_id)
        if not isinstance(channel, discord.TextChannel):
            log.error(f"AIEngine: Channel {self.channel_id} is not a TextChannel or not found")
            return

        target = await self._get_ticker_thread(channel, parsed["ticker"])

        embed = self._build_embed(parsed)
        content = ai_summary if ai_summary else None

        log.info(f"AIEngine: sending alert for {parsed['ticker']} to channel/thread {target.id}")
        try:
            await target.send(content=content, embed=embed)
        except Exception:
            log.exception("AIEngine: error sending message to Discord")

    def _parse_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expects flat JSON like:
        {
          "ticker": "QQQ",
          "direction": "LONG",
          "entry": 420.55,
          "target": 428.00,
          "stop": 417.50,
          "probability": 0.68,
          "timeframe": "15m",
          "time": "2025-11-24 16:00:00"
        }
        Missing fields are handled gracefully.
        """
        ticker = str(payload.get("ticker") or payload.get("symbol") or "UNKNOWN").upper()
        direction = str(payload.get("direction", "")).upper()

        def _to_float(key: str) -> Optional[float]:
            v = payload.get(key)
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        entry = _to_float("entry") or _to_float("price")
        target = _to_float("target")
        stop = _to_float("stop") or _to_float("stop_loss")
        probability = payload.get("probability")
        try:
            if isinstance(probability, str):
                probability = float(probability)
            elif isinstance(probability, (int, float)):
                probability = float(probability)
            else:
                probability = None
        except (TypeError, ValueError):
            probability = None

        timeframe = str(payload.get("timeframe") or payload.get("tf") or "").strip()
        timestamp = str(payload.get("time") or payload.get("timestamp") or "").strip()

        return {
            "ticker": ticker,
            "direction": direction,
            "entry": entry,
            "target": target,
            "stop": stop,
            "probability": probability,
            "timeframe": timeframe,
            "timestamp": timestamp,
        }

    async def _get_ticker_thread(
        self, channel: discord.TextChannel, ticker: str
    ) -> discord.abc.Messageable:
        """
        Use one thread per ticker, e.g. "QQQ Alerts".
        If thread exists -> reuse; else create.
        If thread creation fails, fall back to base channel.
        """
        if not ticker or ticker == "UNKNOWN":
            return channel

        existing_id = self.threads.get(ticker)
        if existing_id:
            thread = channel.get_thread(existing_id)
            if thread is not None:
                return thread

        thread_name = f"{ticker} Alerts"
        try:
            thread = await channel.create_thread(
                name=thread_name,
                type=discord.ChannelType.public_thread,
            )
            self.threads[ticker] = thread.id
            log.info(f"AIEngine: created thread {thread_name} ({thread.id})")
            return thread
        except Exception:
            log.exception("AIEngine: failed to create thread; using base channel")
            return channel

    def _build_embed(self, p: Dict[str, Any]) -> discord.Embed:
        """Builds a clean embed showing direction, R/R components, probability, etc."""
        direction = p["direction"]
        ticker = p["ticker"]

        # Color by direction
        dir_upper = direction.upper()
        if dir_upper in ("LONG", "BUY", "CALL"):
            color = discord.Color.green()
        elif dir_upper in ("SHORT", "SELL", "PUT"):
            color = discord.Color.red()
        else:
            color = discord.Color.blue()

        title = f"{ticker} — {direction or 'Signal'}"

        embed = discord.Embed(
            title=title,
            color=color,
        )

        # Core trade numbers
        def fmt_price(v: Optional[float]) -> str:
            if v is None:
                return "—"
            return f"{v:.2f}"

        embed.add_field(name="Entry", value=fmt_price(p["entry"]), inline=True)
        embed.add_field(name="Target", value=fmt_price(p["target"]), inline=True)
        embed.add_field(name="Stop", value=fmt_price(p["stop"]), inline=True)

        # Probability
        prob = p["probability"]
        if prob is not None:
            if prob <= 1.0:
                prob_pct = prob * 100.0
            else:
                prob_pct = prob
            embed.add_field(name="Probability", value=f"{prob_pct:.1f}%", inline=True)

        # Timeframe
        if p["timeframe"]:
            embed.add_field(name="Timeframe", value=p["timeframe"], inline=True)

        # Rough RR estimate if both target & stop & entry present
        entry = p["entry"]
        target = p["target"]
        stop = p["stop"]
        rr_text = "—"
        if entry is not None and target is not None and stop is not None:
            try:
                reward = abs(target - entry)
                risk = abs(entry - stop)
                if risk > 0:
                    rr = reward / risk
                    rr_text = f"{rr:.2f} : 1"
            except Exception:
                rr_text = "—"
        embed.add_field(name="Est. R:R", value=rr_text, inline=True)

        # Footer timestamp if provided
        if p["timestamp"]:
            embed.set_footer(text=f"Alert time: {p['timestamp']}")

        return embed

    async def _generate_ai_summary(self, p: Dict[str, Any]) -> Optional[str]:
        """
        If OPENAI_API_KEY is set, generate a short tech-style summary.
        If not, return a simple text summary based on numbers.
        """
        # Fallback summary if no OpenAI
        if not self.openai_client:
            return (
                f"{p['direction']} setup on {p['ticker']} "
                f"{f'({p['timeframe']})' if p['timeframe'] else ''} — "
                f"Entry {p['entry']}, Target {p['target']}, Stop {p['stop']}, "
                f"Prob {p['probability']:.2f} if p['probability'] is not None else 'N/A'. "
                "Manage risk accordingly."
            )

        # Build prompt
        bullet = []
        bullet.append(f"Ticker: {p['ticker']}")
        bullet.append(f"Direction: {p['direction'] or 'N/A'}")
        if p["entry"] is not None:
            bullet.append(f"Entry: {p['entry']}")
        if p["target"] is not None:
            bullet.append(f"Target: {p['target']}")
        if p["stop"] is not None:
            bullet.append(f"Stop: {p['stop']}")
        if p["probability"] is not None:
            bullet.append(f"Probability: {p['probability']}")
        if p["timeframe"]:
            bullet.append(f"Timeframe: {p['timeframe']}")

        prompt = (
            "You are an experienced day/swing trader. "
            "Given this trade setup, write a short (1–3 sentences) calm, technical summary "
            "focused on risk management and key levels.\n\n"
            + "\n".join(bullet)
        )

        try:
            resp = await self.openai_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "You are a disciplined technical trader."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=120,
            )
            text = resp.choices[0].message.content.strip()
            return text
        except Exception:
            log.exception("AIEngine: OpenAI error, falling back to simple summary")
            return (
                f"{p['direction']} setup on {p['ticker']} "
                f"{f'({p['timeframe']})' if p['timeframe'] else ''}. "
                "Manage risk and size appropriately."
            )
