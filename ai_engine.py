import asyncio
import logging
from queue import Queue
from typing import Any, Dict, Optional, Tuple

import discord

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ai_engine")


class AIEngine:
    """
    R Black Hybrid System v1 engine.

    Flow:
      - webhook -> JSON payload -> enqueue_alert()
      - worker -> _handle_alert() -> grade setup -> build journal-style message
      - send message to Discord channel or per-ticker thread
    """

    def __init__(self, bot: discord.Client, channel_id: int):
        self.bot = bot
        self.channel_id = int(channel_id)
        self.queue: Queue[Dict[str, Any]] = Queue()
        # simple cache: ticker -> thread_id
        self.threads: Dict[str, int] = {}

    # --- called by webhook ---
    def enqueue_alert(self, payload: Dict[str, Any]) -> None:
        """Public method for webhook to push an alert in."""
        log.info(f"AIEngine: queued alert payload -> {payload}")
        self.queue.put(payload)

    # --- worker loop started from main.py on_ready ---
    async def _worker(self) -> None:
        log.info("AIEngine worker started")
        # wait until Discord bot is ready
        while not self.bot.is_ready():
            await asyncio.sleep(1)
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

    # --- core alert handler ---
    async def _handle_alert(self, payload: Dict[str, Any]) -> None:
        parsed = self._parse_payload(payload)
        log.info(f"AIEngine: parsed alert -> {parsed}")

        rr, risk, reward = self._compute_rr(parsed)
        grade, reasons = self._grade_setup(parsed, rr)

        msg = self._build_journal_message(parsed, rr, risk, reward, grade, reasons)

        channel = self.bot.get_channel(self.channel_id)
        if not isinstance(channel, discord.TextChannel):
            log.error(f"AIEngine: Channel {self.channel_id} not found or not a TextChannel")
            return

        # send to per-ticker thread if possible
        target = await self._get_ticker_thread(channel, parsed["ticker"])

        log.info(
            f"AIEngine: sending message for {parsed['ticker']} "
            f"to channel/thread {target.id} with grade {grade}"
        )
        try:
            await target.send(msg)
        except Exception:
            log.exception("AIEngine: error sending message to Discord")

    # --- parsing & helpers ---

    def _parse_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        We expect TradingView to send JSON shaped roughly like:

        {
          "ticker": "QQQ",
          "direction": "LONG",     // or BUY/SELL/SHORT
          "entry": "431.20",
          "stop": "427.80",
          "target": "441.40",
          "probability": "0.68",
          "timeframe": "15",
          "timestamp": "2025-11-24T12:30Z",

          // optional R Black flags (from Pine, if you add them):
          "htf_bias": "bullish" or "bearish" or "unclear",
          "volume_factor": 1.4,        // current volume / 20-bar avg
          "volume_ok": true,           // precomputed in Pine
          "macd_rsi_agree": true,      // precomputed in Pine
          "chop": false,               // true if range/choppy session
          "ema_zone_touch": true,      // price near 9/20 EMA or HTF zone
          "structure_ok": true         // engulfing / trend candle etc.
        }

        Missing fields are handled gracefully.
        """

        def _to_float(v: Any) -> Optional[float]:
            if v is None:
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        def _to_bool(v: Any) -> Optional[bool]:
            if isinstance(v, bool):
                return v
            if isinstance(v, str):
                lower = v.lower()
                if lower in ("true", "yes", "1"):
                    return True
                if lower in ("false", "no", "0"):
                    return False
            return None

        ticker = str(payload.get("ticker") or payload.get("symbol") or "UNKNOWN").upper()

        direction_raw = str(payload.get("direction") or payload.get("side") or "").upper()
        # normalize direction to LONG / SHORT / UNKNOWN
        if direction_raw in ("BUY", "LONG", "CALL"):
            direction = "LONG"
        elif direction_raw in ("SELL", "SHORT", "PUT"):
            direction = "SHORT"
        else:
            direction = direction_raw or "UNKNOWN"

        entry = _to_float(payload.get("entry") or payload.get("price"))
        stop = _to_float(payload.get("stop") or payload.get("stop_loss"))
        target = _to_float(payload.get("target") or payload.get("tp") or payload.get("take_profit"))

        prob = payload.get("probability")
        probability = None
        try:
            if prob is not None:
                probability = float(prob)
        except (TypeError, ValueError):
            probability = None

        timeframe = str(payload.get("timeframe") or payload.get("tf") or "").strip()
        timestamp = str(payload.get("timestamp") or payload.get("time") or "").strip()

        htf_bias = str(payload.get("htf_bias") or "").lower()  # "bullish", "bearish", "unclear"
        volume_factor = _to_float(payload.get("volume_factor"))
        volume_ok = _to_bool(payload.get("volume_ok"))
        macd_rsi_agree = _to_bool(payload.get("macd_rsi_agree"))
        chop = _to_bool(payload.get("chop"))
        ema_zone_touch = _to_bool(payload.get("ema_zone_touch"))
        structure_ok = _to_bool(payload.get("structure_ok"))

        return {
            "ticker": ticker,
            "direction": direction,
            "entry": entry,
            "stop": stop,
            "target": target,
            "probability": probability,      # 0–1 or 0–100, we’ll normalize later
            "timeframe": timeframe,
            "timestamp": timestamp,

            # R Black flags
            "htf_bias": htf_bias,
            "volume_factor": volume_factor,
            "volume_ok": volume_ok,
            "macd_rsi_agree": macd_rsi_agree,
            "chop": chop,
            "ema_zone_touch": ema_zone_touch,
            "structure_ok": structure_ok,
        }

    def _compute_rr(
        self,
        p: Dict[str, Any]
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Compute risk, reward, and RR based only on entry/stop/target.

        risk  = |entry - stop|
        reward= |target - entry|
        RR    = reward / risk
        """
        entry = p["entry"]
        stop = p["stop"]
        target = p["target"]

        if entry is None or stop is None or target is None:
            return None, None, None

        try:
            risk = abs(entry - stop)
            reward = abs(target - entry)
            if risk <= 0:
                return None, None, None
            rr = reward / risk
            return rr, risk, reward
        except Exception:
            return None, None, None

    def _grade_setup(
        self,
        p: Dict[str, Any],
        rr: Optional[float]
    ) -> Tuple[str, Dict[str, str]]:
        """
        Apply R Black Hybrid v1 logic to assign grade A+ / B / C and reasons.

        Rules used:
          - Must be 5m / 15m execution
          - No 1m scalps
          - HTF bias must be clear for A+ (bullish/bearish, not 'unclear')
          - Direction must agree with bias for A+
          - Volume > 20-bar avg (volume_ok or volume_factor > 1)
          - MACD & RSI must agree
          - Price must interact with EMA/HTF zone (ema_zone_touch)
          - Structure must be clean (structure_ok)
          - RR ≥ 3 for A+ (RR < 3 → not A+)
          - chop = True → not A+
        """

        reasons: Dict[str, str] = {}

        # --- timeframe rules ---
        tf = p["timeframe"].lower()
        # we treat 5/5m/05, 15/15m as valid execution; reject 1min
        is_5 = tf in ("5", "5m", "05")
        is_15 = tf in ("15", "15m")
        timeframe_ok = is_5 or is_15
        if not timeframe_ok:
            reasons["timeframe"] = "Execution timeframe is not 5m or 15m."

        # --- HTF bias + direction ---
        htf_bias = p["htf_bias"]  # bullish / bearish / unclear / ""
        direction = p["direction"]  # LONG / SHORT / UNKNOWN

        def dir_to_bias(d: str) -> Optional[str]:
            if d == "LONG":
                return "bullish"
            if d == "SHORT":
                return "bearish"
            return None

        dir_bias = dir_to_bias(direction)
        htf_clear = htf_bias in ("bullish", "bearish")
        direction_aligned = htf_clear and (dir_bias == htf_bias)

        if not htf_clear:
            reasons["htf_bias"] = "HTF trend is unclear or not provided."
        elif not direction_aligned:
            reasons["direction_htf"] = "Direction does not align with HTF bias."

        # --- volume ---
        vol_ok = False
        if p["volume_ok"] is True:
            vol_ok = True
        elif p["volume_factor"] is not None and p["volume_factor"] > 1.0:
            vol_ok = True

        if not vol_ok:
            reasons["volume"] = "Volume is not clearly above 20-bar average or volume flag is false."

        # --- MACD + RSI agreement ---
        if p["macd_rsi_agree"] is False:
            reasons["momentum"] = "MACD and RSI are not in agreement."

        # --- EMA / structure ---
        if p["ema_zone_touch"] is False:
            reasons["ema_zone"] = "Price is not interacting with 9/20 EMA cluster or HTF zone."

        if p["structure_ok"] is False:
            reasons["structure"] = "Candle structure is not a clean confirmation (engulfing/trend)."

        # --- chop ---
        if p["chop"] is True:
            reasons["chop"] = "Market is flagged as choppy/range — no-trade condition."

        # --- RR requirement ---
        if rr is None:
            reasons["rr"] = "RR could not be computed (missing entry/stop/target)."
        else:
            if rr < 3.0:
                reasons["rr"] = f"RR is {rr:.2f} (< 3R minimum for A+)."

        # --- grade decision ---
        core_ok = (
            timeframe_ok
            and vol_ok
            and p["macd_rsi_agree"] in (True, None)  # if None, we don't penalize
            and p["ema_zone_touch"] in (True, None)
            and p["structure_ok"] in (True, None)
            and p["chop"] is not True
            and direction_aligned
        )

        if rr is not None and rr >= 3.0 and core_ok and not reasons.get("rr"):
            grade = "A+"
        elif rr is not None and rr >= 2.0 and not p["chop"]:
            grade = "B"
        else:
            grade = "C"

        return grade, reasons

    # --- Discord formatting ---

    async def _get_ticker_thread(
        self,
        channel: discord.TextChannel,
        ticker: str
    ) -> discord.abc.Messageable:
        """
        Use a per-ticker thread named "{TICKER} Alerts".
        If cannot create / find, fall back to channel itself.
        """
        if not ticker or ticker == "UNKNOWN":
            return channel

        # reuse if cached
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

    def _build_journal_message(
        self,
        p: Dict[str, Any],
        rr: Optional[float],
        risk: Optional[float],
        reward: Optional[float],
        grade: str,
        reasons: Dict[str, str],
    ) -> str:
        """
        Build Style-2 "journal" message for Discord, based on R Black Hybrid v1.
        """

        # Entry/stop/target formatting
        def fmt_price(x: Optional[float]) -> str:
            if x is None:
                return "—"
            return f"{x:.2f}"

        # Probability formatting (if provided)
        prob_txt = "N/A"
        if p["probability"] is not None:
            prob_val = float(p["probability"])
            if prob_val <= 1.0:
                prob_pct = prob_val * 100.0
            else:
                prob_pct = prob_val
            prob_txt = f"{prob_pct:.1f}%"

        # Timeframe + type guess
        tf = p["timeframe"]
        trade_type = "Day Trade / Swing"
        try:
            tf_num = int(tf)
            if tf_num <= 15:
                trade_type = "Day Trade"
            elif tf_num >= 60:
                trade_type = "Swing"
        except Exception:
            pass

        # RR text
        if rr is None:
            rr_txt = "N/A"
        else:
            rr_txt = f"{rr:.2f}R"

        # Risk text
        risk_txt = "N/A" if risk is None else f"{risk:.2f}"

        # Top-down short summary (based on htf_bias + direction)
        htf_text = "Not provided"
        if p["htf_bias"] == "bullish":
            htf_text = "Bullish HTF bias"
        elif p["htf_bias"] == "bearish":
            htf_text = "Bearish HTF bias"
        elif p["htf_bias"]:
            htf_text = f"HTF bias: {p['htf_bias']}"

        # Reasoning block
        if not reasons:
            notes_header = "Reasoning:"
            notes_lines = [
                "- All core R Black filters satisfied.",
                "- Volume, momentum, EMA zone, and structure are acceptable.",
                "- Meets or exceeds 1:3 RR standard.",
            ]
        else:
            notes_header = "Reasoning / Flags:"
            notes_lines = [f"- {msg}" for msg in reasons.values()]

        lines = []

        # Header
        lines.append(f"📌 R Black Hybrid — {grade} Setup")
        lines.append(f"Symbol: {p['ticker']}")
        lines.append(f"Direction: {p['direction']}")
        lines.append(f"Type: {trade_type}")
        if tf:
            lines.append(f"Timeframe: {tf}m")
        if htf_text:
            lines.append(f"HTF: {htf_text}")
        if p["timestamp"]:
            lines.append(f"Time: {p['timestamp']}")
        lines.append("")  # blank line

        # Core numbers
        lines.append(f"Entry: {fmt_price(p['entry'])}")
        lines.append(f"Stop: {fmt_price(p['stop'])}  (Risk: {risk_txt})")
        lines.append(f"Target: {fmt_price(p['target'])}")
        lines.append(f"RR: {rr_txt}")
        lines.append(f"Probability: {prob_txt}")
        lines.append("")

        # Notes / reasoning
        lines.append(notes_header)
        lines.extend(notes_lines)

        # For quick grading clarity
        if grade == "A+":
            lines.append("")
            lines.append("✅ Qualifies as an A+ trade under R Black Hybrid v1.")
        elif grade == "B":
            lines.append("")
            lines.append("⚠ Solid but not perfect — B-grade setup under R Black rules.")
        else:
            lines.append("")
            lines.append("❌ C-grade setup — does not meet A+ criteria (review reasons above).")

        return "\n".join(lines)
