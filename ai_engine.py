import asyncio
import logging
from queue import Queue
from typing import Any, Dict, Optional, Tuple

import discord

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ai_engine")


class AIEngine:
    """
    R Black Hybrid System v1 engine (refined).

    Flow:
      - webhook -> JSON payload -> enqueue_alert()
      - worker -> _handle_alert() -> apply R Black logic
      - compute RR, grade (A+/B/C), internal probability
      - build journal-style message and send to Discord
    """

    def __init__(self, bot: discord.Client, channel_id: int):
        self.bot = bot
        self.channel_id = int(channel_id)
        self.queue: Queue[Dict[str, Any]] = Queue()
        # simple cache: ticker -> thread_id
        self.threads: Dict[str, int] = {}

    # --- called by webhook ---
    def enqueue_alert(self, payload: Dict[str, Any]) -> None:
        """Preferred enqueue method for webhook."""
        log.info(f"AIEngine: queued alert payload -> {payload}")
        self.queue.put(payload)

    # backward compatible alias if webhook still calls enqueue()
    def enqueue(self, payload: Dict[str, Any]) -> None:
        self.enqueue_alert(payload)

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

        # evaluate HTF, volume & other filters, and compute internal probability
        htf_info = self._evaluate_htf_stack(parsed)
        vol_info = self._evaluate_volume(parsed)
        grade, reasons, internal_prob = self._grade_setup(parsed, rr, htf_info, vol_info)

        msg = self._build_journal_message(
            parsed,
            rr,
            risk,
            reward,
            grade,
            reasons,
            internal_prob,
            htf_info,
            vol_info,
        )

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
        Expected JSON (fields are optional, engine is defensive):

        {
          "ticker": "QQQ",
          "direction": "LONG",     // or BUY/SELL/SHORT/CALL/PUT
          "entry": "431.20",
          "stop": "427.80",
          "target": "441.40",
          "probability": "0.68",   // 0-1 or 0-100

          "timeframe": "15",
          "timestamp": "2025-11-24T12:30Z",

          // Optional R Black flags from Pine:
          "htf_bias": "bullish" or "bearish" or "unclear",

          // Or more detailed HTF stack (optional):
          "htf_monthly": "bullish" | "bearish" | "neutral",
          "htf_weekly":  "bullish" | "bearish" | "neutral",
          "htf_daily":   "bullish" | "bearish" | "neutral",
          "htf_4h":      "bullish" | "bearish" | "neutral",

          // Volume & structure:
          "volume_factor": 1.4,         // current volume / 20-bar average
          "volume_trend": "increasing" | "flat" | "decreasing",
          "volume_breakout": true,      // true if this is breakout candle
          "volume_ok": true,            // precomputed in Pine if you like

          "macd_rsi_agree": true,
          "chop": false,
          "ema_zone_touch": true,       // price near 9/20 EMA cluster / HTF zone
          "structure_ok": true,         // engulfing / strong trend candle

          // Optional pattern tag from Pine:
          "pattern": "bull_engulf" | "bear_engulf" | "hammer" | "shooting_star" | "doji" | ...
        }
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

        # Simple HTF bias if that’s all you send
        htf_bias = str(payload.get("htf_bias") or "").lower()  # "bullish", "bearish", "unclear"

        # Optional detailed HTF stack
        def _norm_bias(x: Any) -> Optional[str]:
            if x is None:
                return None
            s = str(x).lower()
            if s in ("bull", "bullish"):
                return "bullish"
            if s in ("bear", "bearish"):
                return "bearish"
            if s in ("flat", "neutral", "range"):
                return "neutral"
            return None

        htf_monthly = _norm_bias(payload.get("htf_monthly"))
        htf_weekly = _norm_bias(payload.get("htf_weekly"))
        htf_daily = _norm_bias(payload.get("htf_daily"))
        htf_4h = _norm_bias(payload.get("htf_4h"))

        volume_factor = _to_float(payload.get("volume_factor"))
        volume_trend = str(payload.get("volume_trend") or "").lower()  # "increasing", "flat", "decreasing"
        volume_breakout = _to_bool(payload.get("volume_breakout"))
        volume_ok = _to_bool(payload.get("volume_ok"))

        macd_rsi_agree = _to_bool(payload.get("macd_rsi_agree"))
        chop = _to_bool(payload.get("chop"))
        ema_zone_touch = _to_bool(payload.get("ema_zone_touch"))
        structure_ok = _to_bool(payload.get("structure_ok"))

        pattern = str(payload.get("pattern") or "").lower()

        return {
            "ticker": ticker,
            "direction": direction,
            "entry": entry,
            "stop": stop,
            "target": target,
            "probability": probability,      # 0–1 or 0–100, normalized later
            "timeframe": timeframe,
            "timestamp": timestamp,

            # HTF info
            "htf_bias": htf_bias,
            "htf_monthly": htf_monthly,
            "htf_weekly": htf_weekly,
            "htf_daily": htf_daily,
            "htf_4h": htf_4h,

            # Volume / structure
            "volume_factor": volume_factor,
            "volume_trend": volume_trend,
            "volume_breakout": volume_breakout,
            "volume_ok": volume_ok,
            "macd_rsi_agree": macd_rsi_agree,
            "chop": chop,
            "ema_zone_touch": ema_zone_touch,
            "structure_ok": structure_ok,

            # Pattern tag
            "pattern": pattern,
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

    # --- HTF evaluation ---

    def _dir_to_bias(self, d: str) -> Optional[str]:
        if d == "LONG":
            return "bullish"
        if d == "SHORT":
            return "bearish"
        return None

    def _evaluate_htf_stack(self, p: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score HTF stack: Monthly / Weekly / Daily / 4h.

        Weights:
          M: +3 / -3
          W: +2 / -2
          D: +1 / -1
          4h:+1 / -1

        Thresholds:
          score >= +5 -> Bullish stack
          score <= -5 -> Bearish stack
          else        -> Unclear
        """

        def score_bias(bias: Optional[str], weight: int) -> int:
            if bias == "bullish":
                return weight
            if bias == "bearish":
                return -weight
            return 0

        m = p["htf_monthly"]
        w = p["htf_weekly"]
        d = p["htf_daily"]
        h4 = p["htf_4h"]

        # if no detailed info, fall back to simple htf_bias
        if not any([m, w, d, h4]):
            simple = p["htf_bias"]
            label = ""
            score = 0
            if simple == "bullish":
                label = "Bullish HTF bias"
                score = +5
            elif simple == "bearish":
                label = "Bearish HTF bias"
                score = -5
            else:
                label = "HTF bias unclear"

            dir_bias = self._dir_to_bias(p["direction"])
            aligned = (dir_bias is not None and simple and dir_bias == simple)
            reason = ""
            if not simple:
                reason = "HTF bias not provided."
            elif not aligned:
                reason = "Direction does not align with HTF bias."

            return {
                "score": score,
                "label": label,
                "aligned": aligned,
                "reason": reason,
            }

        # detailed stack scoring
        score = 0
        score += score_bias(m, 3)
        score += score_bias(w, 2)
        score += score_bias(d, 1)
        score += score_bias(h4, 1)

        if score >= 5:
            bias_label = "Bullish HTF stack"
            stack_bias = "bullish"
        elif score <= -5:
            bias_label = "Bearish HTF stack"
            stack_bias = "bearish"
        else:
            bias_label = "HTF stack unclear"
            stack_bias = "unclear"

        dir_bias = self._dir_to_bias(p["direction"])
        aligned = (dir_bias is not None and stack_bias in ("bullish", "bearish") and dir_bias == stack_bias)
        reason = ""
        if stack_bias == "unclear":
            reason = "HTF stack not clearly bullish or bearish."
        elif not aligned:
            reason = "Direction does not align with HTF stack bias."

        return {
            "score": score,
            "label": bias_label,
            "aligned": aligned,
            "reason": reason,
        }

    # --- Volume evaluation ---

    def _evaluate_volume(self, p: Dict[str, Any]) -> Dict[str, Any]:
        """
        Advanced volume check.

        Rules (no new indicators, just meta info from Pine):

        - volume_factor > 1.0       => base OK
        - volume_factor >= 1.5      => strong
        - volume_trend increasing   => bonus
        - volume_breakout true      => bonus
        - if volume_factor <= 1.0 and trend decreasing => weak / contracting
        """

        factor = p["volume_factor"]
        trend = p["volume_trend"]  # "increasing" / "flat" / "decreasing"
        breakout = p["volume_breakout"]
        vol_ok_flag = p["volume_ok"]

        score = 0
        ok = False
        reasons = []

        if vol_ok_flag is True:
            ok = True
            score += 2

        if factor is not None:
            if factor > 1.0:
                ok = True
                score += 1
            if factor >= 1.5:
                score += 1
            if factor <= 1.0:
                reasons.append("Volume is not clearly above its 20-bar average.")

        if trend == "increasing":
            score += 1
        elif trend == "decreasing":
            reasons.append("Volume is contracting.")
        # breakout candle volume
        if breakout is True:
            score += 1

        if not ok:
            reasons.append("Volume conditions not clearly met for validated breakout/pullback.")

        reason_text = "; ".join(reasons) if reasons else ""

        return {
            "ok": ok,
            "score": score,
            "reason": reason_text,
        }

    # --- Internal probability ---

    def _compute_internal_probability(
        self,
        p: Dict[str, Any],
        rr: Optional[float],
        htf_info: Dict[str, Any],
        vol_info: Dict[str, Any],
        core_flags: Dict[str, bool],
    ) -> Optional[float]:
        """
        Simple weighted probability engine, staying within your rules.

        Factors:
          - HTF alignment          (30%)
          - Volume quality         (20%)
          - MACD/RSI agreement     (20%)
          - EMA/zone support       (15%)
          - Structure confirmation (10%)
          - No-chop                (5%)

        Output: 0–100 %
        """

        score = 50.0  # baseline

        # RR influence (gently)
        if rr is not None:
            if rr >= 3.0:
                score += 10
            if rr >= 4.0:
                score += 5
            if rr < 3.0:
                score -= 10
            if rr < 2.0:
                score -= 5

        # HTF alignment
        if htf_info["aligned"]:
            score += 20
        elif htf_info["reason"]:
            score -= 15

        # Volume
        if vol_info["ok"]:
            score += min(vol_info["score"] * 5.0, 15.0)
        else:
            score -= 15

        # Momentum (MACD + RSI)
        if core_flags.get("momentum_ok", False):
            score += 10
        elif core_flags.get("momentum_checked", False):
            score -= 10

        # EMA / zones
        if core_flags.get("ema_zone_ok", False):
            score += 7
        elif core_flags.get("ema_zone_checked", False):
            score -= 7

        # Structure
        if core_flags.get("structure_ok", False):
            score += 10
        elif core_flags.get("structure_checked", False):
            score -= 10

        # Chop filter
        if core_flags.get("not_choppy", False):
            score += 5
        elif core_flags.get("choppy", False):
            score -= 25

        # Clamp
        score = max(0.0, min(100.0, score))
        return score

    def _grade_setup(
        self,
        p: Dict[str, Any],
        rr: Optional[float],
        htf_info: Dict[str, Any],
        vol_info: Dict[str, Any],
    ) -> Tuple[str, Dict[str, str], Optional[float]]:
        """
        Apply R Black Hybrid v1 logic to assign grade A+ / B / C and reasons.

        Rules used:
          - Execution must be 5m / 15m (no 1m scalps)
          - HTF stack should support direction for A+
          - Volume must be convincingly strong
          - MACD & RSI must agree
          - Price should interact with EMA/HTF zone
          - Structure should be clean confirmation
          - RR >= 3 for A+ (RR < 3 => not A+)
          - chop = True => not A+
          - We still SHOW B/C setups (as you requested), but label clearly.
        """

        reasons: Dict[str, str] = {}

        # --- timeframe rules ---
        tf = p["timeframe"].lower()
        is_5 = tf in ("5", "5m", "05")
        is_15 = tf in ("15", "15m")
        timeframe_ok = is_5 or is_15
        if not timeframe_ok:
            reasons["timeframe"] = "Execution timeframe is not 5m or 15m (no 1-min scalps allowed)."

        # --- HTF ---
        if htf_info["reason"]:
            reasons["htf"] = htf_info["reason"]

        # --- volume ---
        if not vol_info["ok"]:
            if vol_info["reason"]:
                reasons["volume"] = vol_info["reason"]
            else:
                reasons["volume"] = "Volume is not convincingly above average or expanding."

        # --- MACD + RSI agreement ---
        momentum_ok = False
        if p["macd_rsi_agree"] is True:
            momentum_ok = True
        elif p["macd_rsi_agree"] is False:
            reasons["momentum"] = "MACD and RSI are not in agreement."

        # --- EMA / HTF zone ---
        ema_zone_ok = False
        if p["ema_zone_touch"] is True:
            ema_zone_ok = True
        elif p["ema_zone_touch"] is False:
            reasons["ema_zone"] = "Price is not interacting with 9/20 EMA cluster or HTF zone."

        # --- structure ---
        structure_ok = False
        if p["structure_ok"] is True:
            structure_ok = True
        elif p["structure_ok"] is False:
            reasons["structure"] = "Candle structure is not a clean confirmation (engulfing/trend)."

        # --- chop ---
        not_choppy = True
        if p["chop"] is True:
            not_choppy = False
            reasons["chop"] = "Market is flagged as choppy/range — no-trade condition under R Black."

        # --- RR requirement ---
        if rr is None:
            reasons["rr"] = "RR could not be computed (missing entry/stop/target)."
        else:
            if rr < 3.0:
                reasons["rr"] = f"RR is {rr:.2f} (< 3R minimum A+ standard)."

        core_ok = (
            timeframe_ok
            and htf_info["aligned"]
            and vol_info["ok"]
            and (momentum_ok or p["macd_rsi_agree"] is None)
            and (ema_zone_ok or p["ema_zone_touch"] is None)
            and (structure_ok or p["structure_ok"] is None)
            and not_choppy
        )

        # Grade decision (you chose to still see B/C)
        if rr is not None and rr >= 3.0 and core_ok and not reasons.get("rr"):
            grade = "A+"
        elif rr is not None and rr >= 2.0 and not p["chop"]:
            grade = "B"
        else:
            grade = "C"

        # core flags for probability model
        core_flags = {
            "momentum_ok": momentum_ok,
            "momentum_checked": p["macd_rsi_agree"] is not None,
            "ema_zone_ok": ema_zone_ok,
            "ema_zone_checked": p["ema_zone_touch"] is not None,
            "structure_ok": structure_ok,
            "structure_checked": p["structure_ok"] is not None,
            "not_choppy": not_choppy,
            "choppy": p["chop"] is True,
        }

        internal_prob = self._compute_internal_probability(p, rr, htf_info, vol_info, core_flags)

        return grade, reasons, internal_prob

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
        internal_prob: Optional[float],
        htf_info: Dict[str, Any],
        vol_info: Dict[str, Any],
    ) -> str:
        """
        Build Style-2 "journal" message for Discord, based on R Black Hybrid v1.
        """

        def fmt_price(x: Optional[float]) -> str:
            if x is None:
                return "—"
            return f"{x:.2f}"

        # Probability display:
        # - If Pine sends prob, use that.
        # - Otherwise use internal probability.
        prob_txt = "N/A"
        external = p["probability"]
        if external is not None:
            try:
                val = float(external)
                if val <= 1.0:
                    prob_pct = val * 100.0
                else:
                    prob_pct = val
                prob_txt = f"{prob_pct:.1f}% (Pine)"
            except Exception:
                prob_txt = "N/A"
        elif internal_prob is not None:
            prob_txt = f"{internal_prob:.1f}% (System)"

        # Timeframe + trade type guess
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
        rr_txt = "N/A" if rr is None else f"{rr:.2f}R"
        risk_txt = "N/A" if risk is None else f"{risk:.2f}"

        # HTF summary
        htf_label = htf_info.get("label") or "HTF context not provided"

        # Reasoning block
        if not reasons:
            notes_header = "Reasoning:"
            notes_lines = [
                "- All core R Black filters satisfied.",
                "- HTF stack, volume, momentum, EMA zone, and structure align.",
                "- Meets or exceeds 1:3 RR standard.",
            ]
        else:
            notes_header = "Reasoning / Flags:"
            notes_lines = [f"- {msg}" for msg in reasons.values()]

        # Pattern note (if provided)
        pattern_line = ""
        pat = p.get("pattern") or ""
        if pat:
            pattern_line = f"Pattern: {pat}"

        lines = []

        # Header
        lines.append(f"📌 R Black Hybrid — {grade} Setup")
        lines.append(f"Symbol: {p['ticker']}")
        lines.append(f"Direction: {p['direction']}")
        lines.append(f"Type: {trade_type}")
        if tf:
            lines.append(f"Timeframe: {tf}m")
        if htf_label:
            lines.append(f"HTF: {htf_label}")
        if p["timestamp"]:
            lines.append(f"Time: {p['timestamp']}")
        if pattern_line:
            lines.append(pattern_line)
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

        # Grade summary
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
