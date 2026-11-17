#!/usr/bin/env python3
"""
Finastic 20/50 EMA Crossover  -  TELEGRAM TRADE ALERTS  -  XAU/USD  -  1 MINUTE
(logic copied from finastic_2050_ema_v2.pine, data from Twelve Data)

Alert is sent ONCE, when a trade ACTIVATES (EMA cross armed -> pullback touches EMA20 ->
entry at that bar's close).  Message = entry, stop loss, all TP levels.
TP / SL hits are NOT alerted.  The bot still tracks them silently, because the Pine strategy
only takes a new trade when flat.

Run:   python finastic_alert_1m.py          (live)
       python finastic_alert_1m.py test     (send one test message to Telegram)
"""
from __future__ import annotations

import bisect
import logging
import math
import sys
import time
from collections import deque
from typing import Optional

import pandas as pd
import requests

# ═══════════════════════════════════════════════════════════════════════════
# HARDCODED SETTINGS  -  EDIT HERE
# ═══════════════════════════════════════════════════════════════════════════
TELEGRAM_TOKEN     = "PASTE_YOUR_BOT_TOKEN_HERE"
TELEGRAM_CHAT_ID   = "PASTE_YOUR_CHAT_ID_HERE"
TWELVEDATA_API_KEY = "3410684668ac4b138d3c2485ac3a9c3b"

TF            = 1                 # minutes (this file = 1)
SYMBOL        = "XAU/USD"
DAILY_BUDGET  = 780               # free plan = 800 requests/day. Paid plan: raise this.
PER_MIN_LIMIT = 8                 # free plan = 8 requests/minute
WARMUP_BARS   = 1500              # history replayed at start (silent, no alerts)
FETCH_DELAY   = 3                 # seconds to wait after a bar closes
STALE_BARS    = 3                 # entry older than this many bars is NOT alerted (late data)

# ---- strategy inputs: same defaults as the Pine script ----
FAST, SLOW   = 20, 50
STOP_USD     = 13.0
MAX_WAIT     = 30
USE_TREND    = False              # Pine "Higher-TF trend filter" (off by default)
HTF_MIN      = 60
USE_LONG     = True
USE_SHORT    = True
PIP_SIZE     = 0.10
RUNGS        = [(1.0, 20.0), (1.5, 20.0), (2.0, 20.0), (2.3, 20.0), (2.5, 20.0), (3.0, 0.0)]  # (R, %)
BE_AFTER     = 0
CAPITAL      = 10000.0
QTY_PCT      = 100.0

NAN = float("nan")
log = logging.getLogger("finastic")


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _isnan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


# ═══════════════════════════════════════════════════════════════════════════
# ta.ema  (Pine: first value = first source, then alpha smoothing)
# ═══════════════════════════════════════════════════════════════════════════
class EMA:
    def __init__(self, n: int):
        self.alpha, self.val = 2.0 / (n + 1), NAN

    def update(self, x: float) -> float:
        self.val = x if _isnan(self.val) else self.alpha * x + (1 - self.alpha) * self.val
        return self.val


# ═══════════════════════════════════════════════════════════════════════════
# Telegram
# ═══════════════════════════════════════════════════════════════════════════
def telegram(text: str) -> bool:
    for attempt in range(1, 4):
        try:
            r = requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                              data={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=10)
            r.raise_for_status()
            log.info("Telegram alert sent")
            return True
        except Exception as e:                 # never log the URL: it holds the token
            log.error("Telegram failed (%s) try %d/3", type(e).__name__, attempt)
            time.sleep(2)
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Strategy  (exact port: arming, entry, ladder, TradingView order-fill emulation)
# ═══════════════════════════════════════════════════════════════════════════
class Finastic:
    def __init__(self, on_entry=None):
        self.on_entry = on_entry
        self.alerts_enabled = True
        self.ef, self.es = EMA(FAST), EMA(SLOW)
        self.prev_fast = self.prev_slow = NAN
        self.bar_index = -1

        self.longWait = self.shortWait = False
        self.waitBars = 0

        self.lvlR, self.lvlW = self._build_ladder()
        self.nLvl = len(self.lvlR)

        self.eEntry = self.eSL = NAN
        self.eDir = 0
        self.initQty: Optional[float] = None
        self.tpDone = 0

        self.pos_size = 0.0
        self.equity = CAPITAL
        self.orders: dict = {}
        self.pending_entry: Optional[tuple] = None
        self.fills: list = []

    def lvlPx(self, i): return self.eEntry + self.eDir * self.lvlR[i] * STOP_USD
    def pipsOf(self, r): return r * STOP_USD / PIP_SIZE

    @staticmethod
    def _build_ladder():
        R, W = [], []
        for r, w in RUNGS:                          # rungs must ascend, else dropped
            if r > 0 and w > 0 and r > (R[-1] if R else 0.0):
                R.append(r)
                W.append(w)
        if R:
            wsum = sum(W)
            if wsum > 100.0:
                W = [w * 100.0 / wsum for w in W]   # over-allocated -> scale back
            elif wsum < 100.0:
                W[-1] += 100.0 - wsum               # under-allocated -> last rung takes balance
        return R, W

    # -------- broker emulator: fills strategy.exit orders placed on earlier bars
    def _broker(self, o, h, l, c):
        if self.pos_size == 0 or not self.orders:
            return
        d = 1 if self.pos_size > 0 else -1
        path = [o, h, l, c] if abs(o - h) < abs(o - l) else [o, l, h, c]

        def first_touch(kind, level):
            if kind == "limit" and d * (o - level) >= 0:
                return (-1, 0.0), o
            if kind == "stop" and d * (o - level) <= 0:
                return (-1, 0.0), o
            for s in range(3):
                a, b = path[s], path[s + 1]
                if a != b and min(a, b) <= level <= max(a, b):
                    return (s, abs(level - a) / abs(b - a)), level
            return None, None

        trig = []
        for k, (oid, od) in enumerate(self.orders.items()):
            best = None
            for kind in ("stop", "limit"):          # stop first -> conservative on ties
                lvl = od[kind]
                if lvl is None:
                    continue
                t, px = first_touch(kind, lvl)
                if t is not None and (best is None or t < best[0]):
                    best = (t, px)
            if best:
                trig.append((best[0], k, oid, best[1]))
        trig.sort(key=lambda x: (x[0], x[1]))

        for _, _, oid, px in trig:
            if self.pos_size == 0:
                break
            od = self.orders.pop(oid)
            size = abs(self.pos_size)
            q = size if od["qty"] is None else min(od["qty"], size)
            self.fills.append((q, px))
            self.equity += q * (px - self.eEntry) * d
            self.pos_size -= d * q
            if abs(self.pos_size) < 1e-12:
                self.pos_size = 0.0
        if self.pos_size == 0:
            self.orders.clear()

    # -------- one CLOSED bar
    def on_bar(self, ts, o, h, l, c, htf_bull: bool = False):
        self.bar_index += 1
        self._broker(o, h, l, c)

        fast, slow = self.ef.update(c), self.es.update(c)
        have_prev = not (_isnan(self.prev_fast) or _isnan(self.prev_slow))
        have_now = not (_isnan(fast) or _isnan(slow))
        bullCross = have_prev and have_now and fast > slow and self.prev_fast <= self.prev_slow
        bearCross = have_prev and have_now and fast < slow and self.prev_fast >= self.prev_slow
        self.prev_fast, self.prev_slow = fast, slow

        # arming
        if bullCross:
            self.longWait, self.shortWait, self.waitBars = True, False, 0
        if bearCross:
            self.shortWait, self.longWait, self.waitBars = True, False, 0
        if self.longWait or self.shortWait:
            self.waitBars += 1
            if self.waitBars > MAX_WAIT:
                self.longWait = self.shortWait = False

        nLvl = self.nLvl

        # entry
        flat = self.pos_size == 0 and self.initQty is None
        longSig = USE_LONG and self.longWait and l <= fast and flat and (not USE_TREND or htf_bull)
        shortSig = USE_SHORT and self.shortWait and h >= fast and flat and (not USE_TREND or not htf_bull)

        if (longSig or shortSig) and nLvl > 0:
            self.eDir = 1 if longSig else -1
            self.eEntry = c
            self.eSL = c - self.eDir * STOP_USD
            self.tpDone = 0
            self.initQty = None
            self.fills = []
            self.pending_entry = (self.eDir, self.equity * QTY_PCT / 100.0 / c)
            self.longWait = self.shortWait = False
            self._announce_entry(ts)

        if self.pos_size != 0 and self.initQty is None:
            self.initQty = abs(self.pos_size)

        # ladder management
        if self.pos_size != 0 and self.initQty is not None and nLvl > 0:
            filled = self.initQty - abs(self.pos_size)
            cum, done = 0.0, 0
            for i in range(nLvl):
                qi = self.initQty * self.lvlW[i] / 100.0
                cum += qi
                if filled >= cum - qi * 0.5:
                    done = i + 1
            if done > self.tpDone:
                for i in range(self.tpDone, done):
                    log.info("TP%d hit @ %.2f  [%s]", i + 1, self.lvlPx(i), ts)
                self.tpDone = done
                if BE_AFTER > 0 and self.tpDone >= BE_AFTER:
                    self.eSL = self.eEntry
            fromId = "Long" if self.eDir > 0 else "Short"
            placed = 0.0
            for i in range(nLvl):
                last = i == nLvl - 1
                q = self.initQty - placed if last else self.initQty * self.lvlW[i] / 100.0
                placed += q
                if i >= self.tpDone:
                    if last:
                        self.orders[f"TP{i + 1}"] = dict(qty=None, limit=self.lvlPx(i), stop=self.eSL, frm=fromId)
                    elif q > 0:
                        self.orders[f"TP{i + 1}"] = dict(qty=q, limit=self.lvlPx(i), stop=self.eSL, frm=fromId)

        # trade closed (silent - log only)
        if self.pos_size == 0 and self.initQty is not None:
            cash = sum(abs(q) * (px - self.eEntry) * self.eDir for q, px in self.fills)
            realR = cash / (self.initQty * STOP_USD)
            log.info("Trade closed %s  %+.2fR  (%.0f pips)  [%s]",
                     "LONG" if self.eDir > 0 else "SHORT", realR, self.pipsOf(realR), ts)
            self.initQty = None
            self.tpDone = 0
            self.longWait = self.shortWait = False
            self.orders.clear()

        # entry fills at the close of the signal bar (process_orders_on_close=true)
        if self.pending_entry:
            d, q = self.pending_entry
            self.pos_size = d * q
            self.pending_entry = None

    # -------- the alert
    def _announce_entry(self, ts):
        d = "LONG" if self.eDir > 0 else "SHORT"
        t_close = (ts + pd.Timedelta(minutes=TF)).tz_convert("Asia/Kolkata")
        lines = [f"{'🟢' if self.eDir > 0 else '🔴'} FINASTIC {TF}M  -  {d}  XAUUSD",
                 f"Entry: {self.eEntry:.2f}",
                 f"SL:    {self.eSL:.2f}   ({self.pipsOf(1.0):.0f} pips / ${STOP_USD:g})",
                 ""]
        for i in range(self.nLvl):
            lines.append(f"TP{i + 1}:  {self.lvlPx(i):.2f}   1:{self.lvlR[i]:g}  "
                         f"(+{self.pipsOf(self.lvlR[i]):.0f}p)  {self.lvlW[i]:.0f}%")
        lines += ["", f"Bar closed {t_close:%d %b %H:%M} IST"]
        msg = "\n".join(lines)
        log.info("NEW TRADE\n%s", msg)
        if self.alerts_enabled and self.on_entry:
            self.on_entry(dict(text=msg, ts=ts))


# ═══════════════════════════════════════════════════════════════════════════
# Twelve Data client (budget-aware; 3m / 10m built locally from 1m / 5m bars)
# ═══════════════════════════════════════════════════════════════════════════
URL = "https://api.twelvedata.com/time_series"
BASE = {1: ("1min", 1), 3: ("1min", 1), 5: ("5min", 5), 15: ("15min", 15), 60: ("1h", 60)}


class TwelveData:
    def __init__(self, key: str, symbol: str = SYMBOL):
        self.key, self.symbol = key, symbol
        self.day, self.used = None, 0
        self.recent = deque()

    def _throttle(self):
        while True:
            now = _now()
            if self.day != now.date():
                self.day, self.used = now.date(), 0
            if self.used >= DAILY_BUDGET:
                wait = (pd.Timestamp(now.date(), tz="UTC") + pd.Timedelta(days=1) - now).total_seconds() + 5
                log.warning("Daily request budget (%d) used up - sleeping until 00:00 UTC", DAILY_BUDGET)
                time.sleep(min(wait, 600))
                continue
            while self.recent and (now - self.recent[0]).total_seconds() >= 60:
                self.recent.popleft()
            if len(self.recent) >= PER_MIN_LIMIT:
                time.sleep(max(1.0, 60 - (now - self.recent[0]).total_seconds() + 0.5))
                continue
            self.recent.append(now)
            self.used += 1
            return

    def _raw(self, interval: str, n: int, since=None, tf: int = 1) -> pd.DataFrame:
        self._throttle()
        if since is not None:
            k = tf // BASE[tf][1]
            n = (int((_now() - since) / pd.Timedelta(minutes=tf)) + 3) * k + k
        r = requests.get(URL, timeout=20, params=dict(
            symbol=self.symbol, interval=interval, outputsize=int(min(max(n, 2), 5000)),
            timezone="UTC", order="asc", apikey=self.key))
        j = r.json()
        if j.get("status") == "error" or "values" not in j:
            raise RuntimeError(f"TwelveData: {j.get('message', j)}")
        df = pd.DataFrame(j["values"])
        df.index = pd.to_datetime(df["datetime"], utc=True)          # bar OPEN time
        df = df[["open", "high", "low", "close"]].astype(float).dropna()
        return df[~df.index.duplicated(keep="last")].sort_index()

    def bars(self, tf: int, n: int = 0, since=None) -> pd.DataFrame:
        """Last ~n CLOSED bars of timeframe tf (forming bar removed)."""
        interval, bmin = BASE[tf]
        k = tf // bmin
        df = self._raw(interval, n * k + k, since, tf)
        df = df[df.index + pd.Timedelta(minutes=bmin) <= _now()]
        if df.empty or k == 1:
            return df
        last_close = df.index[-1] + pd.Timedelta(minutes=bmin)
        out = df.resample(f"{tf}min", origin="epoch").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        return out[out.index + pd.Timedelta(minutes=tf) <= last_close]


class HTF:
    """request.security(..., lookahead_off): EMA20>EMA50 on COMPLETED higher-TF bars."""
    def __init__(self, td: TwelveData):
        self.td, self.m = td, HTF_MIN
        self.ef, self.es = EMA(FAST), EMA(SLOW)
        self.close_times, self.bull, self.last = [], [], None

    def refresh(self, n: int = 3):
        for ts, c in self.td.bars(self.m, n)["close"].items():
            if self.last is not None and ts <= self.last:
                continue
            self.bull.append(self.ef.update(c) > self.es.update(c))
            self.close_times.append(ts + pd.Timedelta(minutes=self.m))
            self.last = ts

    def due(self) -> bool:
        return self.last is None or _now() >= self.last + 2 * pd.Timedelta(minutes=self.m) + pd.Timedelta(seconds=FETCH_DELAY)

    def bull_at(self, t_close) -> bool:
        i = bisect.bisect_right(self.close_times, t_close) - 1
        return bool(self.bull[i]) if i >= 0 else False


# ═══════════════════════════════════════════════════════════════════════════
def feed(strat: Finastic, df: pd.DataFrame, htf: Optional[HTF]):
    tfd = pd.Timedelta(minutes=TF)
    for ts, r in df.iterrows():
        strat.on_bar(ts, r.open, r.high, r.low, r.close, htf.bull_at(ts + tfd) if htf else False)


def pos_name(s: Finastic) -> str:
    return "LONG" if s.pos_size > 0 else "SHORT" if s.pos_size < 0 else "flat"


def run_live():
    tfd = pd.Timedelta(minutes=TF)
    td = TwelveData(TWELVEDATA_API_KEY)

    def on_entry(e):
        if _now() - (e["ts"] + tfd) > STALE_BARS * tfd:      # e.g. bot was asleep / data came late
            log.warning("Stale entry (%s) - not alerted", e["ts"])
            return
        telegram(e["text"])

    strat = Finastic(on_entry=on_entry)
    est = int(23 * 60 / TF)
    log.info("Timeframe %dm -> about %d API requests/day while market open (free limit 800)%s", TF, est,
             "  *** NEEDS A PAID TWELVE DATA PLAN ***" if est > 800 else "")

    htf = HTF(td) if USE_TREND else None
    if htf:
        htf.refresh(600)
    k = TF // BASE[TF][1]
    df = td.bars(TF, min(WARMUP_BARS, 5000 // k - 1))
    strat.alerts_enabled = False                              # warm-up: silent
    feed(strat, df, htf)
    strat.alerts_enabled = True
    last = df.index[-1]
    log.info("Warm-up done: %d bars, last closed bar %s UTC, position: %s", len(df), last, pos_name(strat))
    telegram(f"✅ Finastic {TF}M alert bot is LIVE (XAUUSD)")

    retry = 5
    while True:
        try:
            now = _now()
            due = last + 2 * tfd + pd.Timedelta(seconds=FETCH_DELAY)
            if now < due:
                time.sleep(min((due - now).total_seconds(), 60))
                continue
            if htf and htf.due():
                htf.refresh()
            df = td.bars(TF, since=last)
            new = df[df.index > last]
            if new.empty:                                     # data lag / market closed
                time.sleep(retry)
                retry = min(retry * 2, 600)
                continue
            retry = 5
            feed(strat, new, htf)
            last = new.index[-1]
            log.info("bar %s | close %.2f | EMA%d %.2f EMA%d %.2f | pos %s | API today %d",
                     last, new.close.iloc[-1], FAST, strat.prev_fast, SLOW, strat.prev_slow,
                     pos_name(strat), td.used)
        except KeyboardInterrupt:
            return
        except Exception as e:
            log.error("loop error: %s: %s", type(e).__name__, str(e)[:200])
            time.sleep(30)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    if TELEGRAM_TOKEN.startswith("PASTE") or TELEGRAM_CHAT_ID.startswith("PASTE"):
        raise SystemExit("Edit TELEGRAM_TOKEN and TELEGRAM_CHAT_ID at the top of this file first.")
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        telegram(f"✅ Finastic {TF}M test alert - Telegram wiring works.")
        return
    run_live()


if __name__ == "__main__":
    main()
