"""Tushare API client for fetching Chinese government bond futures data.

All Tushare API interaction is encapsulated here:
- fut_basic   — contract listing
- fut_mapping — dominant-contract → actual-contract resolution
- ft_mins     — 5-minute OHLC bars
- fut_daily   — daily OHLC bars

For 5-min data, *ft_mins* requires actual contract codes (e.g. T2609.CFX),
not continuous symbols.  We use *fut_mapping* to resolve dominant contracts
on each trading day, then call *ft_mins* per contract code.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import tushare as ts

logger = logging.getLogger(__name__)

# ── Token ─────────────────────────────────────────────────────────────


def get_token() -> str:
    """Return TUSHARE_TOKEN from environment.

    Also tries loading from .env via python-dotenv if that package is installed.
    """
    token = os.environ.get("TUSHARE_TOKEN")
    if token:
        return token
    # try dotenv as a convenience for local dev
    try:
        from dotenv import load_dotenv as _load  # type: ignore[import-untyped]
        from pathlib import Path as _Path

        env = _Path(__file__).resolve().parents[1] / ".env"
        if env.exists():
            _load(env)
        token = os.environ.get("TUSHARE_TOKEN")
    except ImportError:
        pass
    if not token:
        raise RuntimeError(
            "TUSHARE_TOKEN is not set.\n"
            "  - Set the TUSHARE_TOKEN environment variable, or\n"
            "  - Create a .env file with TUSHARE_TOKEN=your_token, or\n"
            "  - Obtain a free token at https://tushare.pro"
        )
    return token


# ── Pro API singleton ─────────────────────────────────────────────────


_pro: Any = None


def _get_pro() -> Any:
    global _pro
    if _pro is None:
        token = get_token()
        ts.set_token(token)
        _pro = ts.pro_api()
    return _pro


# ── Rate-limit helpers ─────────────────────────────────────────────────


def _rate_sleep() -> None:
    """Sleep briefly between API calls to stay under free-tier limits."""
    time.sleep(0.35)  # ~170 calls / min, well under the 200/min limit


def _call_with_retry(fn, **kwargs) -> pd.DataFrame:
    """Call a tushare API function with retry + backoff on transient errors.

    *fn* is a bound method of the pro_api instance, e.g. ``pro.ft_mins``.
    """
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            result: pd.DataFrame = fn(**kwargs)
            _rate_sleep()
            return result
        except Exception as exc:
            msg = str(exc).lower()
            transient = any(w in msg for w in ("timeout", "connection", "rate", "频率", "too many"))
            if not transient or attempt == max_attempts:
                raise
            wait = 2.0 ** attempt
            logger.warning("Tushare API transient error (attempt %d/%d), waiting %.0fs: %s",
                           attempt, max_attempts, wait, exc)
            time.sleep(wait)
    raise RuntimeError("unreachable")  # pragma: no cover


# ── Contract resolution ────────────────────────────────────────────────


# Known CFFEX government bond futures underlying symbols
# Tushare naming is counter-intuitive:
#   T.CFX   → 10-Year CGB dominant continuous
#   TL0.CFX → 30-Year CGB dominant continuous
#   TL.CFX  → 10-Year CGB (another variant, NOT 30Y!)
CGB_SYMBOLS = {
    "T":   {"name": "10-Year CGB Futures",  "exchange": "CFFEX", "continuous_code": "T.CFX"},
    "TL":  {"name": "30-Year CGB Futures",  "exchange": "CFFEX", "continuous_code": "TL0.CFX"},
}


def resolve_ts_codes(contract: str) -> list[str]:
    """Return the list of actual ts_codes (e.g. 'T2509.CFX') that currently exist
    for *contract* ('T' or 'TL') on the exchange.

    Uses fut_basic to discover listed contracts.
    """
    pro = _get_pro()
    symbol = contract.upper()
    df = _call_with_retry(pro.fut_basic, exchange="CFFEX", fut_type="2")  # 2 = 国债
    if df.empty:
        raise ValueError(f"No futures contracts found on CFFEX for {symbol}")
    codes = sorted(set(df.loc[df["ts_code"].str.startswith(symbol, na=False), "ts_code"].tolist()))
    if not codes:
        raise ValueError(f"No contracts matching {symbol} on CFFEX")
    return codes


def resolve_dominant_contract(contract: str, trade_date: str | None = None) -> str:
    """Return the dominant (most-active) actual contract ts_code for *contract*.

    Uses fut_mapping to map e.g. 'T.CFX' → 'T2609.CFX' on *trade_date*.
    For 30Y we use 'TL0.CFX' (not 'TL.CFX' which is another 10Y variant).
    """
    pro = _get_pro()
    symbol = contract.upper()
    mapping_code = CGB_SYMBOLS.get(symbol, {}).get("continuous_code", f"{symbol}.CFX")
    date_str = trade_date or datetime.now().strftime("%Y%m%d")
    df = _call_with_retry(pro.fut_mapping, ts_code=mapping_code, trade_date=date_str)
    if df.empty:
        raise ValueError(f"No fut_mapping data for {mapping_code} on {date_str}")
    return str(df["mapping_ts_code"].iloc[0])


def get_contract_timeline(contract: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Build a timeline of dominant-contract → actual-ts_code mapping.

    Returns DataFrame with columns: trade_date (str), mapping_ts_code (str).
    One row per unique dominant-contract period.
    """
    pro = _get_pro()
    symbol = contract.upper()
    mapping_code = CGB_SYMBOLS.get(symbol, {}).get("continuous_code", f"{symbol}.CFX")
    df = _call_with_retry(pro.fut_mapping, ts_code=mapping_code)
    # Filter to date range
    df = df.loc[(df["trade_date"] >= start_date) & (df["trade_date"] <= end_date)]
    # Find transition points — when mapping_ts_code changes
    df = df.sort_values("trade_date").reset_index(drop=True)
    df["_prev_code"] = df["mapping_ts_code"].shift(1)
    transitions = df.loc[df["mapping_ts_code"] != df["_prev_code"]]
    if transitions.empty and not df.empty:
        transitions = df.head(1)
    return transitions[["trade_date", "mapping_ts_code"]].copy()


# ── Data fetching (fut_daily) ──────────────────────────────────────────


def fetch_fut_daily(contract: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily OHLC data for *contract* ('T' or 'TL').

    Uses Tushare fut_daily with the continuous contract code.

    Returns standardized DataFrame with columns:
        date, open, high, low, close, volume, open_interest
    """
    pro = _get_pro()
    symbol = contract.upper()
    code = CGB_SYMBOLS.get(symbol, {}).get("continuous_code", f"{symbol}.CFX")

    logger.info("fut_daily: %s  %s → %s", code, start_date, end_date)
    df = _call_with_retry(pro.fut_daily, ts_code=code, start_date=start_date, end_date=end_date)
    if df.empty:
        logger.warning("fut_daily returned empty for %s", code)
        return pd.DataFrame()
    return _normalize_daily(df)


def _normalize_daily(raw: pd.DataFrame) -> pd.DataFrame:
    """Standardize fut_daily column names."""
    df = raw.rename(columns={
        "trade_date": "date",
        "vol": "volume",
        "oi": "open_interest",
    }, errors="ignore")
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    for col in ["open", "high", "low", "close", "volume", "open_interest"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    keep = ["date", "open", "high", "low", "close"]
    for extra in ["volume", "open_interest"]:
        if extra in df.columns:
            keep.append(extra)
    df = df[keep]
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ── Data fetching (ft_mins — intraday) ─────────────────────────────────


def fetch_ft_5min(contract: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch 5-minute OHLC data for *contract* ('T' or 'TL').

    Strategy:
    1. Resolve actual contract codes via fut_mapping timeline.
    2. For each actual contract, call ft_mins(freq='5min', ts_code=...).
    3. Concatenate, deduplicate, sort by datetime.

    Returns standardized DataFrame — same columns as fetch_fut_daily.
    """
    pro = _get_pro()
    symbol = contract.upper()

    # Get the contract timeline
    timeline = get_contract_timeline(contract, start_date, end_date)
    if timeline.empty:
        logger.warning("No contract timeline for %s %s→%s", symbol, start_date, end_date)
        return pd.DataFrame()

    # Also try the continuous code directly (some Tushare instances support this)
    # Actually, from testing: ft_mins does NOT support .CFX continuous codes.
    # We must use actual contract codes from fut_mapping.

    all_frames: list[pd.DataFrame] = []
    seen_codes: set[str] = set()

    for _, row in timeline.iterrows():
        code = row["mapping_ts_code"]
        if code in seen_codes:
            continue
        seen_codes.add(code)

        # Determine the date range for this contract
        t_date = row["trade_date"]
        # Find next transition to set the end date for this contract
        later = timeline.loc[timeline["trade_date"] > t_date, "trade_date"]
        contract_end = later.min() if not later.empty else end_date
        # Overlap one day to ensure continuity
        c_start = max(start_date, str(pd.to_datetime(t_date).date()).replace("-", ""))
        c_end = min(end_date, str(pd.to_datetime(contract_end).date()).replace("-", ""))

        if c_start > c_end:
            continue

        try:
            logger.info("ft_mins: %s  %s → %s", code, c_start, c_end)
            df = _call_with_retry(pro.ft_mins, ts_code=code, freq="5min",
                                  start_date=c_start, end_date=c_end)
            if not df.empty:
                all_frames.append(df)
        except Exception as exc:
            logger.warning("ft_mins failed for %s (%s→%s): %s", code, c_start, c_end, exc)
            continue

    if not all_frames:
        return pd.DataFrame()

    raw = pd.concat(all_frames, ignore_index=True)
    return _normalize_mins(raw)


def _normalize_mins(raw: pd.DataFrame) -> pd.DataFrame:
    """Standardize ft_mins column names.

    ft_mins returns columns like:
        ts_code, trade_time, open, close, high, low, vol, amount, oi
    """
    df = raw.copy()
    # trade_time is already a datetime string like "2026-07-29 15:15:00"
    if "trade_time" in df.columns:
        df["date"] = pd.to_datetime(df["trade_time"])
        df = df.drop(columns=["trade_time"], errors="ignore")
    elif "time" in df.columns and "trade_date" in df.columns:
        df["date"] = pd.to_datetime(
            df["trade_date"].astype(str) + " " + df["time"].astype(str)
        )

    df = df.rename(columns={
        "vol": "volume",
        "oi": "open_interest",
    }, errors="ignore")

    for col in ["open", "high", "low", "close", "volume", "open_interest"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    keep = ["date", "open", "high", "low", "close"]
    for extra in ["volume", "open_interest"]:
        if extra in df.columns:
            keep.append(extra)
    df = df[[c for c in keep if c in df.columns]]

    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.drop_duplicates(subset=["date"], keep="last")
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ── Health check ───────────────────────────────────────────────────────


def health_check() -> dict:
    """Verify Tushare API connectivity and token validity."""
    try:
        pro = _get_pro()
        df = _call_with_retry(pro.trade_cal, exchange="CFFEX",
                               start_date=datetime.now().strftime("%Y%m%d"),
                               end_date=datetime.now().strftime("%Y%m%d"))
        return {"ok": True, "message": "Tushare API reachable", "token_valid": True}
    except Exception as exc:
        msg = str(exc)
        token_ok = "token" not in msg.lower() and "授权" not in msg and "auth" not in msg.lower()
        return {"ok": False, "message": msg, "token_valid": token_ok}
