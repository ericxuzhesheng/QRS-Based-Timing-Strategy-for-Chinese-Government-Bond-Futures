"""Causal QRS research on real-contract Tushare minute bars.

Parameter definitions are declared here before the holdout is evaluated. The
legacy full-sample grid remains available as an explicitly in-sample experiment.
"""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from .qrs_calculator import calculate_qrs_intraday
from .trend_filter import build_daily_trend_filter_from_intraday


CANDIDATES = [
    {"N": n, "M": m, "n": 2.0, **signal}
    for (n, m), signal in product(
        [(16, 600), (20, 800)],
        [
            {"S": 0.3, "trend_method": "ma_compare", "ma_len_days": 5, "compare_lag_days": 2},
            {"S": 0.5, "trend_method": "ma_compare", "ma_len_days": 5, "compare_lag_days": 2},
            {"S": 0.5, "trend_method": "price_compare", "ma_len_days": 5, "compare_lag_days": 2},
        ],
    )
]
BASELINE_ID = 1


def aggregate_minutes(raw: pd.DataFrame, market: str) -> tuple[pd.DataFrame, dict]:
    required = {"datetime", "open", "high", "low", "close", "volume", "open_interest",
                "product", "source_contract", "source", "trade_date", "session_complete"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing source columns: {sorted(missing)}")
    df = raw.copy().sort_values("datetime").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    if df.empty or df["datetime"].duplicated().any() or df["datetime"].isna().any():
        raise ValueError("Minute input must be nonempty with unique valid timestamps")
    if not df["source"].eq("tushare").all() or not df["product"].eq(market).all():
        raise ValueError("Source must be Tushare and match the requested product")
    if not df["source_contract"].str.match(rf"^{market}\d{{4}}\.CFX$").all():
        raise ValueError("Only concrete CFFEX contracts are accepted")
    prices = df[["open", "high", "low", "close"]].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Nonfinite or nonpositive OHLC")
    if ((df["high"] < df[["open", "close", "low"]].max(axis=1)) |
        (df["low"] > df[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("Invalid OHLC envelope")
    if (df["volume"] < 0).any() or not np.isfinite(df["volume"]).all():
        raise ValueError("Invalid volume")
    if (df.groupby("trade_date")["source_contract"].nunique() != 1).any():
        raise ValueError("A research date must use one preselected concrete contract")
    trade_dates = pd.to_datetime(df["trade_date"].astype(str).str.replace("-", "", regex=False), format="%Y%m%d")
    if not df["datetime"].dt.normalize().eq(trade_dates).all():
        raise ValueError("Trade dates disagree with timestamps")
    df["trade_date"] = trade_dates.dt.strftime("%Y%m%d")
    complete = df.groupby("trade_date")["session_complete"].transform("all")
    excluded_days = int(df.loc[~complete, "trade_date"].nunique())
    df = df.loc[complete].copy()
    # Right labels are period ends. Execution time retains the first actual minute.
    df["bar_end"] = df["datetime"].dt.ceil("5min")
    bars = df.groupby(["trade_date", "source_contract", "bar_end"], sort=True).agg(
        date=("datetime", "last"), execution_time=("datetime", "first"),
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"),
        open_interest=("open_interest", "last"),
        first_minute_volume=("volume", "first"), last_minute_volume=("volume", "last"),
        minute_count=("datetime", "size"),
    ).reset_index()
    # Missing bars are excluded, never filled. Five distinct minute timestamps
    # in a five-minute bin imply a complete bin only for whole-minute data.
    if (df["datetime"].dt.second != 0).any() or (df["datetime"].dt.microsecond != 0).any():
        raise ValueError("Expected whole-minute timestamps")
    excluded_bins = int(bars["minute_count"].ne(5).sum())
    bars = bars.loc[bars["minute_count"].eq(5)].sort_values("date").reset_index(drop=True)
    if bars.empty:
        raise ValueError("No complete five-minute bars")
    quality = {
        "minute_rows": len(raw), "minute_start": str(raw["datetime"].min()),
        "minute_end": str(raw["datetime"].max()), "excluded_incomplete_dates": excluded_days,
        "excluded_incomplete_bins": excluded_bins, "five_minute_rows": len(bars),
        "five_minute_start": str(bars["date"].min()), "five_minute_end": str(bars["date"].max()),
        "five_minute_days": int(bars["trade_date"].nunique()),
        "bars_per_day": sorted(int(x) for x in bars.groupby("trade_date").size().unique()),
        "zero_volume_minutes": int(raw["volume"].eq(0).sum()),
        "zero_first_minute_volume_bars": int(bars["first_minute_volume"].eq(0).sum()),
    }
    return bars, quality


def causal_roll_prices(bars: pd.DataFrame) -> pd.DataFrame:
    """Scale only forward at a roll, using the incoming first observed open.

    There is no backward adjustment and no future mapping. These are factor
    coordinates; execution and profit use the unchanged raw prices.
    """
    out = bars.copy()
    scale = np.ones(len(out), dtype=float)
    codes = out["source_contract"].to_numpy()
    closes = out["close"].to_numpy(dtype=float)
    opens = out["open"].to_numpy(dtype=float)
    for i in range(1, len(out)):
        scale[i] = scale[i - 1]
        if codes[i] != codes[i - 1]:
            scale[i] = closes[i - 1] * scale[i - 1] / opens[i]
    for col in ["open", "high", "low", "close"]:
        out[col] = out[col].astype(float) * scale
    return out


def target_positions(bars: pd.DataFrame, params: dict) -> pd.DataFrame:
    factor_prices = causal_roll_prices(bars)
    factor = calculate_qrs_intraday(factor_prices, N=params["N"], M=params["M"], n=params["n"])
    trend = build_daily_trend_filter_from_intraday(
        pd.Series(factor_prices["close"].to_numpy(), index=pd.DatetimeIndex(bars["date"])),
        trend_method=params["trend_method"], ma_len_days=params["ma_len_days"],
        compare_lag_days=params["compare_lag_days"],
    )
    values = factor["qrs"].to_numpy(dtype=float)
    up = trend["trend_up_intraday"].to_numpy()
    down = trend["trend_down_intraday"].to_numpy()
    days = bars["trade_date"].to_numpy()
    raw = np.zeros(len(bars))
    current = 0.0
    for i, qrs in enumerate(values):
        if i == 0 or days[i] != days[i - 1]:
            current = 0.0
        if np.isfinite(qrs):
            if qrs > params["S"] and up[i]:
                current = 1.0
            elif qrs < -params["S"] and down[i]:
                current = -1.0
        raw[i] = current
    out = bars.copy()
    out["qrs"] = values
    out["trend_up"] = up
    out["trend_down"] = down
    out["raw_target"] = raw
    # The preceding CLOSED five-minute bar supplies the next opening target.
    out["target"] = pd.Series(raw).shift(1).fillna(0.0)
    out.loc[out["trade_date"].ne(out["trade_date"].shift()), "target"] = 0.0
    out["signal_time"] = out["date"].shift(1)
    out.loc[out["trade_date"].ne(out["trade_date"].shift()), "signal_time"] = pd.NaT
    return out


def simulate(signals: pd.DataFrame, cost_bp: float = 1.0) -> pd.DataFrame:
    """One-times notional, next-open execution, flat at each day's last close.

    A zero-volume first minute defers position changes. Daily closing liquidity
    is not inferred from OHLC; zero-volume closing exits are explicitly flagged.
    """
    out = signals.copy().reset_index(drop=True)
    day = out["trade_date"].to_numpy()
    close = out["close"].to_numpy(dtype=float)
    opening = out["open"].to_numpy(dtype=float)
    targets = out["target"].to_numpy(dtype=float)
    volume = out["first_minute_volume"].to_numpy(dtype=float)
    contracts = out["source_contract"].to_numpy()
    end_day = np.r_[day[1:] != day[:-1], True]
    position = np.zeros(len(out))
    gap = np.zeros(len(out))
    intra = np.zeros(len(out))
    turnover = np.zeros(len(out))
    exit_turnover = np.zeros(len(out))
    blocked = np.zeros(len(out), dtype=bool)
    previous = 0.0
    for i in range(len(out)):
        if i == 0 or day[i] != day[i - 1]:
            previous = 0.0
        elif contracts[i] != contracts[i - 1]:
            raise ValueError("Intraday contract changes are unsupported")
        if i and day[i] == day[i - 1]:
            gap[i] = previous * (opening[i] / close[i - 1] - 1.0)
        desired = targets[i]
        blocked[i] = volume[i] <= 0 and desired != previous
        actual = previous if blocked[i] else desired
        position[i] = actual
        turnover[i] = abs(actual - previous)
        intra[i] = actual * (close[i] / opening[i] - 1.0)
        if end_day[i]:
            exit_turnover[i] = abs(actual)
        previous = 0.0 if end_day[i] else actual
    out["position"] = position
    out["gap_return"] = gap
    out["intrabar_return"] = intra
    out["open_turnover"] = turnover
    out["close_turnover"] = exit_turnover
    out["turnover"] = turnover + exit_turnover
    out["blocked_zero_volume"] = blocked
    out["zero_volume_close_exit"] = (exit_turnover > 0) & out["last_minute_volume"].le(0)
    cost = float(cost_bp) / 10000
    out["return_gross"] = (1 + gap) * (1 + intra) - 1
    out["return_net"] = (1 + gap) * (1 - turnover * cost) * (1 + intra) * (1 - exit_turnover * cost) - 1
    return out


def daily_returns(ledger: pd.DataFrame) -> pd.Series:
    return (1 + ledger["return_net"]).groupby(ledger["trade_date"]).prod() - 1


def performance(ledger: pd.DataFrame, apply_liquidity_gate: bool = True) -> dict:
    returns = daily_returns(ledger)
    nav = (1 + returns).cumprod()
    vol = float(returns.std(ddof=1) * np.sqrt(252))
    # Include initial NAV=1 in the running peak, including immediate losses.
    peaks = nav.cummax().clip(lower=1.0)
    result = {
        "start": str(ledger["date"].min()), "end": str(ledger["date"].max()),
        "days": len(returns), "bars": len(ledger), "cumulative_return": float(nav.iloc[-1] - 1),
        "annualized_return": float(nav.iloc[-1] ** (252 / len(returns)) - 1),
        "annualized_volatility": vol,
        "sharpe_ratio": float(returns.mean() * 252 / vol) if vol > 0 else np.nan,
        "max_drawdown": float((nav / peaks - 1).min()), "turnover": float(ledger["turnover"].sum()),
        "blocked_zero_volume": int(ledger["blocked_zero_volume"].sum()),
        "zero_volume_close_exits": int(ledger["zero_volume_close_exit"].sum()),
    }
    result["performance_status"] = "ohlc_research" if result["zero_volume_close_exits"] == 0 else "blocked_zero_volume_exit"
    if apply_liquidity_gate and result["zero_volume_close_exits"]:
        for metric in ["cumulative_return", "annualized_return", "annualized_volatility", "sharpe_ratio", "max_drawdown"]:
            result[metric] = np.nan
    return result


def select_candidate(train_metrics: pd.DataFrame) -> int:
    ranked = train_metrics.loc[np.isfinite(train_metrics["sharpe_ratio"])].sort_values(
        ["sharpe_ratio", "candidate_id"], ascending=[False, True], kind="stable")
    if ranked.empty:
        raise ValueError("No candidate has a valid training Sharpe")
    return int(ranked.iloc[0]["candidate_id"])
