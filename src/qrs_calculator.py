from __future__ import annotations

import numpy as np
import pandas as pd


def weighted_low_high_beta_r2(low: np.ndarray, high: np.ndarray) -> tuple[float, float]:
    x = np.asarray(low, dtype=float)
    y = np.asarray(high, dtype=float)
    if len(x) != len(y) or len(x) == 0:
        return np.nan, np.nan
    w = np.ones_like(x, dtype=float) / len(x)
    x_mean = np.sum(w * x)
    y_mean = np.sum(w * y)
    x_centered = x - x_mean
    y_centered = y - y_mean
    denom = np.sum(w * x_centered * x_centered)
    if denom == 0:
        return np.nan, np.nan
    beta = np.sum(w * x_centered * y_centered) / denom
    alpha = y_mean - beta * x_mean
    y_hat = alpha + beta * x
    sse = np.sum(w * (y - y_hat) ** 2)
    sst = np.sum(w * (y - y_mean) ** 2)
    r2 = np.nan if sst == 0 else 1.0 - sse / sst
    return float(beta), float(r2)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std(ddof=0)
    return (series - mean) / std.replace(0.0, np.nan)


def rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    def rank_last(values: np.ndarray) -> float:
        last = values[-1]
        if not np.isfinite(last):
            return np.nan
        valid = values[np.isfinite(values)]
        if len(valid) == 0:
            return np.nan
        return float((valid <= last).sum() / len(valid))

    return series.rolling(window=window, min_periods=window).apply(rank_last, raw=True)


def rolling_slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    denom = np.mean((x - x_mean) ** 2)

    def slope(values: np.ndarray) -> float:
        y = np.asarray(values, dtype=float)
        if np.isfinite(y).sum() < window:
            return np.nan
        y_mean = y.mean()
        return float(np.mean((x - x_mean) * (y - y_mean)) / denom)

    return series.rolling(window=window, min_periods=window).apply(slope, raw=True)


def calculate_qrs_intraday(
    data: pd.DataFrame,
    N: int = 16,
    M: int = 600,
    n: float = 2.0,
    normalize_penalty: bool = False,
    penalty_mean_window: int | None = None,
) -> pd.DataFrame:
    required = {"date", "high", "low", "close"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns for QRS intraday calculation: {sorted(missing)}")

    df = data.copy().sort_values("date").reset_index(drop=True)
    window = int(N)
    if window < 2 or int(M) < 2:
        raise ValueError("N and M must be at least 2")
    # Match the original preceding-N-bars regression, without a Python loop.
    # A fixed, already-known anchor avoids cancellation for tiny price ranges.
    anchor = float(df["close"].iloc[0]) if len(df) else 0.0
    low = df["low"].astype(float) - anchor
    high = df["high"].astype(float) - anchor
    var_low = low.rolling(window).var(ddof=0).shift(1)
    var_high = high.rolling(window).var(ddof=0).shift(1)
    cov = low.rolling(window).cov(high, ddof=0).shift(1)
    beta = cov / var_low.where(var_low > 0)
    r2 = (cov**2 / (var_low * var_high).where((var_low > 0) & (var_high > 0))).clip(0, 1)

    out = df.copy()
    out["beta"] = beta
    out["r2"] = r2
    out["z_beta"] = rolling_zscore(out["beta"], int(M))
    penalty = out["r2"].clip(lower=0) ** float(n)
    if normalize_penalty:
        mean_window = int(penalty_mean_window) if penalty_mean_window is not None else int(M)
        penalty_mean = penalty.rolling(window=mean_window, min_periods=mean_window).mean()
        penalty = penalty / penalty_mean.replace(0.0, np.nan)
    out["penalty"] = penalty
    out["qrs"] = out["z_beta"] * out["penalty"]
    out["qrs_raw"] = out["beta"]
    out["qrs_r2"] = out["r2"]
    out["qrs_zscore"] = out["z_beta"]
    out["qrs_adjusted"] = out["qrs"]
    return out


def calculate_qrs(
    data: pd.DataFrame,
    rolling_window: int = 18,
    zscore_window: int = 120,
    r2_power: float = 2.0,
    slope_window: int = 20,
    percentile_window: int = 120,
    normalize_penalty: bool = False,
) -> pd.DataFrame:
    required = {"date", "high", "low", "close"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns for QRS calculation: {sorted(missing)}")

    df = data.copy().sort_values("date").reset_index(drop=True)
    n = len(df)
    beta = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    low = df["low"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)

    for i in range(int(rolling_window) - 1, n):
        start = i - int(rolling_window) + 1
        beta[i], r2[i] = weighted_low_high_beta_r2(low[start : i + 1], high[start : i + 1])

    out = df.copy()
    out["qrs_raw"] = beta
    out["qrs_r2"] = r2
    out["qrs_zscore"] = rolling_zscore(out["qrs_raw"], int(zscore_window))
    penalty = out["qrs_r2"].clip(lower=0) ** float(r2_power)
    if normalize_penalty:
        penalty_mean = penalty.rolling(window=int(zscore_window), min_periods=int(zscore_window)).mean()
        penalty = penalty / penalty_mean.replace(0.0, np.nan)
    out["qrs_adjusted"] = out["qrs_zscore"] * penalty
    out["qrs_slope"] = rolling_slope(out["qrs_adjusted"], int(slope_window))
    out["qrs_percentile"] = rolling_percentile(out["qrs_adjusted"], int(percentile_window))
    out["daily_return"] = out["close"].pct_change()
    return out
