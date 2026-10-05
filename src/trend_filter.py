from __future__ import annotations

import pandas as pd


def build_daily_trend_filter_from_intraday(
    close_intraday: pd.Series,
    ma_len_days: int = 5,
    compare_lag_days: int = 2,
    trend_method: str = "ma_compare",
    ma_short: int = 5,
    ma_long: int = 20,
) -> pd.DataFrame:
    close = close_intraday.copy().astype(float)
    if not isinstance(close.index, pd.DatetimeIndex):
        raise TypeError("close_intraday must be indexed by DatetimeIndex")
    close = close.sort_index()
    daily_close = close.resample("1D").last().dropna()

    method = str(trend_method)
    if method == "ma_compare":
        daily_ma = daily_close.rolling(int(ma_len_days)).mean()
        trend_up = daily_ma.shift(1) > daily_ma.shift(1 + int(compare_lag_days))
        trend_down = daily_ma.shift(1) < daily_ma.shift(1 + int(compare_lag_days))
        daily = pd.DataFrame(
            {
                "daily_close": daily_close,
                "ma": daily_ma,
                "trend_up": trend_up,
                "trend_down": trend_down,
            }
        )
    elif method == "ma_cross":
        short_ma = daily_close.rolling(int(ma_short)).mean()
        long_ma = daily_close.rolling(int(ma_long)).mean()
        trend_up = short_ma.shift(1) > long_ma.shift(1)
        trend_down = short_ma.shift(1) < long_ma.shift(1)
        daily = pd.DataFrame(
            {
                "daily_close": daily_close,
                "ma_short": short_ma,
                "ma_long": long_ma,
                "trend_up": trend_up,
                "trend_down": trend_down,
            }
        )
    elif method == "price_compare":
        daily_ma = daily_close.rolling(int(ma_len_days)).mean()
        trend_up = daily_close.shift(1) > daily_ma.shift(1)
        trend_down = daily_close.shift(1) < daily_ma.shift(1)
        daily = pd.DataFrame(
            {
                "daily_close": daily_close,
                "ma": daily_ma,
                "trend_up": trend_up,
                "trend_down": trend_down,
            }
        )
    else:
        raise ValueError(f"Unsupported trend_method: {trend_method}")

    trend_up_map = daily["trend_up"].copy()
    trend_up_map.index = trend_up_map.index.normalize()
    trend_down_map = daily["trend_down"].copy()
    trend_down_map.index = trend_down_map.index.normalize()

    intraday_dates = pd.Series(close.index.normalize(), index=close.index)
    trend_up_intraday = intraday_dates.map(trend_up_map).where(lambda s: s.notna(), False).astype(bool)
    trend_down_intraday = intraday_dates.map(trend_down_map).where(lambda s: s.notna(), False).astype(bool)

    return pd.DataFrame(
        {
            "trend_up_intraday": trend_up_intraday,
            "trend_down_intraday": trend_down_intraday,
        },
        index=close.index,
    )
