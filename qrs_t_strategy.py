"""
QRS（Quantified Resistance/Support）用于国债期货 T（5分钟）择时：

策略

- 做多：QRS > +S 且「前一日 40日均线 > 三日前 40日均线」（短期趋势向上）
- 做空：QRS < -S 且「前一日 40日均线 < 三日前 40日均线」（短期趋势向下）
- 其他：保持上一时刻仓位（状态机）

关键说明：
1) 你是5分钟数据，"40日均线"是日频概念：
   - 先把5分钟 close 聚合成"日收盘价"（每个交易日最后一根5min close）
   - 在日频上计算 MA40
   - 计算趋势条件：MA40(t-1) vs MA40(t-3)
   - 将趋势条件映射回5分钟：同一天所有bar共用当天的趋势条件

2) QRS 因子构造：
   - 用最近 N=20根 (low, high) 做回归 high_t = α + β·low_t + ε
   - 对 β 做 M=800 期滚动 z-score 标准化
   - 用回归 R² 作为惩罚项，QRS = zscore(β) × (R²)^n，n 控制惩罚强度
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Iterable, Union, Any

import os
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools
from tqdm import tqdm

try:
    from sklearn.model_selection import ParameterGrid as SklearnParameterGrid
except Exception:
    SklearnParameterGrid = None

try:
    from sklearn.base import BaseEstimator as SklearnBaseEstimator
    from sklearn.model_selection import GridSearchCV as SklearnGridSearchCV
    from sklearn.model_selection import TimeSeriesSplit as SklearnTimeSeriesSplit
except Exception:
    SklearnBaseEstimator = None
    SklearnGridSearchCV = None
    SklearnTimeSeriesSplit = None


# =========================
# 0) 解决 matplotlib 中文显示
# =========================
def setup_cn_font():
    mpl.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    mpl.rcParams["axes.unicode_minus"] = False

@dataclass
class QRSParams:
    N: int = 20
    M: int = 800    
    n: int = 2
    normalize_penalty: bool = True
    penalty_mean_window: Optional[int] = None


@dataclass
class RunConfig:
    excel_path: str
    sheet_name: Optional[str]
    save_dir: str
    paper_dynamic: bool = False


def _str_to_optional(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    v = str(s).strip()
    if v == "":
        return None
    if v.lower() in {"none", "null"}:
        return None
    return v


def resolve_run_config(argv: Optional[List[str]] = None) -> RunConfig:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--excel-path", default=None)
    parser.add_argument("--sheet-name", default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--paper-dynamic", action="store_true")
    args = parser.parse_args(argv)

    # 地址配置
    default_excel_path = r"D:\Python\浙商证券固收\RSRS QRS择时报告\10年国债期货_5min_3年.xlsx"
    default_sheet_name = "Sheet1"
    default_save_dir = r"D:\Python\浙商证券固收\RSRS QRS择时报告"

    excel_path = _str_to_optional(os.getenv("QRS_EXCEL_PATH")) or default_excel_path
    sheet_name = _str_to_optional(os.getenv("QRS_SHEET_NAME")) or default_sheet_name
    save_dir = _str_to_optional(os.getenv("QRS_SAVE_DIR")) or default_save_dir

    if _str_to_optional(args.excel_path) is not None:
        excel_path = str(args.excel_path)
    if args.sheet_name is not None:
        sheet_name = _str_to_optional(args.sheet_name)
    if _str_to_optional(args.save_dir) is not None:
        save_dir = str(args.save_dir)

    return RunConfig(
        excel_path=excel_path,
        sheet_name=sheet_name,
        save_dir=save_dir,
        paper_dynamic=bool(args.paper_dynamic),
    )


class QRSGridSearchEstimator(SklearnBaseEstimator if SklearnBaseEstimator is not None else object):
    def __init__(
        self,
        S: float = 0.5,
        trend_method: str = "ma_compare",
        ma_len_days: int = 5,
        compare_lag_days: int = 2,
        ma_short: int = 5,
        ma_long: int = 20,
        allow_short: bool = True,
        periods_per_year: int = 252 * 54,
    ):
        self.S = S
        self.trend_method = trend_method
        self.ma_len_days = ma_len_days
        self.compare_lag_days = compare_lag_days
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.allow_short = allow_short
        self.periods_per_year = periods_per_year

    def fit(self, X, y=None):
        return self

    def score(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            factor = X["factor"]
            close = X["close"]
        else:
            arr = np.asarray(X)
            if arr.ndim != 2 or arr.shape[1] < 2:
                raise ValueError("X 需要至少两列：[factor, close]")
            factor = pd.Series(arr[:, 0])
            close = pd.Series(arr[:, 1])

        df = pd.DataFrame({"factor": factor, "close": close}).dropna()
        if df.empty:
            return -np.inf
        factor = df["factor"]
        close = df["close"]

        trend_method = str(self.trend_method)
        if trend_method == "ma_compare":
            trend_df = QRSBacktester.build_daily_trend_filter_from_5m(
                close_5m=close,
                ma_len_days=int(self.ma_len_days),
                compare_lag_days=int(self.compare_lag_days),
                trend_method=trend_method,
            )
        elif trend_method == "ma_cross":
            trend_df = QRSBacktester.build_daily_trend_filter_from_5m(
                close_5m=close,
                trend_method=trend_method,
                ma_short=int(self.ma_short),
                ma_long=int(self.ma_long),
            )
        elif trend_method == "price_compare":
            trend_df = QRSBacktester.build_daily_trend_filter_from_5m(
                close_5m=close,
                ma_len_days=int(self.ma_len_days),
                trend_method=trend_method,
            )
        else:
            raise ValueError(f"不支持的 trend_method: {trend_method}")

        position = QRSBacktester.build_position_strategy_simple(
            factor=factor,
            trend_up_5m=trend_df["trend_up_5m"],
            trend_down_5m=trend_df["trend_down_5m"],
            close=close,
            S=float(self.S),
            allow_short=bool(self.allow_short),
        )
        perf = QRSBacktester.calc_excess_return(close=close, position=position)
        sharpe = QRSBacktester.calc_sharpe_ratio(perf["ret_strat"], periods_per_year=int(self.periods_per_year))
        if not np.isfinite(sharpe):
            return -np.inf
        return float(sharpe)


class QRSBacktester:
    def __init__(self, excel_path: str, sheet_name: Optional[str] = None):
        self.excel_path = excel_path
        self.sheet_name = sheet_name
        self.df_5m: Optional[pd.DataFrame] = None
        self.fac_5m: Optional[pd.DataFrame] = None

    @staticmethod
    def _param_grid_size(param_grid: Union[Dict[str, List], List[Dict[str, List]]]) -> int:
        if isinstance(param_grid, dict):
            total = 1
            for v in param_grid.values():
                total *= len(v)
            return total
        return sum(QRSBacktester._param_grid_size(g) for g in param_grid)

    @staticmethod
    def _iter_param_grid(param_grid: Union[Dict[str, List], List[Dict[str, List]]]) -> Iterable[Dict[str, Any]]:
        if SklearnParameterGrid is not None:
            for p in SklearnParameterGrid(param_grid):
                yield p
            return

        def _iter_single_grid(grid: Dict[str, List]) -> Iterable[Dict[str, Any]]:
            items = list(grid.items())
            keys = [k for k, _ in items]
            values = [v for _, v in items]
            for combo in itertools.product(*values):
                yield dict(zip(keys, combo))

        if isinstance(param_grid, dict):
            yield from _iter_single_grid(param_grid)
            return
        for g in param_grid:
            yield from _iter_single_grid(g)

    @staticmethod
    def _evaluate_metrics_for_params(
        factor: pd.Series,
        close: pd.Series,
        params: Dict[str, Any],
        allow_short: bool,
        periods_per_year: int,
    ) -> Dict[str, float]:
        s = float(params["S"])
        trend_method = str(params["trend_method"])

        if trend_method == "ma_compare":
            ma_len = int(params["ma_len_days"])
            lag = int(params["compare_lag_days"])
            trend_df = QRSBacktester.build_daily_trend_filter_from_5m(
                close_5m=close,
                ma_len_days=ma_len,
                compare_lag_days=lag,
                trend_method=trend_method,
            )
        elif trend_method == "ma_cross":
            ma_short = int(params["ma_short"])
            ma_long = int(params["ma_long"])
            trend_df = QRSBacktester.build_daily_trend_filter_from_5m(
                close_5m=close,
                trend_method=trend_method,
                ma_short=ma_short,
                ma_long=ma_long,
            )
        elif trend_method == "price_compare":
            ma_len = int(params["ma_len_days"])
            trend_df = QRSBacktester.build_daily_trend_filter_from_5m(
                close_5m=close,
                ma_len_days=ma_len,
                trend_method=trend_method,
            )
        else:
            raise ValueError(f"不支持的 trend_method: {trend_method}")

        position = QRSBacktester.build_position_strategy_simple(
            factor=factor,
            trend_up_5m=trend_df["trend_up_5m"],
            trend_down_5m=trend_df["trend_down_5m"],
            close=close,
            S=s,
            allow_short=allow_short,
        )
        perf = QRSBacktester.calc_excess_return(close=close, position=position)

        sharpe_full = QRSBacktester.calc_sharpe_ratio(perf["ret_strat"], periods_per_year=periods_per_year)
        annual_return = perf["ret_strat"].mean() * periods_per_year
        annual_volatility = perf["ret_strat"].std() * np.sqrt(periods_per_year)
        max_drawdown = (perf["nav_strat"] / perf["nav_strat"].expanding(min_periods=1).max() - 1).min()
        return {
            "sharpe_full": float(sharpe_full) if np.isfinite(sharpe_full) else np.nan,
            "annual_return": float(annual_return) if np.isfinite(annual_return) else np.nan,
            "annual_volatility": float(annual_volatility) if np.isfinite(annual_volatility) else np.nan,
            "max_drawdown": float(max_drawdown) if np.isfinite(max_drawdown) else np.nan,
        }

    # =========================
    # 1) Excel读取（自动找表头）
    # =========================
    @staticmethod
    def _to_str(x) -> str:
        return "" if pd.isna(x) else str(x).strip()

    @staticmethod
    def _norm(x: str) -> str:
        return "".join(str(x).lower().split())

    @staticmethod
    def _find_header_row(
        raw: pd.DataFrame, keywords: List[str], search_rows: int = 50
    ) -> Optional[int]:
        kset = set([QRSBacktester._norm(k) for k in keywords])
        n = min(search_rows, len(raw))
        for i in range(n):
            row = [
                QRSBacktester._norm(QRSBacktester._to_str(v))
                for v in raw.iloc[i].tolist()
            ]
            if kset.issubset(set(row)):
                return i
        return None

    @staticmethod
    def _standardize_columns(cols: List[str]) -> Dict[str, str]:
        mapping = {}
        for c in cols:
            cn = QRSBacktester._norm(c)
            if cn in ("time", "datetime") or ("时间" in c):
                mapping[c] = "datetime"
            elif cn == "open" or ("开盘" in c):
                mapping[c] = "open"
            elif cn == "high" or ("最高" in c):
                mapping[c] = "high"
            elif cn == "low" or ("最低" in c):
                mapping[c] = "low"
            elif cn == "close" or ("收盘" in c) or ("结算" in c):
                mapping[c] = "close"
            elif cn in ("volume", "vol") or ("成交量" in c) or ("成交额" in c):
                mapping[c] = "volume"
            elif cn in ("oi", "openinterest") or ("持仓" in c):
                mapping[c] = "oi"
        return mapping

    def load_excel(self) -> pd.DataFrame:
        raw = pd.read_excel(self.excel_path, sheet_name=self.sheet_name, header=None)
        if raw.empty:
            raise ValueError("Excel读取为空，请检查 EXCEL_PATH / SHEET_NAME。")

        header_row = self._find_header_row(
            raw, ["time", "open", "high", "low", "close"], search_rows=50
        )

        if header_row is not None:
            header = [self._to_str(x) for x in raw.iloc[header_row].tolist()]
            data = raw.iloc[header_row + 1 :].copy()
            data.columns = header
        else:
            data = pd.read_excel(self.excel_path, sheet_name=self.sheet_name)

        data = data.dropna(axis=1, how="all")
        col_map = self._standardize_columns(
            [self._to_str(c) for c in list(data.columns)]
        )
        data = data.rename(columns=col_map)

        need = {"datetime", "high", "low", "close"}
        missing = need - set(data.columns)
        if missing:
            raise ValueError(
                f"缺少关键字段：{sorted(missing)}；当前列：{list(data.columns)}"
            )

        data["datetime"] = pd.to_datetime(data["datetime"], errors="coerce")
        data = data.dropna(subset=["datetime"]).sort_values("datetime")

        for c in ["open", "high", "low", "close", "volume", "oi"]:
            if c in data.columns:
                data[c] = pd.to_numeric(data[c], errors="coerce")

        data = data.dropna(subset=["high", "low", "close"])
        data = data.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["high", "low", "close"]
        )

        out_cols = [
            c
            for c in ["open", "high", "low", "close", "volume", "oi"]
            if c in data.columns
        ]
        out = data.set_index("datetime")[out_cols].copy()
        out = out[~out.index.duplicated(keep="last")].sort_index()

        self.df_5m = out
        return out

    # =========================
    # 2) QRS计算
    # =========================
    @staticmethod
    def _wls_beta_r2(
        x: np.ndarray, y: np.ndarray, w: np.ndarray
    ) -> Tuple[float, float]:
        w = np.asarray(w, dtype=float)
        ws = w.sum()
        if ws <= 0:
            return np.nan, np.nan
        w = w / ws

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        xw = np.sum(w * x)
        yw = np.sum(w * y)
        xc = x - xw
        yc = y - yw

        denom = np.sum(w * xc * xc)
        if denom == 0:
            return np.nan, np.nan

        b = np.sum(w * xc * yc) / denom
        a = yw - b * xw
        y_hat = a + b * x

        sse = np.sum(w * (y - y_hat) ** 2)
        sst = np.sum(w * (y - yw) ** 2)
        r2 = np.nan if sst == 0 else 1.0 - sse / sst
        return float(b), float(r2)

    @staticmethod
    def _zscore(s: pd.Series, window: int) -> pd.Series:
        mu = s.rolling(window).mean()
        sd = s.rolling(window).std(ddof=0)
        return (s - mu) / sd

    @staticmethod
    def _rolling_rank_pct_last(s: pd.Series, window: int) -> pd.Series:
        def _rank_last(x: np.ndarray) -> float:
            last = x[-1]
            return float((x <= last).sum() / len(x))

        return s.rolling(window).apply(_rank_last, raw=True)

    def compute_factor(self, p: QRSParams) -> pd.DataFrame:
        if self.df_5m is None:
            raise RuntimeError("请先 load_excel()。")

        df = self.df_5m.copy()
        df["ret"] = df["close"].pct_change()

        n = len(df)
        beta = np.full(n, np.nan)
        r2 = np.full(n, np.nan)

        low = df["low"].to_numpy()
        high = df["high"].to_numpy()

        for i in range(p.N, n):
            x = low[i - p.N : i]
            y = high[i - p.N : i]
            w = np.ones_like(x, dtype=float)
            b, rr = self._wls_beta_r2(x, y, w)
            beta[i] = b
            r2[i] = rr

        fac = pd.DataFrame(index=df.index)
        fac["beta"] = beta
        fac["R2"] = r2
        fac["z_beta"] = self._zscore(fac["beta"], p.M)
        penalty = fac["R2"] ** float(p.n)
        if bool(p.normalize_penalty):
            mw = p.penalty_mean_window if p.penalty_mean_window is not None else int(p.M)
            denom = penalty.rolling(int(mw)).mean()
            penalty = penalty / denom.replace(0.0, np.nan)
        fac["penalty"] = penalty
        fac["qrs"] = fac["z_beta"] * fac["penalty"]

        self.fac_5m = fac.join(df[["close"]], how="left")
        return self.fac_5m

    @staticmethod
    def compute_qrs_from_ohlc(
        ohlc: pd.DataFrame,
        N: int,
        M: int,
        n_power: float,
        normalize_penalty: bool = False,
        penalty_mean_window: Optional[int] = None,
    ) -> pd.DataFrame:
        df = ohlc[["high", "low", "close"]].copy()
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["high", "low", "close"])
        if df.empty:
            raise ValueError("OHLC 为空或清洗后为空")

        n_rows = len(df)
        beta = np.full(n_rows, np.nan)
        r2 = np.full(n_rows, np.nan)

        low = df["low"].to_numpy()
        high = df["high"].to_numpy()

        for i in range(int(N), n_rows):
            x = low[i - int(N) : i]
            y = high[i - int(N) : i]
            w = np.ones_like(x, dtype=float)
            b, rr = QRSBacktester._wls_beta_r2(x, y, w)
            beta[i] = b
            r2[i] = rr

        fac = pd.DataFrame(index=df.index)
        fac["beta"] = beta
        fac["R2"] = r2
        fac["z_beta"] = QRSBacktester._zscore(fac["beta"], int(M))
        penalty = fac["R2"] ** float(n_power)
        if bool(normalize_penalty):
            mw = int(penalty_mean_window) if penalty_mean_window is not None else int(M)
            denom = penalty.rolling(int(mw)).mean()
            penalty = penalty / denom.replace(0.0, np.nan)
        fac["penalty"] = penalty
        fac["qrs"] = fac["z_beta"] * fac["penalty"]
        fac = fac.join(df[["close"]], how="left")
        return fac

    @staticmethod
    def _future_return(close: pd.Series, horizon_bars: int) -> pd.Series:
        h = int(horizon_bars)
        if h <= 0:
            raise ValueError("horizon_bars 必须 > 0")
        return close.shift(-h) / close - 1.0

    @staticmethod
    def calc_timing_ability_coefficient(
        indicator: pd.Series,
        close: pd.Series,
        horizon_bars: int,
        r: float,
        s_min: Optional[float] = None,
        s_max: Optional[float] = None,
        s_step: float = 0.1,
        min_bin_count: int = 10,
    ) -> float:
        df = pd.DataFrame({"ind": indicator, "close": close}).dropna()
        if df.empty:
            return np.nan

        fut = QRSBacktester._future_return(df["close"], horizon_bars=int(horizon_bars))
        df = df.assign(fut=fut).dropna(subset=["ind", "fut"])
        if df.empty:
            return np.nan

        ind = df["ind"].astype(float)
        fut = df["fut"].astype(float)

        if s_min is None:
            s_min = float(np.nanquantile(ind.values, 0.05))
        if s_max is None:
            s_max = float(np.nanquantile(ind.values, 0.95))
        if not np.isfinite(s_min) or not np.isfinite(s_max) or s_max <= s_min:
            return np.nan

        step = float(s_step)
        if step <= 0:
            raise ValueError("s_step 必须 > 0")

        centers = np.arange(s_min, s_max + step * 0.5, step)
        fvals = np.full(len(centers), np.nan, dtype=float)
        rr = float(r)

        ind_values = ind.values
        fut_values = fut.values

        for i, c in enumerate(centers):
            mask = (ind_values >= (c - rr)) & (ind_values <= (c + rr))
            if int(mask.sum()) < int(min_bin_count):
                continue
            fvals[i] = float(np.nanmean(fut_values[mask]))

        ok = np.isfinite(fvals) & np.isfinite(centers)
        if ok.sum() < 3:
            return np.nan

        x = centers[ok]
        y = fvals[ok]
        if np.nanstd(x) == 0 or np.nanstd(y) == 0:
            return np.nan
        return float(np.corrcoef(x, y)[0, 1])

    @staticmethod
    def select_threshold_by_h(
        indicator: pd.Series,
        close: pd.Series,
        horizon_bars: int,
        r: float,
        s_max: float = 3.0,
        s_step: float = 0.1,
        min_bin_count: int = 10,
    ) -> float:
        df = pd.DataFrame({"ind": indicator, "close": close}).dropna()
        if df.empty:
            return np.nan

        fut = QRSBacktester._future_return(df["close"], horizon_bars=int(horizon_bars))
        df = df.assign(fut=fut).dropna(subset=["ind", "fut"])
        if df.empty:
            return np.nan

        ind = df["ind"].astype(float).values
        fut = df["fut"].astype(float).values
        rr = float(r)
        step = float(s_step)
        if step <= 0:
            raise ValueError("s_step 必须 > 0")

        s_grid = np.arange(step, float(s_max) + step * 0.5, step)
        g = np.full(len(s_grid), np.nan, dtype=float)

        for i, s in enumerate(s_grid):
            mask_p = (ind >= (s - rr)) & (ind <= (s + rr))
            mask_n = (ind >= (-s - rr)) & (ind <= (-s + rr))
            if int(mask_p.sum()) < int(min_bin_count) or int(mask_n.sum()) < int(min_bin_count):
                continue
            fp = float(np.nanmean(fut[mask_p]))
            fn = float(np.nanmean(fut[mask_n]))
            g[i] = (fp - fn) / 2.0

        ok = np.isfinite(g)
        if ok.sum() == 0:
            return np.nan

        g2 = g.copy()
        g2[~ok] = np.nan
        h = np.full(len(s_grid), np.nan, dtype=float)
        for i in range(len(s_grid)):
            seg = g2[: i + 1]
            if np.isfinite(seg).sum() == 0:
                continue
            h[i] = float(np.nanmean(seg))

        j = int(np.nanargmax(h))
        return float(s_grid[j])

    @staticmethod
    def dynamic_qrs_paper_model(
        ohlc: pd.DataFrame,
        param_grid: List[Dict[str, Any]],
        window_bars: int,
        horizon_bars: int,
        r: float,
        s_step: float = 0.1,
        min_bin_count: int = 10,
        rebalance_every_bars: int = 54,
        s_max: float = 3.0,
    ) -> pd.DataFrame:
        ohlc = ohlc[["high", "low", "close"]].copy()
        ohlc = ohlc.replace([np.inf, -np.inf], np.nan).dropna(subset=["high", "low", "close"])
        if ohlc.empty:
            raise ValueError("OHLC 为空或清洗后为空")

        qrs_map: Dict[str, pd.Series] = {}
        for p in tqdm(list(param_grid), desc="Compute QRS for grid"):
            key = str(p.get("key", f"N{p['N']}_M{p['M']}_n{p['n']}"))
            fac = QRSBacktester.compute_qrs_from_ohlc(
                ohlc=ohlc,
                N=int(p["N"]),
                M=int(p["M"]),
                n_power=float(p["n"]),
                normalize_penalty=bool(p.get("normalize_penalty", False)),
                penalty_mean_window=p.get("penalty_mean_window", None),
            )
            qrs_map[key] = fac["qrs"]

        idx = ohlc.index
        best_key = pd.Series(index=idx, dtype=object)
        best_score = pd.Series(index=idx, dtype=float)
        best_S = pd.Series(index=idx, dtype=float)

        step = int(rebalance_every_bars)
        win = int(window_bars)
        if step <= 0 or win <= 0:
            raise ValueError("window_bars / rebalance_every_bars 必须 > 0")

        total_steps = max(0, (len(idx) - win + step - 1) // step)
        for end_pos in tqdm(range(win, len(idx), step), total=total_steps, desc="Rolling select params"):
            end_ts = idx[end_pos]
            start_pos = max(0, end_pos - win + 1)
            win_index = idx[start_pos : end_pos + 1]
            close_win = ohlc.loc[win_index, "close"]

            best_k = None
            best_c = -np.inf
            for k, q in qrs_map.items():
                ind_win = q.reindex(win_index)
                c = QRSBacktester.calc_timing_ability_coefficient(
                    indicator=ind_win,
                    close=close_win,
                    horizon_bars=int(horizon_bars),
                    r=float(r),
                    s_step=float(s_step),
                    min_bin_count=int(min_bin_count),
                )
                if not np.isfinite(c):
                    continue
                if c > best_c:
                    best_c = c
                    best_k = k

            if best_k is None:
                continue

            ind_win = qrs_map[best_k].reindex(win_index)
            s_star = QRSBacktester.select_threshold_by_h(
                indicator=ind_win,
                close=close_win,
                horizon_bars=int(horizon_bars),
                r=float(r),
                s_max=float(s_max),
                s_step=float(s_step),
                min_bin_count=int(min_bin_count),
            )

            seg_index = idx[end_pos : min(end_pos + step, len(idx))]
            best_key.loc[seg_index] = best_k
            best_score.loc[seg_index] = float(best_c)
            best_S.loc[seg_index] = float(s_star) if np.isfinite(s_star) else np.nan

        best_key = best_key.ffill()
        best_score = best_score.ffill()
        best_S = best_S.ffill()

        qrs_dynamic = pd.Series(index=idx, dtype=float)
        for k, q in qrs_map.items():
            mask = best_key == k
            if bool(mask.any()):
                qrs_dynamic.loc[mask] = q.loc[mask]

        out = pd.DataFrame(
            {
                "qrs_dynamic": qrs_dynamic.astype(float),
                "S_dynamic": best_S.astype(float),
                "timing_coeff": best_score.astype(float),
                "param_key": best_key,
            },
            index=idx,
        )
        return out

    # =========================
    # 3) 斜率
    # =========================
    @staticmethod
    def rolling_slope(y: pd.Series, window: int) -> pd.Series:
        x = np.arange(window, dtype=float)

        def _slope(arr: np.ndarray) -> float:
            yy = np.asarray(arr, dtype=float)
            if np.all(np.isnan(yy)):
                return np.nan
            xm = x.mean()
            ym = np.nanmean(yy)
            cov = np.nanmean((x - xm) * (yy - ym))
            var = np.mean((x - xm) ** 2)
            return float(cov / var) if var > 0 else np.nan

        return y.rolling(window).apply(_slope, raw=True)

    # =========================
    # 4) 画图：Close + QRS + QRS slope（单图双轴） + 保存
    # =========================
    @staticmethod
    def _safe_filename(s: str) -> str:
        return re.sub(r'[\\/:*?"<>|]+', "_", s)

    @staticmethod
    def plot_close_qrs_qrs_slope(
        fac_df: pd.DataFrame,
        qrs_col: str,
        slope_window: int,
        slope_scale: float = 10.0,
        start: Optional[str] = None,
        title: str = "10Y CGB Futures: Price, QRS & QRS Slope",
        save_dir: Optional[str] = None,
        save_dpi: int = 200,
        fname: Optional[str] = None,
    ) -> Optional[str]:
        df = fac_df.copy()
        if start is not None:
            df = df.loc[pd.to_datetime(start) :]

        if qrs_col not in df.columns:
            raise ValueError(f"qrs_col={qrs_col} 不存在。")

        df = df.dropna(subset=["close"])
        if df.empty:
            raise ValueError("筛选后数据为空，请检查 start 或数据本身。")

        qrs = df[qrs_col]
        qrs_slope = QRSBacktester.rolling_slope(qrs, slope_window)

        fig, ax1 = plt.subplots(figsize=(16, 6))
        ax2 = ax1.twinx()

        (l1,) = ax1.plot(df.index, df["close"], linewidth=1.2, label="Close")
        (l2,) = ax2.plot(df.index, qrs, linestyle="--", linewidth=1.2, label="QRS")
        (l3,) = ax2.plot(
            df.index,
            qrs_slope * slope_scale,
            linewidth=1.2,
            label=f"QRS slope (×{slope_scale:g})",
        )

        ax1.set_ylabel("Close Price")
        ax2.set_ylabel(f"QRS / slope (×{slope_scale:g})")
        ax1.grid(True, alpha=0.3)
        ax1.set_title(title)

        lines = [l1, l2, l3]
        labels = [ln.get_label() for ln in lines]
        ax1.legend(lines, labels, loc="upper center", ncol=3, frameon=False)

        fig.autofmt_xdate()
        plt.tight_layout()

        saved_path = None
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            if fname is None:
                start_tag = df.index.min().strftime("%Y%m%d")
                end_tag = df.index.max().strftime("%Y%m%d")
                fname = f"QRS_{qrs_col}_{start_tag}_{end_tag}.png"
            fname = QRSBacktester._safe_filename(fname)
            saved_path = os.path.join(save_dir, fname)
            plt.savefig(saved_path, dpi=save_dpi, bbox_inches="tight")

        plt.show()
        return saved_path

    # =========================
    # 5) 日频趋势过滤：从5分钟数据生成"日收盘 + MA趋势条件"，再映射回5分钟
    # =========================
    @staticmethod
    def build_daily_trend_filter_from_5m(
        close_5m: pd.Series,
        ma_len_days: int = 20,
        compare_lag_days: int = 2,
        trend_method: str = "ma_compare",
        ma_short: int = 5,
        ma_long: int = 20,
    ) -> pd.DataFrame:
        """
        生成日频趋势条件，并映射回5分钟：

        日频数据：
        - daily_close：每个交易日最后一根5min的close
        - ma：daily_close 的 ma_len_days 日均线

        趋势判断方法（trend_method）：
        - "ma_compare": 比较不同滞后期的均线
          多头趋势：MA(t-1) > MA(t-1-compare_lag_days)
          空头趋势：MA(t-1) < MA(t-1-compare_lag_days)
        - "ma_cross": 比较短均线和长均线
          多头趋势：MA_short(t-1) > MA_long(t-1)
          空头趋势：MA_short(t-1) < MA_long(t-1)
        - "price_compare": 比较价格与均线
          多头趋势：Price(t-1) > MA(t-1)
          空头趋势：Price(t-1) < MA(t-1)

        返回：
        - trend_up_5m：对齐到5分钟index的布尔序列（当日所有bar共享）
        - trend_down_5m：同理
        """
        # 取"日收盘"：每个自然日最后一个bar
        daily_close = close_5m.resample("1D").last().dropna()

        if trend_method == "ma_compare":
            # 方法1：比较不同滞后期的均线
            daily_ma = daily_close.rolling(ma_len_days).mean()

            ma_t1 = daily_ma.shift(1)
            ma_t1_lag = daily_ma.shift(1 + compare_lag_days)

            trend_up_d = ma_t1 > ma_t1_lag
            trend_down_d = ma_t1 < ma_t1_lag

            daily_df = pd.DataFrame(
                {
                    "daily_close": daily_close,
                    "ma": daily_ma,
                    "trend_up": trend_up_d,
                    "trend_down": trend_down_d,
                }
            )

        elif trend_method == "ma_cross":
            # 方法2：比较短均线和长均线
            ma_short_val = daily_close.rolling(ma_short).mean()
            ma_long_val = daily_close.rolling(ma_long).mean()

            trend_up_d = ma_short_val.shift(1) > ma_long_val.shift(1)
            trend_down_d = ma_short_val.shift(1) < ma_long_val.shift(1)

            daily_df = pd.DataFrame(
                {
                    "daily_close": daily_close,
                    "ma_short": ma_short_val,
                    "ma_long": ma_long_val,
                    "trend_up": trend_up_d,
                    "trend_down": trend_down_d,
                }
            )

        elif trend_method == "price_compare":
            # 方法3：比较价格与均线
            daily_ma = daily_close.rolling(ma_len_days).mean()

            trend_up_d = daily_close.shift(1) > daily_ma.shift(1)
            trend_down_d = daily_close.shift(1) < daily_ma.shift(1)

            daily_df = pd.DataFrame(
                {
                    "daily_close": daily_close,
                    "ma": daily_ma,
                    "trend_up": trend_up_d,
                    "trend_down": trend_down_d,
                }
            )

        else:
            raise ValueError(f"未知的趋势判断方法: {trend_method}")

        # 映射回5分钟：用日期对齐（同一天共用）
        # 先构造"日期键"
        idx_5m = close_5m.index
        date_5m = idx_5m.normalize()
        daily_key = daily_df.index.normalize()

        trend_up_map = daily_df["trend_up"].copy()
        trend_up_map.index = daily_key

        trend_down_map = daily_df["trend_down"].copy()
        trend_down_map.index = daily_key

        trend_up_5m = pd.Series(date_5m, index=idx_5m).map(trend_up_map).astype("float")
        trend_down_5m = (
            pd.Series(date_5m, index=idx_5m).map(trend_down_map).astype("float")
        )

        # map 后可能有 NaN（例如最开始没有MA），转回 bool 前先填 False
        trend_up_5m = trend_up_5m.fillna(0.0).astype(bool)
        trend_down_5m = trend_down_5m.fillna(0.0).astype(bool)

        out = pd.DataFrame(
            {
                "trend_up_5m": trend_up_5m,
                "trend_down_5m": trend_down_5m,
            },
            index=idx_5m,
        )

        return out

    # =========================
    # 6) 策略：QRS 阈值 + 日频趋势过滤（状态机）
    # =========================
    @staticmethod
    def build_position_strategy_simple(
        factor: pd.Series,
        trend_up_5m: pd.Series,
        trend_down_5m: pd.Series,
        close: pd.Series,
        S: Union[float, pd.Series],
        allow_short: bool = True,
    ) -> pd.Series:
        """
        简化策略（无风控）：
        - 做多：QRS > +S 且 trend_up_5m=True
        - 做空：QRS < -S 且 trend_down_5m=True
        - 其他：保持上一时刻仓位
        """
        f = factor.copy()
        up = trend_up_5m.reindex(f.index).fillna(False)
        dn = trend_down_5m.reindex(f.index).fillna(False)
        if isinstance(S, pd.Series):
            s_series = S.reindex(f.index).astype(float)
            s_series = s_series.ffill()
        else:
            s_series = pd.Series(float(S), index=f.index, dtype=float)

        pos = np.zeros(len(f), dtype=float)
        prev_pos = 0.0

        for i, cur_f in enumerate(f.values):
            if np.isnan(cur_f):
                pos[i] = prev_pos
                continue

            new_pos = prev_pos
            s_i = float(s_series.iloc[i]) if np.isfinite(s_series.iloc[i]) else np.nan
            if not np.isfinite(s_i):
                pos[i] = prev_pos
                continue
            
            if (cur_f > s_i) and bool(up.iloc[i]):
                new_pos = 1.0
            elif (cur_f < -s_i) and bool(dn.iloc[i]):
                new_pos = -1.0 if allow_short else 0.0

            prev_pos = new_pos
            pos[i] = prev_pos

        return pd.Series(pos, index=f.index, name="position")

    @staticmethod
    def build_signal_detail(
        factor: pd.Series,
        trend_up_5m: pd.Series,
        trend_down_5m: pd.Series,
        position: pd.Series,
        S: Union[float, pd.Series],
        allow_short: bool = True,
    ) -> pd.DataFrame:
        f = factor.copy()
        up = trend_up_5m.reindex(f.index).fillna(False)
        dn = trend_down_5m.reindex(f.index).fillna(False)
        pos = position.reindex(f.index).fillna(0.0)

        if isinstance(S, pd.Series):
            s_series = S.reindex(f.index).astype(float).ffill()
        else:
            s_series = pd.Series(float(S), index=f.index, dtype=float)

        signal_long = ((f > s_series) & up).fillna(False).astype(int)
        signal_short = (
            ((f < -s_series) & dn).fillna(False).astype(int) if allow_short else pd.Series(0, index=f.index, dtype=int)
        )

        return pd.DataFrame(
            {
                "qrs": f,
                "threshold": s_series,
                "trend_up": up.astype(int),
                "trend_down": dn.astype(int),
                "long_signal": signal_long,
                "short_signal": signal_short,
                "signal_raw": pos.astype(int),
            },
            index=f.index,
        )

    # =========================
    # 7) 策略可视化：价格 + 多空区间阴影（可保存）
    # =========================
    @staticmethod
    def plot_price_with_position_shade(
        close: pd.Series,
        position: pd.Series,
        title: str = "样本外回测曲线（起始日至今）",
        price_label: str = "期货收盘价（连续）：国债期货：10年期",
        long_label: str = "做多区间",
        short_label: str = "做空区间",
        alpha: float = 0.18,
        show_short: bool = True,
        save_dir: Optional[str] = None,
        save_dpi: int = 250,
        fname: Optional[str] = None,
    ) -> Optional[str]:
        df = pd.DataFrame({"close": close, "pos": position}).dropna(subset=["close"])
        if df.empty:
            raise ValueError("close/position 为空或对齐后为空，无法画图。")

        x = df.index
        y = df["close"].values
        pos = df["pos"].fillna(0.0).values

        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(x, y, linewidth=2.0, label=price_label)

        def shade(mask: np.ndarray, label: str, color: str):
            in_seg = False
            seg_start = None
            used = False
            for i in range(len(mask)):
                if mask[i] and (not in_seg):
                    in_seg = True
                    seg_start = x[i]
                if in_seg and ((not mask[i]) or (i == len(mask) - 1)):
                    seg_end = x[i] if (not mask[i]) else x[i]
                    ax.axvspan(
                        seg_start,
                        seg_end,
                        alpha=alpha,
                        color=color,
                        label=(label if not used else None),
                    )
                    used = True
                    in_seg = False

        shade(pos == 1.0, long_label, "red")
        if show_short:
            shade(pos == -1.0, short_label, "blue")

        ax.set_title(title, fontsize=16)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper center", ncol=2, frameon=False)
        fig.autofmt_xdate()
        plt.tight_layout()

        saved_path = None
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            if fname is None:
                fname = "strategy_position_shade.png"
            fname = QRSBacktester._safe_filename(fname)
            saved_path = os.path.join(save_dir, fname)
            plt.savefig(saved_path, dpi=save_dpi, bbox_inches="tight")

        plt.show()
        return saved_path

    # =========================
    # 8) 计算策略收益、基准收益、超额收益（相对Buy&Hold）
    # =========================
    @staticmethod
    def calc_excess_return(close: pd.Series, position: pd.Series) -> pd.DataFrame:
        """
        输出：
        - ret_bh：买入持有收益（基准）
        - ret_strat：策略收益（用 position.shift(1) 防止未来函数）
        - ret_excess：超额收益 = 策略收益 - 基准收益
        - nav_bh / nav_strat / nav_excess：对应净值曲线（从1开始）
        """
        df = pd.DataFrame({"close": close, "pos": position}).dropna(subset=["close"])
        df["ret_bh"] = df["close"].pct_change()

        # 关键：用上一根bar的仓位赚这一根bar的收益
        df["ret_strat"] = df["pos"].shift(1).fillna(0.0) * df["ret_bh"]

        df["ret_excess"] = df["ret_strat"] - df["ret_bh"]

        df["nav_bh"] = (1.0 + df["ret_bh"].fillna(0.0)).cumprod()
        df["nav_strat"] = (1.0 + df["ret_strat"].fillna(0.0)).cumprod()

        # 超额净值：用"超额收益"单独复利
        df["nav_excess"] = (1.0 + df["ret_excess"].fillna(0.0)).cumprod()
        return df

    # =========================
    # 9) 计算夏普比率
    # =========================
    @staticmethod
    def calc_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
        """
        计算年化夏普比率
        
        参数:
        - returns: 收益率序列
        - risk_free_rate: 无风险利率（默认为0）
        - periods_per_year: 每年交易期数（日频默认252）
        
        返回:
        - 年化夏普比率
        """
        # 去除NaN值
        clean_returns = returns.dropna()
        if len(clean_returns) < 2:
            return np.nan
            
        # 计算超额收益
        excess_returns = clean_returns - risk_free_rate / periods_per_year
        
        # 计算年化收益率和年化波动率
        annual_return = excess_returns.mean() * periods_per_year
        annual_volatility = excess_returns.std() * np.sqrt(periods_per_year)
        
        # 计算夏普比率
        if annual_volatility == 0:
            return 0.0
            
        return annual_return / annual_volatility

    # =========================
    # 10) 画净值：策略 vs 基准 vs 超额
    # =========================
    @staticmethod
    def plot_nav_compare(
        perf_df: pd.DataFrame,
        title: str = "NAV: Strategy vs Benchmark vs Excess",
        save_dir: Optional[str] = None,
        save_dpi: int = 250,
        fname: Optional[str] = None,
    ) -> Optional[str]:
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(
            perf_df.index, perf_df["nav_strat"], label="Strategy NAV", linewidth=1.6
        )
        ax.plot(perf_df.index, perf_df["nav_bh"], label="Benchmark NAV", linewidth=1.6)
        ax.plot(perf_df.index, perf_df["nav_excess"], label="Excess NAV", linewidth=1.6)

        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", frameon=False)
        fig.autofmt_xdate()
        plt.tight_layout()

        saved_path = None
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            if fname is None:
                fname = "nav_compare.png"
            fname = QRSBacktester._safe_filename(fname)
            saved_path = os.path.join(save_dir, fname)
            plt.savefig(saved_path, dpi=save_dpi, bbox_inches="tight")

        plt.show()
        return saved_path

    # =========================
    # 11) 网格搜索最优参数
    # =========================
    @staticmethod
    def grid_search(
        factor: pd.Series,
        close: pd.Series,
        s_values: List[float],
        trend_methods: List[str],
        ma_len_days_values: List[int],
        compare_lag_days_values: List[int],
        ma_short_values: List[int],
        ma_long_values: List[int],
        allow_short: bool = True,
        start_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        网格搜索最优参数组合，目标是最大化夏普比率
        
        参数:
        - factor: QRS因子序列
        - close: 价格序列
        - s_values: QRS阈值列表
        - trend_methods: 趋势判断方法列表
        - ma_len_days_values: 均线长度列表（用于ma_compare和price_compare）
        - compare_lag_days_values: 比较滞后期列表（用于ma_compare）
        - ma_short_values: 短均线长度列表（用于ma_cross）
        - ma_long_values: 长均线长度列表（用于ma_cross）
        - allow_short: 是否允许做空
        - start_date: 回测起始日期
        
        返回:
        - 包含所有参数组合及其夏普比率的结果表
        """
        # 截取数据
        if start_date is not None:
            factor = factor.loc[pd.to_datetime(start_date):]
            close = close.loc[pd.to_datetime(start_date):]
        
        # 确保数据对齐
        df = pd.DataFrame({"factor": factor, "close": close}).dropna()
        factor = df["factor"]
        close = df["close"]
        
        # 结果存储
        results = []

        method_set = set(trend_methods)
        param_grid: List[Dict[str, List]] = []
        if "ma_compare" in method_set:
            param_grid.append(
                {
                    "S": s_values,
                    "trend_method": ["ma_compare"],
                    "ma_len_days": ma_len_days_values,
                    "compare_lag_days": compare_lag_days_values,
                }
            )
        if "ma_cross" in method_set:
            param_grid.append(
                {
                    "S": s_values,
                    "trend_method": ["ma_cross"],
                    "ma_short": ma_short_values,
                    "ma_long": ma_long_values,
                }
            )
        if "price_compare" in method_set:
            param_grid.append(
                {
                    "S": s_values,
                    "trend_method": ["price_compare"],
                    "ma_len_days": ma_len_days_values,
                }
            )

        if not param_grid:
            raise ValueError("trend_methods 为空或不包含支持的方法")

        periods_per_year = 252 * 54
        total_combinations = QRSBacktester._param_grid_size(param_grid)
        with tqdm(total=total_combinations, desc="Grid Search Progress") as pbar:
            for params in QRSBacktester._iter_param_grid(param_grid):
                s = float(params["S"])
                trend_method = str(params["trend_method"])
                row = {
                    "S": s,
                    "trend_method": trend_method,
                    "ma_len_days": np.nan,
                    "compare_lag_days": np.nan,
                    "ma_short": np.nan,
                    "ma_long": np.nan,
                }

                if trend_method == "ma_compare":
                    ma_len = int(params["ma_len_days"])
                    lag = int(params["compare_lag_days"])
                    row["ma_len_days"] = ma_len
                    row["compare_lag_days"] = lag
                    trend_df = QRSBacktester.build_daily_trend_filter_from_5m(
                        close_5m=close,
                        ma_len_days=ma_len,
                        compare_lag_days=lag,
                        trend_method=trend_method,
                    )
                elif trend_method == "ma_cross":
                    ma_short = int(params["ma_short"])
                    ma_long = int(params["ma_long"])
                    row["ma_short"] = ma_short
                    row["ma_long"] = ma_long
                    trend_df = QRSBacktester.build_daily_trend_filter_from_5m(
                        close_5m=close,
                        trend_method=trend_method,
                        ma_short=ma_short,
                        ma_long=ma_long,
                    )
                elif trend_method == "price_compare":
                    ma_len = int(params["ma_len_days"])
                    row["ma_len_days"] = ma_len
                    trend_df = QRSBacktester.build_daily_trend_filter_from_5m(
                        close_5m=close,
                        ma_len_days=ma_len,
                        trend_method=trend_method,
                    )
                else:
                    raise ValueError(f"不支持的 trend_method: {trend_method}")

                position = QRSBacktester.build_position_strategy_simple(
                    factor=factor,
                    trend_up_5m=trend_df["trend_up_5m"],
                    trend_down_5m=trend_df["trend_down_5m"],
                    close=close,
                    S=s,
                    allow_short=allow_short,
                )

                perf = QRSBacktester.calc_excess_return(close=close, position=position)
                sharpe = QRSBacktester.calc_sharpe_ratio(perf["ret_strat"], periods_per_year=periods_per_year)

                row.update(
                    {
                        "sharpe_ratio": sharpe,
                        "annual_return": perf["ret_strat"].mean() * periods_per_year,
                        "annual_volatility": perf["ret_strat"].std() * np.sqrt(periods_per_year),
                        "max_drawdown": (
                            perf["nav_strat"] / perf["nav_strat"].expanding(min_periods=1).max() - 1
                        ).min(),
                    }
                )
                results.append(row)
                pbar.update(1)
        
        # 转换为DataFrame并排序
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by="sharpe_ratio", ascending=False)
        
        return results_df

    @staticmethod
    def grid_search_cv(
        factor: pd.Series,
        close: pd.Series,
        s_values: List[float],
        trend_methods: List[str],
        ma_len_days_values: List[int],
        compare_lag_days_values: List[int],
        ma_short_values: List[int],
        ma_long_values: List[int],
        allow_short: bool = True,
        start_date: Optional[str] = None,
        n_splits: int = 3,
    ) -> Tuple[Any, pd.DataFrame]:
        if SklearnGridSearchCV is None or SklearnTimeSeriesSplit is None:
            raise ImportError("未检测到 sklearn，无法使用 GridSearchCV")

        if start_date is not None:
            factor = factor.loc[pd.to_datetime(start_date):]
            close = close.loc[pd.to_datetime(start_date):]

        X = pd.DataFrame({"factor": factor, "close": close}).dropna()
        if X.empty:
            raise ValueError("factor/close 对齐后为空，无法网格搜索")

        method_set = set(trend_methods)
        param_grid: List[Dict[str, List]] = []
        if "ma_compare" in method_set:
            param_grid.append(
                {
                    "S": s_values,
                    "trend_method": ["ma_compare"],
                    "ma_len_days": ma_len_days_values,
                    "compare_lag_days": compare_lag_days_values,
                }
            )
        if "ma_cross" in method_set:
            param_grid.append(
                {
                    "S": s_values,
                    "trend_method": ["ma_cross"],
                    "ma_short": ma_short_values,
                    "ma_long": ma_long_values,
                }
            )
        if "price_compare" in method_set:
            param_grid.append(
                {
                    "S": s_values,
                    "trend_method": ["price_compare"],
                    "ma_len_days": ma_len_days_values,
                }
            )
        if not param_grid:
            raise ValueError("trend_methods 为空或不包含支持的方法")

        periods_per_year = 252 * 54
        estimator = QRSGridSearchEstimator(allow_short=allow_short, periods_per_year=periods_per_year)

        n_samples = len(X)
        if n_samples <= max(5, n_splits + 1):
            split = n_samples // 2
            if split <= 0 or split >= n_samples:
                raise ValueError("样本数量过少，无法进行 CV")
            cv = [(np.arange(split), np.arange(split, n_samples))]
        else:
            cv = SklearnTimeSeriesSplit(n_splits=min(int(n_splits), n_samples - 1))

        gs = SklearnGridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring=None,
            cv=cv,
            refit=True,
            n_jobs=1,
        )
        gs.fit(X, y=None)

        cv_df = pd.DataFrame(gs.cv_results_)
        params_df = cv_df["params"].apply(pd.Series)
        results_df = pd.concat(
            [
                params_df,
                cv_df[["mean_test_score", "std_test_score", "rank_test_score"]],
            ],
            axis=1,
        ).rename(
            columns={
                "mean_test_score": "sharpe_ratio",
                "std_test_score": "sharpe_std",
                "rank_test_score": "rank",
            }
        )

        for c in ["ma_len_days", "compare_lag_days", "ma_short", "ma_long"]:
            if c not in results_df.columns:
                results_df[c] = np.nan

        full_metrics = []
        for _, r in results_df.iterrows():
            full_metrics.append(
                QRSBacktester._evaluate_metrics_for_params(
                    factor=X["factor"],
                    close=X["close"],
                    params=r.to_dict(),
                    allow_short=allow_short,
                    periods_per_year=periods_per_year,
                )
            )
        results_df = pd.concat([results_df.reset_index(drop=True), pd.DataFrame(full_metrics)], axis=1)
        results_df = results_df.sort_values(by="sharpe_ratio", ascending=False)
        return gs, results_df


def main():
    # =========================
    # 配置区：你只改这里
    # =========================
    cfg = resolve_run_config()
    SHEET_NAME = cfg.sheet_name

    # RSRS 因子参数
    N = 16
    M = 600
    N_POWER = 2
    QRS_COL = "qrs"

    # 策略阈值
    S = 0.5
    ALLOW_SHORT = True  # 是否允许做空
    
    # 参数（因子斜率）
    SLOPE_L = 54  # 每天交易日数为 54 个 5 分钟bar
    SLOPE_SCALE = 20.0  # 斜率放大倍数
    
    # 图片保存目录
    SAVE_DIR = cfg.save_dir

    BACKTEST_START_DATE = "2025-01-01"
    TL_EXCEL_PATH = r"D:\Python\浙商证券固收\RSRS QRS择时报告\30年国债期货_5min_2年.xlsx"

    # ===== 策略：日频趋势过滤参数 =====
    MA_LEN_DAYS = 5
    COMPARE_LAG_DAYS = 2  # 比较间隔天数
    RUN_PAPER_DYNAMIC = bool(cfg.paper_dynamic)

    setup_cn_font()

    def _slice_by_start(obj: Union[pd.Series, pd.DataFrame], start: Optional[str]) -> Union[pd.Series, pd.DataFrame]:
        if start is None:
            return obj
        ts = pd.to_datetime(start)
        return obj.loc[ts:]

    def run_one(symbol: str, excel_path: str, sheet_name: Optional[str]) -> None:
        save_dir = os.path.join(SAVE_DIR, symbol)
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n========== {symbol} ==========")
        print(f"数据源: {excel_path}")
        print(f"输出目录: {save_dir}")
        print(f"回测/网格搜索起始日期: {BACKTEST_START_DATE}")

        p = QRSParams(N=N, M=M, n=N_POWER)
        bt = QRSBacktester(excel_path, sheet_name)
        bt.load_excel()
        fac = bt.compute_factor(p)

        fac_plot = _slice_by_start(fac, BACKTEST_START_DATE).copy()

        if bool(RUN_PAPER_DYNAMIC) and bt.df_5m is not None:
            print("\n开始运行论文动态参数模型（可能较慢）...")
            t0 = time.time()
            ohlc_all = bt.df_5m.copy()
            ohlc_all = _slice_by_start(ohlc_all, BACKTEST_START_DATE).copy()
            param_grid = []
            for NN in [15, 20, 25]:
                for MM in [240, 600]:
                    for nn in [1, 2]:
                        param_grid.append(
                            {
                                "key": f"N{NN}_M{MM}_n{nn}_pn",
                                "N": NN,
                                "M": MM,
                                "n": nn,
                                "normalize_penalty": True,
                                "penalty_mean_window": MM,
                            }
                        )

            dyn = QRSBacktester.dynamic_qrs_paper_model(
                ohlc=ohlc_all,
                param_grid=param_grid,
                window_bars=int(54 * 60),
                horizon_bars=int(54 * 10),
                r=0.05,
                s_step=0.2,
                min_bin_count=20,
                rebalance_every_bars=54,
                s_max=3.0,
            )
            dyn.to_csv(os.path.join(save_dir, "qrs_paper_dynamic_params.csv"), index=True)

            factor_dyn = dyn["qrs_dynamic"]
            S_dyn = dyn["S_dynamic"]
            trend_dyn = bt.build_daily_trend_filter_from_5m(
                close_5m=ohlc_all["close"],
                ma_len_days=MA_LEN_DAYS,
                compare_lag_days=COMPARE_LAG_DAYS,
            )
            pos_dyn = bt.build_position_strategy_simple(
                factor=factor_dyn,
                trend_up_5m=trend_dyn["trend_up_5m"],
                trend_down_5m=trend_dyn["trend_down_5m"],
                close=ohlc_all["close"],
                S=S_dyn,
                allow_short=ALLOW_SHORT,
            )
            perf_dyn = bt.calc_excess_return(close=ohlc_all["close"], position=pos_dyn)
            perf_dyn.to_csv(os.path.join(save_dir, "qrs_paper_dynamic_perf.csv"), index=True)
            bt.plot_nav_compare(
                perf_df=perf_dyn,
                title="Paper dynamic QRS model NAV",
                save_dir=save_dir,
                fname="NAV_PAPER_DYNAMIC.png",
            )
            print(f"论文动态参数模型完成，用时 {time.time() - t0:.1f}s")
        elif bool(RUN_PAPER_DYNAMIC):
            print("\n未检测到 OHLC 数据，跳过论文动态参数模型。")
        else:
            print("\n未启用论文动态参数模型（如需启用：python qrs_t_strategy.py --paper-dynamic）。")

        factor = fac_plot[QRS_COL]
        trend_df = bt.build_daily_trend_filter_from_5m(
            close_5m=fac_plot["close"],
            ma_len_days=MA_LEN_DAYS,
            compare_lag_days=COMPARE_LAG_DAYS,
        )

        pos2 = bt.build_position_strategy_simple(
            factor=factor,
            trend_up_5m=trend_df["trend_up_5m"],
            trend_down_5m=trend_df["trend_down_5m"],
            close=fac_plot["close"],
            S=S,
            allow_short=ALLOW_SHORT,
        )

        perf2 = bt.calc_excess_return(close=fac_plot["close"], position=pos2)
        bt.plot_nav_compare(
            perf_df=perf2,
            title=f"Strategy NAV (QRS={QRS_COL}, S={S:g}, MA{MA_LEN_DAYS}, lag={COMPARE_LAG_DAYS})",
            save_dir=save_dir,
            fname=f"NAV_STRAT_{QRS_COL}_S{S:g}_MA{MA_LEN_DAYS}_LAG{COMPARE_LAG_DAYS}.png",
        )

        print("\n开始网格搜索最优参数...")

        s_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        trend_methods = ["ma_compare", "ma_cross", "price_compare"]
        ma_len_days_values = [3, 5, 10, 20]
        compare_lag_days_values = [1, 2, 3]
        ma_short_values = [3, 5, 10]
        ma_long_values = [10, 20, 30]

        gs = None
        try:
            gs, grid_results = bt.grid_search_cv(
                factor=fac[QRS_COL],
                close=fac["close"],
                s_values=s_values,
                trend_methods=trend_methods,
                ma_len_days_values=ma_len_days_values,
                compare_lag_days_values=compare_lag_days_values,
                ma_short_values=ma_short_values,
                ma_long_values=ma_long_values,
                allow_short=ALLOW_SHORT,
                start_date=BACKTEST_START_DATE,
                n_splits=3,
            )
            print("\nGridSearchCV best_params_：")
            print(gs.best_params_)
            print(f"GridSearchCV best_score_（mean_test_score）：{gs.best_score_:.6f}")
        except Exception:
            grid_results = bt.grid_search(
                factor=fac[QRS_COL],
                close=fac["close"],
                s_values=s_values,
                trend_methods=trend_methods,
                ma_len_days_values=ma_len_days_values,
                compare_lag_days_values=compare_lag_days_values,
                ma_short_values=ma_short_values,
                ma_long_values=ma_long_values,
                allow_short=ALLOW_SHORT,
                start_date=BACKTEST_START_DATE,
            )

        print("\n网格搜索结果（按夏普比率排序，前10名）：")
        print(grid_results.head(10).to_string(index=False))

        grid_results.to_csv(os.path.join(save_dir, "qrs_grid_search_results.csv"), index=False)
        print(f"\n完整结果已保存到: {os.path.join(save_dir, 'qrs_grid_search_results.csv')}")

        best_params = grid_results.iloc[0]
        print(f"\n最佳参数组合：")
        print(f"S值: {best_params['S']}")
        print(f"趋势方法: {best_params['trend_method']}")

        if best_params["trend_method"] == "ma_compare":
            print(f"均线长度: {best_params['ma_len_days']}")
            print(f"滞后期: {best_params['compare_lag_days']}")
        elif best_params["trend_method"] == "ma_cross":
            print(f"短均线: {best_params['ma_short']}")
            print(f"长均线: {best_params['ma_long']}")
        elif best_params["trend_method"] == "price_compare":
            print(f"均线长度: {best_params['ma_len_days']}")

        print(f"夏普比率: {best_params['sharpe_ratio']:.4f}")
        print(f"年化收益: {best_params['annual_return']:.4f} ({best_params['annual_return']*100:.2f}%)")
        print(f"年化波动: {best_params['annual_volatility']:.4f} ({best_params['annual_volatility']*100:.2f}%)")
        print(f"最大回撤: {best_params['max_drawdown']:.4f} ({best_params['max_drawdown']*100:.2f}%)")

        print("\n使用最佳参数重新运行策略...")

        if best_params["trend_method"] == "ma_compare":
            best_trend_df = bt.build_daily_trend_filter_from_5m(
                close_5m=fac_plot["close"],
                ma_len_days=int(best_params["ma_len_days"]),
                compare_lag_days=int(best_params["compare_lag_days"]),
                trend_method=best_params["trend_method"],
            )
        elif best_params["trend_method"] == "ma_cross":
            best_trend_df = bt.build_daily_trend_filter_from_5m(
                close_5m=fac_plot["close"],
                ma_short=int(best_params["ma_short"]),
                ma_long=int(best_params["ma_long"]),
                trend_method=best_params["trend_method"],
            )
        else:
            best_trend_df = bt.build_daily_trend_filter_from_5m(
                close_5m=fac_plot["close"],
                ma_len_days=int(best_params["ma_len_days"]),
                trend_method=best_params["trend_method"],
            )

        best_pos = bt.build_position_strategy_simple(
            factor=fac_plot[QRS_COL],
            trend_up_5m=best_trend_df["trend_up_5m"],
            trend_down_5m=best_trend_df["trend_down_5m"],
            close=fac_plot["close"],
            S=best_params["S"],
            allow_short=ALLOW_SHORT,
        )

        best_perf = bt.calc_excess_return(close=fac_plot["close"], position=best_pos)

        bt.plot_price_with_position_shade(
            close=fac_plot["close"],
            position=best_pos,
            title=f"最佳策略：QRS={QRS_COL}, S={best_params['S']:.2f}, 趋势方法={best_params['trend_method']}",
            save_dir=save_dir,
            save_dpi=250,
            fname="best_strategy_position.png",
        )

        bt.plot_nav_compare(
            perf_df=best_perf,
            title=f"最佳策略净值：夏普比率={best_params['sharpe_ratio']:.4f}",
            save_dir=save_dir,
            save_dpi=250,
            fname="best_strategy_nav.png",
        )

        signal_detail_5m = bt.build_signal_detail(
            factor=fac_plot[QRS_COL],
            trend_up_5m=best_trend_df["trend_up_5m"],
            trend_down_5m=best_trend_df["trend_down_5m"],
            position=best_pos,
            S=best_params["S"],
            allow_short=ALLOW_SHORT,
        )
        signal_detail_5m["ret"] = fac_plot["close"].pct_change()
        signal_detail_5m["strategy_ret"] = best_pos.shift(1).fillna(0.0) * signal_detail_5m["ret"]
        signal_detail_5m["cum_return"] = (1.0 + signal_detail_5m["strategy_ret"].fillna(0.0)).cumprod()
        signal_detail_5m["benchmark"] = (1.0 + signal_detail_5m["ret"].fillna(0.0)).cumprod()
        signal_detail_5m["final_signal"] = best_pos.shift(1).fillna(0.0).astype(int)

        signal_5m_csv_path = os.path.join(save_dir, "best_strategy_signal_5m.csv")
        signal_detail_5m.to_csv(signal_5m_csv_path, index=True, encoding="utf-8-sig")
        print(f"5分钟信号已保存: {signal_5m_csv_path}")

        daily_close = fac_plot["close"].resample("1D").last().dropna()
        daily_report = pd.DataFrame(index=daily_close.index)
        daily_report["qrs"] = fac_plot[QRS_COL].resample("1D").last().reindex(daily_close.index)
        daily_report["trend_up"] = best_trend_df["trend_up_5m"].resample("1D").last().reindex(daily_close.index).fillna(False).astype(int)
        daily_report["trend_down"] = best_trend_df["trend_down_5m"].resample("1D").last().reindex(daily_close.index).fillna(False).astype(int)
        daily_report["long_signal"] = signal_detail_5m["long_signal"].resample("1D").max().reindex(daily_close.index).fillna(0).astype(int)
        daily_report["short_signal"] = signal_detail_5m["short_signal"].resample("1D").max().reindex(daily_close.index).fillna(0).astype(int)
        daily_report["signal_raw"] = best_pos.resample("1D").last().reindex(daily_close.index).fillna(0.0).astype(int)
        daily_report["final_signal"] = daily_report["signal_raw"].shift(1).fillna(0).astype(int)
        daily_report["close"] = daily_close
        daily_report["ret"] = daily_close.pct_change()
        daily_report["strategy_ret"] = daily_report["final_signal"] * daily_report["ret"]
        daily_report["cum_return"] = (1.0 + daily_report["strategy_ret"].fillna(0.0)).cumprod()
        daily_report["benchmark"] = (1.0 + daily_report["ret"].fillna(0.0)).cumprod()

        daily_signal_csv_path = os.path.join(save_dir, "best_strategy_signal_daily.csv")
        daily_signal_xlsx_path = os.path.join(save_dir, "best_strategy_signal_daily.xlsx")
        daily_report.to_csv(daily_signal_csv_path, index=True, encoding="utf-8-sig")
        daily_report.to_excel(daily_signal_xlsx_path, index=True)
        print(f"日频信号已保存: {daily_signal_csv_path}")
        print(f"日频信号 Excel 已保存: {daily_signal_xlsx_path}")
        print("\n日频信号预览（最近10行）:")
        print(daily_report.tail(10).to_string())

    if not os.path.exists(cfg.excel_path):
        raise FileNotFoundError(f"未找到 T 数据文件: {cfg.excel_path}")
    if not os.path.exists(TL_EXCEL_PATH):
        raise FileNotFoundError(f"未找到 TL 数据文件: {TL_EXCEL_PATH}")

    run_one("T", cfg.excel_path, SHEET_NAME)
    run_one("TL", TL_EXCEL_PATH, SHEET_NAME)

if __name__ == "__main__":
    main()
