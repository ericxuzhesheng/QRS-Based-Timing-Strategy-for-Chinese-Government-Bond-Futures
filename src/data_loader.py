from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .utils import PROJECT_ROOT


DATE_ALIASES = {"date", "datetime", "trade_date", "time", "日期", "时间", "交易日期"}
COLUMN_ALIASES = {
    "date": {"date", "datetime", "trade_date", "time", "日期", "时间", "交易日期"},
    "open": {"open", "open_price", "开盘价", "开盘"},
    "high": {"high", "high_price", "最高价", "最高"},
    "low": {"low", "low_price", "最低价", "最低"},
    "close": {"close", "close_price", "收盘价", "收盘", "结算价", "结算"},
    "volume": {"volume", "vol", "成交量", "成交额"},
    "open_interest": {"open_interest", "openinterest", "oi", "持仓量", "持仓"},
}
REQUIRED_COLUMNS = {"date", "open", "high", "low", "close"}
OPTIONAL_COLUMNS = ["volume", "open_interest"]
SEARCH_KEYWORDS = ("qrs", "t", "国债期货", "cgb", "daily", "latest", "10年", "30年", "tl")


def _normalize_name(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip().lower()
    return "".join(text.replace("_", "").replace("-", "").split())


def _canonical_column(value: object) -> str | None:
    raw = "" if pd.isna(value) else str(value).strip()
    norm = _normalize_name(raw)
    for target, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            alias_norm = _normalize_name(alias)
            if norm == alias_norm or alias in raw:
                return target
    return None


def _find_header_row(raw: pd.DataFrame, search_rows: int = 80) -> int | None:
    n = min(search_rows, len(raw))
    for i in range(n):
        mapped = {_canonical_column(v) for v in raw.iloc[i].tolist()}
        if {"date", "open", "high", "low", "close"}.issubset(mapped):
            return i
    return None


def _read_excel(path: Path, sheet_name: str | int | None = None) -> pd.DataFrame:
    sheet = 0 if sheet_name is None else sheet_name
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    header_row = _find_header_row(raw)
    if header_row is None:
        return pd.read_excel(path, sheet_name=sheet)
    header = raw.iloc[header_row].astype(str).str.strip().tolist()
    data = raw.iloc[header_row + 1 :].copy()
    data.columns = header
    return data


def _read_csv(path: Path, encoding: str | None = None) -> pd.DataFrame:
    encodings = [encoding] if encoding else ["utf-8-sig", "utf-8", "gbk"]
    last_error: Exception | None = None
    for enc in encodings:
        try:
            raw = pd.read_csv(path, encoding=enc, header=None)
            header_row = _find_header_row(raw)
            if header_row is None:
                return pd.read_csv(path, encoding=enc)
            header = raw.iloc[header_row].astype(str).str.strip().tolist()
            data = raw.iloc[header_row + 1 :].copy()
            data.columns = header
            return data
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise ValueError(f"Unable to read CSV: {path}")


def standardize_market_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.dropna(axis=1, how="all").copy()
    rename_map = {}
    used_targets: set[str] = set()
    for col in data.columns:
        target = _canonical_column(col)
        if target and target not in used_targets:
            rename_map[col] = target
            used_targets.add(target)
    data = data.rename(columns=rename_map)

    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Missing required OHLC/date columns: {sorted(missing)}; columns={list(data.columns)}")

    out_cols = ["date", "open", "high", "low", "close", *[c for c in OPTIONAL_COLUMNS if c in data.columns]]
    out = data[out_cols].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])

    for col in ["open", "high", "low", "close", *OPTIONAL_COLUMNS]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    key_missing = out[["open", "high", "low", "close"]].isna().any(axis=1)
    if key_missing.all():
        raise ValueError("All OHLC rows are missing after numeric conversion.")
    out = out.loc[~key_missing].replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    return out


def load_market_data(path: str | Path, sheet_name: str | int | None = None) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input data file not found: {file_path}")
    suffix = file_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        raw = _read_excel(file_path, sheet_name=sheet_name)
    elif suffix == ".csv":
        raw = _read_csv(file_path)
    else:
        raise ValueError(f"Unsupported input format: {suffix}. Use CSV or Excel.")
    return standardize_market_data(raw)


def aggregate_to_daily(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"
    if "open_interest" in df.columns:
        agg["open_interest"] = "last"
    daily = df.resample("1D").agg(agg).dropna(subset=["open", "high", "low", "close"])
    return daily.reset_index()


def discover_input_file(contract: str = "T", search_dirs: Iterable[str | Path] | None = None) -> Path:
    dirs = [PROJECT_ROOT / "data" / "raw", PROJECT_ROOT] if search_dirs is None else [Path(p) for p in search_dirs]
    candidates: list[Path] = []
    for directory in dirs:
        if not directory.exists():
            continue
        for pattern in ("*.csv", "*.xlsx", "*.xls"):
            candidates.extend(directory.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            "No CSV/Excel input found. Put data under data/raw/ or pass --input, e.g. "
            "python scripts/run_qrs_pipeline.py --input data/raw/qrs_timing_latest.xlsx --contract T"
        )

    contract_lower = contract.lower()

    def score(path: Path) -> tuple[int, float]:
        name = path.name.lower()
        value = 0
        if contract_lower == "tl":
             if "tl" in name or "30" in name or "30年" in name:
                 value += 50
        elif contract_lower == "t":
             if ("10" in name or "10年" in name) and ("30" not in name):
                 value += 50
             elif " t" in name or name.startswith("t"):
                 value += 30

        for keyword in SEARCH_KEYWORDS:
            if keyword.lower() in name:
                value += 5

        return value, path.stat().st_mtime

    best_match = sorted(candidates, key=score, reverse=True)[0]
    print(f"Contract {contract}: selected file {best_match.name}")
    return best_match


# ── Tushare data source support ────────────────────────────────────


def load_from_tushare(
    contract: str = "T",
    freq: str = "5min",
    start_date: str = "20240101",
    end_date: str | None = None,
    use_cache: bool = True,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load market data from Tushare API with local Parquet caching.

    Priority:  cache → Tushare API → raise error (caller may fall back
    to local files).

    Parameters
    ----------
    contract: "T" or "TL".
    freq: "5min" (ft_mins) or "daily" (fut_daily).
    start_date: YYYYMMDD — full-fetch start when cache is empty.
    end_date: YYYYMMDD — defaults to today.
    use_cache: Read/write local Parquet cache.
    force_refresh: Ignore cache and re-fetch everything.

    Returns
    -------
    Standardized OHLC DataFrame: date, open, high, low, close, [volume, open_interest].
    """
    from datetime import datetime

    from .cache_manager import incremental_update, read_cache, write_cache
    from .tushare_fetcher import fetch_ft_5min, fetch_fut_daily

    end_str = end_date or datetime.now().strftime("%Y%m%d")

    if freq == "5min":
        fetch_fn = lambda s, e: fetch_ft_5min(contract, s, e)
    elif freq == "daily":
        fetch_fn = lambda s, e: fetch_fut_daily(contract, s, e)
    else:
        raise ValueError(f"Unsupported frequency: {freq}")

    if not use_cache:
        # Direct API call, no caching
        df = fetch_fn(start_date, end_str)
        if df.empty:
            raise RuntimeError(f"Tushare returned empty data for {contract} ({freq})")
        return df

    # Incremental update (cache → api → merge → write)
    df = incremental_update(contract, freq, fetch_fn, start_date=start_date, force=force_refresh)
    if df.empty:
        raise RuntimeError(f"No data available for {contract} ({freq}) — Tushare API returned empty and no cache.")
    return df

