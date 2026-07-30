"""Parquet cache manager for Tushare data.

Design:
- Cache format: Apache Parquet (snappy compression)
- Cache directory: data/cache/
- Metadata tracked in meta.json alongside cache files

Cache keys are `{contract}_{freq}` — e.g. "T_5min" / "TL_daily".
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from .utils import project_path

CACHE_DIR = project_path("data", "cache")
META_PATH = CACHE_DIR / "meta.json"

# ── helpers ───────────────────────────────────────────────────────────


def _cache_key(contract: str, freq: str) -> str:
    return f"{contract.lower()}_{freq}"


def _cache_path(contract: str, freq: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{_cache_key(contract, freq)}.parquet"


# ── read / write ───────────────────────────────────────────────────────


def read_cache(contract: str, freq: str) -> pd.DataFrame | None:
    """Read cached data for a contract + frequency.  Returns None on miss."""
    path = _cache_path(contract, freq)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if df.empty:
            return None
        return df
    except Exception:
        return None


def write_cache(df: pd.DataFrame, contract: str, freq: str) -> Path:
    """Write DataFrame to Parquet cache.  Also updates meta.json."""
    path = _cache_path(contract, freq)
    df.to_parquet(path, index=False, compression="snappy")
    _update_meta_entry(contract, freq, df)
    return path


# ── metadata ───────────────────────────────────────────────────────────


def _load_meta() -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not META_PATH.exists():
        return {}
    try:
        return json.loads(META_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_meta(meta: dict) -> None:
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _update_meta_entry(contract: str, freq: str, df: pd.DataFrame) -> None:
    meta = _load_meta()
    key = _cache_key(contract, freq)
    dates = pd.to_datetime(df["date"])
    meta[key] = {
        "contract": contract,
        "frequency": freq,
        "last_updated": datetime.now().isoformat(),
        "date_start": str(dates.min().date()),
        "date_end": str(dates.max().date()),
        "row_count": int(len(df)),
    }
    _save_meta(meta)


def get_cache_info(contract: str, freq: str) -> dict | None:
    """Return metadata dict for a cache entry, or None."""
    return _load_meta().get(_cache_key(contract, freq))


def is_cache_stale(contract: str, freq: str, max_age_hours: int = 24) -> bool:
    """Return True if cache doesn't exist or is older than *max_age_hours*."""
    info = get_cache_info(contract, freq)
    if info is None or "last_updated" not in info:
        return True
    last = datetime.fromisoformat(info["last_updated"])
    return (datetime.now() - last) > timedelta(hours=max_age_hours)


def get_last_cached_date(contract: str, freq: str) -> str | None:
    """Return 'YYYYMMDD' of the last cached date, or None."""
    info = get_cache_info(contract, freq)
    if info is None or not info.get("date_end"):
        return None
    return info["date_end"].replace("-", "")


# ── incremental update ─────────────────────────────────────────────────


def incremental_update(
    contract: str,
    freq: str,
    fetch_fn,
    start_date: str = "20150101",
    force: bool = False,
) -> pd.DataFrame:
    """Incremental cache update.

    1. Read existing cache.
    2. Determine fetch range — only new dates unless *force* is True.
    3. Call *fetch_fn(start, end)* to get new data.
    4. Merge, deduplicate, write back to cache.

    Parameters
    ----------
    contract: "T" or "TL"
    freq: "5min" or "daily"
    fetch_fn: callable(start_date: str, end_date: str) -> pd.DataFrame
        Must return a DataFrame whose first column is "date" (datetime64).
    start_date: Fallback start when cache is empty (YYYYMMDD).
    force: If True, clear cache and re-fetch everything.

    Returns
    -------
    Full merged DataFrame, possibly empty.
    """
    end_date = datetime.now().strftime("%Y%m%d")

    if force:
        clear_cache(contract, freq)

    cached = read_cache(contract, freq)
    if cached is not None and not cached.empty:
        last_cached = get_last_cached_date(contract, freq)
        if last_cached is None:
            fetch_start = start_date
        else:
            # Start from the day AFTER the last cached date
            fetch_start = (datetime.strptime(last_cached, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")
    else:
        cached = None
        fetch_start = start_date

    # If we're already up-to-date, return cache as-is
    if fetch_start > end_date:
        return cached if cached is not None else pd.DataFrame()

    # Fetch new data
    try:
        new_data = fetch_fn(fetch_start, end_date)
    except Exception:
        if cached is not None:
            return cached  # degrade gracefully
        raise

    if new_data.empty:
        if cached is not None:
            return cached
        return pd.DataFrame()

    # Merge
    if cached is not None and not cached.empty:
        combined = pd.concat([cached, new_data], ignore_index=True)
    else:
        combined = new_data

    # Deduplicate — keep the latest version of each date row
    if "date" in combined.columns:
        combined = combined.drop_duplicates(subset=["date"], keep="last")
        combined = combined.sort_values("date").reset_index(drop=True)

    write_cache(combined, contract, freq)
    return combined


# ── clear ──────────────────────────────────────────────────────────────


def clear_cache(contract: str | None = None, freq: str | None = None) -> None:
    """Clear all or specific cache files.

    With no arguments clears everything.
    """
    if contract is None:
        for f in CACHE_DIR.glob("*.parquet"):
            f.unlink(missing_ok=True)
        META_PATH.unlink(missing_ok=True)
        return

    for c in ([contract] if contract else ["T", "TL"]):
        for fq in ([freq] if freq else ["5min", "daily"]):
            p = _cache_path(c, fq)
            p.unlink(missing_ok=True)
            # also remove the key from meta
            meta = _load_meta()
            meta.pop(_cache_key(c, fq), None)
            _save_meta(meta)
