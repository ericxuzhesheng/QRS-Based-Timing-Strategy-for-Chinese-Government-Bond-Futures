from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── Load .env if present (for TUSHARE_TOKEN and friends) ─────────────
try:
    from dotenv import load_dotenv as _load_dotenv

    _env_file = PROJECT_ROOT / ".env"
    if _env_file.exists():
        _load_dotenv(_env_file)
except ImportError:
    pass


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def drawdown(nav: pd.Series) -> pd.Series:
    nav = nav.astype(float)
    running_max = nav.cummax()
    return nav / running_max - 1.0


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    clean = returns.dropna().astype(float)
    if clean.empty:
        return np.nan
    nav = (1.0 + clean).cumprod()
    years = len(clean) / periods_per_year
    if years <= 0 or nav.iloc[-1] <= 0:
        return np.nan
    return float(nav.iloc[-1] ** (1.0 / years) - 1.0)


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    clean = returns.dropna().astype(float)
    if len(clean) < 2:
        return np.nan
    return float(clean.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    ann_ret = annualized_return(returns, periods_per_year)
    ann_vol = annualized_volatility(returns, periods_per_year)
    if not np.isfinite(ann_ret) or not np.isfinite(ann_vol) or ann_vol == 0:
        return np.nan
    return float(ann_ret / ann_vol)


def max_drawdown(nav: pd.Series) -> float:
    dd = drawdown(nav).dropna()
    return float(dd.min()) if not dd.empty else np.nan


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    return df.to_markdown(index=False)


def format_percent(value: float) -> str:
    if not np.isfinite(value):
        return "N/A"
    return f"{value * 100:.2f}%"


def format_number(value: float) -> str:
    if not np.isfinite(value):
        return "N/A"
    return f"{value:.4f}"


def require_files(paths: Iterable[str | Path]) -> list[Path]:
    missing = [Path(p) for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("Missing required output files: " + ", ".join(str(p) for p in missing))
    return [Path(p) for p in paths]
