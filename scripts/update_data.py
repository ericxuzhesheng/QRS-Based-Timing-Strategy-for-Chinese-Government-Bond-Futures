#!/usr/bin/env python
"""Daily data update script — fetch from Tushare, merge into cache,
then re-run the full QRS pipeline to refresh all charts + report + README.

Usage:
    python scripts/update_data.py                          # Update cache only
    python scripts/update_data.py --run-pipeline           # Update cache + run full pipeline + refresh README
    python scripts/update_data.py --mode full              # Same as --run-pipeline
    python scripts/update_data.py --contract T             # Single contract
    python scripts/update_data.py --force                  # Re-fetch everything

GitHub Actions (default mode: full):
    python scripts/update_data.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cache_manager import clear_cache, get_cache_info
from src.data_loader import load_from_tushare
from src.tushare_fetcher import health_check as tushare_health


ALL_CONTRACTS = ["T", "TL"]
ALL_FREQS = ["5min", "daily"]


def update_one(contract: str, freq: str, force: bool = False) -> dict:
    """Update cache for a single contract+frequency pair."""
    print(f"\n{'=' * 60}")
    print(f"  Updating {contract} / {freq}")
    print(f"{'=' * 60}")

    result = {"contract": contract, "freq": freq, "ok": False, "rows": 0, "start": None, "end": None}

    try:
        if force:
            clear_cache(contract, freq)

        prev_info = get_cache_info(contract, freq) or {}
        prev_rows = prev_info.get("row_count", 0)

        df = load_from_tushare(contract=contract, freq=freq, use_cache=True, force_refresh=force)
        new_rows = len(df) - prev_rows if prev_rows else len(df)

        result["ok"] = True
        result["rows"] = len(df)
        result["new_rows"] = new_rows
        result["start"] = str(df["date"].min())
        result["end"] = str(df["date"].max())
        print(f"  OK  rows={len(df):,}  new={new_rows:,}  {result['start']} -> {result['end']}")

    except Exception as exc:
        result["error"] = str(exc)
        print(f"  FAILED  {exc}")

    return result


def run_full_pipeline() -> bool:
    """Run the full QRS pipeline (static + dynamic + comparison) and refresh README.

    Returns True if pipeline succeeded.
    """
    print("\n" + "=" * 60)
    print("  Running full QRS pipeline (static + dynamic, T + TL) ...")
    print("=" * 60)

    rc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_qrs_pipeline.py"),
         "--mode", "full", "--contract", "ALL", "--fast-mode"],
        cwd=str(ROOT),
    ).returncode

    if rc != 0:
        print(f"  Pipeline exited with code {rc}")
        return False

    print("  Pipeline complete. Refreshing README ...")
    refresh_readme()
    return True


def refresh_readme() -> None:
    """Read latest results and inject updated performance table into README.md."""
    from src.utils import format_percent, format_number

    comparison_path = ROOT / "results" / "tables" / "qrs_static_vs_dynamic_comparison.csv"
    if not comparison_path.exists():
        print("  WARNING: comparison CSV not found, skipping README refresh")
        return

    import pandas as pd
    df = pd.read_csv(comparison_path)

    # Build the performance table rows from latest results
    rows_md = ""
    for _, row in df.iterrows():
        contract = row.get("contract", row.get("portfolio", "?"))
        method = row.get("method", "?")
        method_label = {"static_grid": "Static Grid (In-sample)", "dynamic_walk_forward": "Dynamic WF (Out-of-sample)"}.get(method, method)
        ann_ret = row.get("annualized_return", 0)
        sharpe = row.get("sharpe_ratio", 0)
        mdd = row.get("max_drawdown", 0)
        win = row.get("win_rate", 0)
        rows_md += f"| **{contract}** | **{method_label}** | {format_percent(ann_ret)} | {format_number(sharpe)} | {format_percent(mdd)} | {format_percent(win)} |\n"

    # Read current README
    readme_path = ROOT / "README.md"
    text = readme_path.read_text(encoding="utf-8")

    import re

    # Chinese section: find the header line, separator line, and all subsequent data rows
    zh_pat = r"(\| 合约 \(Asset\) \| 模式 \(Method\).*?\n\|[-\s|:*]+\n)((?:\|.*\n)+)"
    zh_replacement_header = "| 合约 (Asset) | 模式 (Method) | 年化收益 (Ann. Ret) | 夏普比率 (Sharpe) | 最大回撤 (MaxDD) | 胜率 (Win Rate) |\n"
    zh_replacement_sep = "| :--- | :--- | :---: | :---: | :---: | :---: |\n"
    zh_updated = False

    if re.search(zh_pat, text):
        text = re.sub(zh_pat, zh_replacement_header + zh_replacement_sep + rows_md + "\n", text, count=1)
        zh_updated = True

    # English section: find the header line, separator line, and all subsequent data rows
    en_pat = r"(\| Asset \| Method \| Ann\. Return.*?\n\|[-\s|:*]+\n)((?:\|.*\n)+)"
    en_replacement_header = "| Asset | Method | Ann. Return | Sharpe Ratio | Max Drawdown | Win Rate |\n"
    en_replacement_sep = "| :--- | :--- | :---: | :---: | :---: | :---: |\n"

    if re.search(en_pat, text):
        text = re.sub(en_pat, en_replacement_header + en_replacement_sep + rows_md + "\n", text, count=1)
    elif not zh_updated:
        print("  WARNING: could not find performance table markers in README")

    # Update the data range line
    cache_info = get_cache_info("T", "5min")
    if cache_info:
        latest_date = cache_info.get("date_end", "?")
        zh_line = f"基于 {cache_info.get('date_start', '?')} 至今的回测数据"
        en_line = f"Based on backtest data from {cache_info.get('date_start', '?')} to {latest_date}"

        # Replace the date range lines
        text = re.sub(
            r"基于 \d{4}-\d{2}-\d{2} 至今的回测数据（`fast-mode` 演示参数）：",
            zh_line + "（`fast-mode` 演示参数）：",
            text,
        )
        text = re.sub(
            r"Based on backtest data since \d{4}-\d{2}-\d{2} \(`fast-mode` parameters\):",
            en_line + " (`fast-mode` parameters):",
            text,
        )

    readme_path.write_text(text, encoding="utf-8")
    print(f"  README refreshed with {len(df)} performance rows")


def main() -> int:
    parser = argparse.ArgumentParser(description="Update Tushare data cache and/or refresh results")
    parser.add_argument("--contract", default="ALL", help="T, TL, or ALL")
    parser.add_argument("--freq", default="ALL", help="5min, daily, or ALL")
    parser.add_argument("--force", action="store_true", help="Re-fetch everything (clear cache first)")
    parser.add_argument("--check", action="store_true", help="Only run health check")
    parser.add_argument("--run-pipeline", action="store_true", help="Re-run full QRS pipeline + refresh README after cache update")
    parser.add_argument("--mode", choices=["cache-only", "full"], default="cache-only",
                        help="'full' = update cache + run full pipeline + refresh README")
    args = parser.parse_args()

    # Health check
    if args.check:
        status = tushare_health()
        print(f"Tushare API: {status}")
        return 0 if status["ok"] else 1

    contracts = ALL_CONTRACTS if args.contract == "ALL" else [args.contract.upper()]
    freqs = ALL_FREQS if args.freq == "ALL" else [args.freq]

    # Run cache updates
    results = []
    for c in contracts:
        for f in freqs:
            results.append(update_one(c, f, force=args.force))

    # Summary
    print(f"\n{'=' * 60}")
    print("  UPDATE SUMMARY")
    print(f"{'=' * 60}")
    all_ok = True
    for r in results:
        icon = "OK" if r["ok"] else "FAIL"
        if r["ok"]:
            print(f"  [{icon}] {r['contract']}/{r['freq']}  {r['rows']:,} rows  (+{r.get('new_rows', 0):,} new)")
        else:
            all_ok = False
            print(f"  [{icon}] {r['contract']}/{r['freq']}  ERROR: {r.get('error', 'unknown')}")

    # Full pipeline: refresh charts + report + README
    if args.run_pipeline or args.mode == "full":
        pipeline_ok = run_full_pipeline()
        if not pipeline_ok:
            all_ok = False

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
