#!/usr/bin/env python
"""Daily data update script — fetch from Tushare, merge into cache.

Intended for both GitHub Actions cron jobs and local manual use.

Usage:
    python scripts/update_data.py                          # Update T+TL, 5min+daily
    python scripts/update_data.py --contract T             # Single contract
    python scripts/update_data.py --freq 5min              # Single frequency
    python scripts/update_data.py --force                  # Re-fetch everything
    python scripts/update_data.py --check                  # Health check only
    python scripts/update_data.py --run-pipeline           # Also re-run full pipeline

GitHub Actions:
    python scripts/update_data.py --contract ALL --mode cache-only
"""

from __future__ import annotations

import argparse
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
    """Update cache for a single contract+frequency pair.

    Returns a result summary dict.
    """
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
        print(f"  ✓ OK  rows={len(df):,}  new={new_rows:,}  {result['start']} → {result['end']}")

    except Exception as exc:
        result["error"] = str(exc)
        print(f"  ✗ FAILED  {exc}")

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Update Tushare data cache")
    parser.add_argument("--contract", default="ALL", help="T, TL, or ALL")
    parser.add_argument("--freq", default="ALL", help="5min, daily, or ALL")
    parser.add_argument("--force", action="store_true", help="Re-fetch everything (clear cache first)")
    parser.add_argument("--check", action="store_true", help="Only run health check")
    parser.add_argument("--run-pipeline", action="store_true", help="Re-run QRS pipeline after update")
    parser.add_argument("--mode", choices=["cache-only", "full"], default="cache-only",
                        help="cache-only: update cache files only / full: also regenerate processed data")
    args = parser.parse_args()

    # ── Health check ──
    if args.check:
        status = tushare_health()
        print(f"Tushare API: {status}")
        return 0 if status["ok"] else 1

    contracts = ALL_CONTRACTS if args.contract == "ALL" else [args.contract.upper()]
    freqs = ALL_FREQS if args.freq == "ALL" else [args.freq]

    # ── Run updates ──
    results = []
    for c in contracts:
        for f in freqs:
            results.append(update_one(c, f, force=args.force))

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("  UPDATE SUMMARY")
    print(f"{'=' * 60}")
    all_ok = True
    for r in results:
        icon = "✓" if r["ok"] else "✗"
        if r["ok"]:
            print(f"  {icon} {r['contract']}/{r['freq']}  {r['rows']:,} rows  (+{r.get('new_rows', 0):,} new)")
        else:
            all_ok = False
            print(f"  {icon} {r['contract']}/{r['freq']}  ERROR: {r.get('error', 'unknown')}")

    # ── Optional pipeline run ──
    if args.run_pipeline:
        print("\nRunning QRS pipeline with updated data…")
        import subprocess
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_qrs_pipeline.py"),
             "--mode", "static", "--contract", "ALL", "--fast-mode"],
            cwd=str(ROOT),
        )

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
