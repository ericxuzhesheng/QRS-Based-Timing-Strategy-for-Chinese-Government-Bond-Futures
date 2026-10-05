"""Frozen-training, four-product QRS research from the shared minute cache."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.minute_research import (
    BASELINE_ID, CANDIDATES, aggregate_minutes, daily_returns, performance,
    select_candidate, simulate, target_positions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Shared cache root with research/*_1min.parquet")
    parser.add_argument("--contract", choices=["ALL", "TS", "TF", "T", "TL"], default="ALL")
    parser.add_argument("--train-end", default="2024-12-31")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "minute_research")
    args = parser.parse_args()
    cutoff = pd.Timestamp(args.train_end)
    outdir = args.output
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "ledgers").mkdir(exist_ok=True)
    all_summary, all_train, selections, quality_rows, daily_rows = [], [], [], [], []
    markets = ["TS", "TF", "T", "TL"] if args.contract == "ALL" else [args.contract]
    provenance = {
        "run_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "data_dir": str(args.data_dir.resolve()), "source": "tushare",
        "training_end": args.train_end, "holdout_start": str(cutoff.date() + pd.Timedelta(days=1)),
        "grid": CANDIDATES, "selection": "per-product highest training daily net Sharpe at 1 bp per side; stable candidate-id tie break",
        "cost_bp_per_side": [0, 1, 3], "execution": "preceding closed 5min signal at next actual first-minute open; flat at final daily close",
        "liquidity": "zero-volume first-minute changes deferred; closing zero-volume exits flagged; no real bid/ask or guaranteed fills",
        "rolls": "causal forward multiplicative factor adjustment at incoming open; original execution prices; flat overnight",
        "inputs": {}, "status": "running",
    }
    for market in markets:
        path = args.data_dir / "research" / f"{market}_1min.parquet"
        raw = pd.read_parquet(path)
        bars, quality = aggregate_minutes(raw, market)
        quality_rows.append({"product": market, **quality})
        provenance["inputs"][market] = {"path": str(path.resolve()), "bytes": path.stat().st_size, **quality}
        train_mask = bars["date"].dt.normalize().le(cutoff)
        if bars.loc[train_mask, "trade_date"].nunique() < 250 or train_mask.all():
            raise ValueError(f"Insufficient training or holdout for {market}")
        candidate_signals, training = {}, []
        for candidate_id, params in enumerate(CANDIDATES):
            signals = target_positions(bars, params)
            candidate_signals[candidate_id] = signals
            ledger = simulate(signals, 1)
            # Preserve the predeclared ranking even if a historical closing
            # fill is unverified. Its status explicitly limits that ranking.
            metrics = performance(ledger.loc[train_mask], apply_liquidity_gate=False)
            training.append({"product": market, "candidate_id": candidate_id, **params, **metrics})
        training_df = pd.DataFrame(training)
        selected = select_candidate(training_df)
        selections.append({"product": market, "candidate_id": selected, **CANDIDATES[selected]})
        all_train.extend(training)
        for cost_bp in [0, 1, 3]:
            ledger = simulate(candidate_signals[selected], cost_bp)
            ledger.to_parquet(outdir / "ledgers" / f"{market}_selected_{cost_bp}bp.parquet", index=False)
            daily = daily_returns(ledger)
            daily_rows.extend({"product": market, "method": "selected", "cost_bp": cost_bp,
                               "trade_date": d, "return": float(r)} for d, r in daily.items())
            for scope, mask in [("training", train_mask), ("holdout", ~train_mask), ("all", np.ones(len(bars), dtype=bool))]:
                all_summary.append({"product": market, "method": "selected", "candidate_id": selected,
                                    "cost_bp": cost_bp, "scope": scope, **performance(ledger.loc[mask])})
        baseline = simulate(candidate_signals[BASELINE_ID], 1)
        baseline.to_parquet(outdir / "ledgers" / f"{market}_baseline_1bp.parquet", index=False)
        for scope, mask in [("training", train_mask), ("holdout", ~train_mask)]:
            all_summary.append({"product": market, "method": "fixed_baseline", "candidate_id": BASELINE_ID,
                                "cost_bp": 1, "scope": scope, **performance(baseline.loc[mask])})
        benchmark_signals = candidate_signals[selected].copy()
        benchmark_signals["target"] = 1.0
        benchmark = simulate(benchmark_signals, 1)
        # Benchmark's first opening trade has no prior signal: it is a known
        # constant daily allocation, evaluated separately from the QRS ledger.
        benchmark.to_parquet(outdir / "ledgers" / f"{market}_long_intraday_1bp.parquet", index=False)
        all_summary.append({"product": market, "method": "long_intraday", "candidate_id": -1,
                            "cost_bp": 1, "scope": "holdout", **performance(benchmark.loc[~train_mask])})
        print(f"{market}: {quality['five_minute_start']} to {quality['five_minute_end']}; selected candidate {selected}", flush=True)
    summary = pd.DataFrame(all_summary)
    summary.to_csv(outdir / "summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_train).to_csv(outdir / "training_grid.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(selections).to_csv(outdir / "selection.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(quality_rows).to_csv(outdir / "data_quality.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(daily_rows).to_csv(outdir / "daily_returns.csv", index=False, encoding="utf-8-sig")
    provenance["status"] = "complete"
    (outdir / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    report = build_report(summary, pd.DataFrame(quality_rows), pd.DataFrame(selections), args.train_end)
    (outdir / "report.md").write_text(report, encoding="utf-8")
    print(summary.loc[summary["scope"].eq("holdout"), ["product", "method", "cost_bp", "annualized_return", "sharpe_ratio", "max_drawdown"]].to_string(index=False), flush=True)


def build_report(summary, quality, selections, train_end):
    rows = summary.loc[summary["scope"].eq("holdout") & summary["cost_bp"].eq(1)].copy()
    for col in ["cumulative_return", "annualized_return", "max_drawdown"]:
        rows[col] = rows[col].map(lambda x: f"{x:.2%}")
    rows["sharpe_ratio"] = rows["sharpe_ratio"].map(lambda x: f"{x:.3f}")
    table = rows[["product", "method", "days", "cumulative_return", "annualized_return", "sharpe_ratio", "max_drawdown", "turnover"]].to_markdown(index=False)
    ranges = quality[["product", "minute_rows", "minute_start", "minute_end", "five_minute_days", "bars_per_day", "excluded_incomplete_dates", "excluded_incomplete_bins"]].to_markdown(index=False)
    blocked = summary.loc[summary["method"].eq("selected") & summary["cost_bp"].eq(1) & summary["zero_volume_close_exits"].gt(0), ["product", "scope", "zero_volume_close_exits"]]
    gates = blocked.to_markdown(index=False) if len(blocked) else "所有摘要区间均无零成交量日末退出。"
    current = summary.loc[summary["scope"].eq("holdout") & summary["method"].eq("selected") & summary["cost_bp"].eq(1)]
    negative = int(current["cumulative_return"].lt(0).sum())
    return f"""# QRS 四品种分钟研究

本次修复 `ma_cross` 和 `price_compare` 使用当日最终收盘的泄漏，并以 Tushare 真实合约一分钟行情重建五分钟研究。原 README 的高夏普结果包含时间泄漏及忽略成本，不能作为有效证据。

单边 1bp 后本次 {len(current)} 个品种中有 {negative} 个冻结策略在后段亏损。候选排序只比较预先声明的六种设置，不证明存在可交易的净优势。

## 数据范围

{ranges}

完整历史一分钟保存在共享缓存，研究仅使用标记完整的交易日期，并只聚合恰好五个真实分钟的五分钟格。历史开盘时段和到期日上午半日由原始时间与质量标记保留。缺口不补价。右侧时间表示五分钟收盘；成交时间使用下一格第一条实际分钟，例如 09:35 信号只能在 09:36 分钟开盘执行。

## 预先固定的选择规则

每个品种使用最早有效历史至 **{train_end}** 的训练数据，从两个因子定义乘三个信号定义共六个候选中按 **单边 1bp 成本后的日收益夏普** 选择一次。随后固定参数评价 2025 年起的后段，后段不参与选参。候选、训练评分及选择结果分别见 [training_grid.csv](training_grid.csv)、[selection.csv](selection.csv) 和 [provenance.json](provenance.json)。固定基线为 N=16、M=600、n=2、S=0.5、5日均线与2日前比较。

{selections.to_markdown(index=False)}

## 后段结果：单边 1bp

{table}

年化收益按每日净值复利计算；夏普按日收益均值/标准差乘 √252，回撤含起始净值 1。每个品种按一倍名义本金单独评价，不以保证金放大。`selected` 是训练选择后的固定策略；`fixed_baseline` 是预设因果基线；`long_intraday` 为同样日内开盘买入、收盘卖出的比较项。

更多数据和成本情景见 [summary.csv](summary.csv)。优化表示修复方法并限定训练选择，不能保证后段收益提高；负收益与不如固定基线的结果原样保留。

## 交易和换月

日趋势仅使用前一完成交易日的信息。因子回归与标准化仅使用当前已知或更早数据；当根收盘信号下一格开盘才能执行。每天最终收盘平仓，日与日之间不持仓、不把跨合约或隔夜价格差记为收益。因子遇到换月只向前按新合约首格开盘缩放，不回调旧历史，成交与盈亏始终使用原价。

开盘第一分钟成交量为零时推迟调仓；日末零成交量的平仓会单独标记。成本为预先固定的 0/1/3bp 单边总摩擦假设，未使用真实 bid/ask、成交回报或交易所手续费逐合约标定，因此 OHLC 回放与正成交量都不能证明可实际成交。[data_quality.csv](data_quality.csv) 保留零成交量和缺口统计，`summary.csv` 保留被推迟的调仓及零成交量日末退出数量。

以下区间存在不能用成交量支持的日末平仓，完整绩效已屏蔽为缺失值。原回放和训练评分保留用于复核，冻结参数保持不变；训练评分属于有未核实成交假设的排序，不能视为可实现收益。未被屏蔽的后段均无零量收盘退出，首分钟零量请求的推迟次数保存在摘要中。

{gates}

公共源交叉核对发现训练日 **2013-11-19 TF1312.CFX**：规则分钟收盘 90.100，Tushare 日线收盘 90.940，差 0.84，OI 亦不一致。其余少量收盘差不超过 0.005，包括后段 2026-07-27 的 TS/T。原始来源保留，不以日线覆盖分钟；训练排序受这些源质量限制。详见共享缓存 `quality/minute_daily_{{product}}.parquet`，不依据后段重新选参。

## 复现与验证

```powershell
python scripts/run_minute_research.py --data-dir "D:/Github Repository/research-validation-20261005/cgb-market-data"
python scripts/verify_minute_research.py
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
python -m pytest tests/test_minute_research.py tests/test_tushare_daily_backfill.py -q -p no:cacheprovider
```

逐格回放账本位于本地 `ledgers/*.parquet`，默认不进入 Git。[verification.json](verification.json) 为独立现金流和时间规则核对的结果。
"""


if __name__ == "__main__":
    main()
