from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest import (
    intraday_debug_stats,
    intraday_performance_summary,
    performance_summary,
    run_backtest,
    run_intraday_backtest,
)
from src.data_loader import aggregate_to_daily, discover_input_file, load_market_data
from src.grid_search import (
    run_grid_search, 
    FAST_QRS_GRID, FAST_SIGNAL_GRID, 
    FULL_QRS_GRID, FULL_SIGNAL_GRID
)
from src.dynamic_selection import run_dynamic_qrs_selection
from src.qrs_calculator import calculate_qrs, calculate_qrs_intraday
from src.signal_generator import generate_qrs_signals, generate_qrs_intraday_signals
from src.utils import ensure_dir, format_number, format_percent, markdown_table, project_path
from src.visualization import (
    plot_best_strategy_position,
    plot_drawdown,
    plot_nav_comparison,
    plot_qrs_future_return,
    plot_qrs_price_overlay,
    plot_static_vs_dynamic_nav,
    plot_dynamic_param_timeline,
    plot_dynamic_selection_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QRS timing pipeline for Chinese government bond futures.")
    parser.add_argument("--mode", choices=["static", "dynamic", "full", "daily_baseline"], default="static")
    parser.add_argument("--input", default=None, help="CSV/XLSX input file. If omitted, auto-discover a file.")
    parser.add_argument("--sheet-name", default=None, help="Excel sheet name or index. Defaults to the first sheet.")
    parser.add_argument("--contract", default="T", help="Contract label, e.g. T, TL, ALL.")
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--train-window-days", type=int, default=60)
    parser.add_argument("--test-window-days", type=int, default=10)
    parser.add_argument("--rebalance-days", type=int, default=1)
    parser.add_argument("--horizon-bars", type=int, default=54)
    parser.add_argument("--selection-metric", default="rank_ic")
    parser.add_argument("--fast-mode", action="store_true")
    parser.add_argument("--full-grid", action="store_true")
    parser.add_argument("--allow-short", dest="allow_short", action="store_true")
    parser.add_argument("--no-allow-short", dest="allow_short", action="store_false")
    parser.set_defaults(allow_short=True)
    
    # Legacy / Manual Override Args
    parser.add_argument("--N", type=int, default=16)
    parser.add_argument("--M", type=int, default=600)
    parser.add_argument("--rolling-window", type=int, default=18)
    parser.add_argument("--zscore-window", type=int, default=120)
    parser.add_argument("--r2-power", type=float, default=2.0)
    parser.add_argument("--S", type=float, default=0.5)
    parser.add_argument("--slope-window", type=int, default=20)
    parser.add_argument("--percentile-window", type=int, default=120)
    parser.add_argument("--long-threshold", type=float, default=0.7)
    parser.add_argument("--exit-threshold", type=float, default=-0.7)
    parser.add_argument("--trend-method", default="ma_compare", choices=["ma_compare", "ma_cross", "price_compare"])
    parser.add_argument("--ma-len-days", type=int, default=5)
    parser.add_argument("--compare-lag-days", type=int, default=2)
    parser.add_argument("--ma-short", type=int, default=5)
    parser.add_argument("--ma-long", type=int, default=20)
    parser.add_argument("--periods-per-year", type=int, default=252 * 54)
    return parser.parse_args()


def _sheet_value(value: str | None) -> str | int | None:
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip()
    return int(text) if text.isdigit() else text


def _resolve_input(path: str | None, contract: str) -> Path:
    input_path = Path(path) if path else discover_input_file(contract)
    if not input_path.is_absolute():
        input_path = (ROOT / input_path).resolve()
    return input_path


def _vertical_metric_table(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        ("Cumulative Return", "cumulative_return", True),
        ("Annualized Return", "annualized_return", True),
        ("Annualized Volatility", "annualized_volatility", True),
        ("Sharpe Ratio", "sharpe_ratio", False),
        ("Max Drawdown", "max_drawdown", True),
        ("Calmar Ratio", "calmar_ratio", False),
        ("Win Rate", "win_rate", True),
        ("Turnover", "turnover", False),
    ]
    rows = []
    for label, col, is_pct in metrics:
        row: dict[str, str] = {"Metric": label}
        for _, item in summary.iterrows():
            value = float(item[col])
            row[str(item["portfolio"])] = format_percent(value) if is_pct else format_number(value)
        rows.append(row)
    return pd.DataFrame(rows)


def _debug_table(debug: pd.DataFrame) -> pd.DataFrame:
    row = debug.iloc[0].to_dict()
    rows = []
    for key, value in row.items():
        if isinstance(value, float):
            display = format_percent(value) if key.endswith("ratio") or "return" in key or "drawdown" in key else format_number(value)
        else:
            display = str(value)
        rows.append({"Metric": key, "Value": display})
    return pd.DataFrame(rows)


def build_report(
    summary: pd.DataFrame,
    metadata: dict[str, Any],
    debug: pd.DataFrame | None = None,
    best_params: dict[str, Any] | None = None,
) -> str:
    metric_table = markdown_table(_vertical_metric_table(summary))
    debug_table = markdown_table(_debug_table(debug)) if debug is not None and not debug.empty else ""
    best_params_text = json.dumps(best_params or {}, ensure_ascii=False, indent=2)

    return r'''# 基于 QRS 的中国国债期货择时研究报告 | QRS-Based Timing Strategy Report for Chinese Government Bond Futures

<p align="center">
<a href="#zh"><img src="https://img.shields.io/badge/LANGUAGE-%E4%B8%AD%E6%96%87-E84D3D?style=for-the-badge&labelColor=3B3F47" alt="LANGUAGE 中文"></a>
<a href="#en"><img src="https://img.shields.io/badge/LANGUAGE-ENGLISH-2F73C9?style=for-the-badge&labelColor=3B3F47" alt="LANGUAGE ENGLISH"></a>
</p>

<a id="zh"></a>

## 简体中文

当前语言：中文 | [Switch to English](#en)

---

### 1. 项目概述

本报告展示基于 QRS 指标的中国国债期货择时研究流程。
- **理论来源**：本项目核心思路参考自中金公司（CICC）量化研究报告 **《金融工程视角下的技术择时艺术》**。
- **运行环境**：当前运行模式：`__MODE__`；当前运行合约：`__CONTRACT__`；输入数据：`__INPUT_FILE__`；样本区间：`__START_DATE__ 至 __END_DATE__`。


### 2. 核心模型逻辑

#### 2.1 QRS 因子构建
QRS 类指标通过价格区间中的阻力与支撑关系刻画趋势质量. 在 5 分钟 OHLC 上执行局部回归：

$$
high_t = \alpha + \beta \cdot low_t + \varepsilon_t
$$

对斜率 $\beta$ 进行滚动 Z-score 标准化，并引入 $R^2$ 作为惩罚项：

$$
qrs = zscore(\beta, M) \times (R^2)^n
$$

#### 2.2 信号设计与趋势过滤
状态机规则：
- **做多**：$QRS > +S$ 且日频趋势向上, 仓位设为 1；
- **做空**：$QRS < -S$ 且日频趋势向下, 仓位设为 -1（若允许做空）；
- **维持**：维持上一根 5 分钟 Bar 的原始仓位。

$$
Pos_t =
\begin{cases}
1, & QRS_t > S \text{ and } Trend_{up,t} = True \\
-1, & QRS_t < -S \text{ and } Trend_{down,t} = True \\
Pos_{t-1}, & \text{otherwise}
\end{cases}
$$

### 3. 参数搜索空间与最佳参数

默认搜索空间包含 $S$ 阈值、趋势判断方式及均线周期等.

本次回测采用的最佳参数：
```json
__BEST_PARAMS__
```

### 4. 回测结果汇总

__METRIC_TABLE__

### 5. Debug 辅助指标

__DEBUG_TABLE__

### 6. 可视化图表

#### 6.1 因子与价格叠加
![QRS Price Overlay](figures/qrs_price_overlay.png)

#### 6.2 策略净值对比
![Strategy NAV](figures/nav_comparison.png)

#### 6.3 策略回撤
![Drawdown](figures/drawdown.png)

#### 6.4 最佳参数持仓分布
![Best Strategy Position](figures/best_strategy_position.png)

#### 6.5 因子择时能力 (Timing Coefficient)
![QRS Future Return](figures/qrs_future_return.png)

### 7. 免责声明
本报告仅用于量化研究与代码示例，不构成任何投资建议或收益承诺.

---

<a id="en"></a>

## English

Current language: English | [切换到中文](#zh)

---

### 1. Project Overview

This report presents a QRS-based timing workflow for Chinese government bond futures.
- **Source**: The core logic is inspired by the CICC quantitative research report ***The Art of Technical Timing from a Financial Engineering Perspective***.
- **Environment**: Current mode: `__MODE__`; contract: `__CONTRACT__`; input data: `__INPUT_FILE__`; sample period: `__START_DATE__ to __END_DATE__`.
### 2. Core Model Logic

#### 2.1 QRS Factor Construction
QRS indicators describe trend quality through resistance-support relationships. A local regression is performed on 5-minute OHLC bars:

$$
high_t = \alpha + \beta \cdot low_t + \varepsilon_t, \quad t \in \{1,2,\ldots,N\}
$$

The slope $\beta$ is standardized via rolling Z-score and adjusted by an $R^2$ penalty term:

$$
qrs = zscore(\beta, M) \times (R^2)^n
$$

#### 2.2 Signal Design and Trend Filter
The state machine rules:
- **Long**: $QRS > +S$ and daily trend is UP, position becomes 1;
- **Short**: $QRS < -S$ and daily trend is DOWN, position becomes -1 (if allowed);
- **Otherwise**: Carry the previous raw position.

### 3. Parameter Search Space & Best Parameters

The default search space covers $S$ thresholds, trend methods, and MA periods.

Parameters used in this run:
```json
__BEST_PARAMS__
```

### 4. Backtest Results

__METRIC_TABLE__

### 5. Debug Metrics

__DEBUG_TABLE__

### 6. Visualization

#### 6.1 Factor & Price Overlay
![QRS Price Overlay](figures/qrs_price_overlay.png)

#### 6.2 Strategy NAV Comparison
![Strategy NAV](figures/nav_comparison.png)

#### 6.3 Strategy Drawdown
![Drawdown](figures/drawdown.png)

#### 6.4 Best Strategy Position
![Best Strategy Position](figures/best_strategy_position.png)

#### 6.5 Timing Coefficient (QRS Future Return)
![QRS Future Return](figures/qrs_future_return.png)

### 7. Disclaimer
This report is for quantitative research and demonstration only. It does not constitute investment advice or any return guarantee.
'''.replace("__MODE__", str(metadata["mode"])) \
   .replace("__CONTRACT__", str(metadata["contract"])) \
   .replace("__INPUT_FILE__", str(metadata["input_file"])) \
   .replace("__START_DATE__", str(metadata["start_date"])) \
   .replace("__END_DATE__", str(metadata["end_date"])) \
   .replace("__BEST_PARAMS__", best_params_text) \
   .replace("__METRIC_TABLE__", metric_table) \
   .replace("__DEBUG_TABLE__", debug_table)


def _save_common_outputs(bt: pd.DataFrame, summary: pd.DataFrame, metadata: dict[str, Any], debug: pd.DataFrame | None, best_params: dict[str, Any] | None, suffix: str = "") -> None:
    figures_dir = project_path("results", "figures")
    tables_dir = project_path("results", "tables")
    ensure_dir(figures_dir)
    ensure_dir(tables_dir)

    summary.to_csv(tables_dir / f"backtest_summary{suffix}.csv", index=False, encoding="utf-8-sig")
    nav_cols = [c for c in ["date", "ret_strategy", "ret_benchmark", "ret_excess", "nav_strategy", "nav_benchmark", "nav_excess", "position", "turnover"] if c in bt.columns]
    if not nav_cols:
        nav_cols = ["date", "strategy_return", "benchmark_return", "strategy_nav", "benchmark_nav", "position", "turnover"]
    bt[nav_cols].to_csv(tables_dir / f"strategy_nav{suffix}.csv", index=False, encoding="utf-8-sig")

    plot_qrs_price_overlay(bt, figures_dir / f"qrs_price_overlay{suffix}.png")
    plot_nav_comparison(bt, figures_dir / f"nav_comparison{suffix}.png", title=f"Strategy NAV vs Benchmark {suffix}")
    plot_drawdown(bt, figures_dir / f"drawdown{suffix}.png")
    plot_qrs_future_return(bt, figures_dir / f"qrs_future_return{suffix}.png")
    if "position" in bt.columns:
        plot_best_strategy_position(bt, figures_dir / f"best_strategy_position{suffix}.png")

    if suffix == "" or suffix == "_T" or suffix == "_TL":
         project_path("results", "report.md").write_text(build_report(summary, metadata, debug, best_params), encoding="utf-8")


def _get_int(params: dict[str, Any], key: str, default: int) -> int:
    val = params.get(key)
    if val is None or pd.isna(val):
        return default
    return int(val)


def run_static(args: argparse.Namespace, raw: pd.DataFrame, contract: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"--- Running Static Grid Search for {contract} ---")
    qrs_all = calculate_qrs_intraday(raw, N=args.N, M=args.M, n=args.r2_power)
    qrs_eval = qrs_all.loc[pd.to_datetime(qrs_all["date"]) >= pd.to_datetime(args.start_date)].copy().reset_index(drop=True)
    
    tables_dir = project_path("results", "tables")
    ensure_dir(tables_dir)
    
    grid_results, best_params = run_grid_search(qrs_eval, allow_short=args.allow_short, periods_per_year=args.periods_per_year)
    # Convert best_params to JSON-friendly format
    best_params_json = {k: (None if pd.isna(v) else v) for k, v in best_params.items()}
    grid_results.to_csv(tables_dir / f"static_grid_results_{contract}.csv", index=False, encoding="utf-8-sig")
    (tables_dir / f"best_params_static_{contract}.json").write_text(json.dumps(best_params_json, ensure_ascii=False, indent=2), encoding="utf-8")

    signal_df = generate_qrs_intraday_signals(
        qrs_eval,
        S=float(best_params["S"]),
        trend_method=str(best_params["trend_method"]),
        ma_len_days=_get_int(best_params, "ma_len_days", args.ma_len_days),
        compare_lag_days=_get_int(best_params, "compare_lag_days", args.compare_lag_days),
        ma_short=_get_int(best_params, "ma_short", args.ma_short),
        ma_long=_get_int(best_params, "ma_long", args.ma_long),
        allow_short=args.allow_short,
    )
    bt = run_intraday_backtest(signal_df, periods_per_year=args.periods_per_year)
    summary = intraday_performance_summary(bt, periods_per_year=args.periods_per_year)
    debug = intraday_debug_stats(bt)
    
    metadata = {
        "mode": "static",
        "contract": contract,
        "input_file": "auto",
        "start_date": pd.to_datetime(bt["date"].iloc[0]).strftime("%Y-%m-%d %H:%M:%S"),
        "end_date": pd.to_datetime(bt["date"].iloc[-1]).strftime("%Y-%m-%d %H:%M:%S"),
        "data_frequency": "5-minute intraday",
        "allow_short": str(args.allow_short),
        "periods_per_year": str(args.periods_per_year),
        "N": str(args.N),
        "M": str(args.M),
        "r2_power": str(args.r2_power),
        "trend_method": str(best_params["trend_method"]),
    }
    _save_common_outputs(bt, summary, metadata, debug, best_params, suffix=f"_{contract}")
    summary.to_csv(tables_dir / f"static_summary_{contract}.csv", index=False)
    return bt, summary


def run_dynamic(args: argparse.Namespace, raw: pd.DataFrame, contract: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"--- Running Dynamic Walk-Forward for {contract} ---")
    qrs_grid = FAST_QRS_GRID if args.fast_mode else FULL_QRS_GRID
    sig_grid = FAST_SIGNAL_GRID if args.fast_mode else FULL_SIGNAL_GRID
    
    bt, param_trace = run_dynamic_qrs_selection(
        raw, 
        qrs_param_grid=qrs_grid,
        signal_param_grid=sig_grid,
        train_window_bars=args.train_window_days * 54,
        rebalance_every_bars=args.rebalance_days * 54,
        horizon_bars=args.horizon_bars,
        selection_metric=args.selection_metric,
        allow_short=args.allow_short,
        periods_per_year=args.periods_per_year
    )
    
    if bt.empty:
        print(f"Dynamic selection returned empty result for {contract}")
        return pd.DataFrame(), pd.DataFrame()
        
    summary = intraday_performance_summary(bt, periods_per_year=args.periods_per_year)
    debug = intraday_debug_stats(bt)
    
    tables_dir = project_path("results", "tables")
    figures_dir = project_path("results", "figures")
    ensure_dir(tables_dir)
    ensure_dir(figures_dir)
    
    bt.to_csv(tables_dir / f"dynamic_strategy_nav_{contract}.csv", index=False, encoding="utf-8-sig")
    param_trace.to_csv(tables_dir / f"dynamic_params_{contract}.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(tables_dir / f"dynamic_summary_{contract}.csv", index=False)
    
    plot_nav_comparison(bt, figures_dir / f"dynamic_nav_comparison_{contract}.png", title=f"Dynamic Strategy NAV vs Benchmark {contract}")
    plot_dynamic_param_timeline(param_trace, figures_dir / f"dynamic_param_timeline_{contract}.png")
    plot_dynamic_selection_score(param_trace, figures_dir / f"dynamic_selection_score_{contract}.png")
    
    metadata = {
        "mode": "dynamic",
        "contract": contract,
        "input_file": "auto",
        "start_date": pd.to_datetime(bt["date"].iloc[0]).strftime("%Y-%m-%d %H:%M:%S"),
        "end_date": pd.to_datetime(bt["date"].iloc[-1]).strftime("%Y-%m-%d %H:%M:%S"),
        "data_frequency": "5-minute intraday",
        "allow_short": str(args.allow_short),
        "periods_per_year": str(args.periods_per_year),
    }
    _save_common_outputs(bt, summary, metadata, debug, {"mode": "dynamic"}, suffix=f"_dynamic_{contract}")
    return bt, summary


def run_pipeline(args: argparse.Namespace) -> None:
    contracts = ["T", "TL"] if args.contract == "ALL" else [args.contract]
    all_results = []
    
    for contract in contracts:
        try:
            input_path = _resolve_input(args.input, contract)
            raw = load_market_data(input_path, sheet_name=_sheet_value(args.sheet_name))
        except Exception as e:
            print(f"Error loading data for {contract}: {e}")
            continue
            
        static_bt = None
        dynamic_bt = None
        
        if args.mode in ["static", "full"]:
            static_bt, static_summary = run_static(args, raw, contract)
            s_res = static_summary.loc[static_summary["portfolio"] == "QRS Strategy"].iloc[0].to_dict()
            s_res.update({"contract": contract, "method": "static_grid"})
            all_results.append(s_res)
            
        if args.mode in ["dynamic", "full"]:
            dynamic_bt, dynamic_summary = run_dynamic(args, raw, contract)
            if not dynamic_summary.empty:
                d_res = dynamic_summary.loc[dynamic_summary["portfolio"] == "QRS Strategy"].iloc[0].to_dict()
                d_res.update({"contract": contract, "method": "dynamic_walk_forward"})
                all_results.append(d_res)
                
        if args.mode == "full" and static_bt is not None and dynamic_bt is not None:
            figures_dir = project_path("results", "figures")
            plot_static_vs_dynamic_nav(static_bt, dynamic_bt, figures_dir / f"static_vs_dynamic_nav_{contract}.png", title=f"Static vs Dynamic QRS NAV ({contract})")
            
        if args.mode == "daily_baseline":
            run_daily_baseline(args, raw, input_path)

    if all_results:
        res_df = pd.DataFrame(all_results)
        res_df.to_csv(project_path("results", "tables", "qrs_static_vs_dynamic_comparison.csv"), index=False)
        print("\n=== Comparison Summary ===")
        print(res_df[["contract", "method", "annualized_return", "sharpe_ratio", "max_drawdown"]].to_string(index=False))


if __name__ == "__main__":
    run_pipeline(parse_args())
