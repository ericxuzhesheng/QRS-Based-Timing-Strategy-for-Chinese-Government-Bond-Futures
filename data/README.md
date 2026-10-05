# Data

The current four-product research reads `research/{TS,TF,T,TL}_1min.parquet` from the shared Tushare cache passed to `scripts/run_minute_research.py --data-dir`. Its schema is `datetime,open,high,low,close,volume,open_interest,product,source_contract,source,trade_date,roll_flag,session_complete`. `source` must be `tushare`; contracts are concrete CFFEX contracts selected using the preceding exchange date. The downloader retains source snapshots separately, supplies historical session quality flags and does not fill missing prices. Only complete dates and five-real-minute bins enter the study. Derived execution/cashflow ledgers are kept locally under `results/minute_research/ledgers/` and ignored by Git.

- `data/raw/`: place local raw CSV/Excel files here. Raw data is ignored by git to avoid committing large or proprietary files.
- `data/processed/qrs_daily.csv`: generated daily QRS, signal, and backtest data from `scripts/run_qrs_pipeline.py`.

The loader supports CSV and Excel files with common English/Chinese market-data fields such as `date`, `time`, `open`, `high`, `low`, `close`, `volume`, `open_interest`, `日期`, `开盘价`, `最高价`, `最低价`, `收盘价`, `成交量`, and `持仓量`.
