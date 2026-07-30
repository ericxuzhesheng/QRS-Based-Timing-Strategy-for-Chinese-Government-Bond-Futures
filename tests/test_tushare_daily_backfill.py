from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pandas as pd

from src.cache_manager import incremental_update
from src.tushare_fetcher import fetch_fut_daily


def _raw_daily(code: str, dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": [code] * len(dates),
            "trade_date": dates,
            "open": [100.0] * len(dates),
            "high": [101.0] * len(dates),
            "low": [99.0] * len(dates),
            "close": [100.5] * len(dates),
            "vol": [10.0] * len(dates),
            "oi": [20.0] * len(dates),
        }
    )


class TushareDailyBackfillTests(unittest.TestCase):
    @patch("src.tushare_fetcher._call_with_retry")
    @patch("src.tushare_fetcher.get_contract_timeline")
    @patch("src.tushare_fetcher._get_pro")
    def test_tl_daily_is_stitched_from_mapped_contracts(
        self,
        get_pro: Mock,
        get_timeline: Mock,
        call_with_retry: Mock,
    ) -> None:
        get_pro.return_value = Mock(fut_daily=Mock())
        get_timeline.return_value = pd.DataFrame(
            {
                "trade_date": ["20240102", "20240311"],
                "mapping_ts_code": ["TL2403.CFX", "TL2406.CFX"],
            }
        )
        call_with_retry.side_effect = [
            _raw_daily("TL2403.CFX", ["20240102", "20240311"]),
            _raw_daily("TL2406.CFX", ["20240311", "20240312"]),
        ]

        result = fetch_fut_daily("TL", "20240101", "20240312")

        self.assertEqual(
            result["date"].dt.strftime("%Y%m%d").tolist(),
            ["20240102", "20240311", "20240312"],
        )
        self.assertEqual(call_with_retry.call_count, 2)
        self.assertEqual(
            call_with_retry.call_args_list[0].kwargs["ts_code"],
            "TL2403.CFX",
        )

    @patch("src.cache_manager.write_cache")
    @patch("src.cache_manager.get_last_cached_date")
    @patch("src.cache_manager.read_cache")
    def test_incremental_update_repairs_large_historical_gap(
        self,
        read_cache: Mock,
        get_last_cached_date: Mock,
        write_cache: Mock,
    ) -> None:
        today_dt = datetime.now()
        first_cached_dt = today_dt - timedelta(days=10)
        today = today_dt.strftime("%Y-%m-%d")
        read_cache.return_value = pd.DataFrame(
            {
                "date": pd.to_datetime([first_cached_dt.strftime("%Y-%m-%d"), today]),
                "close": [1.0, 2.0],
            }
        )
        get_last_cached_date.return_value = today_dt.strftime("%Y%m%d")
        fetch = Mock(
            return_value=pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-02", "2026-07-19"]),
                    "close": [0.5, 0.9],
                }
            )
        )

        result = incremental_update("TL", "daily", fetch, start_date="20240101")

        fetch.assert_called_once_with(
            "20240101",
            (first_cached_dt - timedelta(days=1)).strftime("%Y%m%d"),
        )
        self.assertEqual(result["date"].min(), pd.Timestamp("2024-01-02"))
        self.assertEqual(len(result), 4)
        write_cache.assert_called_once()


if __name__ == "__main__":
    unittest.main()
