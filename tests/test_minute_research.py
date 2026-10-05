from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.minute_research import aggregate_minutes, causal_roll_prices, performance, select_candidate, simulate, target_positions
from src.qrs_calculator import calculate_qrs_intraday, weighted_low_high_beta_r2
from src.trend_filter import build_daily_trend_filter_from_intraday
from qrs_t_strategy import QRSBacktester


def bars():
    date = pd.to_datetime(["2024-01-02 09:35", "2024-01-02 09:40", "2024-01-02 09:45", "2024-01-03 09:35", "2024-01-03 09:40"])
    return pd.DataFrame({
        "date": date, "execution_time": date - pd.Timedelta(minutes=4),
        "trade_date": date.strftime("%Y%m%d"), "source_contract": ["T2403.CFX"] * 3 + ["T2406.CFX"] * 2,
        "open": [100., 101., 102., 120., 121.], "close": [101., 102., 103., 121., 120.],
        "high": [102., 103., 104., 122., 122.], "low": [99., 100., 101., 119., 119.],
        "first_minute_volume": [1.] * 5, "last_minute_volume": [1.] * 5,
    })


@pytest.mark.parametrize("method", ["ma_compare", "ma_cross", "price_compare"])
@pytest.mark.parametrize("implementation", [build_daily_trend_filter_from_intraday, QRSBacktester.build_daily_trend_filter_from_5m])
def test_daily_trend_does_not_see_later_current_day_close(method, implementation):
    timestamps = pd.DatetimeIndex([d + pd.Timedelta(hours=h) for d in pd.bdate_range("2024-01-01", periods=12) for h in [10, 15]])
    close = pd.Series(np.repeat(np.arange(100., 112.), 2), index=timestamps)
    changed = close.copy()
    changed.iloc[-1] = 1.
    kwargs = dict(trend_method=method, ma_len_days=3, compare_lag_days=1, ma_short=2, ma_long=4)
    original = implementation(close, **kwargs)
    mutant = implementation(changed, **kwargs)
    pd.testing.assert_frame_equal(original, mutant)
    # A prefix ending at the current morning must agree with that morning in
    # the full dataset, despite the final close becoming observable later.
    prefix = implementation(close.iloc[:-1], **kwargs)
    pd.testing.assert_frame_equal(prefix, original.iloc[:-1])


def test_vectorized_regression_matches_local_ols_and_is_prefix_causal():
    rng = np.random.default_rng(2026)
    low = 100 + np.cumsum(rng.normal(0, .01, 160))
    high = low + rng.uniform(.001, .02, len(low))
    frame = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=len(low), freq="5min"),
                          "low": low, "high": high, "close": (high + low) / 2})
    result = calculate_qrs_intraday(frame, N=16, M=20)
    expected = np.array([weighted_low_high_beta_r2(low[i - 16:i], high[i - 16:i]) for i in range(16, len(low))])
    np.testing.assert_allclose(result.loc[16:, ["beta", "r2"]], expected, rtol=1e-9, atol=1e-10)
    prefix = calculate_qrs_intraday(frame.iloc[:110], N=16, M=20)
    pd.testing.assert_frame_equal(prefix, result.iloc[:110])


def test_real_minute_aggregation_keeps_actual_next_open_time_and_half_day():
    times = pd.date_range("2024-01-02 09:31", periods=10, freq="min")
    raw = pd.DataFrame({"datetime": times, "open": 100., "high": 101., "low": 99., "close": 100.,
                        "volume": 1., "open_interest": 10., "product": "T", "source": "tushare",
                        "source_contract": "T2403.CFX", "trade_date": "2024-01-02", "session_complete": True})
    aggregated, _ = aggregate_minutes(raw, "T")
    assert aggregated["date"].tolist() == [pd.Timestamp("2024-01-02 09:35"), pd.Timestamp("2024-01-02 09:40")]
    assert aggregated["execution_time"].tolist() == [pd.Timestamp("2024-01-02 09:31"), pd.Timestamp("2024-01-02 09:36")]
    missing = raw.drop(index=1)
    aggregated, quality = aggregate_minutes(missing, "T")
    assert len(aggregated) == 1
    assert quality["excluded_incomplete_bins"] == 1
    wrong = raw.copy()
    wrong["source"] = "synthetic"
    with pytest.raises(ValueError, match="Tushare"):
        aggregate_minutes(wrong, "T")


def test_next_open_pnl_excludes_overnight_roll_and_includes_two_sided_costs():
    signals = bars()
    signals["target"] = [0, 1, 1, 0, -1]
    ledger = simulate(signals, 1)
    assert ledger.loc[3, "gap_return"] == 0
    assert ledger.loc[3, "return_net"] == 0
    assert ledger["turnover"].sum() == 4
    # Day 1 buys at 101, sells at 103: both one-bp costs at their actual marks.
    actual = (1 + ledger.loc[:2, "return_net"]).prod()
    expected = (1 - .0001) * (103 / 101) * (1 - .0001)
    assert actual == pytest.approx(expected, abs=1e-14)
    # Day 2 short is established at 121 only after its first closed bar.
    assert ledger.loc[4, "return_net"] == pytest.approx((1-.0001)*(1 + 1/121)*(1-.0001)-1)


def test_zero_volume_open_changes_are_deferred_and_close_is_flagged():
    signals = bars().iloc[:3].copy()
    signals["target"] = [0, 1, 1]
    signals.loc[1, "first_minute_volume"] = 0
    signals.loc[2, "last_minute_volume"] = 0
    ledger = simulate(signals, 1)
    assert ledger["position"].tolist() == [0, 0, 1]
    assert ledger["blocked_zero_volume"].tolist() == [False, True, False]
    assert ledger["zero_volume_close_exit"].tolist() == [False, False, True]
    gated = performance(ledger)
    assert gated["performance_status"] == "blocked_zero_volume_exit"
    assert np.isnan(gated["annualized_return"])
    # Ranking receipt can retain the old hypothetical return without presenting
    # it as an executable performance result or changing frozen parameters.
    assert np.isfinite(performance(ledger, apply_liquidity_gate=False)["annualized_return"])


def test_roll_adjustment_never_rewrites_known_history_or_depends_on_future():
    frame = bars()
    adjusted = causal_roll_prices(frame)
    np.testing.assert_array_equal(adjusted.loc[:2, "close"], frame.loc[:2, "close"])
    assert adjusted.loc[3, "open"] == pytest.approx(frame.loc[2, "close"])
    future = frame.copy()
    future.loc[4, ["high", "close"]] = [500., 400.]
    pd.testing.assert_frame_equal(causal_roll_prices(future).iloc[:4], adjusted.iloc[:4])
    # Regression/trend/state machine remain prefix causal as well.
    params = {"N": 2, "M": 2, "n": 2., "S": .1, "trend_method": "price_compare", "ma_len_days": 2, "compare_lag_days": 1}
    pd.testing.assert_frame_equal(target_positions(frame.iloc[:4], params), target_positions(frame, params).iloc[:4])


def test_selection_uses_training_scores_and_deterministic_tie_break():
    training = pd.DataFrame({"candidate_id": [2, 1, 0], "sharpe_ratio": [np.nan, .4, .4]})
    assert select_candidate(training) == 0
