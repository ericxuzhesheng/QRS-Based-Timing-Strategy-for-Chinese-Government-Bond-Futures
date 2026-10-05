"""Independently reconstruct opening/closing cashflows from saved QRS ledgers."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "minute_research")
    args = parser.parse_args()
    receipt, total = [], 0
    summaries = pd.read_csv(args.output / "summary.csv")
    cutoff = json.loads((args.output / "provenance.json").read_text(encoding="utf-8"))["training_end"]
    selections = pd.read_csv(args.output / "selection.csv")
    train = pd.read_csv(args.output / "training_grid.csv")
    for row in selections.itertuples():
        ranked = train.loc[train["product"].eq(row.product)].sort_values(["sharpe_ratio", "candidate_id"], ascending=[False, True])
        assert int(ranked.iloc[0]["candidate_id"]) == row.candidate_id
        assert pd.to_datetime(ranked["end"]).le(pd.Timestamp(cutoff) + pd.Timedelta(days=1)).all()
    for path in sorted((args.output / "ledgers").glob("*_selected_*bp.parquet")):
        market, _, cost_text = path.stem.split("_")
        cost_bp = int(cost_text[:-2])
        cost = cost_bp / 10000
        frame = pd.read_parquet(path)
        days = frame["trade_date"]
        same_day = days.eq(days.shift()).to_numpy()
        previous = frame["position"].shift().fillna(0).to_numpy(copy=True)
        previous[~same_day] = 0
        opening = frame["open"].to_numpy()
        close = frame["close"].to_numpy()
        before = frame["close"].shift().fillna(frame["open"]).to_numpy()
        gap = previous * (opening / before - 1)
        held = frame["position"].to_numpy()
        mark = held * (close / opening - 1)
        change = abs(held - previous)
        last = ~days.eq(days.shift(-1)).to_numpy()
        exit_change = np.where(last, abs(held), 0)
        net = (1 + gap) * (1 - change * cost) * (1 + mark) * (1 - exit_change * cost) - 1
        np.testing.assert_allclose(frame["return_net"], net, rtol=0, atol=1e-13)
        np.testing.assert_array_equal(frame["open_turnover"], change)
        np.testing.assert_array_equal(frame["close_turnover"], exit_change)
        assert (frame.loc[~same_day, "position"] == 0).all()
        assert (frame.loc[same_day, "source_contract"].to_numpy() == frame["source_contract"].shift().loc[same_day].to_numpy()).all()
        assert (frame.loc[same_day, "signal_time"] < frame.loc[same_day, "execution_time"]).all()
        assert (frame.loc[frame["first_minute_volume"].le(0), "open_turnover"] == 0).all()
        for scope, scope_mask in [("training", frame["date"].dt.normalize().le(pd.Timestamp(cutoff))),
                                  ("all", np.ones(len(frame), dtype=bool))]:
            if frame.loc[scope_mask, "zero_volume_close_exit"].any():
                blocked = summaries.loc[summaries["product"].eq(market) & summaries["method"].eq("selected") &
                                        summaries["scope"].eq(scope) & summaries["cost_bp"].eq(cost_bp)]
                assert blocked["performance_status"].eq("blocked_zero_volume_exit").all()
                assert blocked[["cumulative_return", "annualized_return", "sharpe_ratio", "max_drawdown"]].isna().all().all()
        daily = pd.Series(1 + net).groupby(days).prod() - 1
        mask = pd.to_datetime(daily.index.astype(str), format="%Y%m%d") > pd.Timestamp(cutoff)
        actual = (1 + daily.loc[mask]).prod() - 1
        recorded = summaries.loc[summaries["product"].eq(market) & summaries["method"].eq("selected") &
                                 summaries["scope"].eq("holdout") & summaries["cost_bp"].eq(cost_bp), "cumulative_return"].iloc[0]
        np.testing.assert_allclose(actual, recorded, rtol=0, atol=1e-12)
        total += len(frame)
        receipt.append({"product": market, "cost_bp": cost_bp, "bars": len(frame), "holdout_cumulative_return": actual})
    assert len(receipt) == len(selections) * 3
    result = {"status": "passed", "verified_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "bars_reconstructed": total, "checked": ["training-only selection", "signal precedes next-open execution",
              "daily-flat execution", "no intraday contract transitions", "zero-volume opening deferral", "zero-volume exit performance gate", "independent net cashflows", "saved holdout cumulative returns"],
              "ledgers": receipt}
    (args.output / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
