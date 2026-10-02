"""
Relative Value Engine for MCX Gold Futures Contracts.
Computes pairwise normalized spreads, rolling statistics, and no-lookahead Z-score signals.
"""

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

from src.normalization import add_normalized_columns, CONTRACT_SPECS


@dataclass
class SpreadPoint:
    trade_date: date
    symbol_a: str
    expiry_a: date
    norm_price_a: float
    raw_price_a: float
    volume_a: int
    oi_a: int
    
    symbol_b: str
    expiry_b: date
    norm_price_b: float
    raw_price_b: float
    volume_b: int
    oi_b: int

    spread_inr_per_g: float     # P_a - P_b in ₹/g fine gold
    spread_pct: float           # ((P_a - P_b) / P_b) * 100
    rolling_mean: Optional[float]
    rolling_std: Optional[float]
    z_score: Optional[float]
    signal: str                 # "LONG_SPREAD", "SHORT_SPREAD", "FLAT"


def compute_cross_sectional_spreads(
    cleaned_df: pd.DataFrame,
    matching_mode: str = "exact_expiry"  # "exact_expiry", "nearest_cycle", or "all_pairs"
) -> pd.DataFrame:
    """
    Compute pairwise relative value spreads across all gold contracts for available trade dates.
    - 'exact_expiry': Only pair contracts with identical expiry_date (e.g. GOLDTEN vs GOLDGUINEA vs GOLDPETAL).
    - 'nearest_cycle': Pairs contracts with matching maturity order (M1 vs M1, M2 vs M2, etc.).
    - 'all_pairs': All cross-contract combinations.
    """
    if cleaned_df.empty:
        return pd.DataFrame()

    df = add_normalized_columns(cleaned_df)
    spread_records = []

    # Group by trade date
    for trade_date, date_group in df.groupby("trade_date"):
        # Assign maturity rank per symbol
        date_grp = date_group.copy()
        date_grp["maturity_rank"] = date_grp.groupby("symbol")["expiry_date"].rank(method="dense").astype(int)

        if matching_mode == "exact_expiry":
            expiry_groups = [(str(exp), grp) for exp, grp in date_grp.groupby("expiry_date")]
        elif matching_mode == "nearest_cycle":
            expiry_groups = [(f"M{rank}", grp) for rank, grp in date_grp.groupby("maturity_rank")]
        else: # all_pairs
            expiry_groups = [("ALL", date_grp)]

        for group_label, grp in expiry_groups:
            contracts = grp.to_dict("records")
            n = len(contracts)
            for i in range(n):
                for j in range(i + 1, n):
                    c_a = contracts[i]
                    c_b = contracts[j]

                    if c_a["symbol"] == c_b["symbol"] and c_a["expiry_date"] == c_b["expiry_date"]:
                        continue

                    p_a = c_a["norm_close"]
                    p_b = c_b["norm_close"]

                    if pd.isna(p_a) or pd.isna(p_b) or p_b <= 0:
                        continue

                    spread_inr = p_a - p_b
                    spread_pct = (spread_inr / p_b) * 100.0

                    pair_name = f"{c_a['symbol']} vs {c_b['symbol']}"

                    spread_records.append({
                        "trade_date": trade_date,
                        "cycle_group": group_label,
                        "pair": pair_name,
                        "symbol_a": c_a["symbol"],
                        "expiry_a": c_a["expiry_date"],
                        "norm_close_a": p_a,
                        "raw_close_a": c_a["close"],
                        "volume_a": c_a["volume_lots"],
                        "oi_a": c_a["open_interest_lots"],
                        "symbol_b": c_b["symbol"],
                        "expiry_b": c_b["expiry_date"],
                        "norm_close_b": p_b,
                        "raw_close_b": c_b["close"],
                        "volume_b": c_b["volume_lots"],
                        "oi_b": c_b["open_interest_lots"],
                        "spread_inr_g": spread_inr,
                        "spread_pct": spread_pct,
                        "expiry_matched": c_a["expiry_date"] == c_b["expiry_date"],
                        "days_to_expiry_diff": abs((c_a["expiry_date"] - c_b["expiry_date"]).days) if pd.notna(c_a["expiry_date"]) and pd.notna(c_b["expiry_date"]) else 0,
                    })

    if not spread_records:
        return pd.DataFrame()

    res_df = pd.DataFrame(spread_records)
    res_df.sort_values(by=["pair", "expiry_a", "trade_date"], inplace=True)
    res_df.reset_index(drop=True, inplace=True)
    return res_df


def calculate_rolling_zscores(
    spread_series: pd.DataFrame,
    lookback_window: int = 10,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    stop_threshold: float = 3.5
) -> pd.DataFrame:
    """
    Calculate strictly no-lookahead rolling mean, std, and Z-score for each contract pair.
    
    Z-score formula:
        Z_t = (Spread_t - Mean_{t-1..t-w}) / Std_{t-1..t-w}
    or expanding if history length < lookback_window.
    """
    if spread_series.empty:
        return spread_series

    df = spread_series.copy()
    df["rolling_mean"] = np.nan
    df["rolling_std"] = np.nan
    df["z_score"] = np.nan
    df["signal"] = "NO_DATA"

    # Process each pair and expiry slice independently
    grouped = df.groupby(["pair", "expiry_a", "expiry_b"])
    
    result_chunks = []
    for (pair, exp_a, exp_b), group in grouped:
        grp = group.sort_values("trade_date").copy()
        n = len(grp)
        spreads = grp["spread_inr_g"].values
        means = np.full(n, np.nan)
        stds = np.full(n, np.nan)
        z_scores = np.full(n, np.nan)
        signals = ["INSUFFICIENT_HISTORY"] * n

        for i in range(n):
            # Baseline is formed strictly from PAST data prior to index i
            # when i >= 2, we use past observations [start_idx : i]
            if i >= 2:
                start_idx = max(0, i - lookback_window)
                past_window = spreads[start_idx : i]
                m = float(np.mean(past_window))
                s = float(np.std(past_window, ddof=1)) if len(past_window) > 1 else float(np.std(past_window))
                means[i] = m
                stds[i] = s

                if s > 1e-8:
                    z = (spreads[i] - m) / s
                    z_scores[i] = z
                    
                    if z > entry_threshold:
                        signals[i] = "SHORT_A_LONG_B"  # A overpriced vs B
                    elif z < -entry_threshold:
                        signals[i] = "LONG_A_SHORT_B"  # A underpriced vs B
                    elif abs(z) <= exit_threshold:
                        signals[i] = "MEAN_REVERTED_FLAT"
                    else:
                        signals[i] = "HOLD_OR_NEUTRAL"
                else:
                    z_scores[i] = 0.0
                    signals[i] = "ZERO_DISPERSION"
            else:
                means[i] = spreads[i]
                stds[i] = 0.0
                z_scores[i] = 0.0
                signals[i] = "INSUFFICIENT_HISTORY"

        grp["rolling_mean"] = means
        grp["rolling_std"] = stds
        grp["z_score"] = z_scores
        grp["signal"] = signals
        result_chunks.append(grp)

    if result_chunks:
        final_df = pd.concat(result_chunks, ignore_index=True)
        return final_df
    return df
