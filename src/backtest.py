"""
Walk-Forward Backtesting Engine for MCX Gold Relative Value Strategies.
Strict no-lookahead execution (Signal at t, Execution at t+1), contract expiry handling,
realistic friction modeling, and statistical sufficiency checks.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

from src.filters import CostModel, LiquidityFilter
from src.normalization import CONTRACT_SPECS, get_contract_spec
from src.relative_value import calculate_rolling_zscores, compute_cross_sectional_spreads


@dataclass
class BacktestTrade:
    trade_id: int
    pair: str
    symbol_a: str
    expiry_a: date
    symbol_b: str
    expiry_b: date
    direction: str             # "LONG_A_SHORT_B" or "SHORT_A_LONG_B"
    signal_date: date
    entry_date: date
    exit_date: Optional[date]
    entry_norm_a: float
    entry_norm_b: float
    entry_raw_a: float
    entry_raw_b: float
    exit_norm_a: Optional[float] = None
    exit_norm_b: Optional[float] = None
    exit_raw_a: Optional[float] = None
    exit_raw_b: Optional[float] = None
    lots_a: int = 1
    lots_b: int = 1
    fine_gold_grams: float = 0.0
    gross_pnl_inr: float = 0.0
    total_costs_inr: float = 0.0
    net_pnl_inr: float = 0.0
    return_pct: float = 0.0
    exit_reason: str = ""


@dataclass
class BacktestReport:
    is_sufficient_data: bool
    total_days_available: int
    min_days_required: int
    status_message: str
    total_trades: int = 0
    winning_trades: int = 0         # Net winning trades (net_pnl > 0)
    losing_trades: int = 0          # Net losing trades (net_pnl <= 0)
    win_rate_pct: float = 0.0       # Net win rate %
    gross_winning_trades: int = 0   # Gross winning trades (gross_pnl > 0)
    gross_losing_trades: int = 0    # Gross losing trades (gross_pnl < 0)
    gross_win_rate_pct: float = 0.0 # Gross win rate %
    total_gross_pnl_inr: float = 0.0
    total_costs_inr: float = 0.0
    total_net_pnl_inr: float = 0.0
    gross_profit_factor: float = 0.0 # Sum(Gross Gains) / Sum(|Gross Losses|)
    net_profit_factor: float = 0.0   # Sum(Net Gains) / Sum(|Net Losses|)
    profit_factor: float = 0.0       # Alias for net_profit_factor
    avg_trade_net_pnl_inr: float = 0.0
    median_trade_net_pnl_inr: float = 0.0
    max_drawdown_inr: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: Optional[float] = None
    trades: List[Dict[str, Any]] = field(default_factory=list)
    daily_equity_curve: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_sufficient_data": self.is_sufficient_data,
            "total_days_available": self.total_days_available,
            "min_days_required": self.min_days_required,
            "status_message": self.status_message,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate_pct": self.win_rate_pct,
            "gross_winning_trades": self.gross_winning_trades,
            "gross_losing_trades": self.gross_losing_trades,
            "gross_win_rate_pct": self.gross_win_rate_pct,
            "total_gross_pnl_inr": self.total_gross_pnl_inr,
            "total_costs_inr": self.total_costs_inr,
            "total_net_pnl_inr": self.total_net_pnl_inr,
            "gross_profit_factor": self.gross_profit_factor,
            "net_profit_factor": self.net_profit_factor,
            "profit_factor": self.profit_factor,
            "avg_trade_net_pnl_inr": self.avg_trade_net_pnl_inr,
            "median_trade_net_pnl_inr": self.median_trade_net_pnl_inr,
            "max_drawdown_inr": self.max_drawdown_inr,
            "max_drawdown_pct": self.max_drawdown_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "trades": self.trades,
            "daily_equity_curve": self.daily_equity_curve,
        }


class WalkForwardBacktester:
    """
    Simulates cross-contract relative value arbitrage with strict execution timing:
    - Signals generated at close of Day t.
    - Orders executed at Open (or next available close) of Day t+1.
    - Positions closed prior to physical delivery notice / contract expiry.
    """
    def __init__(
        self,
        cost_model: Optional[CostModel] = None,
        liq_filter: Optional[LiquidityFilter] = None,
        min_lookback_days: int = 15,
        days_before_expiry_exit: int = 2,
        entry_z_threshold: float = 2.0,
        exit_z_threshold: float = 0.5,
        stop_z_threshold: float = 3.5,
        slippage_bps: float = 2.0
    ):
        self.cost_model = cost_model or CostModel()
        self.liq_filter = liq_filter or LiquidityFilter()
        self.min_lookback_days = min_lookback_days
        self.days_before_expiry_exit = days_before_expiry_exit
        self.entry_z_threshold = entry_z_threshold
        self.exit_z_threshold = exit_z_threshold
        self.stop_z_threshold = stop_z_threshold
        self.slippage_bps = slippage_bps

    def run(self, master_df: pd.DataFrame) -> BacktestReport:
        """
        Execute walk-forward backtest over available gold dataset.
        Checks for statistical sufficiency before generating results.
        """
        if master_df.empty:
            return BacktestReport(
                is_sufficient_data=False,
                total_days_available=0,
                min_days_required=self.min_lookback_days,
                status_message="No data provided for backtest.",
            )

        unique_dates = sorted(master_df["trade_date"].unique().tolist())
        num_days = len(unique_dates)

        if num_days < self.min_lookback_days:
            return BacktestReport(
                is_sufficient_data=False,
                total_days_available=num_days,
                min_days_required=self.min_lookback_days,
                status_message=(
                    f"Insufficient historical data: Only {num_days} trading date(s) available in dataset. "
                    f"A minimum of {self.min_lookback_days} daily Bhavcopy files are required to compute "
                    f"reliable rolling Z-scores without fabrication."
                ),
            )

        # Multi-day walk forward simulation
        spreads_df = compute_cross_sectional_spreads(master_df, matching_mode="exact_expiry")
        if spreads_df.empty:
            return BacktestReport(
                is_sufficient_data=False,
                total_days_available=num_days,
                min_days_required=self.min_lookback_days,
                status_message="No matching expiry pairs found to construct spread series.",
            )

        z_df = calculate_rolling_zscores(
            spreads_df,
            lookback_window=min(10, num_days - 1),
            entry_threshold=self.entry_z_threshold,
            exit_threshold=self.exit_z_threshold,
            stop_threshold=self.stop_z_threshold
        )

        # Simulate trades walk-forward
        completed_trades: List[BacktestTrade] = []
        open_positions: Dict[Tuple[str, date, str, date], BacktestTrade] = {}
        trade_id_counter = 1

        # Date lookup maps for quick price querying
        df_indexed = master_df.set_index(["trade_date", "symbol", "expiry_date"])

        for t_idx in range(len(unique_dates) - 1):
            curr_date = unique_dates[t_idx]
            next_date = unique_dates[t_idx + 1]

            day_z = z_df[z_df["trade_date"] == curr_date]

            # 1. Manage and close existing positions
            active_keys = list(open_positions.keys())
            for pos_key in active_keys:
                pos = open_positions[pos_key]
                sym_a, exp_a, sym_b, exp_b = pos_key

                # Check expiry rule
                days_to_exp_a = (exp_a - next_date).days
                days_to_exp_b = (exp_b - next_date).days
                is_near_expiry = (days_to_exp_a <= self.days_before_expiry_exit) or (days_to_exp_b <= self.days_before_expiry_exit)

                # Check if prices exist on next_date
                try:
                    row_next_a = df_indexed.loc[(next_date, sym_a, exp_a)]
                    row_next_b = df_indexed.loc[(next_date, sym_b, exp_b)]
                except KeyError:
                    # Contract no longer traded or missing
                    is_near_expiry = True
                    row_next_a = None
                    row_next_b = None

                # Find current Z-score signal on curr_date
                pair_z_row = day_z[(day_z["symbol_a"] == sym_a) & (day_z["expiry_a"] == exp_a) &
                                   (day_z["symbol_b"] == sym_b) & (day_z["expiry_b"] == exp_b)]

                curr_z = pair_z_row["z_score"].values[0] if not pair_z_row.empty else 0.0

                should_exit = False
                exit_reason = ""

                if is_near_expiry:
                    should_exit = True
                    exit_reason = "EXPIRY_ROLL_OR_SETTLEMENT"
                elif abs(curr_z) <= self.exit_z_threshold:
                    should_exit = True
                    exit_reason = "MEAN_REVERSION_TARGET_MET"
                elif abs(curr_z) >= self.stop_z_threshold:
                    should_exit = True
                    exit_reason = "STOP_LOSS_TRIGGERED"

                if should_exit:
                    # Execute exit on next_date
                    if row_next_a is not None and row_next_b is not None:
                        exit_price_a = float(row_next_a.get("open", row_next_a["close"]))
                        exit_price_b = float(row_next_b.get("open", row_next_b["close"]))
                        exit_norm_a = float(row_next_a["norm_close"])
                        exit_norm_b = float(row_next_b["norm_close"])
                    else:
                        exit_price_a = pos.entry_raw_a
                        exit_price_b = pos.entry_raw_b
                        exit_norm_a = pos.entry_norm_a
                        exit_norm_b = pos.entry_norm_b

                    pos.exit_date = next_date
                    pos.exit_raw_a = exit_price_a
                    pos.exit_raw_b = exit_price_b
                    pos.exit_norm_a = exit_norm_a
                    pos.exit_norm_b = exit_norm_b
                    pos.exit_reason = exit_reason

                    # Calculate PnL based on normalized fine grams
                    grams = pos.fine_gold_grams
                    if pos.direction == "LONG_A_SHORT_B":
                        # Long A gain: (exit_a - entry_a) * grams; Short B gain: (entry_b - exit_b) * grams
                        pnl_a = (exit_norm_a - pos.entry_norm_a) * grams
                        pnl_b = (pos.entry_norm_b - exit_norm_b) * grams
                    else: # SHORT_A_LONG_B
                        pnl_a = (pos.entry_norm_a - exit_norm_a) * grams
                        pnl_b = (exit_norm_b - pos.entry_norm_b) * grams

                    gross_pnl = pnl_a + pnl_b
                    
                    # Transaction costs and slippage for roundtrip (entry + exit)
                    notional_a = exit_norm_a * grams
                    notional_b = exit_norm_b * grams
                    cost_a = self.cost_model.calculate_round_trip_cost(notional_a, self.slippage_bps)["total_cost_inr"]
                    cost_b = self.cost_model.calculate_round_trip_cost(notional_b, self.slippage_bps)["total_cost_inr"]
                    total_friction = cost_a + cost_b

                    pos.gross_pnl_inr = gross_pnl
                    pos.total_costs_inr = total_friction
                    pos.net_pnl_inr = gross_pnl - total_friction
                    pos.return_pct = (pos.net_pnl_inr / (notional_a + notional_b)) * 100.0 if (notional_a + notional_b) > 0 else 0.0

                    completed_trades.append(pos)
                    del open_positions[pos_key]

            # 2. Check for new signals on curr_date -> enter on next_date
            for _, row in day_z.iterrows():
                sig = row["signal"]
                if sig in ["LONG_A_SHORT_B", "SHORT_A_LONG_B"]:
                    pos_key = (row["symbol_a"], row["expiry_a"], row["symbol_b"], row["expiry_b"])
                    if pos_key in open_positions:
                        continue  # already in position

                    # Check next_date availability
                    try:
                        next_a = df_indexed.loc[(next_date, row["symbol_a"], row["expiry_a"])]
                        next_b = df_indexed.loc[(next_date, row["symbol_b"], row["expiry_b"])]
                    except KeyError:
                        continue

                    # Liquidity check
                    l_ok, _ = self.liq_filter.assess_contract(next_a)
                    if not l_ok:
                        continue

                    entry_price_a = float(next_a.get("open", next_a["close"]))
                    entry_price_b = float(next_b.get("open", next_b["close"]))
                    entry_norm_a = float(next_a["norm_close"])
                    entry_norm_b = float(next_b["norm_close"])

                    # Standardize position size to 100 grams fine gold equivalent (standard 1 mini lot equivalent)
                    fine_gold_grams = 100.0

                    trade = BacktestTrade(
                        trade_id=trade_id_counter,
                        pair=row["pair"],
                        symbol_a=row["symbol_a"],
                        expiry_a=row["expiry_a"],
                        symbol_b=row["symbol_b"],
                        expiry_b=row["expiry_b"],
                        direction=sig,
                        signal_date=curr_date,
                        entry_date=next_date,
                        exit_date=None,
                        entry_norm_a=entry_norm_a,
                        entry_norm_b=entry_norm_b,
                        entry_raw_a=entry_price_a,
                        entry_raw_b=entry_price_b,
                        fine_gold_grams=fine_gold_grams,
                    )
                    open_positions[pos_key] = trade
                    trade_id_counter += 1

        # Calculate summary statistics
        total_trades = len(completed_trades)
        net_wins = [t for t in completed_trades if t.net_pnl_inr > 0]
        net_losses = [t for t in completed_trades if t.net_pnl_inr <= 0]
        net_win_rate = (len(net_wins) / total_trades * 100.0) if total_trades > 0 else 0.0

        gross_wins = [t for t in completed_trades if t.gross_pnl_inr > 0]
        gross_losses = [t for t in completed_trades if t.gross_pnl_inr < 0]
        gross_win_rate = (len(gross_wins) / total_trades * 100.0) if total_trades > 0 else 0.0

        total_gross = sum(t.gross_pnl_inr for t in completed_trades)
        total_costs = sum(t.total_costs_inr for t in completed_trades)
        total_net = sum(t.net_pnl_inr for t in completed_trades)

        # Gross Profit Factor: Sum(Gross Gains) / Sum(|Gross Losses|)
        gross_gains_sum = sum(t.gross_pnl_inr for t in gross_wins)
        gross_losses_sum = abs(sum(t.gross_pnl_inr for t in gross_losses))
        gross_profit_factor = (gross_gains_sum / gross_losses_sum) if gross_losses_sum > 0 else (99.0 if gross_gains_sum > 0 else 0.0)

        # Net Profit Factor: Sum(Net Gains) / Sum(|Net Losses|)
        net_gains_sum = sum(t.net_pnl_inr for t in net_wins)
        net_losses_sum = abs(sum(t.net_pnl_inr for t in net_losses))
        net_profit_factor = (net_gains_sum / net_losses_sum) if net_losses_sum > 0 else (99.0 if net_gains_sum > 0 else 0.0)

        avg_trade_net = (total_net / total_trades) if total_trades > 0 else 0.0
        median_trade_net = float(np.median([t.net_pnl_inr for t in completed_trades])) if completed_trades else 0.0

        # Equity curve & Max Drawdown
        cum_pnl = 0.0
        peak = 0.0
        max_dd = 0.0
        eq_curve = []

        for t in completed_trades:
            cum_pnl += t.net_pnl_inr
            if cum_pnl > peak:
                peak = cum_pnl
            dd = peak - cum_pnl
            if dd > max_dd:
                max_dd = dd
            eq_curve.append({
                "exit_date": str(t.exit_date),
                "trade_id": t.trade_id,
                "net_pnl": t.net_pnl_inr,
                "cumulative_net_pnl": cum_pnl,
                "drawdown": dd,
            })

        trade_dicts = [
            {
                "trade_id": t.trade_id,
                "pair": t.pair,
                "direction": t.direction,
                "entry_date": str(t.entry_date),
                "exit_date": str(t.exit_date),
                "gross_pnl_inr": round(t.gross_pnl_inr, 2),
                "total_costs_inr": round(t.total_costs_inr, 2),
                "net_pnl_inr": round(t.net_pnl_inr, 2),
                "return_pct": round(t.return_pct, 2),
                "exit_reason": t.exit_reason,
            }
            for t in completed_trades
        ]

        return BacktestReport(
            is_sufficient_data=True,
            total_days_available=num_days,
            min_days_required=self.min_lookback_days,
            status_message=f"Walk-forward simulation executed successfully across {num_days} trading dates.",
            total_trades=total_trades,
            winning_trades=len(net_wins),
            losing_trades=len(net_losses),
            win_rate_pct=net_win_rate,
            gross_winning_trades=len(gross_wins),
            gross_losing_trades=len(gross_losses),
            gross_win_rate_pct=gross_win_rate,
            total_gross_pnl_inr=total_gross,
            total_costs_inr=total_costs,
            total_net_pnl_inr=total_net,
            gross_profit_factor=gross_profit_factor,
            net_profit_factor=net_profit_factor,
            profit_factor=net_profit_factor,
            avg_trade_net_pnl_inr=avg_trade_net,
            median_trade_net_pnl_inr=median_trade_net,
            max_drawdown_inr=max_dd,
            trades=trade_dicts,
            daily_equity_curve=eq_curve,
        )
