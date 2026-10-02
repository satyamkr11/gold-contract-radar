"""
Cost and Liquidity Filter module for MCX Gold Contracts.
Models exchange fees, statutory taxes, brokerage, slippage, and liquidity conditions.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np


@dataclass
class CostModel:
    """
    Configurable transaction cost model for MCX Commodity Futures.
    All rates are expressed as decimals (e.g. 0.0001 = 0.01%).
    """
    brokerage_per_order_inr: float = 20.0       # Flat brokerage per executed order
    exchange_turnover_fee_rate: float = 0.000021 # 0.0021% of turnover (MCX fee)
    ctt_sell_rate: float = 0.00010              # 0.01% on sell side (Commodity Transaction Tax)
    stamp_duty_buy_rate: float = 0.000020        # 0.002% on buy side (State Stamp Duty)
    sebi_fee_rate: float = 0.000001             # 0.0001% of turnover
    gst_rate: float = 0.18                      # 18% on (Brokerage + Exchange Fees + SEBI Fees)
    default_slippage_bps: float = 2.0           # 2 basis points (0.02%) per leg default slippage

    def calculate_round_trip_cost(
        self,
        notional_value: float,
        slippage_bps: float = 2.0,
        num_orders: int = 2
    ) -> Dict[str, float]:
        """
        Calculate total round-trip friction for buying and later selling 1 leg.
        Turnover = Buy notional + Sell notional ~ 2 * notional_value
        """
        if notional_value <= 0:
            return {"total_cost_inr": 0.0, "cost_pct": 0.0}

        buy_turnover = notional_value
        sell_turnover = notional_value
        total_turnover = buy_turnover + sell_turnover

        brokerage = self.brokerage_per_order_inr * num_orders
        exchange_fee = total_turnover * self.exchange_turnover_fee_rate
        sebi_fee = total_turnover * self.sebi_fee_rate
        gst = (brokerage + exchange_fee + sebi_fee) * self.gst_rate
        ctt = sell_turnover * self.ctt_sell_rate
        stamp_duty = buy_turnover * self.stamp_duty_buy_rate

        statutory_and_brokerage = brokerage + exchange_fee + sebi_fee + gst + ctt + stamp_duty
        slippage_cost = total_turnover * (slippage_bps / 10000.0)
        total_friction = statutory_and_brokerage + slippage_cost

        return {
            "brokerage_inr": brokerage,
            "exchange_fee_inr": exchange_fee,
            "sebi_fee_inr": sebi_fee,
            "gst_inr": gst,
            "ctt_inr": ctt,
            "stamp_duty_inr": stamp_duty,
            "statutory_brokerage_inr": statutory_and_brokerage,
            "slippage_inr": slippage_cost,
            "total_cost_inr": total_friction,
            "cost_pct": (total_friction / notional_value) * 100.0,
            "cost_per_gram_fine_gold": 0.0, # Filled when fine gold weight is provided
        }


@dataclass
class LiquidityFilter:
    """
    Liquidity criteria to prevent executing in phantom or illiquid contracts.
    """
    min_volume_lots: int = 5
    min_open_interest_lots: int = 10
    min_traded_value_lacs: float = 1.0

    def assess_contract(self, row: pd.Series) -> Tuple[bool, str]:
        """Check if contract meets minimum volume and open interest thresholds."""
        vol = row.get("volume_lots", 0)
        oi = row.get("open_interest_lots", 0)
        val = row.get("value_lacs", 0.0)

        reasons = []
        if vol < self.min_volume_lots:
            reasons.append(f"Low Vol ({vol} < {self.min_volume_lots} lots)")
        if oi < self.min_open_interest_lots:
            reasons.append(f"Low OI ({oi} < {self.min_open_interest_lots} lots)")
        if val < self.min_traded_value_lacs:
            reasons.append(f"Low Traded Val (₹{val:.2f}L < ₹{self.min_traded_value_lacs:.2f}L)")

        if reasons:
            return False, "; ".join(reasons)
        return True, "Liquid"


def evaluate_spread_liquidity(
    leg1_row: pd.Series,
    leg2_row: pd.Series,
    liq_filter: LiquidityFilter
) -> Tuple[bool, str]:
    """Evaluate whether both legs of a relative value pair meet liquidity criteria."""
    l1_ok, l1_msg = liq_filter.assess_contract(leg1_row)
    l2_ok, l2_msg = liq_filter.assess_contract(leg2_row)

    if l1_ok and l2_ok:
        return True, "Both legs liquid"
    
    issues = []
    if not l1_ok:
        issues.append(f"Leg 1 ({leg1_row.get('symbol', 'Leg1')} {leg1_row.get('expiry_date', '')}): {l1_msg}")
    if not l2_ok:
        issues.append(f"Leg 2 ({leg2_row.get('symbol', 'Leg2')} {leg2_row.get('expiry_date', '')}): {l2_msg}")
    return False, " | ".join(issues)
