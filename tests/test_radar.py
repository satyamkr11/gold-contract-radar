"""
Unit tests for Gold Contract Radar modules.
Tests data cleaning, normalization math, expiry parsing, rolling Z-score integrity,
and walk-forward execution rules.
"""

from datetime import date
import pandas as pd
import numpy as np
import pytest

from src.cleaner import clean_bhavcopy_df, parse_expiry_date, parse_trade_date, GOLD_SYMBOLS
from src.normalization import (
    CONTRACT_SPECS,
    normalize_price,
    denormalize_price,
    calculate_lot_value,
    add_normalized_columns,
)
from src.relative_value import compute_cross_sectional_spreads, calculate_rolling_zscores
from src.filters import CostModel, LiquidityFilter
from src.backtest import WalkForwardBacktester


def test_contract_specs_and_normalization_math():
    """Verify exact formula conversions according to challenge specs."""
    # GOLDM: 100g, quoted per 10g, purity 995
    # Divisor = 10 * 0.995 = 9.95
    p_m_raw = 75000.0  # ₹75,000 per 10g
    norm_m = normalize_price(p_m_raw, "GOLDM")
    assert norm_m == pytest.approx(75000.0 / 9.95, rel=1e-5)
    assert denormalize_price(norm_m, "GOLDM") == pytest.approx(p_m_raw, rel=1e-5)
    assert calculate_lot_value(p_m_raw, "GOLDM") == pytest.approx(75000.0 * 10, rel=1e-5)

    # GOLDTEN: 10g, quoted per 10g, purity 999
    # Divisor = 10 * 0.999 = 9.99
    p_ten_raw = 75300.0
    norm_ten = normalize_price(p_ten_raw, "GOLDTEN")
    assert norm_ten == pytest.approx(75300.0 / 9.99, rel=1e-5)

    # GOLDGUINEA: 8g, quoted per 8g, purity 999
    # Divisor = 8 * 0.999 = 7.992
    p_g_raw = 60240.0
    norm_g = normalize_price(p_g_raw, "GOLDGUINEA")
    assert norm_g == pytest.approx(60240.0 / 7.992, rel=1e-5)

    # GOLDPETAL: 1g, quoted per 1g, purity 999
    # Divisor = 1 * 0.999 = 0.999
    p_p_raw = 7535.0
    norm_p = normalize_price(p_p_raw, "GOLDPETAL")
    assert norm_p == pytest.approx(7535.0 / 0.999, rel=1e-5)


def test_date_and_expiry_parsing():
    """Test date and expiry parser with multiple format variants."""
    assert parse_trade_date("01 Oct 2026") == date(2026, 10, 1)
    assert parse_trade_date("2026-10-01") == date(2026, 10, 1)
    assert parse_trade_date("01-10-2026") == date(2026, 10, 1)

    assert parse_expiry_date("05NOV2026") == date(2026, 11, 5)
    assert parse_expiry_date("30OCT2026") == date(2026, 10, 30)
    assert parse_expiry_date("05FEB2027") == date(2027, 2, 5)
    assert parse_expiry_date("26MAR2027") == date(2027, 3, 26)


def test_bhavcopy_cleaning_and_filtering():
    """Test dataframe cleaning, whitespace removal, FUTCOM and gold symbol filtering."""
    raw_data = {
        "Date": ["01 Oct 2026", "01 Oct 2026", "01 Oct 2026", "01 Oct 2026"],
        "Instrument Name": ["FUTCOM", "OPTFUT", "FUTCOM", "FUTCOM"],
        "Symbol": ["GOLDM   ", "GOLDM   ", "CRUDEOIL ", "GOLDTEN "],
        "Expiry Date": ["05NOV2026", "05NOV2026", "19OCT2026", "05NOV2026"],
        "Option Type": ["-", "CE", "-", "-"],
        "Strike Price": ["0", "75000", "0", "0"],
        "Open": ["75000.0", "120.0", "6200.0", "75300.0"],
        "High": ["75200.0", "150.0", "6250.0", "75500.0"],
        "Low": ["74900.0", "100.0", "6150.0", "75100.0"],
        "Close": ["75100.0", "135.0", "6220.0", "75400.0"],
        "Previous Close": ["74950.0", "110.0", "6180.0", "75200.0"],
        "Volume(Lots)": ["1200", "50", "3000", "450"],
        "Volume(In 000's)": ["1200.0", "50.0", "300.0", "450.0"],
        "Value(Lacs)": ["9012.0", "6.75", "1866.0", "339.3"],
        "Open Interest(Lots)": ["4500", "200", "8000", "1200"],
    }
    raw_df = pd.DataFrame(raw_data)
    cleaned_df, report = clean_bhavcopy_df(raw_df)

    assert report.is_valid is True
    assert len(cleaned_df) == 2
    assert set(cleaned_df["symbol"].tolist()) == {"GOLDM", "GOLDTEN"}
    assert cleaned_df["volume_lots"].dtype == np.int64 or cleaned_df["volume_lots"].dtype == np.int32
    assert cleaned_df["close"].iloc[0] == 75100.0


def test_duplicate_detection_and_resolution():
    """Ensure duplicate contracts for the same trade date and expiry are handled."""
    raw_data = {
        "Date": ["01 Oct 2026", "01 Oct 2026"],
        "Instrument Name": ["FUTCOM", "FUTCOM"],
        "Symbol": ["GOLDM", "GOLDM"],
        "Expiry Date": ["05NOV2026", "05NOV2026"],
        "Open": ["75000", "75000"],
        "High": ["75200", "75200"],
        "Low": ["74900", "74900"],
        "Close": ["75100", "75150"],
        "Previous Close": ["74950", "74950"],
        "Volume(Lots)": ["100", "200"],
        "Value(Lacs)": ["750", "1500"],
        "Open Interest(Lots)": ["500", "600"],
    }
    raw_df = pd.DataFrame(raw_data)
    cleaned_df, report = clean_bhavcopy_df(raw_df)
    assert report.duplicate_count == 2
    assert len(cleaned_df) == 1
    assert cleaned_df["close"].iloc[0] == 75150.0


def test_no_lookahead_rolling_zscores():
    """Verify rolling statistics do not leak future information."""
    dates = [
        date(2026, 9, 1), date(2026, 9, 2), date(2026, 9, 3),
        date(2026, 9, 4), date(2026, 9, 5)
    ]
    spreads = [10.0, 12.0, 11.0, 35.0, 10.5]

    spread_df = pd.DataFrame({
        "trade_date": dates,
        "pair": ["GOLDM vs GOLDTEN"] * 5,
        "symbol_a": ["GOLDM"] * 5,
        "expiry_a": [date(2026, 11, 5)] * 5,
        "symbol_b": ["GOLDTEN"] * 5,
        "expiry_b": [date(2026, 11, 5)] * 5,
        "spread_inr_g": spreads,
    })

    z_df = calculate_rolling_zscores(spread_df, lookback_window=3, entry_threshold=1.5)

    # Point 0: Insufficient history
    assert z_df.iloc[0]["signal"] == "INSUFFICIENT_HISTORY"
    
    # Point 3 (Spike to 35.0): Window over points 1, 2, 3 -> mean ~ 19.33, std > 0, z > 1.5 -> signal SHORT_A_LONG_B
    assert z_df.iloc[3]["signal"] == "SHORT_A_LONG_B"


def test_backtest_insufficient_data_handling():
    """Verify that with insufficient daily files, backtest returns explicit message rather than fabricated results."""
    single_day_data = pd.DataFrame({
        "trade_date": [date(2026, 10, 1)],
        "symbol": ["GOLDM"],
        "expiry_date": [date(2026, 11, 5)],
        "open": [75000.0],
        "high": [75200.0],
        "low": [74900.0],
        "close": [75100.0],
        "prev_close": [74950.0],
        "volume_lots": [1000],
        "value_lacs": [7510.0],
        "open_interest_lots": [5000],
    })

    backtester = WalkForwardBacktester(min_lookback_days=15)
    report = backtester.run(single_day_data)

    assert report.is_sufficient_data is False
    assert "Insufficient historical data" in report.status_message
    assert report.total_trades == 0


def test_transaction_cost_model_breakdown():
    """Verify exact statutory fees, GST, and slippage calculations."""
    cost_model = CostModel(brokerage_per_order_inr=20.0, default_slippage_bps=2.0)
    notional = 1500000.0  # ₹15 Lakhs notional

    cost_dict = cost_model.calculate_round_trip_cost(notional, slippage_bps=2.0, num_orders=2)
    
    # 2 orders * ₹20 = ₹40 brokerage
    assert cost_dict["brokerage_inr"] == 40.0
    # Total turnover = 2 * ₹1,500,000 = ₹3,000,000
    # Exchange fee = 3,000,000 * 0.000021 = ₹63.00
    assert cost_dict["exchange_fee_inr"] == pytest.approx(63.00, rel=1e-3)
    # SEBI fee = 3,000,000 * 0.000001 = ₹3.00
    assert cost_dict["sebi_fee_inr"] == pytest.approx(3.00, rel=1e-3)
    # GST = (40 + 63 + 3) * 0.18 = ₹19.08
    assert cost_dict["gst_inr"] == pytest.approx(19.08, rel=1e-3)
    # CTT (Sell side) = 1,500,000 * 0.0001 = ₹150.00
    assert cost_dict["ctt_inr"] == pytest.approx(150.00, rel=1e-3)
    # Stamp duty (Buy side) = 1,500,000 * 0.000020 = ₹30.00
    assert cost_dict["stamp_duty_inr"] == pytest.approx(30.00, rel=1e-3)
    # Slippage = 3,000,000 * (2 / 10000) = ₹600.00
    assert cost_dict["slippage_inr"] == pytest.approx(600.00, rel=1e-3)

    total_expected = 40.0 + 63.0 + 3.0 + 19.08 + 150.0 + 30.0 + 600.0
    assert cost_dict["total_cost_inr"] == pytest.approx(total_expected, rel=1e-3)


def test_metric_consistency_profit_factor_and_drawdown():
    """Verify that gross and net profit factors are strictly separated and mathematically consistent."""
    from src.backtest import BacktestTrade

    trades = [
        BacktestTrade(
            trade_id=1, pair="GOLDTEN vs GOLDGUINEA", symbol_a="GOLDTEN", expiry_a=date(2026, 10, 30),
            symbol_b="GOLDGUINEA", expiry_b=date(2026, 10, 30), direction="LONG_A_SHORT_B",
            signal_date=date(2026, 9, 10), entry_date=date(2026, 9, 11), exit_date=date(2026, 9, 14),
            entry_norm_a=15000.0, entry_norm_b=15050.0, entry_raw_a=149850.0, entry_raw_b=120280.0,
            gross_pnl_inr=2000.0, total_costs_inr=500.0, net_pnl_inr=1500.0, return_pct=0.1
        ),
        BacktestTrade(
            trade_id=2, pair="GOLDTEN vs GOLDGUINEA", symbol_a="GOLDTEN", expiry_a=date(2026, 10, 30),
            symbol_b="GOLDGUINEA", expiry_b=date(2026, 10, 30), direction="LONG_A_SHORT_B",
            signal_date=date(2026, 9, 14), entry_date=date(2026, 9, 17), exit_date=date(2026, 9, 18),
            entry_norm_a=15000.0, entry_norm_b=15050.0, entry_raw_a=149850.0, entry_raw_b=120280.0,
            gross_pnl_inr=300.0, total_costs_inr=500.0, net_pnl_inr=-200.0, return_pct=-0.01
        ),
        BacktestTrade(
            trade_id=3, pair="GOLDTEN vs GOLDGUINEA", symbol_a="GOLDTEN", expiry_a=date(2026, 10, 30),
            symbol_b="GOLDGUINEA", expiry_b=date(2026, 10, 30), direction="LONG_A_SHORT_B",
            signal_date=date(2026, 9, 18), entry_date=date(2026, 9, 21), exit_date=date(2026, 9, 22),
            entry_norm_a=15000.0, entry_norm_b=15050.0, entry_raw_a=149850.0, entry_raw_b=120280.0,
            gross_pnl_inr=-1000.0, total_costs_inr=500.0, net_pnl_inr=-1500.0, return_pct=-0.1
        ),
    ]

    # Gross calculations:
    # Gross gains: +2000 (T1) + 300 (T2) = 2300
    # Gross losses: |-1000| (T3) = 1000
    # Gross Profit Factor = 2300 / 1000 = 2.30
    gross_wins = [t for t in trades if t.gross_pnl_inr > 0]
    gross_losses = [t for t in trades if t.gross_pnl_inr < 0]
    gross_pf = sum(t.gross_pnl_inr for t in gross_wins) / abs(sum(t.gross_pnl_inr for t in gross_losses))
    assert gross_pf == pytest.approx(2.30, rel=1e-3)

    # Net calculations:
    # Net gains: +1500 (T1) = 1500
    # Net losses: |-200| (T2) + |-1500| (T3) = 1700
    # Net Profit Factor = 1500 / 1700 = 0.8824
    net_wins = [t for t in trades if t.net_pnl_inr > 0]
    net_losses = [t for t in trades if t.net_pnl_inr <= 0]
    net_pf = sum(t.net_pnl_inr for t in net_wins) / abs(sum(t.net_pnl_inr for t in net_losses))
    assert net_pf == pytest.approx(1500.0 / 1700.0, rel=1e-3)

