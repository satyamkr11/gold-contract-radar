"""
Contract normalization module for MCX Gold Futures contracts.
Converts raw exchange quotations to a common ₹/gram fine-gold-equivalent metric.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


@dataclass(frozen=True)
class ContractSpec:
    symbol: str
    name: str
    lot_size_grams: float         # Physical delivery / lot size in grams
    quote_unit_grams: float       # Quoted unit in grams (e.g. 10g, 8g, 1g)
    purity_fineness: float        # Fineness in parts per thousand (e.g. 995, 999)
    tick_size_inr: float          # Minimum price tick in INR
    description: str

    @property
    def purity_fraction(self) -> float:
        return self.purity_fineness / 1000.0

    @property
    def fine_gold_per_quote_unit(self) -> float:
        """Fine gold content in grams within 1 quotation unit."""
        return self.quote_unit_grams * self.purity_fraction

    @property
    def fine_gold_per_lot(self) -> float:
        """Fine gold content in grams per full lot."""
        return self.lot_size_grams * self.purity_fraction


# MCX Gold Contract Master Specifications
CONTRACT_SPECS: Dict[str, ContractSpec] = {
    "GOLDM": ContractSpec(
        symbol="GOLDM",
        name="Gold Mini",
        lot_size_grams=100.0,
        quote_unit_grams=10.0,
        purity_fineness=995.0,
        tick_size_inr=1.0,
        description="100g lot, quoted per 10g, 995 fineness (99.5% pure gold)",
    ),
    "GOLDTEN": ContractSpec(
        symbol="GOLDTEN",
        name="Gold 10",
        lot_size_grams=10.0,
        quote_unit_grams=10.0,
        purity_fineness=999.0,
        tick_size_inr=1.0,
        description="10g lot, quoted per 10g, 999 fineness (99.9% pure gold)",
    ),
    "GOLDGUINEA": ContractSpec(
        symbol="GOLDGUINEA",
        name="Gold Guinea",
        lot_size_grams=8.0,
        quote_unit_grams=8.0,
        purity_fineness=999.0,
        tick_size_inr=1.0,
        description="8g lot, quoted per 8g, 999 fineness (99.9% pure gold)",
    ),
    "GOLDPETAL": ContractSpec(
        symbol="GOLDPETAL",
        name="Gold Petal",
        lot_size_grams=1.0,
        quote_unit_grams=1.0,
        purity_fineness=999.0,
        tick_size_inr=1.0,
        description="1g lot, quoted per 1g, 999 fineness (99.9% pure gold)",
    ),
}


def get_contract_spec(symbol: str) -> ContractSpec:
    """Retrieve specifications for a given gold symbol."""
    sym = symbol.upper().strip()
    if sym not in CONTRACT_SPECS:
        raise ValueError(f"Unknown contract symbol '{symbol}'. Supported: {list(CONTRACT_SPECS.keys())}")
    return CONTRACT_SPECS[sym]


def normalize_price(price: float, symbol: str) -> float:
    """
    Convert raw quoted price to ₹ per gram of 100% fine gold equivalent.

    Formula:
        P_normalized = Raw_Quote / (Quote_Unit_Grams * (Purity_Fineness / 1000.0))
                     = Raw_Quote / Fine_Gold_Per_Quote_Unit
    """
    if pd.isna(price) or price <= 0:
        return np.nan
    spec = get_contract_spec(symbol)
    return float(price) / spec.fine_gold_per_quote_unit


def denormalize_price(norm_price: float, symbol: str) -> float:
    """Convert normalized ₹/g fine gold price back to contract's raw quote units."""
    if pd.isna(norm_price) or norm_price <= 0:
        return np.nan
    spec = get_contract_spec(symbol)
    return float(norm_price) * spec.fine_gold_per_quote_unit


def calculate_lot_value(price: float, symbol: str) -> float:
    """Calculate the total rupee notional value of 1 full contract lot."""
    if pd.isna(price) or price <= 0:
        return np.nan
    spec = get_contract_spec(symbol)
    # lot value = price * (lot_size_grams / quote_unit_grams)
    multiplier = spec.lot_size_grams / spec.quote_unit_grams
    return float(price) * multiplier


def add_normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append normalized price columns to cleaned gold dataframe.
    Columns added:
    - norm_close (₹/g fine gold)
    - norm_open (₹/g fine gold)
    - norm_high (₹/g fine gold)
    - norm_low (₹/g fine gold)
    - lot_value_inr (INR notional per lot)
    - fine_gold_lot_grams (fine gold in grams per lot)
    """
    df_out = df.copy()
    
    fine_units = df_out["symbol"].apply(lambda s: CONTRACT_SPECS[s].fine_gold_per_quote_unit if s in CONTRACT_SPECS else np.nan)
    lot_multipliers = df_out["symbol"].apply(lambda s: CONTRACT_SPECS[s].lot_size_grams / CONTRACT_SPECS[s].quote_unit_grams if s in CONTRACT_SPECS else np.nan)
    fine_gold_lots = df_out["symbol"].apply(lambda s: CONTRACT_SPECS[s].fine_gold_per_lot if s in CONTRACT_SPECS else np.nan)

    df_out["norm_close"] = df_out["close"] / fine_units
    df_out["norm_open"] = df_out["open"] / fine_units
    df_out["norm_high"] = df_out["high"] / fine_units
    df_out["norm_low"] = df_out["low"] / fine_units
    df_out["lot_value_inr"] = df_out["close"] * lot_multipliers
    df_out["fine_gold_lot_grams"] = fine_gold_lots

    return df_out
