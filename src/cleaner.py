"""
Data inspection and cleaning module for MCX Bhavcopy.
Standardizes fields, parses dates, validates data quality, and filters Gold futures.
"""

from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

# Supported target Gold contracts
GOLD_SYMBOLS = ["GOLDM", "GOLDTEN", "GOLDGUINEA", "GOLDPETAL"]

TARGET_COLUMNS = [
    "Date",
    "Instrument Name",
    "Symbol",
    "Expiry Date",
    "Option Type",
    "Strike Price",
    "Open",
    "High",
    "Low",
    "Close",
    "Previous Close",
    "Volume(Lots)",
    "Volume(In 000's)",
    "Value(Lacs)",
    "Open Interest(Lots)",
]


@dataclass
class DataQualityReport:
    total_raw_rows: int
    futcom_rows: int
    gold_rows: int
    unique_symbols: List[str]
    unique_expiries: List[str]
    dates_found: List[str]
    missing_ohlc_count: int
    zero_volume_count: int
    zero_oi_count: int
    duplicate_count: int
    is_valid: bool
    issues: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_raw_rows": self.total_raw_rows,
            "futcom_rows": self.futcom_rows,
            "gold_rows": self.gold_rows,
            "unique_symbols": self.unique_symbols,
            "unique_expiries": self.unique_expiries,
            "dates_found": self.dates_found,
            "missing_ohlc_count": self.missing_ohlc_count,
            "zero_volume_count": self.zero_volume_count,
            "zero_oi_count": self.zero_oi_count,
            "duplicate_count": self.duplicate_count,
            "is_valid": self.is_valid,
            "issues": self.issues,
        }


def parse_trade_date(date_str: Any) -> Optional[date]:
    """Parse Bhavcopy trade date string into datetime.date."""
    if pd.isna(date_str) or not str(date_str).strip():
        return None
    s = str(date_str).strip().strip('"')
    # Try standard MCX formats: "01 Oct 2026", "2026-10-01", "01-10-2026", "01-Oct-2026"
    for fmt in ("%d %b %Y", "%Y-%m-%d", "%d-%m-%Y", "%d-%b-%Y", "%d%b%Y", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    try:
        return pd.to_datetime(s).date()
    except Exception:
        return None


def parse_expiry_date(expiry_str: Any) -> Optional[date]:
    """Parse Bhavcopy expiry date string (e.g. '05NOV2026', '30OCT2026', '05-Nov-2026') into datetime.date."""
    if pd.isna(expiry_str) or not str(expiry_str).strip():
        return None
    s = str(expiry_str).strip().strip('"').upper()
    for fmt in ("%d%b%Y", "%d-%b-%Y", "%d %b %Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    try:
        return pd.to_datetime(s).date()
    except Exception:
        return None


def clean_bhavcopy_df(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, DataQualityReport]:
    """
    Clean, validate and filter raw MCX Bhavcopy DataFrame.
    """
    issues = []
    total_raw_rows = len(raw_df)

    # Standardize column headers: strip whitespace and quotes
    df = raw_df.copy()
    df.columns = [c.strip().strip('"') for c in df.columns]

    # Required column presence check
    expected_cols = [
        "Date", "Instrument Name", "Symbol", "Expiry Date",
        "Open", "High", "Low", "Close", "Previous Close",
        "Volume(Lots)", "Value(Lacs)", "Open Interest(Lots)"
    ]
    for col in expected_cols:
        if col not in df.columns:
            # Check case-insensitive match
            matches = [c for c in df.columns if c.lower() == col.lower()]
            if matches:
                df.rename(columns={matches[0]: col}, inplace=True)
            else:
                issues.append(f"Missing expected column: '{col}'")

    if issues:
        report = DataQualityReport(
            total_raw_rows=total_raw_rows,
            futcom_rows=0,
            gold_rows=0,
            unique_symbols=[],
            unique_expiries=[],
            dates_found=[],
            missing_ohlc_count=0,
            zero_volume_count=0,
            zero_oi_count=0,
            duplicate_count=0,
            is_valid=False,
            issues=issues,
        )
        return pd.DataFrame(), report

    # Strip whitespace on string columns
    for str_col in ["Instrument Name", "Symbol", "Expiry Date", "Option Type"]:
        if str_col in df.columns:
            df[str_col] = df[str_col].astype(str).str.strip().str.strip('"')

    # Filter for Commodity Futures (FUTCOM)
    futcom_mask = df["Instrument Name"].str.upper() == "FUTCOM"
    futcom_rows = int(futcom_mask.sum())
    df_fut = df[futcom_mask].copy()

    # Filter for targeted Gold contracts
    gold_mask = df_fut["Symbol"].str.upper().isin(GOLD_SYMBOLS)
    gold_rows = int(gold_mask.sum())
    df_gold = df_fut[gold_mask].copy()

    if df_gold.empty:
        issues.append("No Gold contracts (GOLDM, GOLDTEN, GOLDGUINEA, GOLDPETAL) found in FUTCOM records.")
        report = DataQualityReport(
            total_raw_rows=total_raw_rows,
            futcom_rows=futcom_rows,
            gold_rows=0,
            unique_symbols=[],
            unique_expiries=[],
            dates_found=[],
            missing_ohlc_count=0,
            zero_volume_count=0,
            zero_oi_count=0,
            duplicate_count=0,
            is_valid=False,
            issues=issues,
        )
        return pd.DataFrame(), report

    # Parse and validate dates
    df_gold["trade_date"] = df_gold["Date"].apply(parse_trade_date)
    df_gold["expiry_date"] = df_gold["Expiry Date"].apply(parse_expiry_date)

    if df_gold["trade_date"].isna().any():
        issues.append(f"Found {df_gold['trade_date'].isna().sum()} rows with unparseable trade date.")
    if df_gold["expiry_date"].isna().any():
        issues.append(f"Found {df_gold['expiry_date'].isna().sum()} rows with unparseable expiry date.")

    # Numeric conversions
    numeric_cols = [
        "Open", "High", "Low", "Close", "Previous Close",
        "Volume(Lots)", "Value(Lacs)", "Open Interest(Lots)"
    ]
    for ncol in numeric_cols:
        if ncol in df_gold.columns:
            # replace blank strings / non-numeric with NaN
            df_gold[ncol] = pd.to_numeric(
                df_gold[ncol].astype(str).str.replace(",", "").str.strip(),
                errors="coerce"
            )

    # Check duplicates on primary contract key: (trade_date, Symbol, expiry_date)
    dup_mask = df_gold.duplicated(subset=["trade_date", "Symbol", "expiry_date"], keep=False)
    duplicate_count = int(dup_mask.sum())
    if duplicate_count > 0:
        issues.append(f"Found {duplicate_count} duplicate contract entries for same trade date and expiry.")
        # Deduplicate by keeping the one with higher volume/OI or last
        df_gold = df_gold.drop_duplicates(subset=["trade_date", "Symbol", "expiry_date"], keep="last")

    # Metrics on data health
    missing_ohlc_count = int(df_gold["Close"].isna().sum())
    zero_volume_count = int((df_gold["Volume(Lots)"].fillna(0) == 0).sum())
    zero_oi_count = int((df_gold["Open Interest(Lots)"].fillna(0) == 0).sum())

    # Standardized output columns
    standardized_df = pd.DataFrame({
        "trade_date": df_gold["trade_date"],
        "symbol": df_gold["Symbol"].str.upper(),
        "expiry_date": df_gold["expiry_date"],
        "open": df_gold["Open"],
        "high": df_gold["High"],
        "low": df_gold["Low"],
        "close": df_gold["Close"],
        "prev_close": df_gold["Previous Close"],
        "volume_lots": df_gold["Volume(Lots)"].fillna(0).astype(int),
        "value_lacs": df_gold["Value(Lacs)"].fillna(0.0).astype(float),
        "open_interest_lots": df_gold["Open Interest(Lots)"].fillna(0).astype(int),
    })

    # Sort deterministically
    standardized_df.sort_values(by=["trade_date", "symbol", "expiry_date"], inplace=True)
    standardized_df.reset_index(drop=True, inplace=True)

    unique_symbols = sorted(standardized_df["symbol"].unique().tolist())
    unique_expiries = sorted([str(d) for d in standardized_df["expiry_date"].dropna().unique().tolist()])
    dates_found = sorted([str(d) for d in standardized_df["trade_date"].dropna().unique().tolist()])

    report = DataQualityReport(
        total_raw_rows=total_raw_rows,
        futcom_rows=futcom_rows,
        gold_rows=len(standardized_df),
        unique_symbols=unique_symbols,
        unique_expiries=unique_expiries,
        dates_found=dates_found,
        missing_ohlc_count=missing_ohlc_count,
        zero_volume_count=zero_volume_count,
        zero_oi_count=zero_oi_count,
        duplicate_count=duplicate_count,
        is_valid=len(issues) == 0,
        issues=issues,
    )

    return standardized_df, report


def process_bhavcopy_file(
    file_path: Path | str,
    output_dir: Optional[Path | str] = None
) -> Tuple[pd.DataFrame, DataQualityReport]:
    """
    Load raw Bhavcopy CSV, clean, validate, and optionally save processed CSV/Parquet.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Bhavcopy file not found at: {file_path}")

    # Read CSV with flexible encoding/handling
    raw_df = pd.read_csv(file_path, dtype=str)
    cleaned_df, report = clean_bhavcopy_df(raw_df)

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        # Save processed CSV and Parquet
        date_str = report.dates_found[0].replace("-", "") if report.dates_found else "all"
        csv_out = out_path / f"gold_processed_{date_str}.csv"
        parquet_out = out_path / f"gold_processed_{date_str}.parquet"
        
        cleaned_df.to_csv(csv_out, index=False)
        try:
            cleaned_df.to_parquet(parquet_out, index=False)
        except Exception:
            pass

    return cleaned_df, report
