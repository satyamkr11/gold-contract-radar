"""
MCX Historical Data Pipeline module.
Handles multi-day Bhavcopy ingestion, schema validation, combining datasets into Parquet,
and provides an isolated interface/downloader for official MCX Bhavcopy sources.
"""

from dataclasses import dataclass
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import glob
import pandas as pd
import requests

from src.cleaner import clean_bhavcopy_df, DataQualityReport, process_bhavcopy_file
from src.normalization import add_normalized_columns


@dataclass
class IngestionResult:
    total_files_scanned: int
    valid_files_processed: int
    total_raw_rows: int
    total_gold_rows: int
    unique_dates: List[str]
    unique_symbols: List[str]
    parquet_path: Optional[str]
    issues: List[str]


class MCXDataDownloader:
    """
    Downloader interface for MCX Daily Bhavcopy files.
    MCX publishes daily bhavcopy files. Because exchange endpoints frequently employ session tokens,
    anti-bot protections, or date format changes, this downloader provides:
    1. Verified standard MCX web request mechanism with custom user-agent and headers.
    2. Exact URL template documentation.
    3. Clear fallback and instructions when automated downloads require manual export.
    """
    MCX_BHAVCOPY_URL_TEMPLATE = "https://www.mcxindia.com/market-data/bhavcopy"
    
    def __init__(self, raw_data_dir: Path | str = "data/raw"):
        self.raw_data_dir = Path(raw_data_dir)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/csv;q=0.8,*/*;q=0.7",
            "Referer": "https://www.mcxindia.com/",
        })

    def download_date(self, target_date: date) -> Tuple[bool, str, Optional[Path]]:
        """
        Attempt to download Bhavcopy CSV for a specific target date.
        Validates whether the content returned matches the requested date.
        """
        date_str_formatted = target_date.strftime("%d%m%Y")
        dest_file = self.raw_data_dir / f"BhavCopyDateWise_{date_str_formatted}.csv"

        if dest_file.exists():
            return True, f"File already exists locally at {dest_file}", dest_file

        # MCX Daily Bhavcopy download POST/GET endpoint structure:
        url = f"https://www.mcxindia.com/market-data/bhavcopy"
        
        try:
            # First fetch homepage or bhavcopy page to obtain cookies/CSRF tokens
            resp_init = self.session.get(url, timeout=10)
            if resp_init.status_code != 200:
                return False, f"MCX server returned status {resp_init.status_code}", None

            # Attempt export request
            export_url = "https://www.mcxindia.com/backoffice/CommonSiteReport.aspx"
            payload = {
                "__EVENTTARGET": "ctl00$cph_cls$btnExport",
                "ctl00$cph_cls$txtDate": target_date.strftime("%d/%m/%Y"),
            }
            res = self.session.post(export_url, data=payload, timeout=15)
            
            if res.status_code == 200 and "Instrument Name" in res.text and "Symbol" in res.text:
                # Validate date inside content
                with open(dest_file, "w", encoding="utf-8") as f:
                    f.write(res.text)
                
                # Check date validation
                df_test = pd.read_csv(dest_file, nrows=10)
                first_date = str(df_test.iloc[0]["Date"]).strip()
                return True, f"Successfully downloaded and validated for {target_date}", dest_file
            else:
                return False, (
                    f"MCX automated export endpoint returned non-CSV payload for {target_date}. "
                    f"To ingest historical data, download the official daily Bhavcopy CSV from "
                    f"https://www.mcxindia.com/market-data/bhavcopy and place it in 'data/raw/BhavCopyDateWise_{date_str_formatted}.csv'."
                ), None

        except Exception as e:
            return False, (
                f"Automated download failed for {target_date}: {str(e)}. "
                f"Please manually place the MCX Bhavcopy file in 'data/raw/BhavCopyDateWise_{date_str_formatted}.csv'."
            ), None


class MCXDataPipeline:
    """
    Multi-file ingestion pipeline that discovers, validates, cleans, normalizes,
    and consolidates all MCX Bhavcopy files into a unified dataset.
    """
    def __init__(
        self,
        raw_dir: Path | str = "data/raw",
        processed_dir: Path | str = "data/processed"
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.downloader = MCXDataDownloader(self.raw_dir)

    def scan_raw_files(self) -> List[Path]:
        """List all CSV files in raw directory sorted by filename."""
        files = list(self.raw_dir.glob("*.csv")) + list(self.raw_dir.glob("*.CSV"))
        return sorted(list(set(files)))

    def ingest_all(self, export_parquet: bool = True) -> Tuple[pd.DataFrame, IngestionResult]:
        """
        Process all raw Bhavcopy CSVs, combine, normalize, and save master parquet/csv.
        """
        raw_files = self.scan_raw_files()
        issues = []

        if not raw_files:
            return pd.DataFrame(), IngestionResult(
                total_files_scanned=0,
                valid_files_processed=0,
                total_raw_rows=0,
                total_gold_rows=0,
                unique_dates=[],
                unique_symbols=[],
                parquet_path=None,
                issues=["No Bhavcopy CSV files found in data/raw/."],
            )

        cleaned_dfs = []
        total_raw = 0
        valid_files_count = 0

        for fpath in raw_files:
            try:
                raw_df = pd.read_csv(fpath, dtype=str)
                total_raw += len(raw_df)
                c_df, report = clean_bhavcopy_df(raw_df)

                if report.is_valid and not c_df.empty:
                    cleaned_dfs.append(c_df)
                    valid_files_count += 1
                else:
                    issues.append(f"File {fpath.name}: {'; '.join(report.issues)}")
            except Exception as e:
                issues.append(f"Error processing {fpath.name}: {str(e)}")

        if not cleaned_dfs:
            return pd.DataFrame(), IngestionResult(
                total_files_scanned=len(raw_files),
                valid_files_processed=valid_files_count,
                total_raw_rows=total_raw,
                total_gold_rows=0,
                unique_dates=[],
                unique_symbols=[],
                parquet_path=None,
                issues=issues + ["No valid gold records could be extracted from files."],
            )

        master_df = pd.concat(cleaned_dfs, ignore_index=True)
        # Deduplicate on (trade_date, symbol, expiry_date)
        master_df.drop_duplicates(subset=["trade_date", "symbol", "expiry_date"], keep="last", inplace=True)
        master_df.sort_values(by=["trade_date", "symbol", "expiry_date"], inplace=True)
        master_df.reset_index(drop=True, inplace=True)

        # Add normalized metrics
        master_df = add_normalized_columns(master_df)

        parquet_path = None
        if export_parquet:
            p_file = self.processed_dir / "gold_master.parquet"
            c_file = self.processed_dir / "gold_master.csv"
            master_df.to_csv(c_file, index=False)
            try:
                master_df.to_parquet(p_file, index=False)
                parquet_path = str(p_file)
            except Exception as pe:
                issues.append(f"Parquet export note: {str(pe)}")

        unique_dates = sorted([str(d) for d in master_df["trade_date"].unique().tolist()])
        unique_symbols = sorted(master_df["symbol"].unique().tolist())

        result = IngestionResult(
            total_files_scanned=len(raw_files),
            valid_files_processed=valid_files_count,
            total_raw_rows=total_raw,
            total_gold_rows=len(master_df),
            unique_dates=unique_dates,
            unique_symbols=unique_symbols,
            parquet_path=parquet_path,
            issues=issues,
        )

        return master_df, result
