"""
SIX Delayed Publication Bonds Downloader

Downloads delayed publication bond data from SIX Group.
These files contain details of OTC bond trades from the previous trading day.

URL Pattern: https://www.six-group.com/exchanges/dwh_download/delayed_publication/delayed_publication_YYYYMMDD.csv
"""

import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
import time


BASE_URL = "https://www.six-group.com/exchanges/dwh_download/delayed_publication/delayed_publication_{date}.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "delayed_publications"


def download_single_file(date: datetime, output_dir: Path = DEFAULT_OUTPUT_DIR,
                         skip_existing: bool = True) -> Tuple[bool, str]:
    """
    Download delayed publication file for a specific date.

    Args:
        date: The date to download
        output_dir: Directory to save the file
        skip_existing: If True, skip download if file already exists

    Returns:
        Tuple of (success: bool, message: str)
    """
    date_str = date.strftime("%Y%m%d")
    filename = f"delayed_publication_{date_str}.csv"
    filepath = output_dir / filename

    if skip_existing and filepath.exists():
        return True, f"Skipped (exists): {filename}"

    url = BASE_URL.format(date=date_str)

    try:
        response = requests.get(url, timeout=30)

        if response.status_code == 200:
            # Check if we got actual CSV content (not an error page)
            content = response.content  # Use bytes, not text
            if content.strip() and not content.startswith(b'<!DOCTYPE'):
                filepath.write_bytes(content)  # Write raw bytes to preserve encoding
                return True, f"Downloaded: {filename}"
            else:
                return False, f"No data for: {date_str}"
        elif response.status_code == 404:
            return False, f"Not found: {date_str} (likely weekend/holiday)"
        else:
            return False, f"HTTP {response.status_code} for: {date_str}"

    except requests.RequestException as e:
        return False, f"Error for {date_str}: {str(e)}"


def download_date_range(start_date: datetime, end_date: datetime,
                        output_dir: Path = DEFAULT_OUTPUT_DIR,
                        skip_existing: bool = True,
                        delay_seconds: float = 0.5) -> dict:
    """
    Download delayed publication files for a date range.

    Args:
        start_date: Start of the date range
        end_date: End of the date range
        output_dir: Directory to save files
        skip_existing: If True, skip files that already exist
        delay_seconds: Delay between requests to be polite to the server

    Returns:
        Dictionary with 'downloaded', 'skipped', 'failed' counts and 'errors' list
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'downloaded': 0,
        'skipped': 0,
        'failed': 0,
        'errors': []
    }

    current_date = start_date
    total_days = (end_date - start_date).days + 1
    processed = 0

    print(f"Downloading delayed publications from {start_date.date()} to {end_date.date()}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    print("-" * 60, flush=True)

    while current_date <= end_date:
        processed += 1
        success, message = download_single_file(current_date, output_dir, skip_existing)

        if success:
            if "Skipped" in message:
                results['skipped'] += 1
            else:
                results['downloaded'] += 1
                print(f"[{processed}/{total_days}] {message}", flush=True)
        else:
            results['failed'] += 1
            if "Not found" not in message:  # Don't log expected 404s for weekends
                results['errors'].append(message)

        current_date += timedelta(days=1)

        # Be polite to the server
        if delay_seconds > 0:
            time.sleep(delay_seconds)

    print("-" * 60)
    print(f"Complete! Downloaded: {results['downloaded']}, Skipped: {results['skipped']}, Failed/Missing: {results['failed']}")

    return results


def download_all_available(output_dir: Path = DEFAULT_OUTPUT_DIR,
                           start_year: int = 2020,
                           skip_existing: bool = True) -> dict:
    """
    Download all available delayed publication files from start_year to today.

    Args:
        output_dir: Directory to save files
        start_year: Year to start downloading from
        skip_existing: If True, skip files that already exist

    Returns:
        Dictionary with download statistics
    """
    start_date = datetime(start_year, 1, 1)
    end_date = datetime.now()

    return download_date_range(start_date, end_date, output_dir, skip_existing)


def parse_delayed_publication(filepath: Path, encoding: str = 'utf-8') -> Optional[pd.DataFrame]:
    """
    Parse a delayed publication CSV file.

    Args:
        filepath: Path to the CSV file
        encoding: File encoding

    Returns:
        DataFrame with parsed data, or None if parsing fails
    """
    try:
        # Use utf-8-sig to handle BOM (Byte Order Mark)
        df = pd.read_csv(filepath, sep=';', encoding='utf-8-sig')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(filepath, sep=';', encoding='latin-1')
        except Exception as e:
            print(f"Failed to parse {filepath}: {e}")
            return None
    except Exception as e:
        print(f"Failed to parse {filepath}: {e}")
        return None

    # Normalize column names (remove BOM and clean up)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    # Remove BOM character from first column if present
    df.columns = [col.replace('\ufeff', '').replace('ï»¿', '') for col in df.columns]

    return df


def calculate_yield_from_price(price: float, coupon: float, remaining_years: float) -> Optional[float]:
    """
    Calculate approximate yield-to-maturity from trade price.

    Uses simplified YTM approximation formula:
    YTM ≈ (Coupon + (100 - Price) / Years) / ((100 + Price) / 2)

    Args:
        price: Clean price as percentage (e.g., 95.5 for 95.5%)
        coupon: Annual coupon rate (e.g., 2.5 for 2.5%)
        remaining_years: Years until maturity

    Returns:
        Approximate YTM as percentage, or None if calculation not possible
    """
    if pd.isna(price) or pd.isna(coupon) or pd.isna(remaining_years):
        return None
    if remaining_years <= 0 or price <= 0:
        return None

    try:
        # Simplified YTM approximation
        annual_capital_gain = (100 - price) / remaining_years
        average_price = (100 + price) / 2
        ytm = (coupon + annual_capital_gain) / average_price * 100
        return round(ytm, 3)
    except:
        return None


def enrich_with_constituents(delayed_df: pd.DataFrame,
                              constituents_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich delayed publication data with segment/rating info from constituents.
    Also calculates yield based on trade price.

    Args:
        delayed_df: DataFrame with delayed publication trades
        constituents_df: DataFrame with index constituents (from sbi_parser)

    Returns:
        Enriched DataFrame with sector_code, rating, domicile, segment_level1, trade_yield
    """
    if delayed_df.empty or constituents_df.empty:
        return delayed_df

    # Select relevant columns from constituents (including coupon and remaining_years for yield calc)
    const_cols = ['isin', 'name', 'sector_code', 'rating', 'domicile', 'nominal', 'duration',
                  'coupon', 'remaining_years', 'maturity_date']
    available_cols = [c for c in const_cols if c in constituents_df.columns]

    # Join on ISIN
    enriched = delayed_df.merge(
        constituents_df[available_cols],
        left_on='product_isin',
        right_on='isin',
        how='left',
        suffixes=('', '_const')
    )

    # Add segment hierarchy (Domestic vs Foreign)
    def get_segment_level1(sector):
        if pd.isna(sector):
            return 'Unknown'
        sector_str = str(sector)
        if 'Swiss' in sector_str or 'Domestic' in sector_str:
            return 'Domestic'
        elif 'Foreign' in sector_str:
            return 'Foreign'
        return 'Other'

    enriched['segment_level1'] = enriched['sector_code'].apply(get_segment_level1)

    # Mark if bond is in index
    enriched['in_index'] = enriched['isin'].notna()

    # Calculate yield based on trade price
    if 'trade_price' in enriched.columns and 'coupon' in enriched.columns and 'remaining_years' in enriched.columns:
        enriched['trade_yield'] = enriched.apply(
            lambda row: calculate_yield_from_price(
                row.get('trade_price'),
                row.get('coupon'),
                row.get('remaining_years')
            ),
            axis=1
        )
    else:
        enriched['trade_yield'] = None

    return enriched


def load_all_delayed_publications(folder: Path = DEFAULT_OUTPUT_DIR) -> pd.DataFrame:
    """
    Load all delayed publication files from a folder into a single DataFrame.

    Args:
        folder: Folder containing the CSV files

    Returns:
        Combined DataFrame with all data
    """
    all_files = sorted(folder.glob("delayed_publication_*.csv"))

    if not all_files:
        print(f"No files found in {folder}")
        return pd.DataFrame()

    dfs = []
    for filepath in all_files:
        df = parse_delayed_publication(filepath)
        if df is not None and not df.empty:
            # Extract date from filename
            date_str = filepath.stem.replace("delayed_publication_", "")
            df['file_date'] = pd.to_datetime(date_str, format='%Y%m%d')
            dfs.append(df)

    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(dfs)} files with {len(combined)} total records")
        return combined

    return pd.DataFrame()


if __name__ == '__main__':
    # Download files from the last 30 days (older data is not available on Six servers)
    print("SIX Delayed Publication Downloader")
    print("=" * 60)

    # Six only keeps data for approximately 2-3 weeks
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()

    results = download_date_range(
        start_date=start_date,
        end_date=end_date,
        output_dir=DEFAULT_OUTPUT_DIR,
        skip_existing=True,
        delay_seconds=0.3  # Faster since we have fewer files
    )

    if results['errors']:
        print("\nErrors encountered:")
        for error in results['errors'][:10]:  # Show first 10 errors
            print(f"  - {error}")
