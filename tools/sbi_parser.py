"""
SBI Parser - Data loading and normalization for SIX Bond Index data
"""

import pandas as pd
from pathlib import Path
from typing import Optional
import warnings

# Sektor-Code Mapping (aus SIX Rulebook Appendix B)
SECTOR_MAPPING = {
    # Public Sector (50000000)
    '51010100': 'Government',
    '51010200': 'Government (Sub)',
    '51510100': 'Supranational UN',
    '51515100': 'Supranational Org',
    '52010100': 'Agency Guaranteed',
    '52015100': 'Agency Not Guaranteed',
    '52510100': 'Local Authority Regional Guaranteed',
    '52515100': 'Local Authority Regional Not Guaranteed',
    '53010100': 'Local Authority Cities Guaranteed',
    '53015100': 'Local Authority Cities Not Guaranteed',
    # Securitized (60000000)
    '61010100': 'Pfandbrief',
    '61010200': 'Pfandbrief (Sub)',
    '61510100': 'Covered Bonds Public Sector',
    '61515100': 'Covered Bonds Mortgage',
    '62010100': 'Asset Backed',
    '62510100': 'GICs / Funding Agreement',
    # Corporate (70000000)
    '71010100': 'Energy',
    '71510100': 'Materials',
    '72010100': 'Industrials',
    '72015100': 'Non-retail services',
    '72020100': 'Transportation',
    '72510100': 'Automotive',
    '72515100': 'Retail goods',
    '73010100': 'Essential retail',
    '73015100': 'Food & Beverage',
    '73510100': 'Health Equipment',
    '73515100': 'Pharma & Biotech',
    '74010100': 'Banks',
    '74010200': 'Banks (Sub)',
    '74015100': 'Financial Services',
    '74020100': 'Insurance',
    '75010100': 'Telecom',
    '75510100': 'Utilities',
    '76010100': 'REITs',
    '76015100': 'Real Estate',
}

# Domicile Mapping
DOMICILE_MAPPING = {
    'DN': 'Domestic',
    'DG': 'Domestic Government',
    'FC': 'Foreign Corporate',
    'FG': 'Foreign Government',
    'FS': 'Foreign Supranational',
}


def parse_constituents(filepath: str, encoding: str = 'utf-8') -> pd.DataFrame:
    """
    Parse the SBI constituent file (e.g., close_sbr14d.csv)

    Returns DataFrame with columns:
    - date, isin, name, nominal, price, accrued_interest
    - ytw (yield to worst), duration, weight
    - sector_code, sector_name, rating, domicile
    - maturity_date, remaining_years
    """
    # Try different encodings
    encodings_to_try = [encoding, 'latin-1', 'cp1252', 'utf-8-sig']

    df = None
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(filepath, sep=';', encoding=enc)
            break
        except UnicodeDecodeError:
            continue

    if df is None:
        raise ValueError(f"Could not read file {filepath} with any encoding")

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # Rename key columns for consistency
    rename_map = {
        'date': 'date',
        'isin': 'isin',
        'instrument_name': 'name',
        'nominal_amount': 'nominal',
        'close_unadjusted_local': 'price',
        'accrued_interest': 'accrued_interest',
        'yield_to_worst': 'ytw',
        'duration_to_worst': 'duration',
        'weight': 'weight',
        'sbi_sector_code': 'sector_code',
        'composite_rating': 'rating',
        'domicile_code': 'domicile',
        'expiration_date': 'maturity_date',
        'remaining_of_maturity': 'remaining_years',
        'mcap_units_index_currency': 'market_cap',
    }

    # Only rename columns that exist
    existing_renames = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=existing_renames)

    # Convert date columns
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    if 'maturity_date' in df.columns:
        df['maturity_date'] = pd.to_datetime(df['maturity_date'])

    # Convert numeric columns
    numeric_cols = ['nominal', 'price', 'accrued_interest', 'ytw', 'duration',
                    'weight', 'remaining_years', 'market_cap']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Add sector name and level1 classification
    # The numeric sector codes are in 'guarantee_collateral_code' (e.g. 74010100)
    # 'sector_code' contains text labels (e.g. "Domestic Non Government")
    gcol = 'guarantee_collateral_code'
    if gcol in df.columns:
        df[gcol] = df[gcol].astype(str)
        df['sector_name'] = df[gcol].map(SECTOR_MAPPING).fillna(df.get('sector_code', 'Unknown'))

        df['sector_level1'] = df[gcol].apply(lambda x:
            'Public Sector' if str(x).startswith('5') else
            'Securitized' if str(x).startswith('6') else
            'Corporate' if str(x).startswith('7') else 'Unknown'
        )
    elif 'sector_code' in df.columns:
        df['sector_code'] = df['sector_code'].astype(str)
        df['sector_name'] = df['sector_code'].map(SECTOR_MAPPING).fillna('Unknown')
        df['sector_level1'] = df['sector_code'].apply(lambda x:
            'Public Sector' if str(x).startswith('5') else
            'Securitized' if str(x).startswith('6') else
            'Corporate' if str(x).startswith('7') else 'Unknown'
        )

    return df


def parse_forecast(filepath: str, encoding: str = 'utf-8') -> pd.DataFrame:
    """
    Parse the SBI forecast file (e.g., forecast_bonds.csv)

    Returns DataFrame with columns:
    - admission_date, cancellation_date, isin, coupon, name
    - maturity_date, sector, rating, nominal, comment
    """
    # Try different encodings
    encodings_to_try = [encoding, 'latin-1', 'cp1252', 'utf-8-sig']

    df = None
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(filepath, sep=';', encoding=enc)
            break
        except UnicodeDecodeError:
            continue

    if df is None:
        raise ValueError(f"Could not read file {filepath} with any encoding")

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # Rename key columns
    rename_map = {
        'admission_date': 'admission_date',
        'cancellation_date': 'cancellation_date',
        'isin': 'isin',
        'coupon': 'coupon',
        'name': 'name',
        'maturity_date': 'maturity_date',
        'sector': 'sector',
        'composite_rating': 'rating',
        'issue_amount_in_chf': 'nominal',
        'comment': 'comment',
        'capital_increase/decrease': 'capital_change',
        'guarantee_type_&_collateral': 'guarantee_code',
        'updated_on': 'updated_on',
    }

    existing_renames = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=existing_renames)

    # Convert date columns (handle European date format DD.MM.YYYY)
    date_cols = ['admission_date', 'cancellation_date', 'maturity_date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format='%d.%m.%Y', errors='coerce')

    if 'updated_on' in df.columns:
        df['updated_on'] = pd.to_datetime(df['updated_on'], errors='coerce')

    # Convert numeric columns
    if 'nominal' in df.columns:
        df['nominal'] = pd.to_numeric(df['nominal'], errors='coerce')
    if 'coupon' in df.columns:
        df['coupon'] = pd.to_numeric(df['coupon'], errors='coerce')
    if 'capital_change' in df.columns:
        df['capital_change'] = pd.to_numeric(df['capital_change'], errors='coerce')

    # Map sector codes to domicile
    if 'sector' in df.columns:
        df['domicile'] = df['sector'].map(DOMICILE_MAPPING).fillna(df['sector'])

    # Categorize by event type
    if 'comment' in df.columns:
        df['event_type'] = df['comment'].str.lower().apply(lambda x:
            'admission' if 'admission' in str(x) else
            'deletion' if 'deletion' in str(x) else
            'capital_increase' if 'capital increase' in str(x) else
            'capital_decrease' if 'capital decrease' in str(x) else
            'rating_change' if 'rating' in str(x) else
            'other'
        )

    return df


def load_historical_forecasts(folder: str) -> dict:
    """
    Load all historical forecast files from a folder

    Returns dict with keys like '202601' and DataFrames as values
    """
    folder_path = Path(folder)
    forecasts = {}

    for file in folder_path.glob('SBI_Forecast_*.csv'):
        # Extract YYYYMM from filename
        period = file.stem.replace('SBI_Forecast_', '')
        try:
            df = parse_forecast(str(file))
            forecasts[period] = df
        except Exception as e:
            warnings.warn(f"Could not load {file}: {e}")

    return forecasts


def get_admissions(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """Filter forecast for new admissions only"""
    if 'event_type' in forecast_df.columns:
        return forecast_df[forecast_df['event_type'] == 'admission'].copy()
    elif 'comment' in forecast_df.columns:
        return forecast_df[forecast_df['comment'].str.lower().str.contains('admission', na=False)].copy()
    return pd.DataFrame()


def get_deletions(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """Filter forecast for deletions only"""
    if 'event_type' in forecast_df.columns:
        return forecast_df[forecast_df['event_type'] == 'deletion'].copy()
    elif 'comment' in forecast_df.columns:
        return forecast_df[forecast_df['comment'].str.lower().str.contains('deletion', na=False)].copy()
    return pd.DataFrame()


def get_capital_changes(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """Filter forecast for capital increases/decreases"""
    if 'event_type' in forecast_df.columns:
        return forecast_df[forecast_df['event_type'].isin(['capital_increase', 'capital_decrease'])].copy()
    elif 'comment' in forecast_df.columns:
        return forecast_df[forecast_df['comment'].str.lower().str.contains('capital', na=False)].copy()
    return pd.DataFrame()


if __name__ == '__main__':
    # Test with sample files
    import sys

    base_path = Path(__file__).parent.parent

    # Test constituents
    const_file = base_path / 'close_sbr14d.csv'
    if const_file.exists():
        print("Testing constituents parser...")
        df = parse_constituents(str(const_file))
        print(f"  Loaded {len(df)} rows")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  Unique ISINs: {df['isin'].nunique()}")
        print()

    # Test forecast
    forecast_file = base_path / 'forecast_bonds.csv'
    if forecast_file.exists():
        print("Testing forecast parser...")
        df = parse_forecast(str(forecast_file))
        print(f"  Loaded {len(df)} rows")
        print(f"  Event types: {df['event_type'].value_counts().to_dict()}")
        print(f"  Admissions: {len(get_admissions(df))}")
        print(f"  Deletions: {len(get_deletions(df))}")
        print()

    # Test historical forecasts
    hist_folder = base_path / 'extracted'
    if hist_folder.exists():
        print("Testing historical forecasts...")
        forecasts = load_historical_forecasts(str(hist_folder))
        print(f"  Loaded {len(forecasts)} periods")
        print(f"  Periods: {sorted(forecasts.keys())[:5]}...{sorted(forecasts.keys())[-5:]}")
