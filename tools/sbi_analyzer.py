"""
SBI Analyzer - Index composition analysis for SBI rebalancing strategy
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta


# Helper to get the numeric sector code column
# In SIX data: 'guarantee_collateral_code' has numeric codes (e.g. 74010100)
# 'sector_code' has text labels (e.g. "Domestic Non Government")
def _get_sector_col(df: pd.DataFrame) -> str:
    """Return the column name containing numeric sector codes."""
    if 'guarantee_collateral_code' in df.columns:
        return 'guarantee_collateral_code'
    return 'sector_code'


def _sector_filter(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Filter by numeric sector code prefix (e.g. '7' for Corporate)."""
    col = _get_sector_col(df)
    return df[df[col].astype(str).str.startswith(prefix)]


# Segment definitions based on SIX taxonomy
# Numeric sector code prefixes:
#   51 = Government, 52/53 = Agency/Local Authority
#   61 = Pfandbrief/Covered Bonds, 62 = Asset-Backed
#   7x = Corporate (71=Energy, 72=Industrials, 73=Consumer, 74=Financials, 75=Telecom/Utilities, 76=Real Estate)
SEGMENTS = {
    'total': {
        'name': 'SBI AAA-BBB Total',
        'filter': lambda df: df,  # No filter - all bonds
    },
    'domestic': {
        'name': 'SBI Domestic',
        'filter': lambda df: df[df['domicile'] == 'CH'],
    },
    'foreign': {
        'name': 'SBI Foreign',
        'filter': lambda df: df[df['domicile'] != 'CH'],
    },
    'government': {
        'name': 'SBI Government',
        'filter': lambda df: _sector_filter(df, '51'),
    },
    'agency': {
        'name': 'SBI Agency / Local Authority',
        'filter': lambda df: pd.concat([_sector_filter(df, '52'), _sector_filter(df, '53')]).drop_duplicates(),
    },
    'pfandbrief': {
        'name': 'SBI Pfandbrief / Covered Bonds',
        'filter': lambda df: _sector_filter(df, '61'),
    },
    'corporate': {
        'name': 'SBI Corporate',
        'filter': lambda df: _sector_filter(df, '7'),
    },
    'financials': {
        'name': 'SBI Financials',
        'filter': lambda df: _sector_filter(df, '74'),
    },
    'domestic_corporate': {
        'name': 'SBI Domestic Corporate',
        'filter': lambda df: _sector_filter(df[df['domicile'] == 'CH'], '7'),
    },
    'foreign_corporate': {
        'name': 'SBI Foreign Corporate',
        'filter': lambda df: _sector_filter(df[df['domicile'] != 'CH'], '7'),
    },
}


def calc_index_mcap(df: pd.DataFrame) -> float:
    """
    Calculate total market cap of index in CHF
    Market Cap = Sum of (Nominal × Price / 100) for all bonds
    """
    if 'market_cap' in df.columns:
        return df['market_cap'].sum()
    elif 'nominal' in df.columns and 'price' in df.columns:
        return (df['nominal'] * df['price'] / 100).sum()
    else:
        raise ValueError("DataFrame must have 'market_cap' or 'nominal'+'price' columns")


def calc_segment_mcap(df: pd.DataFrame, segment: str = 'total') -> float:
    """Calculate market cap for a specific segment"""
    if segment not in SEGMENTS:
        raise ValueError(f"Unknown segment: {segment}. Available: {list(SEGMENTS.keys())}")

    filtered = SEGMENTS[segment]['filter'](df)
    return calc_index_mcap(filtered)


def calc_segment_duration(df: pd.DataFrame, segment: str = 'total') -> float:
    """
    Calculate market-cap-weighted duration for a segment
    Duration = Σ(Duration_i × MCap_i) / Σ(MCap_i)
    """
    if segment not in SEGMENTS:
        raise ValueError(f"Unknown segment: {segment}")

    filtered = SEGMENTS[segment]['filter'](df)

    if len(filtered) == 0:
        return 0.0

    if 'market_cap' in filtered.columns:
        mcap = filtered['market_cap']
    else:
        mcap = filtered['nominal'] * filtered['price'] / 100

    if 'duration' not in filtered.columns:
        raise ValueError("DataFrame must have 'duration' column")

    # Exclude bonds with missing or negative duration from both numerator and denominator
    valid = filtered['duration'].notna() & (filtered['duration'] >= 0)
    mcap = mcap[valid]
    dur = filtered.loc[valid, 'duration']

    total_mcap = mcap.sum()
    if total_mcap == 0:
        return 0.0

    return (dur * mcap).sum() / total_mcap


def calc_segment_yield(df: pd.DataFrame, segment: str = 'total') -> float:
    """Calculate market-cap-weighted yield for a segment"""
    if segment not in SEGMENTS:
        raise ValueError(f"Unknown segment: {segment}")

    filtered = SEGMENTS[segment]['filter'](df)

    if len(filtered) == 0:
        return 0.0

    if 'market_cap' in filtered.columns:
        mcap = filtered['market_cap']
    else:
        mcap = filtered['nominal'] * filtered['price'] / 100

    if 'ytw' not in filtered.columns:
        raise ValueError("DataFrame must have 'ytw' column")

    # Exclude bonds with missing yield from both numerator and denominator
    valid = filtered['ytw'].notna()
    mcap = mcap[valid]
    ytw = filtered.loc[valid, 'ytw']

    total_mcap = mcap.sum()
    if total_mcap == 0:
        return 0.0

    return (ytw * mcap).sum() / total_mcap


def calc_segment_stats(df: pd.DataFrame, segment: str = 'total') -> Dict:
    """Calculate comprehensive statistics for a segment"""
    if segment not in SEGMENTS:
        raise ValueError(f"Unknown segment: {segment}")

    filtered = SEGMENTS[segment]['filter'](df)

    if len(filtered) == 0:
        return {
            'name': SEGMENTS[segment]['name'],
            'count': 0,
            'market_cap': 0,
            'weight_pct': 0,
            'duration': 0,
            'yield': 0,
        }

    total_mcap = calc_index_mcap(df)
    segment_mcap = calc_index_mcap(filtered)

    return {
        'name': SEGMENTS[segment]['name'],
        'count': len(filtered),
        'market_cap': segment_mcap,
        'weight_pct': (segment_mcap / total_mcap * 100) if total_mcap > 0 else 0,
        'duration': calc_segment_duration(df, segment),
        'yield': calc_segment_yield(df, segment),
    }


def get_all_segment_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Get statistics for all segments"""
    stats = []
    for segment in SEGMENTS:
        stats.append(calc_segment_stats(df, segment))
    return pd.DataFrame(stats)


def calc_bond_weight(df: pd.DataFrame, isin: str) -> float:
    """Calculate weight of a specific bond in the index"""
    total_mcap = calc_index_mcap(df)
    if total_mcap == 0:
        return 0.0

    bond = df[df['isin'] == isin]
    if len(bond) == 0:
        return 0.0

    if 'market_cap' in bond.columns:
        bond_mcap = bond['market_cap'].iloc[0]
    else:
        bond_mcap = bond['nominal'].iloc[0] * bond['price'].iloc[0] / 100

    return bond_mcap / total_mcap


def identify_maturity_exits(df: pd.DataFrame, horizon_days: int = 30) -> pd.DataFrame:
    """
    Identify bonds that will exit the index due to residual term < 1 year

    Args:
        df: Constituent DataFrame
        horizon_days: Look-ahead period in days

    Returns:
        DataFrame of bonds exiting within horizon
    """
    if 'remaining_years' not in df.columns:
        raise ValueError("DataFrame must have 'remaining_years' column")

    # Bonds with less than 1 year + horizon_days remaining will exit
    threshold_years = 1 + horizon_days / 365

    exits = df[df['remaining_years'] <= threshold_years].copy()

    # Sort by remaining time
    exits = exits.sort_values('remaining_years')

    return exits


def identify_maturity_bucket_changes(df: pd.DataFrame, horizon_days: int = 30) -> Dict[str, pd.DataFrame]:
    """
    Identify bonds that will move between maturity buckets

    Buckets: 1-3y, 3-5y, 5-7y, 7-10y, 10+y
    """
    bucket_limits = [1, 3, 5, 7, 10, float('inf')]
    bucket_names = ['1-3y', '3-5y', '5-7y', '7-10y', '10+y']

    horizon_years = horizon_days / 365

    changes = {}

    for i, (lower, upper) in enumerate(zip(bucket_limits[:-1], bucket_limits[1:])):
        bucket_name = bucket_names[i]

        # Bonds currently in this bucket
        in_bucket = df[(df['remaining_years'] >= lower) & (df['remaining_years'] < upper)]

        # Bonds that will move out of this bucket (remaining - horizon < lower bound)
        moving_out = in_bucket[in_bucket['remaining_years'] - horizon_years < lower]

        if len(moving_out) > 0:
            changes[f'exit_{bucket_name}'] = moving_out

    return changes


def compare_snapshots(df_old: pd.DataFrame, df_new: pd.DataFrame) -> Dict:
    """
    Compare two constituent snapshots to identify changes

    Returns dict with:
    - additions: Bonds in new but not in old
    - deletions: Bonds in old but not in new
    - duration_change: Change in weighted duration
    - mcap_change: Change in total market cap
    """
    old_isins = set(df_old['isin'])
    new_isins = set(df_new['isin'])

    additions = df_new[df_new['isin'].isin(new_isins - old_isins)]
    deletions = df_old[df_old['isin'].isin(old_isins - new_isins)]

    old_duration = calc_segment_duration(df_old, 'total')
    new_duration = calc_segment_duration(df_new, 'total')

    old_mcap = calc_index_mcap(df_old)
    new_mcap = calc_index_mcap(df_new)

    return {
        'additions': additions,
        'deletions': deletions,
        'additions_count': len(additions),
        'deletions_count': len(deletions),
        'duration_old': old_duration,
        'duration_new': new_duration,
        'duration_change': new_duration - old_duration,
        'mcap_old': old_mcap,
        'mcap_new': new_mcap,
        'mcap_change': new_mcap - old_mcap,
        'mcap_change_pct': (new_mcap - old_mcap) / old_mcap * 100 if old_mcap > 0 else 0,
    }


def get_top_bonds_by_weight(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Get top N bonds by weight in the index"""
    total_mcap = calc_index_mcap(df)

    result = df.copy()
    if 'market_cap' in result.columns:
        result['weight_pct'] = result['market_cap'] / total_mcap * 100
    else:
        result['weight_pct'] = (result['nominal'] * result['price'] / 100) / total_mcap * 100

    return result.nlargest(n, 'weight_pct')[['isin', 'name', 'weight_pct', 'duration', 'rating', 'sector_name']]


def get_rating_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Get market cap breakdown by rating"""
    if 'rating' not in df.columns:
        raise ValueError("DataFrame must have 'rating' column")

    total_mcap = calc_index_mcap(df)

    breakdown = []
    for rating in ['AAA', 'AA', 'A', 'BBB']:
        filtered = df[df['rating'] == rating]
        if len(filtered) > 0:
            mcap = calc_index_mcap(filtered)
            breakdown.append({
                'rating': rating,
                'count': len(filtered),
                'market_cap': mcap,
                'weight_pct': mcap / total_mcap * 100 if total_mcap > 0 else 0,
                'duration': calc_segment_duration(filtered, 'total') if len(filtered) > 0 else 0,
            })

    return pd.DataFrame(breakdown)


if __name__ == '__main__':
    # Test with actual data
    from sbi_parser import parse_constituents
    from pathlib import Path

    base_path = Path(__file__).parent.parent
    const_file = base_path / 'close_sbr14d.csv'

    if const_file.exists():
        print("Loading constituents...")
        df = parse_constituents(str(const_file))

        print(f"\n=== SBI Index Overview ===")
        print(f"Date: {df['date'].iloc[0]}")
        print(f"Total Bonds: {len(df)}")
        print(f"Total Market Cap: CHF {calc_index_mcap(df)/1e9:.2f} Mrd")
        print(f"Weighted Duration: {calc_segment_duration(df, 'total'):.2f} years")
        print(f"Weighted Yield: {calc_segment_yield(df, 'total'):.3f}%")

        print(f"\n=== Segment Breakdown ===")
        stats = get_all_segment_stats(df)
        for _, row in stats.iterrows():
            if row['count'] > 0:
                print(f"{row['name']}: {row['count']} bonds, "
                      f"{row['weight_pct']:.1f}% weight, "
                      f"Duration {row['duration']:.2f}y")

        print(f"\n=== Rating Breakdown ===")
        ratings = get_rating_breakdown(df)
        for _, row in ratings.iterrows():
            print(f"{row['rating']}: {row['count']} bonds, "
                  f"{row['weight_pct']:.1f}% weight, "
                  f"Duration {row['duration']:.2f}y")

        print(f"\n=== Top 10 Bonds by Weight ===")
        top = get_top_bonds_by_weight(df, 10)
        for _, row in top.iterrows():
            print(f"{row['isin']}: {row['weight_pct']:.2f}% - {row['name'][:40]}")

        print(f"\n=== Bonds Exiting (Residual < 1.5y) ===")
        exits = identify_maturity_exits(df, horizon_days=180)
        print(f"Found {len(exits)} bonds exiting within 6 months")
        if len(exits) > 0:
            for _, row in exits.head(5).iterrows():
                print(f"  {row['isin']}: {row['remaining_years']:.2f}y remaining - {row['name'][:30]}")
