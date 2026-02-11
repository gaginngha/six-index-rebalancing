"""
SBI Flow Estimator - Quantify expected flows from index rebalancing

Based on SIX Rulebook v2.30:
- Index is market-cap weighted (Nominal × Price)
- Rebalancing effective on first trading day of month
- Cut-off date: 20th of month
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime

from sbi_parser import parse_constituents, parse_forecast, get_admissions, get_deletions, get_capital_changes
from sbi_analyzer import calc_index_mcap, calc_bond_weight, calc_segment_duration, SEGMENTS


# Default AUM assumptions (in CHF)
DEFAULT_AUM = {
    'sbi_aaa_bbb_total': 3_000_000_000,    # Main SBI tracker ~3 Mrd
    'sbi_corporate': 500_000_000,           # Corporate-only trackers
    'sbi_foreign': 800_000_000,             # Foreign segment
    'sbi_domestic': 2_000_000_000,          # Domestic focused
}


def load_config(config_path: Optional[str] = None) -> Dict:
    """Load AUM configuration from JSON file"""
    if config_path is None:
        config_path = Path(__file__).parent.parent / 'config.json'
    else:
        config_path = Path(config_path)

    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return DEFAULT_AUM


def estimate_admission_flow(
    bond_nominal: float,
    bond_price: float,
    constituents_df: pd.DataFrame,
    aum: float
) -> Dict:
    """
    Estimate flow from a new bond admission

    Flow = AUM × new_weight
    new_weight = bond_mcap / (total_mcap + bond_mcap)
    """
    bond_mcap = bond_nominal * bond_price / 100  # Price is in %
    current_mcap = calc_index_mcap(constituents_df)
    new_total_mcap = current_mcap + bond_mcap

    new_weight = bond_mcap / new_total_mcap
    flow = aum * new_weight

    return {
        'bond_mcap': bond_mcap,
        'new_weight_pct': new_weight * 100,
        'flow_chf': flow,
        'flow_direction': 'BUY',
    }


def estimate_deletion_flow(
    isin: str,
    constituents_df: pd.DataFrame,
    aum: float
) -> Dict:
    """
    Estimate flow from a bond deletion

    Flow = AUM × current_weight
    """
    current_weight = calc_bond_weight(constituents_df, isin)

    bond = constituents_df[constituents_df['isin'] == isin]
    if len(bond) == 0:
        return {
            'bond_mcap': 0,
            'current_weight_pct': 0,
            'flow_chf': 0,
            'flow_direction': 'SELL',
            'error': f'Bond {isin} not found in constituents',
        }

    if 'market_cap' in bond.columns:
        bond_mcap = bond['market_cap'].iloc[0]
    else:
        bond_mcap = bond['nominal'].iloc[0] * bond['price'].iloc[0] / 100

    flow = aum * current_weight

    return {
        'bond_mcap': bond_mcap,
        'current_weight_pct': current_weight * 100,
        'flow_chf': flow,
        'flow_direction': 'SELL',
    }


def estimate_capital_change_flow(
    isin: str,
    delta_nominal: float,
    constituents_df: pd.DataFrame,
    aum: float
) -> Dict:
    """
    Estimate flow from capital increase/decrease

    Flow = AUM × Δ(weight)
    """
    bond = constituents_df[constituents_df['isin'] == isin]
    if len(bond) == 0:
        return {
            'error': f'Bond {isin} not found in constituents',
            'flow_chf': 0,
        }

    current_mcap = calc_index_mcap(constituents_df)
    current_weight = calc_bond_weight(constituents_df, isin)

    # Get bond price
    price = bond['price'].iloc[0]

    # Calculate new weight
    delta_mcap = delta_nominal * price / 100
    new_total_mcap = current_mcap + delta_mcap

    if 'market_cap' in bond.columns:
        old_bond_mcap = bond['market_cap'].iloc[0]
    else:
        old_bond_mcap = bond['nominal'].iloc[0] * price / 100

    new_bond_mcap = old_bond_mcap + delta_mcap
    new_weight = new_bond_mcap / new_total_mcap

    delta_weight = new_weight - current_weight
    flow = aum * delta_weight

    return {
        'old_mcap': old_bond_mcap,
        'new_mcap': new_bond_mcap,
        'delta_mcap': delta_mcap,
        'old_weight_pct': current_weight * 100,
        'new_weight_pct': new_weight * 100,
        'delta_weight_pct': delta_weight * 100,
        'flow_chf': flow,
        'flow_direction': 'BUY' if delta_weight > 0 else 'SELL',
    }


def generate_flow_report(
    forecast_df: pd.DataFrame,
    constituents_df: pd.DataFrame,
    aum_config: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Generate comprehensive flow report for all forecast events

    Returns DataFrame with estimated flows for each event
    """
    if aum_config is None:
        aum_config = load_config()

    # Use total AUM as default
    aum = aum_config.get('sbi_aaa_bbb_total', DEFAULT_AUM['sbi_aaa_bbb_total'])

    results = []

    # Process admissions
    admissions = get_admissions(forecast_df)
    for _, row in admissions.iterrows():
        # Use par (100) as price estimate for new bonds
        price_estimate = 100
        flow_data = estimate_admission_flow(
            row['nominal'], price_estimate, constituents_df, aum
        )

        results.append({
            'isin': row['isin'],
            'name': row.get('name', 'Unknown'),
            'event_type': 'ADMISSION',
            'admission_date': row.get('admission_date'),
            'nominal': row['nominal'],
            'rating': row.get('rating', 'Unknown'),
            'sector': row.get('sector', 'Unknown'),
            'est_weight_pct': flow_data['new_weight_pct'],
            'flow_chf': flow_data['flow_chf'],
            'flow_direction': flow_data['flow_direction'],
        })

    # Process deletions
    deletions = get_deletions(forecast_df)
    for _, row in deletions.iterrows():
        flow_data = estimate_deletion_flow(row['isin'], constituents_df, aum)

        results.append({
            'isin': row['isin'],
            'name': row.get('name', 'Unknown'),
            'event_type': 'DELETION',
            'cancellation_date': row.get('cancellation_date'),
            'nominal': row.get('nominal', 0),
            'rating': row.get('rating', 'Unknown'),
            'sector': row.get('sector', 'Unknown'),
            'est_weight_pct': flow_data.get('current_weight_pct', 0),
            'flow_chf': flow_data['flow_chf'],
            'flow_direction': flow_data['flow_direction'],
        })

    # Process capital changes
    cap_changes = get_capital_changes(forecast_df)
    for _, row in cap_changes.iterrows():
        delta = row.get('capital_change', 0)
        if pd.isna(delta) or delta == 0:
            continue

        flow_data = estimate_capital_change_flow(
            row['isin'], delta, constituents_df, aum
        )

        results.append({
            'isin': row['isin'],
            'name': row.get('name', 'Unknown'),
            'event_type': 'CAPITAL_CHANGE',
            'admission_date': row.get('admission_date'),
            'nominal': row.get('nominal', 0),
            'capital_change': delta,
            'rating': row.get('rating', 'Unknown'),
            'sector': row.get('sector', 'Unknown'),
            'est_weight_pct': flow_data.get('new_weight_pct', 0),
            'delta_weight_pct': flow_data.get('delta_weight_pct', 0),
            'flow_chf': flow_data.get('flow_chf', 0),
            'flow_direction': flow_data.get('flow_direction', 'N/A'),
        })

    return pd.DataFrame(results)


def estimate_duration_from_maturity(maturity_years: float, coupon: float = 0.5) -> float:
    """
    Estimate modified duration from maturity for bonds without duration data.

    Uses approximation: Duration ≈ Maturity × (1 - coupon_factor)
    For low-coupon bonds (typical CHF market): Duration ≈ 0.92 × Maturity
    """
    if pd.isna(maturity_years) or maturity_years <= 0:
        return 0.0

    # Low coupon approximation (CHF bonds typically have low coupons)
    coupon_factor = 0.08  # ~8% discount from maturity
    return maturity_years * (1 - coupon_factor)


def calculate_duration_impact(
    forecast_df: pd.DataFrame,
    constituents_df: pd.DataFrame,
    segment: str = 'total'
) -> Dict:
    """
    Calculate the duration impact from rebalancing for a given segment.

    Returns dict with:
    - current_duration: Current weighted duration
    - new_duration: Estimated duration after rebalancing
    - duration_change: Delta in years
    - duration_change_bps: Rough estimate of price impact (duration × 100bps rate move)
    """
    from datetime import datetime

    # Get segment filter
    if segment not in SEGMENTS:
        segment = 'total'

    segment_filter = SEGMENTS[segment]['filter']
    filtered_const = segment_filter(constituents_df)

    if len(filtered_const) == 0:
        return {
            'segment': segment,
            'segment_name': SEGMENTS[segment]['name'],
            'current_duration': 0,
            'new_duration': 0,
            'duration_change': 0,
            'error': 'No bonds in segment'
        }

    # Current duration and mcap
    current_duration = calc_segment_duration(constituents_df, segment)
    current_mcap = calc_index_mcap(filtered_const)

    # Calculate contribution from deletions (will be removed)
    deletions = get_deletions(forecast_df)
    deletion_duration_contribution = 0.0
    deletion_mcap = 0.0
    segment_deletion_count = 0

    for _, row in deletions.iterrows():
        isin = row['isin']
        bond = filtered_const[filtered_const['isin'] == isin]
        if len(bond) > 0:
            segment_deletion_count += 1
            bond_dur = bond['duration'].iloc[0] if 'duration' in bond.columns else 0
            if 'market_cap' in bond.columns:
                bond_mcap = bond['market_cap'].iloc[0]
            else:
                bond_mcap = bond['nominal'].iloc[0] * bond['price'].iloc[0] / 100
            deletion_duration_contribution += bond_dur * bond_mcap
            deletion_mcap += bond_mcap

    # Calculate contribution from admissions (will be added)
    admissions = get_admissions(forecast_df)
    admission_duration_contribution = 0.0
    admission_mcap = 0.0
    segment_admission_count = 0

    for _, row in admissions.iterrows():
        # Check if this bond belongs to the segment
        # guarantee_code contains the sector code (e.g., 52510100, 75510100)
        sector_code = str(row.get('guarantee_code', ''))
        # sector/domicile contains DN, FC, FG, FS, DG
        sector_type = str(row.get('sector', ''))
        domicile = str(row.get('domicile', sector_type))

        # Segment matching based on sector codes and domicile
        include_in_segment = True
        if segment == 'domestic':
            # DN = Domestic, DG = Domestic Government
            include_in_segment = sector_type in ['DN', 'DG'] or domicile in ['Domestic', 'Domestic Government']
        elif segment == 'foreign':
            # FC = Foreign Corporate, FG = Foreign Gov, FS = Foreign Supranational
            include_in_segment = sector_type in ['FC', 'FG', 'FS'] or domicile in ['Foreign Corporate', 'Foreign Government', 'Foreign Supranational']
        elif segment == 'corporate':
            # 7x = Corporate sector codes
            include_in_segment = sector_code.startswith('7')
        elif segment == 'pfandbrief':
            # 61 = Pfandbrief/Covered Bonds
            include_in_segment = sector_code.startswith('61')
        elif segment == 'government':
            # 51 = Government
            include_in_segment = sector_code.startswith('51')

        if include_in_segment:
            segment_admission_count += 1
            # Estimate duration from maturity
            maturity_date = row.get('maturity_date')
            if pd.notna(maturity_date):
                if isinstance(maturity_date, str):
                    try:
                        maturity_date = pd.to_datetime(maturity_date)
                    except:
                        maturity_date = None

                if maturity_date is not None:
                    years_to_maturity = (maturity_date - pd.Timestamp.now()).days / 365
                    est_duration = estimate_duration_from_maturity(years_to_maturity, row.get('coupon', 0.5))
                else:
                    est_duration = 5.0  # Default assumption
            else:
                est_duration = 5.0

            # Estimate market cap (assume price = 100)
            nominal = row.get('nominal', 0) or 0
            bond_mcap = nominal  # Price at par

            admission_duration_contribution += est_duration * bond_mcap
            admission_mcap += bond_mcap

    # Calculate new duration
    # New Duration = (Old_Dur × Old_MCap - Del_Dur_Contrib + Adm_Dur_Contrib) / New_MCap
    new_mcap = current_mcap - deletion_mcap + admission_mcap

    if new_mcap > 0:
        old_duration_contribution = current_duration * current_mcap
        new_duration_contribution = old_duration_contribution - deletion_duration_contribution + admission_duration_contribution
        new_duration = new_duration_contribution / new_mcap
    else:
        new_duration = current_duration

    duration_change = new_duration - current_duration

    return {
        'segment': segment,
        'segment_name': SEGMENTS[segment]['name'],
        'current_duration': round(current_duration, 3),
        'new_duration': round(new_duration, 3),
        'duration_change': round(duration_change, 3),
        'duration_change_bps': round(duration_change * 100, 1),  # Price impact per 100bps rate move
        'deletions_count': segment_deletion_count,
        'admissions_count': segment_admission_count,
        'deletion_mcap': deletion_mcap,
        'admission_mcap': admission_mcap,
        'current_mcap': current_mcap,
        'new_mcap': new_mcap,
    }


def get_all_segment_duration_impact(
    forecast_df: pd.DataFrame,
    constituents_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate duration impact for all major segments.

    Returns DataFrame with duration changes per segment.
    """
    segments_to_analyze = ['total', 'domestic', 'foreign', 'corporate', 'pfandbrief', 'government']

    results = []
    for segment in segments_to_analyze:
        try:
            impact = calculate_duration_impact(forecast_df, constituents_df, segment)
            results.append(impact)
        except Exception as e:
            results.append({
                'segment': segment,
                'segment_name': SEGMENTS.get(segment, {}).get('name', segment),
                'current_duration': 0,
                'new_duration': 0,
                'duration_change': 0,
                'error': str(e)
            })

    return pd.DataFrame(results)


def _match_forecast_to_segment(row: pd.Series, segment: str) -> bool:
    """Check if a forecast row belongs to a given segment."""
    if segment == 'total':
        return True

    sector_code = str(row.get('guarantee_code', ''))
    sector_type = str(row.get('sector', ''))

    if segment == 'domestic':
        return sector_type in ['DN', 'DG']
    elif segment == 'foreign':
        return sector_type in ['FC', 'FG', 'FS']
    elif segment == 'corporate':
        return sector_code.startswith('7')
    elif segment == 'pfandbrief':
        return sector_code.startswith('61')
    elif segment == 'government':
        return sector_code.startswith('51')
    elif segment == 'agency':
        return sector_code.startswith('52') or sector_code.startswith('53')
    elif segment == 'financials':
        return sector_code.startswith('74')
    elif segment == 'domestic_corporate':
        return sector_type in ['DN', 'DG'] and sector_code.startswith('7')
    elif segment == 'foreign_corporate':
        return sector_type in ['FC', 'FG', 'FS'] and sector_code.startswith('7')
    return True


def get_duration_detail_by_segment(
    forecast_df: pd.DataFrame,
    constituents_df: pd.DataFrame,
    segment: str = 'total'
) -> pd.DataFrame:
    """
    Return per-bond duration detail for a segment.

    Shows each admission/deletion with its estimated duration,
    nominal, MCap, and duration contribution to the segment.
    """
    if segment not in SEGMENTS:
        segment = 'total'

    segment_filter = SEGMENTS[segment]['filter']
    filtered_const = segment_filter(constituents_df)
    current_mcap = calc_index_mcap(filtered_const) if len(filtered_const) > 0 else 0

    rows = []

    # Deletions
    deletions = get_deletions(forecast_df)
    for _, row in deletions.iterrows():
        isin = row['isin']
        bond = filtered_const[filtered_const['isin'] == isin]
        if len(bond) > 0:
            bond_dur = bond['duration'].iloc[0] if 'duration' in bond.columns else 0
            if 'market_cap' in bond.columns:
                bond_mcap = bond['market_cap'].iloc[0]
            else:
                bond_mcap = bond['nominal'].iloc[0] * bond['price'].iloc[0] / 100
            nominal = bond['nominal'].iloc[0] if 'nominal' in bond.columns else 0

            rows.append({
                'event': 'DELETION',
                'isin': isin,
                'name': row.get('name', bond['name'].iloc[0] if 'name' in bond.columns else ''),
                'rating': row.get('rating', bond.get('rating', pd.Series([''])).iloc[0] if 'rating' in bond.columns else ''),
                'nominal_mio': round(nominal / 1e6, 0),
                'duration': round(bond_dur, 2),
                'mcap_mio': round(bond_mcap / 1e6, 1),
                'weight_pct': round(bond_mcap / current_mcap * 100, 4) if current_mcap > 0 else 0,
                'duration_source': 'Index',
            })

    # Admissions
    admissions = get_admissions(forecast_df)
    for _, row in admissions.iterrows():
        if not _match_forecast_to_segment(row, segment):
            continue

        maturity_date = row.get('maturity_date')
        if pd.notna(maturity_date):
            if isinstance(maturity_date, str):
                try:
                    maturity_date = pd.to_datetime(maturity_date)
                except:
                    maturity_date = None
            if maturity_date is not None:
                years_to_maturity = (maturity_date - pd.Timestamp.now()).days / 365
                est_duration = estimate_duration_from_maturity(years_to_maturity, row.get('coupon', 0.5))
            else:
                est_duration = 5.0
        else:
            est_duration = 5.0

        nominal = row.get('nominal', 0) or 0
        bond_mcap = nominal  # at par

        rows.append({
            'event': 'ADMISSION',
            'isin': row['isin'],
            'name': row.get('name', ''),
            'rating': row.get('rating', ''),
            'nominal_mio': round(nominal / 1e6, 0),
            'duration': round(est_duration, 2),
            'mcap_mio': round(bond_mcap / 1e6, 1),
            'weight_pct': round(bond_mcap / current_mcap * 100, 4) if current_mcap > 0 else 0,
            'duration_source': 'Geschätzt',
        })

    return pd.DataFrame(rows)


def simulate_custom_bond_impact(
    constituents_df: pd.DataFrame,
    bond_nominal: float,
    bond_duration: float,
    bond_rating: str = 'A',
    event_type: str = 'ADMISSION',
    segment: str = 'total'
) -> Dict:
    """
    Simulate the impact of a hypothetical bond on the index.

    Args:
        constituents_df: Current index constituents
        bond_nominal: Bond nominal in CHF
        bond_duration: Bond duration in years
        bond_rating: Rating (AAA, AA, A, BBB)
        event_type: 'ADMISSION' or 'DELETION'
        segment: Which segment to analyze

    Returns:
        Dict with impact metrics
    """
    if segment not in SEGMENTS:
        segment = 'total'

    filtered = SEGMENTS[segment]['filter'](constituents_df)
    if len(filtered) == 0:
        return {'error': 'No bonds in segment'}

    current_mcap = calc_index_mcap(filtered)
    current_duration = calc_segment_duration(constituents_df, segment)
    current_count = len(filtered)

    # Assume price at par for the custom bond
    bond_mcap = bond_nominal
    bond_dur_contribution = bond_duration * bond_mcap

    if event_type == 'ADMISSION':
        new_mcap = current_mcap + bond_mcap
        new_count = current_count + 1
        new_dur_contribution = current_duration * current_mcap + bond_dur_contribution
    else:  # DELETION
        new_mcap = current_mcap - bond_mcap
        new_count = current_count - 1
        new_dur_contribution = current_duration * current_mcap - bond_dur_contribution

    new_duration = new_dur_contribution / new_mcap if new_mcap > 0 else 0
    duration_change = new_duration - current_duration
    new_weight = bond_mcap / new_mcap * 100 if new_mcap > 0 else 0

    return {
        'segment': segment,
        'segment_name': SEGMENTS[segment]['name'],
        'event_type': event_type,
        'bond_nominal_mio': round(bond_nominal / 1e6, 0),
        'bond_duration': bond_duration,
        'bond_rating': bond_rating,
        'bond_weight_pct': round(new_weight, 4),
        'current_bonds': current_count,
        'new_bonds': new_count,
        'current_mcap_mrd': round(current_mcap / 1e9, 2),
        'new_mcap_mrd': round(new_mcap / 1e9, 2),
        'current_duration': round(current_duration, 3),
        'new_duration': round(new_duration, 3),
        'duration_change': round(duration_change, 3),
    }


def summarize_flows(report_df: pd.DataFrame) -> Dict:
    """Summarize total flows by direction and event type"""
    summary = {
        'total_buy_flow': report_df[report_df['flow_direction'] == 'BUY']['flow_chf'].sum(),
        'total_sell_flow': report_df[report_df['flow_direction'] == 'SELL']['flow_chf'].sum(),
        'net_flow': report_df['flow_chf'].sum() * (
            1 if report_df['flow_direction'].iloc[0] == 'BUY' else -1
        ) if len(report_df) > 0 else 0,
        'by_event_type': report_df.groupby('event_type')['flow_chf'].sum().to_dict(),
        'count_by_event': report_df.groupby('event_type').size().to_dict(),
    }

    # Calculate net properly
    buys = report_df[report_df['flow_direction'] == 'BUY']['flow_chf'].sum()
    sells = report_df[report_df['flow_direction'] == 'SELL']['flow_chf'].sum()
    summary['net_flow'] = buys - sells

    return summary


def print_flow_report(report_df: pd.DataFrame, top_n: int = 10):
    """Print formatted flow report"""
    print("\n" + "="*80)
    print("SBI INDEX REBALANCING - FLOW REPORT")
    print("="*80)

    summary = summarize_flows(report_df)

    print(f"\nTOTAL FLOWS:")
    print(f"  Buy pressure:  CHF {summary['total_buy_flow']/1e6:>10.2f} Mio")
    print(f"  Sell pressure: CHF {summary['total_sell_flow']/1e6:>10.2f} Mio")
    print(f"  Net flow:      CHF {summary['net_flow']/1e6:>10.2f} Mio")

    print(f"\nBY EVENT TYPE:")
    for event, flow in summary['by_event_type'].items():
        count = summary['count_by_event'][event]
        print(f"  {event}: {count} events, CHF {flow/1e6:.2f} Mio")

    # Top buy flows
    buys = report_df[report_df['flow_direction'] == 'BUY'].nlargest(top_n, 'flow_chf')
    if len(buys) > 0:
        print(f"\nTOP {min(top_n, len(buys))} BUY FLOWS:")
        for _, row in buys.iterrows():
            print(f"  {row['isin']}: CHF {row['flow_chf']/1e6:.2f} Mio "
                  f"({row['est_weight_pct']:.2f}%) - {row['name'][:30]}")

    # Top sell flows
    sells = report_df[report_df['flow_direction'] == 'SELL'].nlargest(top_n, 'flow_chf')
    if len(sells) > 0:
        print(f"\nTOP {min(top_n, len(sells))} SELL FLOWS:")
        for _, row in sells.iterrows():
            print(f"  {row['isin']}: CHF {row['flow_chf']/1e6:.2f} Mio "
                  f"({row['est_weight_pct']:.2f}%) - {row['name'][:30]}")


if __name__ == '__main__':
    from pathlib import Path

    base_path = Path(__file__).parent.parent

    # Load data
    print("Loading constituents...")
    constituents = parse_constituents(str(base_path / 'close_sbr14d.csv'))

    print("Loading forecast...")
    forecast = parse_forecast(str(base_path / 'forecast_bonds.csv'))

    # Generate report
    print("Generating flow report...")
    report = generate_flow_report(forecast, constituents)

    # Print report
    print_flow_report(report)

    # Save to CSV
    output_path = base_path / '.tmp'
    output_path.mkdir(exist_ok=True)
    report.to_csv(output_path / 'flow_report.csv', index=False)
    print(f"\nReport saved to: {output_path / 'flow_report.csv'}")
