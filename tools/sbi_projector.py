"""
SBI Projector - Forward-looking index projection and bond change simulation
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from pathlib import Path
from datetime import datetime
import warnings

from sbi_analyzer import (calc_index_mcap, calc_segment_duration, calc_segment_yield,
                          get_all_segment_stats, SEGMENTS)
from sbi_flow_estimator import estimate_duration_from_maturity
from sbi_parser import parse_forecast, get_admissions, get_deletions, get_capital_changes


# German month abbreviations
MONTH_LABELS_DE = {
    1: 'Jan', 2: 'Feb', 3: 'Mrz', 4: 'Apr', 5: 'Mai', 6: 'Jun',
    7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Okt', 11: 'Nov', 12: 'Dez'
}


def project_index_forward(
    constituents_df: pd.DataFrame,
    months: int = 12,
    min_residual_years: float = 1.0
) -> List[Dict]:
    """
    Project the index composition forward by removing bonds that breach
    the minimum residual maturity threshold each month.

    Args:
        constituents_df: Current constituent DataFrame with 'remaining_years', etc.
        months: Number of months to project forward (default 12)
        min_residual_years: Minimum residual years threshold (default 1.0)

    Returns:
        List of dicts, one per month, each containing projection data.
    """
    working_df = constituents_df.copy()

    # Store original remaining_years and duration for proportional scaling
    working_df['_orig_remaining_years'] = working_df['remaining_years']
    working_df['_orig_duration'] = working_df['duration']

    # Determine base date from data
    if 'date' in working_df.columns and working_df['date'].notna().any():
        base_date = working_df['date'].iloc[0]
        if isinstance(base_date, str):
            base_date = pd.to_datetime(base_date)
    else:
        base_date = pd.Timestamp.now()

    # Baseline metrics
    base_mcap = calc_index_mcap(working_df)
    base_duration = calc_segment_duration(working_df, 'total')
    base_ytw = calc_segment_yield(working_df, 'total')
    base_count = len(working_df)

    results = []

    for m in range(1, months + 1):
        elapsed = 1 / 12.0  # Each iteration = 1 month forward

        # Reduce remaining_years by 1 month for all bonds
        valid_remaining = working_df['remaining_years'].notna()
        working_df.loc[valid_remaining, 'remaining_years'] -= elapsed

        # Adjust duration proportionally: duration scales with remaining_years
        # new_duration = orig_duration * (new_remaining / orig_remaining)
        orig_ok = (working_df['_orig_remaining_years'].notna() &
                   (working_df['_orig_remaining_years'] > 0))
        working_df.loc[orig_ok, 'duration'] = (
            working_df.loc[orig_ok, '_orig_duration'] *
            working_df.loc[orig_ok, 'remaining_years'] /
            working_df.loc[orig_ok, '_orig_remaining_years']
        ).clip(lower=0)

        # Identify bonds that now breach the threshold.
        # Strict < (not <=) means a bond at exactly 1.000 years is kept one more month,
        # matching SIX rules which require a full year of remaining maturity.
        exit_mask = valid_remaining & (working_df['remaining_years'] < min_residual_years)
        exiting_bonds = working_df[exit_mask].copy()

        # Remove exiting bonds from working set
        working_df = working_df[~exit_mask].copy()

        # Calculate month label
        target_month = base_date.month + m
        target_year = base_date.year + (target_month - 1) // 12
        target_month = ((target_month - 1) % 12) + 1
        month_label = f"M+{m} ({MONTH_LABELS_DE[target_month]} {target_year})"

        # Calculate metrics on remaining bonds (with adjusted durations)
        if len(working_df) > 0:
            current_mcap = calc_index_mcap(working_df)
            current_duration = calc_segment_duration(working_df, 'total')
            current_ytw = calc_segment_yield(working_df, 'total')
            seg_stats = get_all_segment_stats(working_df)
        else:
            current_mcap = 0.0
            current_duration = 0.0
            current_ytw = 0.0
            seg_stats = pd.DataFrame()

        results.append({
            'month_offset': m,
            'month_label': month_label,
            'exiting_bonds': exiting_bonds,
            'exit_count': len(exiting_bonds),
            'remaining_bonds_count': len(working_df),
            'total_mcap': current_mcap,
            'duration': round(current_duration, 3),
            'ytw': round(current_ytw, 4),
            'mcap_change_pct': round((current_mcap - base_mcap) / base_mcap * 100, 2) if base_mcap > 0 else 0,
            'duration_change': round(current_duration - base_duration, 3),
            'ytw_change': round(current_ytw - base_ytw, 4),
            'segment_stats': seg_stats,
        })

    return results


def apply_bond_changes(
    constituents_df: pd.DataFrame,
    changes_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Apply bond additions/adjustments to the index.

    Args:
        constituents_df: Current constituents
        changes_df: Uploaded changes with columns:
            Required: isin, nominal
            Optional: price, duration, rating, sector_code, domicile, maturity_date

    Returns:
        Modified constituents DataFrame with changes applied.
    """
    modified = constituents_df.copy()

    # Calculate default values from current index
    default_ytw = calc_segment_yield(constituents_df, 'total')

    for _, row in changes_df.iterrows():
        isin = str(row['isin']).strip()
        nominal = float(row['nominal'])

        existing = modified[modified['isin'] == isin]

        if len(existing) > 0:
            # Existing bond: update nominal
            idx = existing.index[0]
            modified.at[idx, 'nominal'] = nominal
            price = modified.at[idx, 'price']
            modified.at[idx, 'market_cap'] = nominal * price / 100
        else:
            # New bond: create row
            price = float(row.get('price', 100)) if pd.notna(row.get('price')) else 100.0
            market_cap = nominal * price / 100

            # Duration: from upload, or estimate from maturity, or default
            duration = None
            if 'duration' in row and pd.notna(row.get('duration')):
                duration = float(row['duration'])
            elif 'maturity_date' in row and pd.notna(row.get('maturity_date')):
                mat_date = row['maturity_date']
                if isinstance(mat_date, str):
                    try:
                        mat_date = pd.to_datetime(mat_date, dayfirst=True)
                    except Exception:
                        mat_date = None
                if mat_date is not None:
                    years_to_mat = (mat_date - pd.Timestamp.now()).days / 365
                    coupon = float(row.get('coupon', 0.5)) if pd.notna(row.get('coupon')) else 0.5
                    duration = estimate_duration_from_maturity(years_to_mat, coupon)

            if duration is None:
                duration = 5.0

            rating = str(row.get('rating', 'A')) if pd.notna(row.get('rating')) else 'A'
            domicile = str(row.get('domicile', 'CH')) if pd.notna(row.get('domicile')) else 'CH'
            name = str(row.get('name', isin)) if pd.notna(row.get('name')) else isin

            # Remaining years
            remaining_years = 5.0
            if 'maturity_date' in row and pd.notna(row.get('maturity_date')):
                mat_date = row['maturity_date']
                if isinstance(mat_date, str):
                    try:
                        mat_date = pd.to_datetime(mat_date, dayfirst=True)
                    except Exception:
                        mat_date = None
                if mat_date is not None:
                    remaining_years = (mat_date - pd.Timestamp.now()).days / 365

            # Build new row with same columns as existing data
            new_row = {
                'isin': isin,
                'name': name,
                'nominal': nominal,
                'price': price,
                'market_cap': market_cap,
                'duration': duration,
                'ytw': float(row.get('ytw', default_ytw)) if pd.notna(row.get('ytw')) else default_ytw,
                'rating': rating,
                'domicile': domicile,
                'remaining_years': remaining_years,
                'weight': 0,  # Will be recalculated
                'accrued_interest': 0,
            }

            # Optional columns
            if 'guarantee_collateral_code' in row and pd.notna(row.get('guarantee_collateral_code')):
                new_row['guarantee_collateral_code'] = str(row['guarantee_collateral_code'])
            elif 'guarantee_collateral_code' in modified.columns:
                new_row['guarantee_collateral_code'] = '74010100'  # Default: Banks

            if 'sector_code' in row and pd.notna(row.get('sector_code')):
                new_row['sector_code'] = str(row['sector_code'])

            if 'date' in modified.columns and modified['date'].notna().any():
                new_row['date'] = modified['date'].iloc[0]

            if 'maturity_date' in row and pd.notna(row.get('maturity_date')):
                mat_date = row['maturity_date']
                if isinstance(mat_date, str):
                    try:
                        new_row['maturity_date'] = pd.to_datetime(mat_date, dayfirst=True)
                    except Exception:
                        pass

            new_row_df = pd.DataFrame([new_row])
            modified = pd.concat([modified, new_row_df], ignore_index=True)

    return modified


def validate_changes_csv(raw_df: pd.DataFrame) -> Dict:
    """
    Validate an uploaded bond changes CSV.

    Returns:
        {'valid': bool, 'errors': list, 'warnings': list, 'parsed_df': DataFrame}
    """
    errors = []
    warnings = []

    # Normalize column names
    df = raw_df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # Check required columns
    if 'isin' not in df.columns:
        errors.append("Spalte 'isin' fehlt (Pflichtfeld)")
    if 'nominal' not in df.columns:
        errors.append("Spalte 'nominal' fehlt (Pflichtfeld)")

    if errors:
        return {'valid': False, 'errors': errors, 'warnings': warnings, 'parsed_df': pd.DataFrame()}

    # Clean ISIN
    df['isin'] = df['isin'].astype(str).str.strip()

    # Validate ISINs
    invalid_isins = df[df['isin'].str.len() < 3]
    if len(invalid_isins) > 0:
        errors.append(f"{len(invalid_isins)} ungueltige ISIN(s) gefunden (zu kurz)")

    # Non-CH ISINs
    non_ch = df[~df['isin'].str.startswith('CH')]
    if len(non_ch) > 0:
        warnings.append(f"{len(non_ch)} ISIN(s) sind nicht CH-Bonds")

    # Convert nominal
    df['nominal'] = pd.to_numeric(df['nominal'], errors='coerce')
    nan_nominals = df['nominal'].isna().sum()
    if nan_nominals > 0:
        errors.append(f"{nan_nominals} Zeile(n) mit ungueltigem Nominal")

    # Check for negative nominals (only valid as adjustments)
    neg_nominals = df[df['nominal'] < 0]
    if len(neg_nominals) > 0:
        warnings.append(f"{len(neg_nominals)} Zeile(n) mit negativem Nominal (nur fuer bestehende Bonds gueltig)")

    # Check for duplicates
    dup_isins = df[df['isin'].duplicated(keep=False)]
    if len(dup_isins) > 0:
        warnings.append(f"{len(dup_isins)} doppelte ISINs gefunden - Nominale werden summiert")
        # Sum duplicates
        agg_cols = {'nominal': 'sum'}
        other_cols = [c for c in df.columns if c not in ['isin', 'nominal']]
        for c in other_cols:
            agg_cols[c] = 'first'
        df = df.groupby('isin', as_index=False).agg(agg_cols)

    # Optional: validate price
    if 'price' in df.columns:
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        out_of_range = df[(df['price'].notna()) & ((df['price'] < 50) | (df['price'] > 200))]
        if len(out_of_range) > 0:
            warnings.append(f"{len(out_of_range)} Bond(s) mit unueblichem Preis (< 50 oder > 200)")

    # Optional: validate rating
    valid_ratings = ['AAA', 'AA', 'A', 'BBB']
    if 'rating' in df.columns:
        df['rating'] = df['rating'].astype(str).str.strip().str.upper()
        invalid_ratings = df[~df['rating'].isin(valid_ratings + ['NAN', 'NONE', ''])]
        if len(invalid_ratings) > 0:
            warnings.append(f"{len(invalid_ratings)} Bond(s) mit unbekanntem Rating")

    # Optional: parse admission_date
    if 'admission_date' in df.columns:
        df['admission_date'] = pd.to_datetime(df['admission_date'], dayfirst=True, errors='coerce')
        unparsed_adm = df['admission_date'].isna().sum()
        orig_na = raw_df.columns.str.strip().str.lower().str.replace(' ', '_')
        if 'admission_date' in orig_na.tolist():
            orig_col = raw_df.iloc[:, orig_na.tolist().index('admission_date')]
            unparsed_adm -= orig_col.isna().sum()
        if unparsed_adm > 0:
            warnings.append(f"{unparsed_adm} Admission-Datum(e) konnten nicht geparst werden")

    # Optional: parse maturity_date
    if 'maturity_date' in df.columns:
        df['maturity_date'] = pd.to_datetime(df['maturity_date'], dayfirst=True, errors='coerce')
        unparsed = df['maturity_date'].isna().sum() - raw_df.get('maturity_date', pd.Series()).isna().sum()
        if unparsed > 0:
            warnings.append(f"{unparsed} Maturity-Datum(e) konnten nicht geparst werden")

    # Optional: parse duration
    if 'duration' in df.columns:
        df['duration'] = pd.to_numeric(df['duration'], errors='coerce')

    # Eligibility warnings
    small_bonds = df[df['nominal'] < 100_000_000]
    small_bonds = small_bonds[small_bonds['nominal'] > 0]
    if len(small_bonds) > 0:
        warnings.append(f"{len(small_bonds)} Bond(s) unter CHF 100 Mio Minimum-Nominal")

    valid = len(errors) == 0 and df['nominal'].notna().all()

    return {
        'valid': valid,
        'errors': errors,
        'warnings': warnings,
        'parsed_df': df
    }


def compare_index_states(
    original_df: pd.DataFrame,
    modified_df: pd.DataFrame,
    changes_df: pd.DataFrame
) -> Dict:
    """
    Compare original and modified index states.

    Returns dict with summary, segment_comparison, top_weight_changes, uploaded_bonds.
    """
    # Summary metrics
    orig_mcap = calc_index_mcap(original_df)
    new_mcap = calc_index_mcap(modified_df)
    orig_duration = calc_segment_duration(original_df, 'total')
    new_duration = calc_segment_duration(modified_df, 'total')
    orig_ytw = calc_segment_yield(original_df, 'total')
    new_ytw = calc_segment_yield(modified_df, 'total')

    summary = {
        'original_mcap': orig_mcap,
        'new_mcap': new_mcap,
        'mcap_change': new_mcap - orig_mcap,
        'mcap_change_pct': (new_mcap - orig_mcap) / orig_mcap * 100 if orig_mcap > 0 else 0,
        'original_count': len(original_df),
        'new_count': len(modified_df),
        'original_duration': orig_duration,
        'new_duration': new_duration,
        'duration_change': new_duration - orig_duration,
        'original_ytw': orig_ytw,
        'new_ytw': new_ytw,
        'ytw_change': new_ytw - orig_ytw,
    }

    # Segment comparison
    orig_stats = get_all_segment_stats(original_df)
    new_stats = get_all_segment_stats(modified_df)
    seg_comparison = orig_stats.merge(new_stats, on='name', suffixes=('_aktuell', '_neu'))

    # Weight calculations for all bonds
    orig_weights = original_df[['isin', 'name']].copy()
    orig_weights['weight_aktuell'] = original_df['market_cap'] / orig_mcap * 100 if orig_mcap > 0 else 0

    mod_weights = modified_df[['isin', 'name']].copy()
    mod_weights['weight_neu'] = modified_df['market_cap'] / new_mcap * 100 if new_mcap > 0 else 0

    merged = orig_weights.merge(mod_weights, on='isin', how='outer', suffixes=('', '_mod'))
    merged['name'] = merged['name'].fillna(merged.get('name_mod', ''))
    if 'name_mod' in merged.columns:
        merged = merged.drop(columns=['name_mod'])
    merged['weight_aktuell'] = merged['weight_aktuell'].fillna(0)
    merged['weight_neu'] = merged['weight_neu'].fillna(0)
    merged['weight_delta'] = merged['weight_neu'] - merged['weight_aktuell']
    merged['weight_delta_abs'] = merged['weight_delta'].abs()

    top_changes = merged.nlargest(20, 'weight_delta_abs')[
        ['isin', 'name', 'weight_aktuell', 'weight_neu', 'weight_delta']
    ].copy()
    top_changes['weight_aktuell'] = top_changes['weight_aktuell'].round(4)
    top_changes['weight_neu'] = top_changes['weight_neu'].round(4)
    top_changes['weight_delta'] = top_changes['weight_delta'].round(4)

    # Uploaded bonds detail
    uploaded_isins = changes_df['isin'].tolist()
    uploaded_detail = modified_df[modified_df['isin'].isin(uploaded_isins)].copy()
    if len(uploaded_detail) > 0 and new_mcap > 0:
        uploaded_detail['weight_pct'] = (uploaded_detail['market_cap'] / new_mcap * 100).round(4)
    else:
        uploaded_detail['weight_pct'] = 0

    return {
        'summary': summary,
        'segment_comparison': seg_comparison,
        'top_weight_changes': top_changes,
        'uploaded_bonds': uploaded_detail,
    }


def analyze_historical_forecasts(folder: str) -> pd.DataFrame:
    """
    Analyze all historical forecast files to track index changes over time.

    For each month, computes:
    - Number of admissions and deletions
    - Total nominal in/out
    - Estimated weighted-average duration of admissions vs deletions
    - Net nominal change
    - Capital changes

    Args:
        folder: Path to the folder containing SBI_Forecast_YYYYMM.csv files

    Returns:
        DataFrame with one row per month, sorted chronologically.
    """
    folder_path = Path(folder)
    results = []

    for file in sorted(folder_path.glob('SBI_Forecast_*.csv')):
        period = file.stem.replace('SBI_Forecast_', '')
        try:
            year = int(period[:4])
            month = int(period[4:])
        except (ValueError, IndexError):
            continue

        try:
            df = parse_forecast(str(file))
        except Exception as e:
            warnings.warn(f"Could not load {file}: {e}")
            continue

        admissions = get_admissions(df)
        deletions = get_deletions(df)
        cap_changes = get_capital_changes(df)

        # Compute estimated duration for admissions
        adm_durations = []
        adm_nominals = []
        for _, row in admissions.iterrows():
            nominal = row.get('nominal', 0)
            if pd.isna(nominal) or nominal <= 0:
                continue
            mat_date = row.get('maturity_date')
            adm_date = row.get('admission_date')
            coupon = row.get('coupon', 0.5)
            if pd.isna(coupon):
                coupon = 0.5

            if pd.notna(mat_date) and pd.notna(adm_date):
                years_to_mat = (mat_date - adm_date).days / 365.25
            elif pd.notna(mat_date):
                years_to_mat = (mat_date - pd.Timestamp(year, month, 1)).days / 365.25
            else:
                years_to_mat = 5.0  # fallback

            if years_to_mat > 0:
                dur = estimate_duration_from_maturity(years_to_mat, coupon)
                adm_durations.append(dur)
                adm_nominals.append(nominal)

        # Compute estimated duration for deletions
        del_durations = []
        del_nominals = []
        for _, row in deletions.iterrows():
            nominal = row.get('nominal', 0)
            if pd.isna(nominal) or nominal <= 0:
                continue
            mat_date = row.get('maturity_date')
            cancel_date = row.get('cancellation_date')
            coupon = row.get('coupon', 0.5)
            if pd.isna(coupon):
                coupon = 0.5

            # Deletions typically have ~1 year remaining
            if pd.notna(mat_date) and pd.notna(cancel_date):
                years_to_mat = (mat_date - cancel_date).days / 365.25
            elif pd.notna(mat_date):
                years_to_mat = (mat_date - pd.Timestamp(year, month, 1)).days / 365.25
            else:
                years_to_mat = 1.0  # deletions are typically near 1y

            if years_to_mat > 0:
                dur = estimate_duration_from_maturity(years_to_mat, coupon)
                del_durations.append(dur)
                del_nominals.append(nominal)

        # Weighted average durations
        adm_total_nom = sum(adm_nominals) if adm_nominals else 0
        del_total_nom = sum(del_nominals) if del_nominals else 0

        adm_wavg_dur = (sum(d * n for d, n in zip(adm_durations, adm_nominals)) / adm_total_nom
                        if adm_total_nom > 0 else 0)
        del_wavg_dur = (sum(d * n for d, n in zip(del_durations, del_nominals)) / del_total_nom
                        if del_total_nom > 0 else 0)

        # Capital changes
        cap_increase = 0
        cap_decrease = 0
        if len(cap_changes) > 0 and 'capital_change' in cap_changes.columns:
            cap_vals = cap_changes['capital_change'].dropna()
            cap_increase = cap_vals[cap_vals > 0].sum()
            cap_decrease = cap_vals[cap_vals < 0].sum()

        results.append({
            'period': period,
            'year': year,
            'month': month,
            'date': pd.Timestamp(year, month, 1),
            'admissions_count': len(admissions),
            'deletions_count': len(deletions),
            'admissions_nominal': adm_total_nom,
            'deletions_nominal': del_total_nom,
            'net_nominal': adm_total_nom - del_total_nom,
            'admissions_avg_duration': round(adm_wavg_dur, 2),
            'deletions_avg_duration': round(del_wavg_dur, 2),
            'duration_diff': round(adm_wavg_dur - del_wavg_dur, 2),
            'capital_increase': cap_increase,
            'capital_decrease': cap_decrease,
            'cap_changes_count': len(cap_changes),
            'total_events': len(df),
        })

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results).sort_values('date').reset_index(drop=True)
    return result_df


# Segment classification for forecast bonds
_FORECAST_SEGMENT_RULES = {
    'total': lambda sector, gc: True,
    'domestic': lambda sector, gc: str(sector).startswith('D'),
    'foreign': lambda sector, gc: str(sector).startswith('F'),
    'government': lambda sector, gc: str(gc).startswith('51'),
    'agency': lambda sector, gc: str(gc).startswith('52') or str(gc).startswith('53'),
    'pfandbrief': lambda sector, gc: str(gc).startswith('61'),
    'corporate': lambda sector, gc: str(gc).startswith('7'),
    'financials': lambda sector, gc: str(gc).startswith('74'),
}

_SEGMENT_NAMES = {
    'total': 'SBI AAA-BBB Total',
    'domestic': 'SBI Domestic',
    'foreign': 'SBI Foreign',
    'government': 'SBI Government',
    'agency': 'SBI Agency',
    'pfandbrief': 'SBI Pfandbrief',
    'corporate': 'SBI Corporate',
    'financials': 'SBI Financials',
}


def _classify_forecast_bond(row):
    """Return list of segment keys a forecast bond belongs to."""
    sector = str(row.get('sector', '')) if pd.notna(row.get('sector')) else ''
    gc = str(row.get('guarantee_code', '')) if pd.notna(row.get('guarantee_code')) else ''
    return [seg for seg, rule in _FORECAST_SEGMENT_RULES.items() if rule(sector, gc)]


def _estimate_bond_duration(row, year, month):
    """Estimate duration of a forecast bond from its maturity date."""
    mat_date = row.get('maturity_date')
    ref_date = row.get('admission_date') or row.get('cancellation_date')
    coupon = row.get('coupon', 0.5)
    if pd.isna(coupon):
        coupon = 0.5

    if pd.notna(mat_date) and pd.notna(ref_date):
        ytm = (mat_date - ref_date).days / 365.25
    elif pd.notna(mat_date):
        ytm = (mat_date - pd.Timestamp(year, month, 1)).days / 365.25
    else:
        return None

    if ytm <= 0:
        return None
    return estimate_duration_from_maturity(ytm, coupon)


def reconstruct_historical_durations(constituents_df: pd.DataFrame, folder: str) -> pd.DataFrame:
    """
    Reconstruct actual historical index duration per segment by tracking
    the index composition backwards through forecast files.

    Approach:
    1. Start with current constituent ISINs (from constituents_df)
    2. Go backwards month by month through forecast files
    3. For each month: reverse admissions (remove) and deletions (add back)
    4. At each month: compute nominal-weighted avg duration per segment

    Bond properties come from:
    - Current constituents (for bonds still in the index)
    - Forecast files (for bonds that have been deleted)

    Duration at historical time T is estimated from maturity:
      remaining_years_at_T = (maturity_date - T).days / 365.25
      duration ≈ estimate_duration_from_maturity(remaining_years, coupon)

    Returns DataFrame with columns:
        date, segment, segment_name, duration, bond_count, total_nominal
    """
    folder_path = Path(folder)

    # ---- Step 1: Build bond property lookup ----
    # From current constituents: isin -> {nominal, coupon, maturity_date, sector, gc, domicile}
    bond_props = {}
    for _, row in constituents_df.iterrows():
        isin = row.get('isin')
        if pd.isna(isin):
            continue
        mat = row.get('maturity_date')
        coupon = row.get('coupon', 0.5)
        if pd.isna(coupon):
            coupon = 0.5
        gc = str(row.get('guarantee_collateral_code', '')) if pd.notna(row.get('guarantee_collateral_code')) else ''
        # Map sector_code/domicile to forecast-style sector (DN/FC/etc.)
        domicile = str(row.get('domicile', ''))
        sector_name = str(row.get('sector_level1', ''))
        if domicile == 'CH':
            fc_sector = 'DN'  # Domestic
        else:
            fc_sector = 'FC'  # Foreign Corporate (default for non-CH)
        bond_props[isin] = {
            'nominal': row.get('nominal', 0),
            'coupon': coupon,
            'maturity_date': mat,
            'sector': fc_sector,
            'gc': gc,
        }

    # ---- Step 2: Load all forecast files sorted chronologically ----
    forecast_files = sorted(folder_path.glob('SBI_Forecast_*.csv'))
    forecasts = []
    for file in forecast_files:
        period = file.stem.replace('SBI_Forecast_', '')
        try:
            year = int(period[:4])
            month = int(period[4:])
        except (ValueError, IndexError):
            continue
        try:
            df = parse_forecast(str(file))
        except Exception:
            continue
        forecasts.append({
            'period': period,
            'year': year,
            'month': month,
            'date': pd.Timestamp(year, month, 1),
            'df': df,
        })

    if not forecasts:
        return pd.DataFrame()

    # Also enrich bond_props from forecast files (for deleted bonds not in current constituents)
    for fc in forecasts:
        for _, row in fc['df'].iterrows():
            isin = row.get('isin')
            if pd.isna(isin) or isin in bond_props:
                continue
            mat = row.get('maturity_date')
            coupon = row.get('coupon', 0.5)
            if pd.isna(coupon):
                coupon = 0.5
            sector = str(row.get('sector', '')) if pd.notna(row.get('sector')) else ''
            gc = str(row.get('guarantee_code', '')) if pd.notna(row.get('guarantee_code')) else ''
            nominal = row.get('nominal', 0)
            if pd.isna(nominal):
                nominal = 0
            bond_props[isin] = {
                'nominal': nominal,
                'coupon': coupon,
                'maturity_date': mat,
                'sector': sector,
                'gc': gc,
            }

    # ---- Step 3: Track index composition backwards ----
    # Start with current ISINs
    current_isins = set(constituents_df['isin'].dropna().unique())

    # Sort forecasts newest first for backward traversal
    forecasts_desc = sorted(forecasts, key=lambda f: f['date'], reverse=True)

    # Build snapshots: for each month, store the ISIN set BEFORE that month's rebalancing
    # (i.e., the index composition at the start of that month)
    snapshots = []

    # The current state is AFTER the latest forecast has been applied
    # So the "current" snapshot is the index as of now
    latest_date = forecasts_desc[0]['date']
    snapshots.append({
        'date': latest_date,
        'isins': set(current_isins),
    })

    working_isins = set(current_isins)

    for fc in forecasts_desc:
        adm = get_admissions(fc['df'])
        dels = get_deletions(fc['df'])
        cap = get_capital_changes(fc['df'])

        # Reverse this month's changes to get the state BEFORE this month
        # Remove admissions (they weren't in the index before this month)
        for _, row in adm.iterrows():
            isin = row.get('isin')
            if pd.notna(isin):
                working_isins.discard(isin)

        # Add back deletions (they were in the index before this month)
        for _, row in dels.iterrows():
            isin = row.get('isin')
            if pd.notna(isin):
                working_isins.add(isin)

        # Update nominals from capital changes (use old amount if available)
        for _, row in cap.iterrows():
            isin = row.get('isin')
            change = row.get('capital_change', 0)
            if pd.notna(isin) and pd.notna(change) and isin in bond_props:
                # Reverse the capital change to get previous nominal
                bond_props[isin]['_nominal_adj'] = bond_props[isin].get('_nominal_adj', {})
                # We note this but the nominal in bond_props stays as latest known

        # Compute the date for the month BEFORE this forecast
        # The forecast for month M changes the index from end of M-1 to start of M
        prev_year = fc['year'] if fc['month'] > 1 else fc['year'] - 1
        prev_month = fc['month'] - 1 if fc['month'] > 1 else 12
        prev_date = pd.Timestamp(prev_year, prev_month, 1)

        snapshots.append({
            'date': prev_date,
            'isins': set(working_isins),
        })

    # Reverse so snapshots are chronological
    snapshots = sorted(snapshots, key=lambda s: s['date'])

    # ---- Step 4: Compute duration per segment per month ----
    results = []
    segments = list(_FORECAST_SEGMENT_RULES.keys())

    for snap in snapshots:
        ref_date = snap['date']

        # Per-segment accumulators: {seg: {'dur_x_nom': 0, 'total_nom': 0, 'count': 0}}
        seg_acc = {seg: {'dur_x_nom': 0.0, 'total_nom': 0.0, 'count': 0} for seg in segments}

        for isin in snap['isins']:
            props = bond_props.get(isin)
            if props is None:
                continue

            mat = props['maturity_date']
            if pd.isna(mat):
                continue

            remaining = (mat - ref_date).days / 365.25
            if remaining <= 0:
                continue  # Bond has matured by this date

            coupon = props['coupon']
            dur = estimate_duration_from_maturity(remaining, coupon)
            nominal = props['nominal']
            if pd.isna(nominal) or nominal <= 0:
                continue

            # Classify into segments
            sector = props.get('sector', '')
            gc = props.get('gc', '')
            for seg, rule in _FORECAST_SEGMENT_RULES.items():
                if rule(sector, gc):
                    seg_acc[seg]['dur_x_nom'] += dur * nominal
                    seg_acc[seg]['total_nom'] += nominal
                    seg_acc[seg]['count'] += 1

        for seg in segments:
            acc = seg_acc[seg]
            if acc['total_nom'] > 0:
                wavg_dur = acc['dur_x_nom'] / acc['total_nom']
            else:
                wavg_dur = 0

            results.append({
                'date': ref_date,
                'segment': seg,
                'segment_name': _SEGMENT_NAMES.get(seg, seg),
                'duration': round(wavg_dur, 3),
                'bond_count': acc['count'],
                'total_nominal': acc['total_nom'],
            })

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).sort_values(['date', 'segment']).reset_index(drop=True)


def analyze_historical_by_segment(folder: str) -> pd.DataFrame:
    """
    Analyze historical forecasts with per-segment breakdown.

    Returns DataFrame with columns:
    period, date, segment, segment_name,
    adm_count, del_count, adm_nominal, del_nominal,
    adm_avg_duration, del_avg_duration
    """
    folder_path = Path(folder)
    results = []

    for file in sorted(folder_path.glob('SBI_Forecast_*.csv')):
        period = file.stem.replace('SBI_Forecast_', '')
        try:
            year = int(period[:4])
            month = int(period[4:])
        except (ValueError, IndexError):
            continue

        try:
            df = parse_forecast(str(file))
        except Exception:
            continue

        admissions = get_admissions(df)
        deletions = get_deletions(df)
        date = pd.Timestamp(year, month, 1)

        # Collect per-segment data
        seg_data = {}
        for seg in _FORECAST_SEGMENT_RULES:
            seg_data[seg] = {'adm_durs': [], 'adm_noms': [], 'del_durs': [], 'del_noms': []}

        for _, row in admissions.iterrows():
            nominal = row.get('nominal', 0)
            if pd.isna(nominal) or nominal <= 0:
                continue
            dur = _estimate_bond_duration(row, year, month)
            if dur is None:
                continue
            for seg in _classify_forecast_bond(row):
                seg_data[seg]['adm_durs'].append(dur)
                seg_data[seg]['adm_noms'].append(nominal)

        for _, row in deletions.iterrows():
            nominal = row.get('nominal', 0)
            if pd.isna(nominal) or nominal <= 0:
                continue
            dur = _estimate_bond_duration(row, year, month)
            if dur is None:
                dur = estimate_duration_from_maturity(1.0, 0.5)
            for seg in _classify_forecast_bond(row):
                seg_data[seg]['del_durs'].append(dur)
                seg_data[seg]['del_noms'].append(nominal)

        for seg, data in seg_data.items():
            adm_total = sum(data['adm_noms']) if data['adm_noms'] else 0
            del_total = sum(data['del_noms']) if data['del_noms'] else 0

            adm_wavg = (sum(d * n for d, n in zip(data['adm_durs'], data['adm_noms'])) / adm_total
                        if adm_total > 0 else 0)
            del_wavg = (sum(d * n for d, n in zip(data['del_durs'], data['del_noms'])) / del_total
                        if del_total > 0 else 0)

            results.append({
                'period': period,
                'date': date,
                'segment': seg,
                'segment_name': _SEGMENT_NAMES.get(seg, seg),
                'adm_count': len(data['adm_noms']),
                'del_count': len(data['del_noms']),
                'adm_nominal': adm_total,
                'del_nominal': del_total,
                'adm_avg_duration': round(adm_wavg, 2),
                'del_avg_duration': round(del_wavg, 2),
            })

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).sort_values(['date', 'segment']).reset_index(drop=True)
