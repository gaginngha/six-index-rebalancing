# Dashboard Improvements Design
**Date:** 2026-02-28
**Project:** six-index-rebalancing (SBI AAA-BBB Rebalancing Dashboard)

## Context

Following a full codebase analysis and bug fixes (duration estimation coupon bug, NaN handling in weighted averages), this document covers four improvement areas approved for implementation.

---

## Feature 1: Forecast Duration Activity History

### Purpose
Visualize 94 months (June 2018–Jan 2026) of SIX SBI rebalancing activity as estimated duration impacts, providing historical context for typical rebalancing magnitudes and trends.

### Data Constraint
Only forecast event files are available (`extracted/SBI_Forecast_*.csv`), not historical constituent snapshots. The feature therefore shows *estimated activity* (not measured accuracy) — this is clearly labelled in the UI.

### Backend: `tools/sbi_historical.py` (new file)

Function `load_forecast_activity_history(extracted_dir) -> pd.DataFrame`:
- Iterates all `SBI_Forecast_*.csv` files in chronological order
- Parses each using the existing `parse_forecast()` from `sbi_parser.py`
- For each month computes:
  - `admission_count`, `deletion_count`
  - `admission_nominal_chf`, `deletion_nominal_chf`
  - `avg_admission_duration` — weighted by nominal using `estimate_duration_from_maturity(years_to_maturity, coupon)`
  - `net_duration_activity` — Σ(adm_dur × adm_nominal) − Σ(del_dur × del_nominal) in CHF-years
  - `forecast_month` — derived from filename (e.g. `201806` → `2018-06-01`)
- Returns one-row-per-month DataFrame, 94 rows

### UI: New tab "📈 Historische Aktivität" in `app.py`

- Line chart: `net_duration_activity` over time (CHF-years)
- Grouped bar chart: admission vs deletion nominal CHF per month
- Summary table: all columns, downloadable as CSV
- Disclaimer: "Schätzung basiert auf Coupon/Laufzeit-Daten aus den Prognosen. Historische Indexzusammensetzungen sind nicht verfügbar."

---

## Feature 2: Stale Data Warning

### Purpose
Alert users when the constituent file (`close_sbr14d.csv`) has not been updated recently, preventing analysis on outdated data.

### Implementation (in `app.py`, near data loading)

```python
import os
from datetime import datetime, timedelta

constituent_path = 'close_sbr14d.csv'
mtime = datetime.fromtimestamp(os.path.getmtime(constituent_path))
days_old = (datetime.now() - mtime).days

if days_old > 5:
    st.warning(f"⚠️ Konstituentendaten veraltet: Letzte Änderung {mtime.strftime('%d.%m.%Y')} ({days_old} Tage). Bitte aktualisieren.")
```

Additionally: display `mtime.strftime('%d.%m.%Y')` in the sidebar under the existing index metrics.

---

## Feature 3: Bonds Approaching Exit

### Purpose
Dedicated table of bonds near the minimum residual maturity threshold, grouped by segment, so traders can anticipate upcoming index deletions without manual search.

### UI: New tab "⏳ Baldige Exits" in `app.py`

**Controls:**
- Slider: "Zeithorizont" 1.0–3.0 years (default 2.0), step 0.5

**Content:**
- Summary metrics row: count of bonds, total CHF nominal, weighted duration contribution
- Table per segment (only segments with qualifying bonds shown):
  - Columns: ISIN, Name, Rating, Sector, Verbleibende Monate, Duration (Jahre), Nominal (CHF Mio), Indexgewicht (%)
  - Sorted by maturity date ascending within each segment
- Total row per segment showing subtotal nominal and avg duration

**No new backend code** — uses existing `constituents_df`, existing helper functions, and the `remaining_years` column already computed by `sbi_parser.py`.

---

## Feature 4: Segment-Level Projection + Yield

### Purpose
Show how duration evolves per segment (government, pfandbrief, corporate) alongside the total, and add yield-to-worst to the projection view.

### Backend change: `tools/sbi_projector.py`

In `project_index_forward()`, each result dict already has `seg_stats` from `get_all_segment_stats(working_df)`. Extend the result dict with:
```python
'segment_durations': {
    'total': current_duration,
    'government': calc_segment_duration(working_df, 'government'),
    'pfandbrief': calc_segment_duration(working_df, 'pfandbrief'),
    'corporate': calc_segment_duration(working_df, 'corporate'),
},
'segment_yields': {
    'total': current_ytw,
    'government': calc_segment_yield(working_df, 'government'),
    'pfandbrief': calc_segment_yield(working_df, 'pfandbrief'),
    'corporate': calc_segment_yield(working_df, 'corporate'),
},
```

### UI change: existing "🔮 Index Projektion" tab

- Replace single-line duration chart with multi-line chart (4 lines: total, government, pfandbrief, corporate)
- Add second chart: yield-to-worst projection (same 4 lines)
- Legend in German: Gesamt, Staatsanleihen, Pfandbriefe, Unternehmensanleihen

---

## Feature 5: Unit Tests

### Structure
```
tests/
  __init__.py
  test_calculations.py
```

### Coverage

**`test_calculations.py`:**

`estimate_duration_from_maturity`:
- Zero-coupon 10y → ~10.0
- 1% coupon 10y → ~9.13 (Macaulay at-par approximation)
- 5% coupon 10y → ~7.72
- maturity=0 → 0.0
- NaN maturity → 0.0
- Negative coupon → treated as zero-coupon

`calc_segment_duration`:
- 2-bond DataFrame, known weights → expected float
- 1 bond with NaN duration → excluded from calculation correctly
- Empty DataFrame → 0.0

`calc_segment_yield`:
- Same patterns as duration

`estimate_duration_from_maturity` regression:
- Old formula was `maturity × 0.92`; test that new formula gives ≤ maturity for any positive coupon

### Running tests
```bash
cd /path/to/six-index-rebalancing/tools  # for imports
python -m pytest ../tests/ -v
```

---

## Files Changed

| File | Change |
|------|--------|
| `tools/sbi_historical.py` | New — historical forecast activity |
| `tools/sbi_projector.py` | Extend result dict with per-segment duration/yield |
| `app.py` | Stale data warning + 2 new tabs + updated projection charts |
| `tests/__init__.py` | New (empty) |
| `tests/test_calculations.py` | New — unit tests |

## Files Unchanged

| File | Reason |
|------|--------|
| `tools/sbi_analyzer.py` | Already fixed (NaN handling) |
| `tools/sbi_flow_estimator.py` | Already fixed (coupon bug + comments) |
| `tools/sbi_parser.py` | No changes needed |
| `tools/delayed_publication_downloader.py` | Out of scope |
