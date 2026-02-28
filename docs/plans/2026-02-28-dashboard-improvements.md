# Dashboard Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add unit tests for core calculation functions, a stale data warning, a "bonds approaching exit" tab, and segment-level duration/yield charts to the 12-month projection tab.

**Architecture:** Pure-additive changes. New `tests/` directory for pytest. Two `app.py` modifications (new tab + projection chart extension). One `tools/sbi_projector.py` extension to expose per-segment data. No existing logic removed.

**Note on Historical Analysis:** `analyze_historical_forecasts`, `analyze_historical_by_segment`, and `reconstruct_historical_durations` already exist in `tools/sbi_projector.py` and are fully wired into tab8 ("Historische Entwicklung"). That feature is complete — skip it.

**Tech Stack:** Python 3, pytest, pandas, Streamlit, Plotly

---

### Task 1: Test Infrastructure

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

**Step 1: Create the tests directory and empty `__init__.py`**

```bash
cd "c:/Users/gagin/OneDrive/Dokumente/12 AI/six-index-rebalancing"
mkdir tests
touch tests/__init__.py
```

**Step 2: Create `tests/conftest.py`**

```python
import sys
from pathlib import Path

# Make tool modules importable from tests
sys.path.insert(0, str(Path(__file__).parent.parent / 'tools'))
```

**Step 3: Verify pytest finds the project**

```bash
python -m pytest tests/ --collect-only
```

Expected: `no tests ran` (no tests yet, but no import errors)

**Step 4: Commit**

```bash
git add tests/__init__.py tests/conftest.py
git commit -m "test: add pytest infrastructure with tools/ path setup"
```

---

### Task 2: Unit Tests for `estimate_duration_from_maturity`

**Files:**
- Create: `tests/test_calculations.py`
- Validates: `tools/sbi_flow_estimator.py:245-264`

**Step 1: Write the tests**

Create `tests/test_calculations.py`:

```python
import math
import numpy as np
import pandas as pd
import pytest
from sbi_flow_estimator import estimate_duration_from_maturity


class TestEstimateDuration:
    def test_zero_coupon_equals_maturity(self):
        """Zero-coupon bond: modified duration = maturity (exact)."""
        assert estimate_duration_from_maturity(10, 0) == 10.0

    def test_one_percent_coupon_ten_year(self):
        """1% coupon 10y par bond: modified duration = (1/0.01)*(1 - 1.01^-10) ≈ 9.471."""
        result = estimate_duration_from_maturity(10, 1)
        assert abs(result - 9.471) < 0.01

    def test_five_percent_coupon_ten_year(self):
        """5% coupon 10y par bond: modified duration = (1/0.05)*(1 - 1.05^-10) ≈ 7.722."""
        result = estimate_duration_from_maturity(10, 5)
        assert abs(result - 7.722) < 0.01

    def test_zero_maturity_returns_zero(self):
        assert estimate_duration_from_maturity(0, 1) == 0.0

    def test_negative_maturity_returns_zero(self):
        assert estimate_duration_from_maturity(-5, 1) == 0.0

    def test_nan_maturity_returns_zero(self):
        assert estimate_duration_from_maturity(float('nan'), 1) == 0.0

    def test_negative_coupon_treated_as_zero_coupon(self):
        """Negative coupon → falls back to zero-coupon: duration = maturity."""
        assert estimate_duration_from_maturity(10, -1) == 10.0

    def test_nan_coupon_treated_as_zero_coupon(self):
        assert estimate_duration_from_maturity(10, float('nan')) == 10.0

    def test_duration_less_than_maturity_for_positive_coupon(self):
        """For any positive coupon, modified duration < maturity."""
        result = estimate_duration_from_maturity(10, 2)
        assert result < 10.0
        assert result > 0.0

    def test_higher_coupon_means_shorter_duration(self):
        """Higher coupon → cash flows arrive sooner → shorter duration."""
        d_low = estimate_duration_from_maturity(10, 1)
        d_high = estimate_duration_from_maturity(10, 5)
        assert d_low > d_high
```

**Step 2: Run tests — expect PASS (implementation already updated)**

```bash
python -m pytest tests/test_calculations.py::TestEstimateDuration -v
```

Expected: all 10 tests PASS

**Step 3: Commit**

```bash
git add tests/test_calculations.py
git commit -m "test: add unit tests for estimate_duration_from_maturity"
```

---

### Task 3: Unit Tests for `calc_segment_duration` and `calc_segment_yield`

**Files:**
- Modify: `tests/test_calculations.py`
- Validates: `tools/sbi_analyzer.py:98-158` (NaN fix we already applied)

**Step 1: Append to `tests/test_calculations.py`**

```python
from sbi_analyzer import calc_segment_duration, calc_segment_yield


def make_bonds(durations, nominals, prices=None, ytws=None):
    """Helper: build a minimal constituent DataFrame for 'total' segment tests."""
    n = len(durations)
    return pd.DataFrame({
        'nominal': nominals,
        'price': prices if prices else [100.0] * n,
        'duration': durations,
        'ytw': ytws if ytws else [1.0] * n,
    })


class TestCalcSegmentDuration:
    def test_equal_weights(self):
        df = make_bonds([5.0, 5.0], [1_000_000, 1_000_000])
        assert calc_segment_duration(df, 'total') == 5.0

    def test_mcap_weighted_correctly(self):
        # Bond 1: mcap=1M dur=2y, Bond 2: mcap=2M dur=8y
        # Expected: (2*1 + 8*2) / 3 = 18/3 = 6.0
        df = make_bonds([2.0, 8.0], [1_000_000, 2_000_000])
        assert abs(calc_segment_duration(df, 'total') - 6.0) < 0.001

    def test_nan_duration_excluded_from_numerator_and_denominator(self):
        # Bond 2 has NaN duration — must be excluded from BOTH num and denom.
        # If only excluded from numerator, result would be 5*1M / 3M = 1.67 (wrong).
        # Correct: 5*1M / 1M = 5.0
        df = make_bonds([5.0, np.nan], [1_000_000, 2_000_000])
        assert calc_segment_duration(df, 'total') == 5.0

    def test_all_nan_durations_returns_zero(self):
        df = make_bonds([np.nan, np.nan], [1_000_000, 1_000_000])
        assert calc_segment_duration(df, 'total') == 0.0

    def test_empty_dataframe_returns_zero(self):
        df = make_bonds([], [])
        assert calc_segment_duration(df, 'total') == 0.0

    def test_single_bond(self):
        df = make_bonds([7.5], [500_000_000])
        assert calc_segment_duration(df, 'total') == 7.5


class TestCalcSegmentYield:
    def test_nan_yield_excluded_from_numerator_and_denominator(self):
        # Bond 2 has NaN ytw — must be excluded from both sides.
        # Correct: 1.0 (only bond 1 contributes)
        df = make_bonds([5.0, 5.0], [1_000_000, 2_000_000], ytws=[1.0, np.nan])
        assert calc_segment_yield(df, 'total') == 1.0

    def test_weighted_yield(self):
        # Bond 1: mcap=1M ytw=1%, Bond 2: mcap=3M ytw=2%
        # Expected: (1*1 + 2*3) / 4 = 7/4 = 1.75%
        df = make_bonds([5.0, 5.0], [1_000_000, 3_000_000], ytws=[1.0, 2.0])
        assert abs(calc_segment_yield(df, 'total') - 1.75) < 0.001

    def test_empty_dataframe_returns_zero(self):
        df = make_bonds([], [])
        assert calc_segment_yield(df, 'total') == 0.0
```

**Step 2: Run tests — expect all PASS**

```bash
python -m pytest tests/test_calculations.py -v
```

Expected: all tests PASS (NaN fix already applied)

**Step 3: Commit**

```bash
git add tests/test_calculations.py
git commit -m "test: add unit tests for calc_segment_duration and calc_segment_yield"
```

---

### Task 4: Extend `project_index_forward` with Per-Segment Data

**Files:**
- Modify: `tools/sbi_projector.py:94-119` (the results.append block)
- Test in: `tests/test_projector.py` (new file)

**Step 1: Write the failing test first**

Create `tests/test_projector.py`:

```python
import numpy as np
import pandas as pd
from sbi_projector import project_index_forward


def make_projection_df(n=10, sector_code='74010100', remaining_years=5.0):
    """Minimal constituent DataFrame for projection tests."""
    return pd.DataFrame({
        'nominal': [100_000_000] * n,
        'price': [100.0] * n,
        'duration': [5.0] * n,
        'ytw': [1.0] * n,
        'remaining_years': [remaining_years] * n,
        'guarantee_collateral_code': [sector_code] * n,
        'domicile': ['CH'] * n,
        'date': [pd.Timestamp('2025-01-30')] * n,
    })


class TestProjectIndexForward:
    def test_result_has_segment_durations_key(self):
        df = make_projection_df()
        results = project_index_forward(df, months=1)
        assert 'segment_durations' in results[0]

    def test_result_has_segment_yields_key(self):
        df = make_projection_df()
        results = project_index_forward(df, months=1)
        assert 'segment_yields' in results[0]

    def test_segment_durations_has_required_segments(self):
        df = make_projection_df()
        results = project_index_forward(df, months=1)
        for seg in ['total', 'government', 'pfandbrief', 'corporate']:
            assert seg in results[0]['segment_durations']

    def test_corporate_duration_matches_total_for_all_corporate(self):
        """When all bonds are corporate, corporate duration should equal total."""
        df = make_projection_df(sector_code='74010100')  # corporate financials
        results = project_index_forward(df, months=1)
        r = results[0]
        assert abs(r['segment_durations']['corporate'] - r['segment_durations']['total']) < 0.001

    def test_government_duration_zero_when_no_government_bonds(self):
        """When no government bonds, government duration should be 0."""
        df = make_projection_df(sector_code='74010100')  # all corporate
        results = project_index_forward(df, months=1)
        assert results[0]['segment_durations']['government'] == 0.0

    def test_returns_correct_number_of_months(self):
        df = make_projection_df()
        results = project_index_forward(df, months=6)
        assert len(results) == 6
```

**Step 2: Run test — expect FAIL on `segment_durations` key missing**

```bash
python -m pytest tests/test_projector.py -v
```

Expected: FAIL with `KeyError: 'segment_durations'`

**Step 3: Implement — modify `tools/sbi_projector.py`**

Find the `results.append({` block (around line 106). Add `segment_durations` and `segment_yields` to it:

```python
        # Calculate metrics on remaining bonds (with adjusted durations)
        if len(working_df) > 0:
            current_mcap = calc_index_mcap(working_df)
            current_duration = calc_segment_duration(working_df, 'total')
            current_ytw = calc_segment_yield(working_df, 'total')
            seg_stats = get_all_segment_stats(working_df)
            # Per-segment duration and yield for time-series charts
            segment_durations = {
                seg: round(calc_segment_duration(working_df, seg), 3)
                for seg in ['total', 'government', 'pfandbrief', 'corporate']
            }
            segment_yields = {
                seg: round(calc_segment_yield(working_df, seg), 4)
                for seg in ['total', 'government', 'pfandbrief', 'corporate']
            }
        else:
            current_mcap = 0.0
            current_duration = 0.0
            current_ytw = 0.0
            seg_stats = pd.DataFrame()
            segment_durations = {'total': 0.0, 'government': 0.0, 'pfandbrief': 0.0, 'corporate': 0.0}
            segment_yields = {'total': 0.0, 'government': 0.0, 'pfandbrief': 0.0, 'corporate': 0.0}

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
            'segment_durations': segment_durations,
            'segment_yields': segment_yields,
        })
```

**Step 4: Run tests — expect all PASS**

```bash
python -m pytest tests/test_projector.py -v
```

Expected: all 6 tests PASS

**Step 5: Run full test suite to check no regressions**

```bash
python -m pytest tests/ -v
```

Expected: all tests PASS

**Step 6: Commit**

```bash
git add tools/sbi_projector.py tests/test_projector.py
git commit -m "feat: add per-segment duration/yield to projection results"
```

---

### Task 5: Stale Data Warning in `app.py`

**Files:**
- Modify: `app.py` — two locations:
  1. After `data_loaded = True` (~line 138): add stale check and warning
  2. Sidebar Daten-Info (~line 169): show file age alongside date

**Step 1: Add stale data check after successful data load**

Find this block in `app.py` (around line 136):
```python
try:
    constituents_df, forecast_df, config = load_data()
    data_loaded = True
except Exception as e:
```

Add `import os` to the imports at the top if not already present (it isn't — add it with the other stdlib imports around line 9).

After `data_loaded = True`, insert:
```python
    data_loaded = True
    # Check constituent file freshness
    _const_path = Path(__file__).parent / 'close_sbr14d.csv'
    _mtime = datetime.fromtimestamp(os.path.getmtime(_const_path))
    _data_age_days = (datetime.now() - _mtime).days
```

**Step 2: Show the stale warning in the main content area**

In `app.py`, after the title and before the `if data_loaded:` block (around line 145), insert:
```python
if data_loaded and _data_age_days > 5:
    st.warning(
        f"⚠️ Konstituentendaten veraltet: Letzte Änderung {_mtime.strftime('%d.%m.%Y')} "
        f"({_data_age_days} Tage). Bitte `close_sbr14d.csv` aktualisieren."
    )
```

**Step 3: Update sidebar Daten-Info to show file age**

Find (around line 169):
```python
    st.sidebar.subheader("📅 Daten-Info")
    if 'date' in constituents_df.columns:
        st.sidebar.write(f"Stand: {constituents_df['date'].iloc[0].strftime('%d.%m.%Y')}")
    st.sidebar.write(f"Anzahl Bonds: {len(constituents_df)}")
```

Replace with:
```python
    st.sidebar.subheader("📅 Daten-Info")
    if 'date' in constituents_df.columns:
        data_date_str = constituents_df['date'].iloc[0].strftime('%d.%m.%Y')
        age_label = f" ⚠️ ({_data_age_days}d)" if _data_age_days > 5 else f" ✓ ({_data_age_days}d)"
        st.sidebar.write(f"Stand: {data_date_str}{age_label}")
    st.sidebar.write(f"Anzahl Bonds: {len(constituents_df)}")
```

**Step 4: Add `import os` to top-level imports**

Find the existing stdlib imports near the top of `app.py`:
```python
from pathlib import Path
import sys
```

Add `import os` on the next line.

**Step 5: Verify the app runs without error**

```bash
cd "c:/Users/gagin/OneDrive/Dokumente/12 AI/six-index-rebalancing"
streamlit run app.py --server.headless true &
# Wait 3 seconds, then kill
```

Or just do a Python syntax check:
```bash
python -c "import ast; ast.parse(open('app.py').read()); print('OK')"
```

Expected: `OK`

**Step 6: Commit**

```bash
git add app.py
git commit -m "feat: add stale data warning when constituent file > 5 days old"
```

---

### Task 6: Bonds Approaching Exit Tab

**Files:**
- Modify: `app.py` — add tab9 to the tabs array and implement its content

**Step 1: Add tab9 to the st.tabs call**

Find (line 175):
```python
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📈 Index Übersicht", "💹 Flow Analyse", "📋 Upcoming Events",
        "🔍 Bond Suche", "📊 Delayed Trades",
        "🔮 Index Projektion", "📝 Bond Anpassungen", "📅 Historische Entwicklung"
    ])
```

Replace with:
```python
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📈 Index Übersicht", "💹 Flow Analyse", "📋 Upcoming Events",
        "🔍 Bond Suche", "📊 Delayed Trades",
        "🔮 Index Projektion", "📝 Bond Anpassungen", "📅 Historische Entwicklung",
        "⏳ Baldige Exits"
    ])
```

**Step 2: Add the tab9 content block at the end of the `if data_loaded:` section**

Append after the `with tab8:` block (at the end of the file, around line 1540):

```python
    # ============ TAB 9: Bonds Approaching Exit ============
    with tab9:
        st.header("Baldige Index-Exits")
        st.markdown("*Bonds die die Mindest-Restlaufzeit von 1 Jahr bald unterschreiten*")

        threshold_years = st.slider(
            "Zeithorizont (Jahre)", min_value=1.0, max_value=3.0, value=2.0, step=0.5,
            key="exit_threshold",
            help="Zeigt Bonds mit Restlaufzeit ≤ diesem Wert"
        )

        upcoming = constituents_df[
            constituents_df['remaining_years'].notna() &
            (constituents_df['remaining_years'] <= threshold_years)
        ].copy().sort_values('remaining_years')

        if len(upcoming) == 0:
            st.info(f"Keine Bonds mit Restlaufzeit ≤ {threshold_years} Jahren gefunden.")
        else:
            total_mcap_all = calc_index_mcap(constituents_df)
            total_mcap_exits = calc_index_mcap(upcoming)
            avg_dur_exits = calc_segment_duration(upcoming, 'total')

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Anzahl Bonds", len(upcoming))
            with col2:
                st.metric("MCap Total", f"CHF {total_mcap_exits / 1e9:.2f} Mrd")
            with col3:
                st.metric("Index-Anteil", f"{total_mcap_exits / total_mcap_all * 100:.1f}%")
            with col4:
                st.metric("Ø Duration", f"{avg_dur_exits:.2f}y")

            st.markdown("---")

            # Per-segment breakdown
            st.subheader("Nach Segment")
            key_segs = ['government', 'agency', 'pfandbrief', 'corporate', 'domestic', 'foreign']
            seg_rows = []
            for seg in key_segs:
                seg_bonds = SEGMENTS[seg]['filter'](upcoming)
                if len(seg_bonds) > 0:
                    seg_rows.append({
                        'Segment': SEGMENTS[seg]['name'],
                        'Bonds': len(seg_bonds),
                        'MCap (Mrd CHF)': round(calc_index_mcap(seg_bonds) / 1e9, 2),
                        'Ø Duration (J)': round(calc_segment_duration(seg_bonds, 'total'), 2),
                        'MCap-Anteil (%)': round(calc_index_mcap(seg_bonds) / total_mcap_all * 100, 1),
                    })

            if seg_rows:
                st.dataframe(pd.DataFrame(seg_rows), hide_index=True, use_container_width=True)

            st.markdown("---")

            # Full bond list
            st.subheader(f"Bond-Liste ({len(upcoming)} Bonds)")
            display_cols = [c for c in ['isin', 'name', 'rating', 'remaining_years', 'duration', 'nominal']
                            if c in upcoming.columns]
            display = upcoming[display_cols].copy()
            if 'remaining_years' in display.columns:
                display['remaining_years'] = display['remaining_years'].round(2)
            if 'duration' in display.columns:
                display['duration'] = display['duration'].round(2)
            if 'nominal' in display.columns:
                display['nominal'] = (display['nominal'] / 1e6).round(1)

            col_labels = {
                'isin': 'ISIN', 'name': 'Name', 'rating': 'Rating',
                'remaining_years': 'Restlaufzeit (J)', 'duration': 'Duration (J)',
                'nominal': 'Nominal (Mio CHF)'
            }
            display = display.rename(columns=col_labels)
            st.dataframe(display, hide_index=True, use_container_width=True)

            csv = display.to_csv(index=False, sep=';').encode('utf-8-sig')
            st.download_button(
                "📥 Download CSV", data=csv,
                file_name=f"baldige_exits_{threshold_years}y_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
```

**Step 3: Syntax check**

```bash
python -c "import ast; ast.parse(open('app.py').read()); print('OK')"
```

Expected: `OK`

**Step 4: Commit**

```bash
git add app.py
git commit -m "feat: add 'Baldige Exits' tab showing bonds approaching minimum residual maturity"
```

---

### Task 7: Segment-Level Duration and Yield Charts in Projection Tab

**Files:**
- Modify: `app.py` — update the tab6 chart section (~line 960–993)

**Step 1: Locate the chart data construction in tab6**

Find this block (~line 960):
```python
            chart_data = pd.DataFrame([{
                'Monat': m['month_label'],
                'Bonds': m['remaining_bonds_count'],
                'Duration': m['duration'],
                'Exits': m['exit_count'],
            } for m in projection])
```

**Step 2: Replace with extended chart data including per-segment and yield**

```python
            chart_data = pd.DataFrame([{
                'Monat': m['month_label'],
                'Bonds': m['remaining_bonds_count'],
                'Duration': m['duration'],
                'Duration_Gov': m['segment_durations'].get('government', 0),
                'Duration_Pfand': m['segment_durations'].get('pfandbrief', 0),
                'Duration_Corp': m['segment_durations'].get('corporate', 0),
                'Yield': m['ytw'],
                'Yield_Gov': m['segment_yields'].get('government', 0),
                'Yield_Pfand': m['segment_yields'].get('pfandbrief', 0),
                'Yield_Corp': m['segment_yields'].get('corporate', 0),
                'Exits': m['exit_count'],
            } for m in projection])
```

**Step 3: After the existing exits/duration chart, add the segment duration chart**

Find where the existing chart ends (after `st.plotly_chart(fig, use_container_width=True)`) and insert before `st.markdown("---")`:

```python
            # Segment duration breakdown chart
            st.subheader("Duration nach Segment")
            fig_seg = go.Figure()
            seg_traces = [
                ('Duration', 'Gesamt', COLOR_PRIMARY, dict(width=3)),
                ('Duration_Gov', 'Staatsanleihen', '#f0a500', dict(width=2, dash='dot')),
                ('Duration_Pfand', 'Pfandbriefe', '#a0c4ff', dict(width=2, dash='dot')),
                ('Duration_Corp', 'Unternehmensanleihen', '#ff6b6b', dict(width=2, dash='dot')),
            ]
            for col, label, color, line_style in seg_traces:
                # Only show segments that have non-zero values
                if chart_data[col].max() > 0:
                    fig_seg.add_trace(go.Scatter(
                        x=chart_data['Monat'], y=chart_data[col],
                        name=label,
                        line=dict(color=color, **line_style),
                        mode='lines+markers',
                        marker=dict(size=4),
                        hovertemplate='%{x}<br>' + label + ': %{y:.3f}y<extra></extra>',
                    ))
            fig_seg.update_layout(
                **PLOTLY_LAYOUT,
                yaxis=dict(title='Duration (Jahre)', gridcolor="#2d3548",
                           zerolinecolor="#2d3548", tickfont=dict(color="#8b949e")),
                height=300,
                legend=dict(orientation='h', y=-0.25, font=dict(color="#8b949e"),
                            bgcolor="rgba(0,0,0,0)"),
                margin=dict(b=70),
                hovermode='x unified',
            )
            st.plotly_chart(fig_seg, use_container_width=True)

            # Yield projection chart
            st.subheader("Yield-to-Worst nach Segment")
            fig_ytw = go.Figure()
            ytw_traces = [
                ('Yield', 'Gesamt', COLOR_PRIMARY, dict(width=3)),
                ('Yield_Gov', 'Staatsanleihen', '#f0a500', dict(width=2, dash='dot')),
                ('Yield_Pfand', 'Pfandbriefe', '#a0c4ff', dict(width=2, dash='dot')),
                ('Yield_Corp', 'Unternehmensanleihen', '#ff6b6b', dict(width=2, dash='dot')),
            ]
            for col, label, color, line_style in ytw_traces:
                if chart_data[col].max() > 0:
                    fig_ytw.add_trace(go.Scatter(
                        x=chart_data['Monat'], y=chart_data[col],
                        name=label,
                        line=dict(color=color, **line_style),
                        mode='lines+markers',
                        marker=dict(size=4),
                        hovertemplate='%{x}<br>' + label + ': %{y:.4f}%<extra></extra>',
                    ))
            fig_ytw.update_layout(
                **PLOTLY_LAYOUT,
                yaxis=dict(title='YTW (%)', gridcolor="#2d3548",
                           zerolinecolor="#2d3548", tickfont=dict(color="#8b949e")),
                height=300,
                legend=dict(orientation='h', y=-0.25, font=dict(color="#8b949e"),
                            bgcolor="rgba(0,0,0,0)"),
                margin=dict(b=70),
                hovermode='x unified',
            )
            st.plotly_chart(fig_ytw, use_container_width=True)
```

**Step 4: Syntax check**

```bash
python -c "import ast; ast.parse(open('app.py').read()); print('OK')"
```

Expected: `OK`

**Step 5: Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: all tests PASS

**Step 6: Commit**

```bash
git add app.py
git commit -m "feat: add segment duration and yield charts to 12-month projection tab"
```

---

## Verification

After all tasks complete, start the app and verify:

```bash
streamlit run app.py
```

1. **Stale data warning** — temporarily rename `close_sbr14d.csv`, touch it with an old date, restart; confirm warning appears in header and sidebar shows age with ⚠️
2. **Baldige Exits tab** — navigate to ⏳ tab; adjust slider from 1y to 3y; verify bond count changes and segment breakdown updates; download CSV
3. **Segment projection** — navigate to 🔮 tab; run projection; confirm two new charts appear below the existing chart (Segment Duration, Yield by Segment); verify lines only appear for segments with bonds
4. **All tests green** — `python -m pytest tests/ -v` shows all PASS
