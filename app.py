"""
SBI Index Rebalancing Dashboard
Streamlit-basiertes Frontend für Index-Analyse und Flow-Schätzung
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add tools directory to path
tools_path = Path(__file__).parent / 'tools'
sys.path.insert(0, str(tools_path))

from sbi_parser import parse_constituents, parse_forecast, get_admissions, get_deletions, get_capital_changes
from sbi_analyzer import (calc_index_mcap, calc_segment_mcap, calc_segment_duration, calc_segment_yield,
                          get_all_segment_stats, get_rating_breakdown, get_top_bonds_by_weight,
                          identify_maturity_exits, SEGMENTS)
from sbi_flow_estimator import (generate_flow_report, summarize_flows, load_config,
                                get_all_segment_duration_impact, get_duration_detail_by_segment,
                                simulate_custom_bond_impact)
from sbi_projector import (project_index_forward, apply_bond_changes,
                           validate_changes_csv, compare_index_states,
                           analyze_historical_forecasts, analyze_historical_by_segment,
                           reconstruct_historical_durations, _SEGMENT_NAMES)
from delayed_publication_downloader import (load_all_delayed_publications, enrich_with_constituents,
                                             download_date_range, DEFAULT_OUTPUT_DIR)
import plotly.express as px
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="SBI Rebalancing Tool",
    page_icon="📊",
    layout="wide"
)

# Dark Finance Theme CSS
DARK_CSS = """
<style>
/* Metric cards */
[data-testid="stMetric"] {
    background: #1a1f2e;
    border: 1px solid #2d3548;
    border-radius: 8px;
    padding: 16px;
}
[data-testid="stMetricValue"] { color: #e6edf3; font-weight: 600; }
[data-testid="stMetricLabel"] { color: #8b949e; }

/* Expanders */
[data-testid="stExpander"] {
    border: 1px solid #2d3548;
    border-radius: 8px;
    background: #1a1f2e;
}

/* Alert boxes */
[data-testid="stAlert"] > div[data-baseweb="notification"] {
    border-radius: 6px;
}

/* Buttons */
.stDownloadButton > button {
    background: transparent;
    border: 1px solid #2d3548;
    color: #e6edf3;
    border-radius: 6px;
}
.stDownloadButton > button:hover {
    border-color: #00d4aa;
    color: #00d4aa;
}

/* Horizontal rules */
hr { border-color: #2d3548 !important; }

/* Tab styling */
.stTabs [data-baseweb="tab"] { color: #8b949e; }
.stTabs [aria-selected="true"] { color: #e6edf3; }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# Plotly dark layout template
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
    font=dict(color="#e6edf3", size=12),
    xaxis=dict(gridcolor="#2d3548", zerolinecolor="#2d3548", tickfont=dict(color="#8b949e")),
    hoverlabel=dict(bgcolor="#242b3d", font_color="#e6edf3", bordercolor="#2d3548"),
)
# Separate constants so callers can override without duplicate-kwarg TypeError
YAXIS_STYLE = dict(gridcolor="#2d3548", zerolinecolor="#2d3548", tickfont=dict(color="#8b949e"))
LEGEND_STYLE = dict(font=dict(color="#8b949e"), bgcolor="rgba(0,0,0,0)")

CHART_COLORS = ["#00d4aa", "#58a6ff", "#d4a017", "#ff6b6b", "#8b5cf6", "#06b6d4", "#f59e0b", "#ef4444"]

# Semantic color aliases
COLOR_POSITIVE = CHART_COLORS[0]   # #00d4aa - teal/green for admissions, positive values
COLOR_PRIMARY  = CHART_COLORS[1]   # #58a6ff - blue for duration, netto, primary lines
COLOR_ACCENT   = CHART_COLORS[2]   # #d4a017 - gold for rolling averages, secondary
COLOR_NEGATIVE = CHART_COLORS[3]   # #ff6b6b - red for exits, deletions, negative values

# Password protection
def check_password():
    """Returns True if the user has entered the correct password."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.title("SBI Rebalancing Tool")
    password = st.text_input("Password", type="password", key="password_input")
    if st.button("Login"):
        if password == st.secrets["passwords"]["dashboard_password"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password")
    return False

if not check_password():
    st.stop()

# Load data
@st.cache_data
def load_data():
    base_path = Path(__file__).parent
    constituents = parse_constituents(str(base_path / 'close_sbr14d.csv'))
    forecast = parse_forecast(str(base_path / 'forecast_bonds.csv'))
    config = load_config(str(base_path / 'config.json'))
    return constituents, forecast, config

try:
    constituents_df, forecast_df, config = load_data()
    data_loaded = True
except Exception as e:
    st.error(f"Fehler beim Laden der Daten: {e}")
    data_loaded = False

# Header
st.title("📊 SBI Index Rebalancing Tool")
st.markdown("*Analyse und Flow-Schätzung für SBI AAA-BBB Index*")

if data_loaded:
    # Sidebar
    st.sidebar.header("⚙️ Einstellungen")

    # AUM Slider
    st.sidebar.subheader("AUM Annahmen")
    aum_total = st.sidebar.slider(
        "SBI AAA-BBB Total (Mrd CHF)",
        min_value=1.0,
        max_value=10.0,
        value=config.get('aum', {}).get('sbi_aaa_bbb_total', 3_000_000_000) / 1e9,
        step=0.5
    ) * 1e9

    # Update config with slider value
    config_updated = config.copy()
    if 'aum' not in config_updated:
        config_updated['aum'] = {}
    config_updated['aum']['sbi_aaa_bbb_total'] = aum_total

    # Data info
    st.sidebar.markdown("---")
    st.sidebar.subheader("📅 Daten-Info")
    if 'date' in constituents_df.columns:
        st.sidebar.write(f"Stand: {constituents_df['date'].iloc[0].strftime('%d.%m.%Y')}")
    st.sidebar.write(f"Anzahl Bonds: {len(constituents_df)}")

    # Main content - Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📈 Index Übersicht", "💹 Flow Analyse", "📋 Upcoming Events",
        "🔍 Bond Suche", "📊 Delayed Trades",
        "🔮 Index Projektion", "📝 Bond Anpassungen", "📅 Historische Entwicklung"
    ])

    # ============ TAB 1: Index Overview ============
    with tab1:
        st.header("Index Übersicht")

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        total_mcap = calc_index_mcap(constituents_df)
        duration = calc_segment_duration(constituents_df, 'total')
        ytw = calc_segment_yield(constituents_df, 'total')

        with col1:
            st.metric("Total Market Cap", f"CHF {total_mcap/1e9:.1f} Mrd")
        with col2:
            st.metric("Anzahl Bonds", f"{len(constituents_df)}")
        with col3:
            st.metric("Duration", f"{duration:.2f} Jahre")
        with col4:
            st.metric("Yield to Worst", f"{ytw:.3f}%")

        st.markdown("---")

        # Segment breakdown
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Segment Breakdown")
            segment_stats = get_all_segment_stats(constituents_df)
            segment_display = segment_stats[segment_stats['count'] > 0][['name', 'count', 'weight_pct', 'duration']].copy()
            segment_display.columns = ['Segment', 'Anzahl', 'Gewicht %', 'Duration']
            segment_display['Gewicht %'] = segment_display['Gewicht %'].round(1)
            segment_display['Duration'] = segment_display['Duration'].round(2)
            st.dataframe(segment_display, hide_index=True, width='stretch')

        with col2:
            st.subheader("Rating Breakdown")
            rating_stats = get_rating_breakdown(constituents_df)
            rating_display = rating_stats[['rating', 'count', 'weight_pct', 'duration']].copy()
            rating_display.columns = ['Rating', 'Anzahl', 'Gewicht %', 'Duration']
            rating_display['Gewicht %'] = rating_display['Gewicht %'].round(1)
            rating_display['Duration'] = rating_display['Duration'].round(2)
            st.dataframe(rating_display, hide_index=True, width='stretch')

        st.markdown("---")

        # Top bonds
        st.subheader("Top 20 Bonds nach Gewichtung")
        top_bonds = get_top_bonds_by_weight(constituents_df, 20)
        top_display = top_bonds.copy()
        top_display['weight_pct'] = top_display['weight_pct'].round(3)
        top_display['duration'] = top_display['duration'].round(2)
        top_display.columns = ['ISIN', 'Name', 'Gewicht %', 'Duration', 'Rating', 'Sektor']
        st.dataframe(top_display, hide_index=True, width='stretch')

    # ============ TAB 2: Flow Analysis ============
    with tab2:
        st.header("Flow Analyse")
        st.markdown(f"*Basierend auf AUM von CHF {aum_total/1e9:.1f} Mrd*")

        # Generate flow report
        flow_report = generate_flow_report(forecast_df, constituents_df, config_updated)

        if len(flow_report) > 0:
            summary = summarize_flows(flow_report)

            # Key flow metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Buy Pressure",
                    f"CHF {summary['total_buy_flow']/1e6:.1f} Mio",
                    delta=None
                )
            with col2:
                st.metric(
                    "Sell Pressure",
                    f"CHF {summary['total_sell_flow']/1e6:.1f} Mio",
                    delta=None
                )
            with col3:
                net_color = "normal" if summary['net_flow'] >= 0 else "inverse"
                st.metric(
                    "Net Flow",
                    f"CHF {summary['net_flow']/1e6:.1f} Mio",
                    delta=f"{'BUY' if summary['net_flow'] > 0 else 'SELL'}"
                )

            st.markdown("---")

            # Duration Impact Section
            st.subheader("📐 Duration Impact nach Rebalancing")
            st.markdown("*Geschätzte Änderung der Index-Duration durch Admissions/Deletions*")

            try:
                duration_impact = get_all_segment_duration_impact(forecast_df, constituents_df)

                # Display duration changes for key segments
                col1, col2, col3 = st.columns(3)

                # Total
                total_impact = duration_impact[duration_impact['segment'] == 'total'].iloc[0] if len(duration_impact[duration_impact['segment'] == 'total']) > 0 else None
                if total_impact is not None:
                    with col1:
                        delta_val = total_impact['duration_change']
                        delta_str = f"{delta_val:+.3f}y" if delta_val != 0 else "0"
                        st.metric(
                            "Total Index",
                            f"{total_impact['new_duration']:.2f}y",
                            delta=delta_str,
                            delta_color="inverse" if delta_val < 0 else "normal"
                        )
                        st.caption(f"Aktuell: {total_impact['current_duration']:.2f}y")

                # Domestic
                domestic_impact = duration_impact[duration_impact['segment'] == 'domestic'].iloc[0] if len(duration_impact[duration_impact['segment'] == 'domestic']) > 0 else None
                if domestic_impact is not None:
                    with col2:
                        delta_val = domestic_impact['duration_change']
                        delta_str = f"{delta_val:+.3f}y" if delta_val != 0 else "0"
                        st.metric(
                            "Domestic",
                            f"{domestic_impact['new_duration']:.2f}y",
                            delta=delta_str,
                            delta_color="inverse" if delta_val < 0 else "normal"
                        )
                        st.caption(f"Aktuell: {domestic_impact['current_duration']:.2f}y")

                # Foreign
                foreign_impact = duration_impact[duration_impact['segment'] == 'foreign'].iloc[0] if len(duration_impact[duration_impact['segment'] == 'foreign']) > 0 else None
                if foreign_impact is not None:
                    with col3:
                        delta_val = foreign_impact['duration_change']
                        delta_str = f"{delta_val:+.3f}y" if delta_val != 0 else "0"
                        st.metric(
                            "Foreign",
                            f"{foreign_impact['new_duration']:.2f}y",
                            delta=delta_str,
                            delta_color="inverse" if delta_val < 0 else "normal"
                        )
                        st.caption(f"Aktuell: {foreign_impact['current_duration']:.2f}y")

                # Second row for more segments
                col4, col5, col6 = st.columns(3)

                # Corporate
                corp_impact = duration_impact[duration_impact['segment'] == 'corporate'].iloc[0] if len(duration_impact[duration_impact['segment'] == 'corporate']) > 0 else None
                if corp_impact is not None:
                    with col4:
                        delta_val = corp_impact['duration_change']
                        delta_str = f"{delta_val:+.3f}y" if delta_val != 0 else "0"
                        st.metric(
                            "Corporate",
                            f"{corp_impact['new_duration']:.2f}y",
                            delta=delta_str,
                            delta_color="inverse" if delta_val < 0 else "normal"
                        )
                        st.caption(f"Aktuell: {corp_impact['current_duration']:.2f}y")

                # Pfandbrief
                pfand_impact = duration_impact[duration_impact['segment'] == 'pfandbrief'].iloc[0] if len(duration_impact[duration_impact['segment'] == 'pfandbrief']) > 0 else None
                if pfand_impact is not None:
                    with col5:
                        delta_val = pfand_impact['duration_change']
                        delta_str = f"{delta_val:+.3f}y" if delta_val != 0 else "0"
                        st.metric(
                            "Pfandbrief",
                            f"{pfand_impact['new_duration']:.2f}y",
                            delta=delta_str,
                            delta_color="inverse" if delta_val < 0 else "normal"
                        )
                        st.caption(f"Aktuell: {pfand_impact['current_duration']:.2f}y")

                # Government
                gov_impact = duration_impact[duration_impact['segment'] == 'government'].iloc[0] if len(duration_impact[duration_impact['segment'] == 'government']) > 0 else None
                if gov_impact is not None:
                    with col6:
                        delta_val = gov_impact['duration_change']
                        delta_str = f"{delta_val:+.3f}y" if delta_val != 0 else "0"
                        st.metric(
                            "Government",
                            f"{gov_impact['new_duration']:.2f}y",
                            delta=delta_str,
                            delta_color="inverse" if delta_val < 0 else "normal"
                        )
                        st.caption(f"Aktuell: {gov_impact['current_duration']:.2f}y")

                # Segment overview table
                with st.expander("📊 Duration-Übersicht alle Segmente"):
                    dur_display = duration_impact[['segment_name', 'current_mcap', 'new_mcap', 'current_duration', 'new_duration', 'duration_change', 'admissions_count', 'deletions_count']].copy()
                    dur_display['current_mcap'] = (dur_display['current_mcap'] / 1e9).round(2)
                    dur_display['new_mcap'] = (dur_display['new_mcap'] / 1e9).round(2)
                    dur_display.columns = ['Segment', 'MCap Akt. (Mrd)', 'MCap Neu (Mrd)', 'Dur. Aktuell', 'Dur. Neu', 'Δ Duration', 'Admissions', 'Deletions']
                    st.dataframe(dur_display, hide_index=True, width='stretch')

                # Per-segment bond-level detail
                st.markdown("**Bond-Level Duration Detail pro Segment**")
                seg_choice = st.selectbox(
                    "Segment wählen",
                    options=['total', 'domestic', 'foreign', 'corporate', 'pfandbrief', 'government'],
                    format_func=lambda x: SEGMENTS[x]['name'],
                    key="dur_detail_segment"
                )

                detail_df = get_duration_detail_by_segment(forecast_df, constituents_df, seg_choice)
                if len(detail_df) > 0:
                    # Split into admissions and deletions
                    adm_detail = detail_df[detail_df['event'] == 'ADMISSION'].copy()
                    del_detail = detail_df[detail_df['event'] == 'DELETION'].copy()

                    col_a, col_d = st.columns(2)

                    with col_a:
                        st.markdown(f"**🟢 Admissions ({len(adm_detail)})**")
                        if len(adm_detail) > 0:
                            adm_show = adm_detail[['isin', 'name', 'rating', 'nominal_mio', 'duration', 'weight_pct', 'duration_source']].copy()
                            adm_show.columns = ['ISIN', 'Name', 'Rating', 'Nom (Mio)', 'Duration', 'Gew %', 'Quelle']
                            st.dataframe(adm_show, hide_index=True, width='stretch', height=300)
                        else:
                            st.info("Keine Admissions in diesem Segment")

                    with col_d:
                        st.markdown(f"**🔴 Deletions ({len(del_detail)})**")
                        if len(del_detail) > 0:
                            del_show = del_detail[['isin', 'name', 'rating', 'nominal_mio', 'duration', 'weight_pct', 'duration_source']].copy()
                            del_show.columns = ['ISIN', 'Name', 'Rating', 'Nom (Mio)', 'Duration', 'Gew %', 'Quelle']
                            st.dataframe(del_show, hide_index=True, width='stretch', height=300)
                        else:
                            st.info("Keine Deletions in diesem Segment")

                    # Summary stats
                    if len(adm_detail) > 0 and len(del_detail) > 0:
                        avg_adm_dur = (adm_detail['duration'] * adm_detail['nominal_mio']).sum() / adm_detail['nominal_mio'].sum()
                        avg_del_dur = (del_detail['duration'] * del_detail['nominal_mio']).sum() / del_detail['nominal_mio'].sum()
                        st.info(
                            f"Ø Duration Admissions: **{avg_adm_dur:.2f}y** ({adm_detail['nominal_mio'].sum():.0f} Mio) | "
                            f"Ø Duration Deletions: **{avg_del_dur:.2f}y** ({del_detail['nominal_mio'].sum():.0f} Mio)"
                        )
                else:
                    st.info("Keine Events in diesem Segment")

                st.markdown("---")

                # What-If Simulator
                st.subheader("🧪 What-If Simulator")
                st.markdown("*Simuliere den Impact eines hypothetischen Bonds auf den Index*")

                sim_col1, sim_col2, sim_col3, sim_col4 = st.columns(4)

                with sim_col1:
                    sim_nominal = st.number_input("Nominal (Mio CHF)", min_value=100, max_value=5000, value=200, step=50, key="sim_nom")
                with sim_col2:
                    sim_duration = st.number_input("Duration (Jahre)", min_value=0.5, max_value=50.0, value=10.0, step=0.5, key="sim_dur")
                with sim_col3:
                    sim_rating = st.selectbox("Rating", ['AAA', 'AA', 'A', 'BBB'], index=2, key="sim_rating")
                with sim_col4:
                    sim_event = st.selectbox("Event", ['ADMISSION', 'DELETION'], key="sim_event")

                sim_segments = ['total', 'domestic', 'foreign', 'corporate', 'pfandbrief', 'government']
                sim_results = []
                for seg in sim_segments:
                    r = simulate_custom_bond_impact(
                        constituents_df,
                        sim_nominal * 1e6,
                        sim_duration,
                        sim_rating,
                        sim_event,
                        seg
                    )
                    if 'error' not in r:
                        sim_results.append(r)

                if sim_results:
                    sim_df = pd.DataFrame(sim_results)
                    sim_display = sim_df[['segment_name', 'current_duration', 'new_duration', 'duration_change', 'bond_weight_pct', 'current_mcap_mrd', 'new_mcap_mrd']].copy()
                    sim_display.columns = ['Segment', 'Dur. Aktuell', 'Dur. Neu', 'Δ Duration', 'Gewicht %', 'MCap Akt. (Mrd)', 'MCap Neu (Mrd)']

                    st.dataframe(sim_display, hide_index=True, width='stretch')

                    # Highlight key numbers
                    total_sim = sim_df[sim_df['segment'] == 'total'].iloc[0]
                    direction = "verlängert" if total_sim['duration_change'] > 0 else "verkürzt"
                    st.success(
                        f"**{sim_event}** von CHF {sim_nominal} Mio mit {sim_duration:.1f}y Duration "
                        f"{direction} den Total-Index um **{total_sim['duration_change']:+.3f}y** "
                        f"(Gewicht: {total_sim['bond_weight_pct']:.3f}%)"
                    )

            except Exception as e:
                st.warning(f"Duration-Impact konnte nicht berechnet werden: {e}")

            st.markdown("---")

            # By event type
            st.subheader("Flows nach Event-Typ")
            event_data = []
            for event, flow in summary['by_event_type'].items():
                count = summary['count_by_event'][event]
                event_data.append({
                    'Event': event,
                    'Anzahl': count,
                    'Flow (Mio CHF)': round(flow / 1e6, 2)
                })
            st.dataframe(pd.DataFrame(event_data), hide_index=True, width='stretch')

            st.markdown("---")

            # Detailed flow report
            st.subheader("Detaillierter Flow Report")

            # Filter options
            col1, col2 = st.columns(2)
            with col1:
                direction_filter = st.selectbox("Flow Richtung", ["Alle", "BUY", "SELL"])
            with col2:
                event_filter = st.selectbox("Event Typ", ["Alle"] + list(flow_report['event_type'].unique()))

            filtered_report = flow_report.copy()
            if direction_filter != "Alle":
                filtered_report = filtered_report[filtered_report['flow_direction'] == direction_filter]
            if event_filter != "Alle":
                filtered_report = filtered_report[filtered_report['event_type'] == event_filter]

            # Display columns
            display_cols = ['isin', 'name', 'event_type', 'flow_direction', 'flow_chf', 'est_weight_pct', 'rating']
            display_df = filtered_report[display_cols].copy()
            display_df['flow_chf'] = (display_df['flow_chf'] / 1e6).round(2)
            display_df['est_weight_pct'] = display_df['est_weight_pct'].round(3)
            display_df.columns = ['ISIN', 'Name', 'Event', 'Richtung', 'Flow (Mio)', 'Gewicht %', 'Rating']

            st.dataframe(display_df.sort_values('Flow (Mio)', ascending=False), hide_index=True, width='stretch')
        else:
            st.info("Keine Flow-Events gefunden.")

    # ============ TAB 3: Upcoming Events ============
    with tab3:
        st.header("Upcoming Events")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🟢 Admissions")
            admissions = get_admissions(forecast_df)
            if len(admissions) > 0:
                adm_display = admissions[['isin', 'name', 'nominal', 'rating', 'admission_date']].copy()
                adm_display['nominal'] = (adm_display['nominal'] / 1e6).round(0).astype(int)
                adm_display.columns = ['ISIN', 'Name', 'Nominal (Mio)', 'Rating', 'Datum']
                st.dataframe(adm_display, hide_index=True, width='stretch')
                st.write(f"**Total: {len(admissions)} Bonds, CHF {admissions['nominal'].sum()/1e6:.0f} Mio**")
            else:
                st.info("Keine Admissions geplant")

        with col2:
            st.subheader("🔴 Deletions")
            deletions = get_deletions(forecast_df)
            if len(deletions) > 0:
                del_display = deletions[['isin', 'name', 'nominal', 'rating', 'cancellation_date']].copy()
                del_display['nominal'] = (del_display['nominal'] / 1e6).round(0).astype(int)
                del_display.columns = ['ISIN', 'Name', 'Nominal (Mio)', 'Rating', 'Datum']
                st.dataframe(del_display, hide_index=True, width='stretch')
                st.write(f"**Total: {len(deletions)} Bonds, CHF {deletions['nominal'].sum()/1e6:.0f} Mio**")
            else:
                st.info("Keine Deletions geplant")

        st.markdown("---")

        # Capital changes
        st.subheader("🔄 Capital Changes")
        cap_changes = get_capital_changes(forecast_df)
        if len(cap_changes) > 0:
            cap_display = cap_changes[['isin', 'name', 'capital_change', 'rating']].copy()
            cap_display['capital_change'] = (cap_display['capital_change'] / 1e6).round(0).astype(int)
            cap_display.columns = ['ISIN', 'Name', 'Änderung (Mio)', 'Rating']
            st.dataframe(cap_display, hide_index=True, width='stretch')
        else:
            st.info("Keine Capital Changes geplant")

        st.markdown("---")

        # Maturity exits
        st.subheader("⏰ Bald ablaufende Bonds (< 1.5 Jahre)")
        exits = identify_maturity_exits(constituents_df, horizon_days=180)
        if len(exits) > 0:
            exit_display = exits[['isin', 'name', 'remaining_years', 'rating']].head(20).copy()
            exit_display['remaining_years'] = exit_display['remaining_years'].round(2)
            exit_display.columns = ['ISIN', 'Name', 'Rest-Laufzeit (Jahre)', 'Rating']
            st.dataframe(exit_display, hide_index=True, width='stretch')
            st.write(f"**Total: {len(exits)} Bonds fallen in den nächsten 6 Monaten raus**")
        else:
            st.info("Keine Bonds fallen in den nächsten 6 Monaten raus")

    # ============ TAB 4: Bond Search ============
    with tab4:
        st.header("Bond Suche")

        search_term = st.text_input("ISIN oder Name eingeben", placeholder="z.B. CH1498422810 oder Zürich")

        if search_term:
            # Search in constituents
            mask = (
                constituents_df['isin'].str.contains(search_term, case=False, na=False) |
                constituents_df['name'].str.contains(search_term, case=False, na=False)
            )
            results = constituents_df[mask]

            if len(results) > 0:
                st.subheader(f"Gefunden: {len(results)} Bonds")

                for _, bond in results.head(10).iterrows():
                    with st.expander(f"{bond['isin']} - {bond['name'][:50]}"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"**ISIN:** {bond['isin']}")
                            st.write(f"**Rating:** {bond.get('rating', 'N/A')}")
                            st.write(f"**Sektor:** {bond.get('sector_name', 'N/A')}")
                        with col2:
                            st.write(f"**Nominal:** CHF {bond.get('nominal', 0)/1e6:.0f} Mio")
                            st.write(f"**Preis:** {bond.get('price', 0):.2f}%")
                            if 'market_cap' in bond:
                                st.write(f"**Market Cap:** CHF {bond['market_cap']/1e6:.1f} Mio")
                        with col3:
                            st.write(f"**Duration:** {bond.get('duration', 0):.2f}")
                            st.write(f"**YTW:** {bond.get('ytw', 0):.3f}%")
                            st.write(f"**Rest-Laufzeit:** {bond.get('remaining_years', 0):.2f} Jahre")

                        # Calculate weight
                        weight = bond.get('market_cap', bond['nominal'] * bond['price'] / 100) / total_mcap * 100
                        st.write(f"**Index-Gewicht:** {weight:.4f}%")
            else:
                st.warning("Keine Bonds gefunden")

        st.markdown("---")
        st.subheader("📊 Quick Stats")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Durchschn. Duration", f"{constituents_df['duration'].mean():.2f}")
        with col2:
            st.metric("Durchschn. YTW", f"{constituents_df['ytw'].mean():.3f}%")
        with col3:
            st.metric("Durchschn. Nominal", f"CHF {constituents_df['nominal'].mean()/1e6:.0f} Mio")

    # ============ TAB 5: Delayed Trades ============
    with tab5:
        st.header("Delayed Publication Trades")
        st.markdown("*OTC Bond-Trades mit Segment- und Rating-Analyse*")

        # Load delayed publication data
        @st.cache_data(ttl=300)
        def load_delayed_data():
            delayed_df = load_all_delayed_publications(DEFAULT_OUTPUT_DIR)
            if not delayed_df.empty:
                delayed_df = enrich_with_constituents(delayed_df, constituents_df)
            return delayed_df

        delayed_df = load_delayed_data()

        if delayed_df.empty:
            st.warning("Keine Delayed Publication Daten vorhanden.")
            if st.button("📥 Daten herunterladen"):
                with st.spinner("Lade Daten von Six..."):
                    start_date = datetime.now() - timedelta(days=30)
                    end_date = datetime.now()
                    results = download_date_range(start_date, end_date, DEFAULT_OUTPUT_DIR, skip_existing=True, delay_seconds=0.3)
                    st.success(f"✅ {results['downloaded']} Dateien heruntergeladen")
                    load_delayed_data.clear()
                    st.rerun()
        else:
            # Sidebar filters for this tab (in expander)
            with st.expander("🔍 Filter", expanded=True):
                col1, col2, col3 = st.columns(3)

                with col1:
                    # Segment Level 1 filter
                    segment_l1_options = ['Alle'] + list(delayed_df['segment_level1'].dropna().unique())
                    segment_l1 = st.selectbox("Segment (Level 1)", segment_l1_options, key="delayed_seg_l1")

                with col2:
                    # Segment (sector_code) filter - depends on Level 1
                    if segment_l1 != 'Alle':
                        sector_options = ['Alle'] + list(delayed_df[delayed_df['segment_level1'] == segment_l1]['sector_code'].dropna().unique())
                    else:
                        sector_options = ['Alle'] + list(delayed_df['sector_code'].dropna().unique())
                    sector_filter = st.selectbox("Segment (Detail)", sector_options, key="delayed_sector")

                with col3:
                    # Rating filter
                    rating_options = delayed_df['rating'].dropna().unique().tolist()
                    rating_filter = st.multiselect("Rating", rating_options, default=rating_options, key="delayed_rating")

                col4, col5 = st.columns(2)
                with col4:
                    # Date filter
                    if 'file_date' in delayed_df.columns:
                        min_date = delayed_df['file_date'].min().date()
                        max_date = delayed_df['file_date'].max().date()
                        date_range = st.date_input("Datumsbereich", value=(min_date, max_date), key="delayed_date")

                with col5:
                    # Only index members
                    only_index = st.checkbox("Nur Index-Mitglieder", value=False, key="delayed_index_only")

            # Apply filters
            filtered_df = delayed_df.copy()

            if segment_l1 != 'Alle':
                filtered_df = filtered_df[filtered_df['segment_level1'] == segment_l1]

            if sector_filter != 'Alle':
                filtered_df = filtered_df[filtered_df['sector_code'] == sector_filter]

            if rating_filter:
                filtered_df = filtered_df[filtered_df['rating'].isin(rating_filter)]

            if 'file_date' in filtered_df.columns and len(date_range) == 2:
                filtered_df = filtered_df[
                    (filtered_df['file_date'].dt.date >= date_range[0]) &
                    (filtered_df['file_date'].dt.date <= date_range[1])
                ]

            if only_index:
                filtered_df = filtered_df[filtered_df['in_index'] == True]

            # Key metrics
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Trades", f"{len(filtered_df):,}")
            with col2:
                total_turnover = filtered_df['turnover_chf'].sum() if 'turnover_chf' in filtered_df.columns else 0
                st.metric("Total Turnover", f"CHF {total_turnover/1e6:.1f} Mio")
            with col3:
                unique_bonds = filtered_df['product_isin'].nunique() if 'product_isin' in filtered_df.columns else 0
                st.metric("Unique Bonds", f"{unique_bonds}")
            with col4:
                avg_size = filtered_df['trade_size'].mean() if 'trade_size' in filtered_df.columns else 0
                st.metric("Avg Trade Size", f"CHF {avg_size/1e3:.0f}k")

            st.markdown("---")

            # Charts
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.subheader("Turnover nach Segment")
                if 'sector_code' in filtered_df.columns and 'turnover_chf' in filtered_df.columns:
                    segment_turnover = filtered_df.groupby('sector_code')['turnover_chf'].sum().reset_index()
                    segment_turnover = segment_turnover[segment_turnover['turnover_chf'] > 0]
                    if not segment_turnover.empty:
                        fig1 = px.pie(segment_turnover, values='turnover_chf', names='sector_code',
                                     title='', hole=0.4,
                                     color_discrete_sequence=CHART_COLORS)
                        fig1.update_layout(height=350, yaxis=YAXIS_STYLE, legend=LEGEND_STYLE, **PLOTLY_LAYOUT)
                        st.plotly_chart(fig1, width='stretch')
                    else:
                        st.info("Keine Segment-Daten für Filter")

            with chart_col2:
                st.subheader("Turnover nach Rating")
                if 'rating' in filtered_df.columns and 'turnover_chf' in filtered_df.columns:
                    rating_turnover = filtered_df.groupby('rating')['turnover_chf'].sum().reset_index()
                    rating_turnover = rating_turnover[rating_turnover['turnover_chf'] > 0]
                    if not rating_turnover.empty:
                        fig2 = px.bar(rating_turnover, x='rating', y='turnover_chf',
                                     labels={'rating': 'Rating', 'turnover_chf': 'Turnover CHF'},
                                     color_discrete_sequence=[CHART_COLORS[1]])
                        fig2.update_layout(height=350, yaxis=YAXIS_STYLE, legend=LEGEND_STYLE, **PLOTLY_LAYOUT)
                        st.plotly_chart(fig2, width='stretch')
                    else:
                        st.info("Keine Rating-Daten für Filter")

            # Daily turnover chart
            st.subheader("Turnover pro Tag")
            if 'file_date' in filtered_df.columns and 'turnover_chf' in filtered_df.columns:
                daily_turnover = filtered_df.groupby('file_date')['turnover_chf'].sum().reset_index()
                fig3 = px.bar(daily_turnover, x='file_date', y='turnover_chf',
                             labels={'file_date': 'Datum', 'turnover_chf': 'Turnover CHF'},
                             color_discrete_sequence=[CHART_COLORS[0]])
                fig3.update_layout(height=300, yaxis=YAXIS_STYLE, legend=LEGEND_STYLE, **PLOTLY_LAYOUT)
                st.plotly_chart(fig3, width='stretch')

            # Top bonds by turnover - with expandable trade details
            st.subheader("Top 15 Bonds (aufklappbar mit Yield)")
            st.markdown("*Klicke auf einen Bond um Einzeltrades mit Yield zu sehen*")

            if 'product_isin' in filtered_df.columns:
                # Aggregate by ISIN
                top_isins = filtered_df.groupby('product_isin').agg({
                    'turnover_chf': 'sum',
                    'product_short_name': 'first',
                    'rating': 'first',
                    'sector_code': 'first',
                    'coupon': 'first',
                    'remaining_years': 'first'
                }).reset_index().nlargest(15, 'turnover_chf')

                for _, bond in top_isins.iterrows():
                    isin = bond['product_isin']
                    name = str(bond.get('product_short_name', isin))[:40]
                    turnover = bond['turnover_chf']
                    isin_trades = filtered_df[filtered_df['product_isin'] == isin]
                    trade_count = len(isin_trades)

                    avg_yield = isin_trades['trade_yield'].mean() if 'trade_yield' in isin_trades.columns else None
                    yield_str = f" | Ø Yield: {avg_yield:.3f}%" if avg_yield and not pd.isna(avg_yield) else ""

                    with st.expander(f"**{isin}** - {name} | {trade_count} Trades | CHF {turnover/1e6:.2f} Mio{yield_str}"):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Turnover", f"CHF {turnover/1e6:.2f} Mio")
                        with col2:
                            rating = bond.get('rating', 'N/A')
                            st.metric("Rating", str(rating) if rating and not pd.isna(rating) else 'N/A')
                        with col3:
                            coupon = bond.get('coupon', None)
                            st.metric("Coupon", f"{coupon:.2f}%" if coupon and not pd.isna(coupon) else 'N/A')
                        with col4:
                            remaining = bond.get('remaining_years', None)
                            st.metric("Rest-LZ", f"{remaining:.1f}J" if remaining and not pd.isna(remaining) else 'N/A')

                        if 'trade_yield' in isin_trades.columns and isin_trades['trade_yield'].notna().any():
                            min_y, max_y = isin_trades['trade_yield'].min(), isin_trades['trade_yield'].max()
                            st.info(f"📊 Yield: {min_y:.3f}% - {max_y:.3f}% (Ø {avg_yield:.3f}%)")

                        st.markdown("**Einzeltrades:**")
                        trades_sorted = isin_trades.sort_values('file_date', ascending=False) if 'file_date' in isin_trades.columns else isin_trades
                        trade_cols = ['file_date', 'trade_price', 'trade_size', 'turnover_chf']
                        if 'trade_yield' in trades_sorted.columns:
                            trade_cols.append('trade_yield')
                        trade_display = trades_sorted[[c for c in trade_cols if c in trades_sorted.columns]].copy()
                        if 'trade_price' in trade_display.columns:
                            trade_display['trade_price'] = trade_display['trade_price'].round(3)
                        if 'trade_yield' in trade_display.columns:
                            trade_display['trade_yield'] = trade_display['trade_yield'].apply(
                                lambda x: f"{x:.3f}%" if pd.notna(x) else ""
                            )
                        if 'turnover_chf' in trade_display.columns:
                            trade_display['turnover_chf'] = trade_display['turnover_chf'].apply(
                                lambda x: f"{x:,.0f}".replace(",", "'") if pd.notna(x) else ""
                            )
                        if 'trade_size' in trade_display.columns:
                            trade_display['trade_size'] = trade_display['trade_size'].apply(
                                lambda x: f"{x:,.0f}".replace(",", "'") if pd.notna(x) else ""
                            )
                        trade_display = trade_display.rename(columns={'file_date': 'Datum', 'trade_price': 'Preis', 'trade_size': 'Volumen', 'turnover_chf': 'Turnover', 'trade_yield': 'Yield'})
                        st.dataframe(trade_display, hide_index=True, width='stretch')

            st.markdown("---")

            # All trades table (sortable by volume)
            st.subheader("📋 Alle Trades (sortierbar)")
            st.markdown("*Klicke auf eine Spaltenüberschrift zum Sortieren*")

            if 'product_isin' in filtered_df.columns:
                all_trades_df = filtered_df.copy()
                all_trades_cols = ['file_date', 'product_isin', 'product_short_name', 'sector_code', 'rating',
                                   'trade_price', 'trade_yield', 'trade_size', 'turnover_chf']
                all_trades_available = [c for c in all_trades_cols if c in all_trades_df.columns]
                all_trades_display = all_trades_df[all_trades_available].sort_values('trade_size', ascending=False).copy()

                if 'trade_price' in all_trades_display.columns:
                    all_trades_display['trade_price'] = all_trades_display['trade_price'].round(3)
                if 'trade_yield' in all_trades_display.columns:
                    all_trades_display['trade_yield'] = all_trades_display['trade_yield'].apply(
                        lambda x: f"{x:.3f}%" if pd.notna(x) else ""
                    )
                if 'turnover_chf' in all_trades_display.columns:
                    all_trades_display['turnover_chf'] = all_trades_display['turnover_chf'].apply(
                        lambda x: f"{x:,.0f}".replace(",", "'") if pd.notna(x) else ""
                    )
                if 'trade_size' in all_trades_display.columns:
                    all_trades_display['trade_size'] = all_trades_display['trade_size'].apply(
                        lambda x: f"{x:,.0f}".replace(",", "'") if pd.notna(x) else ""
                    )

                all_trades_col_names = {
                    'file_date': 'Datum', 'product_isin': 'ISIN', 'product_short_name': 'Name',
                    'sector_code': 'Segment', 'rating': 'Rating', 'trade_price': 'Preis',
                    'trade_yield': 'Yield', 'trade_size': 'Volumen', 'turnover_chf': 'Turnover'
                }
                all_trades_display = all_trades_display.rename(columns=all_trades_col_names)
                st.dataframe(all_trades_display, hide_index=True, width='stretch', height=500)

            st.markdown("---")

            # Yield Analysis
            st.subheader("📈 Yield Analyse")
            if 'trade_yield' in filtered_df.columns and filtered_df['trade_yield'].notna().any():
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**VW Yield nach Segment**")
                    valid_df = filtered_df[filtered_df['trade_yield'].notna()]
                    if 'sector_code' in valid_df.columns:
                        seg_yield = valid_df.groupby('sector_code').apply(lambda x: (x['trade_yield'] * x['turnover_chf']).sum() / x['turnover_chf'].sum() if x['turnover_chf'].sum() > 0 else None).reset_index(name='VW Yield %')
                        seg_yield = seg_yield[seg_yield['VW Yield %'].notna()]
                        seg_yield['VW Yield %'] = seg_yield['VW Yield %'].round(3)
                        st.dataframe(seg_yield, hide_index=True, width='stretch')
                with col2:
                    st.markdown("**VW Yield nach Rating**")
                    if 'rating' in valid_df.columns:
                        rat_yield = valid_df.groupby('rating').apply(lambda x: (x['trade_yield'] * x['turnover_chf']).sum() / x['turnover_chf'].sum() if x['turnover_chf'].sum() > 0 else None).reset_index(name='VW Yield %')
                        rat_yield = rat_yield[rat_yield['VW Yield %'].notna()]
                        rat_yield['VW Yield %'] = rat_yield['VW Yield %'].round(3)
                        st.dataframe(rat_yield, hide_index=True, width='stretch')

                st.markdown("**Yield pro Tag**")
                if 'file_date' in valid_df.columns:
                    daily_y = valid_df.groupby('file_date').apply(lambda x: (x['trade_yield'] * x['turnover_chf']).sum() / x['turnover_chf'].sum() if x['turnover_chf'].sum() > 0 else None).reset_index(name='vw_yield')
                    daily_y = daily_y[daily_y['vw_yield'].notna()]
                    if not daily_y.empty:
                        fig_y = px.line(daily_y, x='file_date', y='vw_yield', labels={'file_date': 'Datum', 'vw_yield': 'VW Yield %'}, markers=True,
                                        color_discrete_sequence=[CHART_COLORS[0]])
                        fig_y.update_layout(height=300, yaxis=YAXIS_STYLE, legend=LEGEND_STYLE, **PLOTLY_LAYOUT)
                        st.plotly_chart(fig_y, width='stretch')
            else:
                st.info("Yield nur für Index-Mitglieder berechenbar")

            st.markdown("---")

            # Detailed table with yield
            st.subheader("Alle Trades (mit Yield)")
            display_cols = ['file_date', 'product_isin', 'product_short_name', 'sector_code', 'rating', 'trade_price', 'trade_yield', 'trade_size', 'turnover_chf']
            available_cols = [c for c in display_cols if c in filtered_df.columns]
            detail_df = filtered_df[available_cols].copy()
            if 'trade_price' in detail_df.columns:
                detail_df['trade_price'] = detail_df['trade_price'].round(2)
            col_names = {'file_date': 'Datum', 'product_isin': 'ISIN', 'product_short_name': 'Name', 'sector_code': 'Segment', 'rating': 'Rating', 'trade_price': 'Preis', 'trade_yield': 'Yield %', 'trade_size': 'Volumen', 'turnover_chf': 'Turnover'}
            detail_df = detail_df.rename(columns=col_names)
            detail_column_config = {
                'Volumen': st.column_config.NumberColumn(format="%d"),
                'Turnover': st.column_config.NumberColumn(format="%d"),
                'Yield %': st.column_config.NumberColumn(format="%.3f%%"),
            }
            st.dataframe(detail_df, hide_index=True, width='stretch', height=400, column_config=detail_column_config)

            # Export
            csv = filtered_df.to_csv(index=False, sep=';')
            st.download_button(
                label="📥 Export als CSV",
                data=csv,
                file_name=f"delayed_trades_export_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

    # ============ TAB 6: Index Projektion ============
    with tab6:
        st.header("Index Projektion")
        st.markdown("*Vorausschau: Wie verändert sich der Index, wenn keine neuen Bonds aufgenommen werden?*")

        # Controls
        col_ctrl1, col_ctrl2 = st.columns([2, 1])
        with col_ctrl1:
            projection_months = st.slider("Projektions-Horizont (Monate)", 1, 24, 12, key="proj_months")
        with col_ctrl2:
            min_residual = config.get('rebalancing', {}).get('min_residual_years', 1.0)
            st.info(f"Min. Restlaufzeit: {min_residual} Jahr")

        # Run projection
        projection = project_index_forward(constituents_df, projection_months, min_residual)

        if projection:
            last_month = projection[-1]
            total_exits = sum(m['exit_count'] for m in projection)

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Exits Total", f"{total_exits}",
                          delta=f"-{total_exits} in {projection_months}M", delta_color="inverse")
            with col2:
                st.metric("MCap Veränderung", f"{last_month['mcap_change_pct']:+.1f}%")
            with col3:
                st.metric("Duration (Ende)", f"{last_month['duration']:.2f}y",
                          delta=f"{last_month['duration_change']:+.3f}y")
            with col4:
                st.metric("Yield (Ende)", f"{last_month['ytw']:.3f}%",
                          delta=f"{last_month['ytw_change']:+.4f}%")

            st.markdown("---")

            # Chart: exits per month + duration line
            import plotly.graph_objects as go
            chart_data = pd.DataFrame([{
                'Monat': m['month_label'],
                'Bonds': m['remaining_bonds_count'],
                'Duration': m['duration'],
                'Exits': m['exit_count'],
            } for m in projection])

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=chart_data['Monat'], y=chart_data['Exits'],
                name='Exits pro Monat', marker_color=COLOR_NEGATIVE,
                yaxis='y'
            ))
            fig.add_trace(go.Scatter(
                x=chart_data['Monat'], y=chart_data['Duration'],
                name='Duration', yaxis='y2',
                line=dict(color=COLOR_PRIMARY, width=3),
                mode='lines+markers'
            ))
            fig.update_layout(
                **PLOTLY_LAYOUT,
                yaxis=dict(title='Exits', side='left', gridcolor="#2d3548", zerolinecolor="#2d3548", tickfont=dict(color="#8b949e")),
                yaxis2=dict(title='Duration (Jahre)', overlaying='y', side='right', gridcolor="#2d3548", zerolinecolor="#2d3548", tickfont=dict(color="#8b949e")),
                height=350,
                legend=dict(orientation='h', y=-0.2, font=dict(color="#8b949e"), bgcolor="rgba(0,0,0,0)"),
                margin=dict(b=60)
            )
            st.plotly_chart(fig, width='stretch')

            st.markdown("---")

            # Monthly timeline with expandable details
            st.subheader("Monatliche Übersicht")
            for month_data in projection:
                n_exits = month_data['exit_count']
                label = month_data['month_label']

                if n_exits > 0:
                    with st.expander(
                        f"{label}: {n_exits} Exits | "
                        f"{month_data['remaining_bonds_count']} Bonds | "
                        f"Duration {month_data['duration']:.2f}y ({month_data['duration_change']:+.3f})"
                    ):
                        exit_df = month_data['exiting_bonds']
                        display_cols = ['isin', 'name', 'remaining_years', 'duration', 'rating']
                        if 'market_cap' in exit_df.columns:
                            display_cols.insert(3, 'market_cap')
                        available_cols = [c for c in display_cols if c in exit_df.columns]
                        display = exit_df[available_cols].copy()
                        if 'market_cap' in display.columns:
                            display['market_cap'] = (display['market_cap'] / 1e6).round(1)
                        if 'remaining_years' in display.columns:
                            display['remaining_years'] = display['remaining_years'].round(2)
                        if 'duration' in display.columns:
                            display['duration'] = display['duration'].round(2)
                        col_rename = {
                            'isin': 'ISIN', 'name': 'Name', 'remaining_years': 'Rest-LZ (J)',
                            'market_cap': 'MCap (Mio)', 'duration': 'Duration', 'rating': 'Rating'
                        }
                        display = display.rename(columns=col_rename)
                        st.dataframe(display, hide_index=True, width='stretch')
                else:
                    st.write(f"{label}: Keine Exits | {month_data['remaining_bonds_count']} Bonds")

            # Segment impact for a specific month
            st.markdown("---")
            st.subheader("Segment-Auswirkung")
            selected_month_idx = st.selectbox(
                "Monat wählen",
                options=list(range(len(projection))),
                format_func=lambda i: projection[i]['month_label'],
                key="proj_seg_month"
            )

            if selected_month_idx is not None:
                proj_seg = projection[selected_month_idx]['segment_stats']
                current_seg = get_all_segment_stats(constituents_df)

                if len(proj_seg) > 0 and len(current_seg) > 0:
                    merged_seg = current_seg[['name', 'count', 'weight_pct', 'duration']].merge(
                        proj_seg[['name', 'count', 'weight_pct', 'duration']],
                        on='name', suffixes=('_aktuell', '_projektion')
                    )
                    merged_seg = merged_seg[merged_seg['count_aktuell'] > 0]
                    merged_seg['count_delta'] = merged_seg['count_projektion'] - merged_seg['count_aktuell']
                    merged_seg['weight_delta'] = (merged_seg['weight_pct_projektion'] - merged_seg['weight_pct_aktuell']).round(2)
                    merged_seg['duration_delta'] = (merged_seg['duration_projektion'] - merged_seg['duration_aktuell']).round(3)
                    merged_seg['weight_pct_aktuell'] = merged_seg['weight_pct_aktuell'].round(1)
                    merged_seg['weight_pct_projektion'] = merged_seg['weight_pct_projektion'].round(1)
                    merged_seg['duration_aktuell'] = merged_seg['duration_aktuell'].round(2)
                    merged_seg['duration_projektion'] = merged_seg['duration_projektion'].round(2)
                    merged_seg.columns = [
                        'Segment', 'Bonds Akt.', 'Gew. % Akt.', 'Dur. Akt.',
                        'Bonds Proj.', 'Gew. % Proj.', 'Dur. Proj.',
                        'Bonds Δ', 'Gew. Δ', 'Dur. Δ'
                    ]
                    st.dataframe(merged_seg, hide_index=True, width='stretch')

    # ============ TAB 7: Bond Anpassungen ============
    with tab7:
        st.header("Bond Anpassungen")
        st.markdown("*Simuliere den Effekt von neuen Bonds oder Nominaländerungen auf den Index*")

        # Format explanation
        st.subheader("CSV Format")
        st.markdown("""
**Pflichtfelder:**
| Spalte | Beschreibung | Beispiel |
|--------|-------------|----------|
| `isin` | ISIN des Bonds | CH1234567890 |
| `nominal` | Nominalbetrag in CHF | 200000000 |

**Optionale Felder:**
| Spalte | Beschreibung | Default | Beispiel |
|--------|-------------|---------|----------|
| `admission_date` | Aufnahmedatum (DD.MM.YYYY) | - | 02.03.2026 |
| `price` | Kurs in % | 100 (par) | 102.5 |
| `duration` | Duration in Jahren | Geschätzt aus Maturity | 5.0 |
| `rating` | Composite Rating | A | AAA, AA, A, BBB |
| `domicile` | Domizil-Code | CH | CH, DE, FR |
| `maturity_date` | Fälligkeitsdatum | - | 31.12.2031 |

**Separator:** Semikolon (`;`) oder Komma (`,`)

**Hinweis:** Wenn die ISIN bereits im Index ist, wird das Nominal angepasst. Neue ISINs werden als Admission simuliert.
""")

        # Template example visible in UI
        st.markdown("**Beispiel:**")
        template_example = pd.DataFrame({
            'isin': ['CH1234567890', 'CH0987654321', 'CH1111222233'],
            'admission_date': ['02.03.2026', '02.03.2026', '02.04.2026'],
            'nominal': [200000000, 500000000, 150000000],
            'price': [100.0, 102.5, 98.0],
            'duration': [5.0, 8.5, 3.2],
            'rating': ['A', 'AA', 'BBB'],
            'domicile': ['CH', 'CH', 'DE'],
            'maturity_date': ['31.12.2031', '15.06.2034', '01.03.2029']
        })
        st.dataframe(template_example, hide_index=True, width='stretch')

        # Template download
        template_csv = "isin;admission_date;nominal;price;duration;rating;domicile;maturity_date\nCH1234567890;02.03.2026;200000000;100;5.0;A;CH;31.12.2031\nCH0987654321;02.03.2026;500000000;102.5;8.5;AA;CH;15.06.2034\nCH1111222233;02.04.2026;150000000;98;3.2;BBB;DE;01.03.2029"
        st.download_button(
            "📥 CSV Template herunterladen",
            data=template_csv,
            file_name="bond_anpassungen_template.csv",
            mime="text/csv"
        )

        st.markdown("---")

        # File uploader
        uploaded_file = st.file_uploader("CSV-Datei hochladen", type=['csv'], key="bond_upload")

        if uploaded_file is not None:
            # Parse uploaded CSV
            try:
                raw_df = pd.read_csv(uploaded_file, sep=';')
                if len(raw_df.columns) <= 1:
                    uploaded_file.seek(0)
                    raw_df = pd.read_csv(uploaded_file, sep=',')
            except Exception as e:
                st.error(f"Fehler beim Lesen der CSV: {e}")
                raw_df = None

            if raw_df is not None:
                validation = validate_changes_csv(raw_df)

                # Show errors
                for err in validation['errors']:
                    st.error(f"Fehler: {err}")

                # Show warnings
                for warn in validation['warnings']:
                    st.warning(f"Hinweis: {warn}")

                if validation['valid']:
                    changes_df_upload = validation['parsed_df']

                    # Preview uploaded bonds
                    st.subheader("Hochgeladene Bonds")
                    preview = changes_df_upload.copy()
                    preview['status'] = preview['isin'].apply(
                        lambda x: 'Nominalanpassung' if x in constituents_df['isin'].values else 'Neu'
                    )
                    preview_cols = ['isin', 'nominal', 'status']
                    if 'admission_date' in preview.columns:
                        preview_cols.insert(1, 'admission_date')
                    preview_display = preview[preview_cols].copy()
                    preview_display['nominal'] = (preview_display['nominal'] / 1e6).round(1)
                    col_names = {'isin': 'ISIN', 'nominal': 'Nominal (Mio)', 'status': 'Typ', 'admission_date': 'Admission Date'}
                    preview_display = preview_display.rename(columns=col_names)
                    if 'rating' in preview.columns:
                        preview_display['Rating'] = preview['rating'].values
                    if 'duration' in preview.columns:
                        preview_display['Duration'] = preview['duration'].values
                    st.dataframe(preview_display, hide_index=True, width='stretch')

                    st.markdown("---")

                    # Apply changes and compare
                    modified_df = apply_bond_changes(constituents_df, changes_df_upload)
                    comparison = compare_index_states(constituents_df, modified_df, changes_df_upload)

                    # Summary metrics
                    st.subheader("Vorher / Nachher Vergleich")
                    s = comparison['summary']
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Market Cap",
                                  f"CHF {s['new_mcap']/1e9:.1f} Mrd",
                                  delta=f"{s['mcap_change_pct']:+.2f}%")
                    with col2:
                        count_delta = s['new_count'] - s['original_count']
                        st.metric("Anzahl Bonds",
                                  f"{s['new_count']}",
                                  delta=f"{count_delta:+d}" if count_delta != 0 else "0")
                    with col3:
                        st.metric("Duration",
                                  f"{s['new_duration']:.2f}y",
                                  delta=f"{s['duration_change']:+.3f}y")
                    with col4:
                        st.metric("Yield",
                                  f"{s['new_ytw']:.3f}%",
                                  delta=f"{s['ytw_change']:+.4f}%")

                    # Segment comparison
                    st.markdown("---")
                    st.subheader("Segment-Vergleich")
                    seg_comp = comparison['segment_comparison']
                    if len(seg_comp) > 0:
                        seg_display = seg_comp[['name', 'count_aktuell', 'count_neu',
                                                'weight_pct_aktuell', 'weight_pct_neu',
                                                'duration_aktuell', 'duration_neu']].copy()
                        seg_display = seg_display[seg_display['count_aktuell'] > 0]
                        seg_display['weight_pct_aktuell'] = seg_display['weight_pct_aktuell'].round(1)
                        seg_display['weight_pct_neu'] = seg_display['weight_pct_neu'].round(1)
                        seg_display['duration_aktuell'] = seg_display['duration_aktuell'].round(2)
                        seg_display['duration_neu'] = seg_display['duration_neu'].round(2)
                        seg_display['gew_delta'] = (seg_display['weight_pct_neu'] - seg_display['weight_pct_aktuell']).round(2)
                        seg_display['dur_delta'] = (seg_display['duration_neu'] - seg_display['duration_aktuell']).round(3)
                        seg_display.columns = ['Segment', 'Bonds Akt.', 'Bonds Neu',
                                              'Gew. % Akt.', 'Gew. % Neu',
                                              'Dur. Akt.', 'Dur. Neu', 'Gew. Δ', 'Dur. Δ']
                        st.dataframe(seg_display, hide_index=True, width='stretch')

                    # Uploaded bonds with their weights
                    st.markdown("---")
                    st.subheader("Simulierte Bonds im Index")
                    uploaded_detail = comparison['uploaded_bonds']
                    if len(uploaded_detail) > 0:
                        ud_display = uploaded_detail[['isin', 'name', 'nominal', 'market_cap',
                                                      'duration', 'rating', 'weight_pct']].copy()
                        ud_display['nominal'] = (ud_display['nominal'] / 1e6).round(1)
                        ud_display['market_cap'] = (ud_display['market_cap'] / 1e6).round(1)
                        ud_display['duration'] = ud_display['duration'].round(2)
                        ud_display['weight_pct'] = ud_display['weight_pct'].round(4)
                        ud_display.columns = ['ISIN', 'Name', 'Nominal (Mio)', 'MCap (Mio)',
                                            'Duration', 'Rating', 'Gewicht %']
                        st.dataframe(ud_display, hide_index=True, width='stretch')

                    # Top weight changes
                    st.markdown("---")
                    st.subheader("Grösste Gewichtungsänderungen")
                    top_changes = comparison['top_weight_changes']
                    if len(top_changes) > 0:
                        tc_display = top_changes.copy()
                        tc_display.columns = ['ISIN', 'Name', 'Gew. % Akt.', 'Gew. % Neu', 'Gew. Δ']
                        st.dataframe(tc_display, hide_index=True, width='stretch')

    # ============ TAB 8: Historische Entwicklung ============
    with tab8:
        st.header("Historische Entwicklung")
        st.markdown("*Rekonstruierte Index-Duration basierend auf historischen Forecast-Dateien*")

        # Load historical data
        @st.cache_data
        def load_historical_analysis():
            hist_folder = Path(__file__).parent / 'extracted'
            if not hist_folder.exists():
                return pd.DataFrame(), pd.DataFrame()
            total_df = analyze_historical_forecasts(str(hist_folder))
            seg_df = analyze_historical_by_segment(str(hist_folder))
            return total_df, seg_df

        @st.cache_data
        def load_reconstructed_durations(_constituents_df):
            hist_folder = Path(__file__).parent / 'extracted'
            if not hist_folder.exists():
                return pd.DataFrame()
            return reconstruct_historical_durations(_constituents_df, str(hist_folder))

        hist_df, hist_seg_df = load_historical_analysis()
        recon_df = load_reconstructed_durations(constituents_df)

        if hist_df.empty:
            st.warning("Keine historischen Forecast-Dateien gefunden im Ordner 'extracted/'")
        else:
            import plotly.graph_objects as go

            # Segment selector + year range
            col_ctrl1, col_ctrl2 = st.columns([1, 2])
            with col_ctrl1:
                seg_options = ['total', 'domestic', 'foreign', 'government', 'agency',
                               'pfandbrief', 'corporate', 'financials']
                seg_available = [s for s in seg_options
                                 if not recon_df.empty and s in recon_df['segment'].unique()]
                selected_segment = st.selectbox(
                    "Segment",
                    options=seg_available,
                    format_func=lambda x: _SEGMENT_NAMES.get(x, x),
                    key="hist_segment"
                )
            with col_ctrl2:
                years_available = sorted(hist_df['year'].unique())
                year_range = st.select_slider(
                    "Zeitraum (Jahre)",
                    options=years_available,
                    value=(years_available[0], years_available[-1]),
                    key="hist_year_range"
                )

            # ---- Reconstructed duration data ----
            recon_seg = recon_df[
                (recon_df['segment'] == selected_segment) &
                (recon_df['date'].dt.year >= year_range[0]) &
                (recon_df['date'].dt.year <= year_range[1])
            ].copy()

            # ---- Forecast change data ----
            seg_filtered = hist_seg_df[
                (hist_seg_df['segment'] == selected_segment) &
                (hist_seg_df['date'].dt.year >= year_range[0]) &
                (hist_seg_df['date'].dt.year <= year_range[1])
            ].copy()

            # Current segment duration (MCap-weighted, actual)
            seg_dur = calc_segment_duration(constituents_df, selected_segment) if selected_segment in SEGMENTS else calc_segment_duration(constituents_df, 'total')

            if len(recon_seg) > 0:
                # ---- Compute month-over-month duration change from reconstruction ----
                recon_seg = recon_seg.sort_values('date').reset_index(drop=True)
                recon_seg['dur_change'] = recon_seg['duration'].diff().round(4)
                recon_seg['bond_count_change'] = recon_seg['bond_count'].diff()

                # ---- Key metrics ----
                avg_monthly_change = recon_seg['dur_change'].mean()
                total_change = recon_seg['duration'].iloc[-1] - recon_seg['duration'].iloc[0]
                n_months = len(recon_seg) - 1

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Aktuelle Duration", f"{seg_dur:.3f}y")
                with col2:
                    st.metric("Ø Δ Duration / Monat", f"{avg_monthly_change:+.3f}y" if pd.notna(avg_monthly_change) else "n/a")
                with col3:
                    st.metric("Gesamtveränderung", f"{total_change:+.3f}y")
                with col4:
                    st.metric(f"Bonds ({recon_seg['date'].iloc[0].strftime('%Y')} → {recon_seg['date'].iloc[-1].strftime('%Y')})",
                              f"{recon_seg['bond_count'].iloc[0]} → {recon_seg['bond_count'].iloc[-1]}")

                st.markdown("---")

                # ---- MAIN CHART: Reconstructed duration over time ----
                st.subheader("Rekonstruierte historische Duration")
                st.markdown(f"*Nominal-gewichtete Durchschnittsduration des **{_SEGMENT_NAMES.get(selected_segment, selected_segment)}***")

                fig_hist_dur = go.Figure()
                fig_hist_dur.add_trace(go.Scatter(
                    x=recon_seg['date'],
                    y=recon_seg['duration'],
                    name='Rekonstruierte Duration',
                    line=dict(color=COLOR_PRIMARY, width=2.5),
                    mode='lines',
                    fill='tozeroy',
                    fillcolor='rgba(88, 166, 255, 0.06)',
                    hovertemplate='%{x|%b %Y}<br>Duration: %{y:.3f}y<extra></extra>'
                ))

                # Current actual duration marker
                fig_hist_dur.add_trace(go.Scatter(
                    x=[recon_seg['date'].iloc[-1]],
                    y=[seg_dur],
                    name=f'Aktuell (MCap-gew.): {seg_dur:.2f}y',
                    mode='markers',
                    marker=dict(size=10, color=COLOR_NEGATIVE, symbol='diamond'),
                ))

                fig_hist_dur.update_layout(
                    **PLOTLY_LAYOUT,
                    yaxis=dict(title='Duration (Jahre)', gridcolor="#2d3548", zerolinecolor="#2d3548", tickfont=dict(color="#8b949e")),
                    height=400,
                    legend=dict(orientation='h', y=-0.15, font=dict(color="#8b949e"), bgcolor="rgba(0,0,0,0)"),
                    margin=dict(b=60),
                    hovermode='x unified'
                )
                st.plotly_chart(fig_hist_dur, width='stretch')

                st.caption(
                    "Rekonstruktion der Index-Zusammensetzung durch Rückwärts-Anwendung der Forecast-Dateien. "
                    "Für jeden Monat wird die nominal-gewichtete Durchschnittsduration aus den rekonstruierten "
                    "Indexbestandteilen berechnet (Duration geschätzt aus Restlaufzeit). "
                    "Der rote Punkt zeigt die aktuelle MCap-gewichtete Duration."
                )

                st.markdown("---")

                # ---- Monthly duration change from reconstruction ----
                st.subheader("Monatliche Duration-Veränderung")
                st.markdown(f"*Tatsächliche Veränderung der {_SEGMENT_NAMES.get(selected_segment, selected_segment)}-Duration von Monat zu Monat*")

                change_data = recon_seg.dropna(subset=['dur_change']).copy()
                if len(change_data) > 0:
                    fig_change = go.Figure()
                    fig_change.add_trace(go.Bar(
                        x=change_data['date'],
                        y=change_data['dur_change'],
                        name='Δ Duration',
                        marker_color=change_data['dur_change'].apply(
                            lambda x: 'rgba(0, 212, 170, 0.7)' if x >= 0 else 'rgba(255, 107, 107, 0.7)'
                        ),
                        hovertemplate='%{x|%b %Y}<br>Δ: %{y:+.4f}y<extra></extra>'
                    ))
                    fig_change.add_hline(y=0, line_width=1, line_color='black', opacity=0.3)

                    # Add rolling 12-month average
                    if len(change_data) >= 12:
                        rolling_avg = change_data['dur_change'].rolling(12).mean()
                        fig_change.add_trace(go.Scatter(
                            x=change_data['date'],
                            y=rolling_avg,
                            name='12M Ø',
                            line=dict(color=COLOR_ACCENT, width=2, dash='dash'),
                            mode='lines',
                        ))

                    fig_change.update_layout(
                        **PLOTLY_LAYOUT,
                        yaxis=dict(title='Δ Duration (Jahre)', zeroline=True, gridcolor="#2d3548", zerolinecolor="#2d3548", tickfont=dict(color="#8b949e")),
                        height=400,
                        legend=dict(orientation='h', y=-0.15, font=dict(color="#8b949e"), bgcolor="rgba(0,0,0,0)"),
                        margin=dict(b=60),
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_change, width='stretch')

                st.markdown("---")

                # ---- Forecast details: Admissions vs Deletions ----
                st.subheader("Rebalancing-Details: Admissions vs. Deletions")
                st.markdown(f"*Aus den Forecast-Dateien: welche Bonds kamen rein / gingen raus?*")

                if len(seg_filtered) > 0:
                    # ---- Duration Admissions vs Deletions ----
                    st.subheader("Duration: Admissions vs. Deletions")
                    st.markdown("*Geschätzte gewichtete Durchschnittsduration der ein-/austretenden Bonds*")

                    fig_dur = go.Figure()
                    fig_dur.add_trace(go.Scatter(
                        x=seg_filtered['date'],
                        y=seg_filtered['adm_avg_duration'],
                        name='Admissions Ø Duration',
                        line=dict(color=COLOR_POSITIVE, width=2),
                        mode='lines+markers',
                        marker=dict(size=4)
                    ))
                    fig_dur.add_trace(go.Scatter(
                        x=seg_filtered['date'],
                        y=seg_filtered['del_avg_duration'],
                        name='Deletions Ø Duration',
                        line=dict(color=COLOR_NEGATIVE, width=2),
                        mode='lines+markers',
                        marker=dict(size=4)
                    ))
                    fig_dur.update_layout(
                        **PLOTLY_LAYOUT,
                        yaxis=dict(title='Duration (Jahre)', gridcolor="#2d3548", zerolinecolor="#2d3548", tickfont=dict(color="#8b949e")),
                        height=350,
                        legend=dict(orientation='h', y=-0.15, font=dict(color="#8b949e"), bgcolor="rgba(0,0,0,0)"),
                        margin=dict(b=60),
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_dur, width='stretch')

                    st.markdown("---")

                    # ---- Admissions vs Deletions count ----
                    st.subheader("Anzahl Admissions & Deletions")
                    fig_count = go.Figure()
                    fig_count.add_trace(go.Bar(
                        x=seg_filtered['date'], y=seg_filtered['adm_count'],
                        name='Admissions', marker_color=COLOR_POSITIVE
                    ))
                    fig_count.add_trace(go.Bar(
                        x=seg_filtered['date'], y=-seg_filtered['del_count'],
                        name='Deletions', marker_color=COLOR_NEGATIVE
                    ))
                    fig_count.update_layout(
                        **PLOTLY_LAYOUT,
                        barmode='relative', yaxis=dict(title='Anzahl', gridcolor="#2d3548", zerolinecolor="#2d3548", tickfont=dict(color="#8b949e")),
                        height=300, legend=dict(orientation='h', y=-0.2, font=dict(color="#8b949e"), bgcolor="rgba(0,0,0,0)"), margin=dict(b=60)
                    )
                    st.plotly_chart(fig_count, width='stretch')

                    st.markdown("---")

                    # ---- Nominal volume ----
                    st.subheader("Nominalvolumen pro Monat")
                    fig_nom = go.Figure()
                    fig_nom.add_trace(go.Bar(
                        x=seg_filtered['date'], y=seg_filtered['adm_nominal'] / 1e9,
                        name='Admissions', marker_color=COLOR_POSITIVE
                    ))
                    fig_nom.add_trace(go.Bar(
                        x=seg_filtered['date'], y=-seg_filtered['del_nominal'] / 1e9,
                        name='Deletions', marker_color=COLOR_NEGATIVE
                    ))
                    net_nom = (seg_filtered['adm_nominal'] - seg_filtered['del_nominal']) / 1e9
                    fig_nom.add_trace(go.Scatter(
                        x=seg_filtered['date'], y=net_nom, name='Netto',
                        line=dict(color=COLOR_PRIMARY, width=2), mode='lines+markers', marker=dict(size=4)
                    ))
                    fig_nom.update_layout(
                        **PLOTLY_LAYOUT,
                        barmode='relative', yaxis=dict(title='Nominal (Mrd CHF)', gridcolor="#2d3548", zerolinecolor="#2d3548", tickfont=dict(color="#8b949e")),
                        height=350, legend=dict(orientation='h', y=-0.15, font=dict(color="#8b949e"), bgcolor="rgba(0,0,0,0)"), margin=dict(b=60)
                    )
                    st.plotly_chart(fig_nom, width='stretch')

                st.markdown("---")

                # ---- Monthly detail table ----
                # Merge reconstructed duration with forecast data
                detail_recon = recon_seg[['date', 'duration', 'bond_count', 'total_nominal', 'dur_change']].copy()
                detail_recon['date_key'] = detail_recon['date'].dt.to_period('M')
                if len(seg_filtered) > 0:
                    detail_fc = seg_filtered[['date', 'adm_count', 'del_count',
                                              'adm_avg_duration', 'del_avg_duration']].copy()
                    detail_fc['date_key'] = detail_fc['date'].dt.to_period('M')
                    detail_merged = detail_recon.merge(detail_fc, on='date_key', how='left', suffixes=('', '_fc'))
                else:
                    detail_merged = detail_recon.copy()
                    detail_merged['adm_count'] = 0
                    detail_merged['del_count'] = 0
                    detail_merged['adm_avg_duration'] = 0
                    detail_merged['del_avg_duration'] = 0

                with st.expander("📋 Monatliche Details"):
                    tbl = detail_merged[[
                        'date', 'duration', 'dur_change', 'bond_count',
                        'total_nominal', 'adm_count', 'del_count',
                        'adm_avg_duration', 'del_avg_duration'
                    ]].copy()
                    tbl['date'] = tbl['date'].dt.strftime('%Y-%m')
                    tbl['total_nominal'] = (tbl['total_nominal'] / 1e9).round(1)
                    tbl['dur_change'] = tbl['dur_change'].round(4)
                    tbl.columns = [
                        'Periode', 'Duration', 'Δ Dur.', 'Bonds',
                        'Nominal (Mrd)', 'Adm.', 'Del.',
                        'Ø Dur. Adm.', 'Ø Dur. Del.'
                    ]
                    st.dataframe(tbl, hide_index=True, width='stretch', height=400)

            else:
                st.warning("Keine Daten für dieses Segment im gewählten Zeitraum")

else:
    st.warning("Bitte stelle sicher, dass die Datenfiles vorhanden sind:")
    st.code("""
    - close_sbr14d.csv
    - forecast_bonds.csv
    - config.json
    """)

# Footer
st.markdown("---")
st.markdown("*SBI Index Rebalancing Tool | Basierend auf SIX Rulebook v2.30*")
