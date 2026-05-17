"""
SIX Delayed Publication Viewer
Standalone Streamlit app - fetches OTC bond trade data directly from SIX Group.
No local index files required.
"""

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from datetime import datetime, timedelta
from io import StringIO

BASE_URL = "https://www.six-group.com/exchanges/dwh_download/delayed_publication/delayed_publication_{date}.csv"

st.set_page_config(
    page_title="SIX Delayed Publications",
    page_icon="📈",
    layout="wide"
)

DARK_CSS = """
<style>
[data-testid="stMetric"] {
    background: #1a1f2e;
    border: 1px solid #2d3548;
    border-radius: 8px;
    padding: 16px;
}
[data-testid="stMetricValue"] { color: #e6edf3; font-weight: 600; }
[data-testid="stMetricLabel"] { color: #8b949e; }
[data-testid="stExpander"] {
    border: 1px solid #2d3548;
    border-radius: 8px;
    background: #1a1f2e;
}
hr { border-color: #2d3548 !important; }
.stDownloadButton > button {
    background: transparent;
    border: 1px solid #2d3548;
    color: #e6edf3;
    border-radius: 6px;
}
.stDownloadButton > button:hover { border-color: #00d4aa; color: #00d4aa; }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
    font=dict(color="#e6edf3", size=12),
    xaxis=dict(gridcolor="#2d3548", zerolinecolor="#2d3548", tickfont=dict(color="#8b949e")),
    hoverlabel=dict(bgcolor="#242b3d", font_color="#e6edf3", bordercolor="#2d3548"),
)
YAXIS_STYLE = dict(gridcolor="#2d3548", zerolinecolor="#2d3548", tickfont=dict(color="#8b949e"))
LEGEND_STYLE = dict(font=dict(color="#8b949e"), bgcolor="rgba(0,0,0,0)")
CHART_COLORS = ["#00d4aa", "#58a6ff", "#d4a017", "#ff6b6b", "#8b5cf6", "#06b6d4", "#f59e0b", "#ef4444"]



def fetch_single_day(date: datetime) -> pd.DataFrame | None:
    url = BASE_URL.format(date=date.strftime("%Y%m%d"))
    try:
        resp = requests.get(url, timeout=20)
        if resp.status_code != 200:
            return None
        content = resp.text
        if not content.strip() or content.strip().startswith("<!DOCTYPE"):
            return None
        df = pd.read_csv(StringIO(content), sep=";", encoding="utf-8-sig")
        df.columns = (
            df.columns.str.strip().str.lower()
            .str.replace(" ", "_")
            .str.replace("\ufeff", "", regex=False)
            .str.replace("ï»¿", "", regex=False)
        )
        df["file_date"] = date.date()
        return df
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_date_range(start: datetime, end: datetime) -> pd.DataFrame:
    frames = []
    current = start
    progress = st.progress(0, text="Lade Daten von SIX...")
    total = max((end - start).days + 1, 1)
    i = 0
    while current <= end:
        df = fetch_single_day(current)
        if df is not None and not df.empty:
            frames.append(df)
        current += timedelta(days=1)
        i += 1
        progress.progress(i / total, text=f"Lade {current.strftime('%d.%m.%Y')}...")
    progress.empty()
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    for col in ["trade_price", "trade_size", "turnover_chf"]:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
    if "estimated_trade_pub_business_date" in combined.columns:
        combined["estimated_trade_pub_business_date"] = pd.to_datetime(
            combined["estimated_trade_pub_business_date"], errors="coerce"
        )
    if "trade_date" in combined.columns:
        combined["trade_date"] = pd.to_datetime(combined["trade_date"], errors="coerce")
    combined["file_date"] = pd.to_datetime(combined["file_date"])
    return combined


st.title("📈 SIX Delayed Publication Trades")
st.markdown("*OTC-Bond-Handelsdaten vom SIX Group — verzögert veröffentlicht*")

# --- Datumsauswahl in der Sidebar ---
with st.sidebar:
    st.header("Zeitraum")
    default_end = datetime.now()
    default_start = default_end - timedelta(days=30)
    start_date = st.date_input("Von", value=default_start.date())
    end_date = st.date_input("Bis", value=default_end.date())
    if st.button("🔄 Neu laden", use_container_width=True):
        fetch_date_range.clear()
        st.rerun()

with st.spinner("Lade Handelsdaten..."):
    df = fetch_date_range(datetime.combine(start_date, datetime.min.time()),
                          datetime.combine(end_date, datetime.min.time()))

if df.empty:
    st.warning("Keine Daten für den gewählten Zeitraum gefunden.")
    st.stop()

# --- Filter ---
with st.expander("🔍 Filter", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        isin_search = st.text_input("ISIN / Bond-Name suchen", placeholder="z.B. CH0550...")
    with col2:
        min_turnover = st.number_input("Min. Turnover CHF", min_value=0, value=0, step=100_000,
                                       format="%d")
    with col3:
        max_rows = st.selectbox("Max. Zeilen", [100, 500, 1000, 5000, 0], index=1,
                                format_func=lambda x: "Alle" if x == 0 else f"{x:,}")

filtered = df.copy()

if isin_search:
    mask = (
        filtered.get("product_isin", pd.Series(dtype=str)).str.contains(isin_search, case=False, na=False) |
        filtered.get("product_short_name", pd.Series(dtype=str)).str.contains(isin_search, case=False, na=False)
    )
    filtered = filtered[mask]

if min_turnover > 0 and "turnover_chf" in filtered.columns:
    filtered = filtered[filtered["turnover_chf"] >= min_turnover]

# --- Kennzahlen ---
st.markdown("---")
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Trades", f"{len(filtered):,}")
with m2:
    turnover = filtered["turnover_chf"].sum() if "turnover_chf" in filtered.columns else 0
    st.metric("Total Turnover", f"CHF {turnover / 1e6:.1f} Mio")
with m3:
    unique_bonds = filtered["product_isin"].nunique() if "product_isin" in filtered.columns else 0
    st.metric("Unique Bonds", f"{unique_bonds:,}")
with m4:
    avg_size = filtered["trade_size"].mean() if "trade_size" in filtered.columns else 0
    st.metric("Ø Trade Size", f"CHF {avg_size / 1e3:.0f}k")

st.markdown("---")

# --- Charts ---
chart1, chart2 = st.columns(2)

with chart1:
    st.subheader("Täglicher Turnover")
    if "file_date" in filtered.columns and "turnover_chf" in filtered.columns:
        daily = filtered.groupby("file_date")["turnover_chf"].sum().reset_index()
        fig = px.bar(daily, x="file_date", y="turnover_chf",
                     labels={"file_date": "Datum", "turnover_chf": "Turnover CHF"},
                     color_discrete_sequence=[CHART_COLORS[0]])
        fig.update_layout(height=320, yaxis=YAXIS_STYLE, **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

with chart2:
    st.subheader("Top 15 Bonds nach Turnover")
    if "product_isin" in filtered.columns and "turnover_chf" in filtered.columns:
        group_cols = [c for c in ["product_isin", "product_short_name"] if c in filtered.columns]
        top = (filtered.groupby(group_cols)["turnover_chf"]
               .sum().reset_index().nlargest(15, "turnover_chf"))
        label_col = "product_short_name" if "product_short_name" in top.columns else "product_isin"
        top["label"] = top[label_col].str[:30]
        fig2 = px.bar(top, x="turnover_chf", y="label", orientation="h",
                      labels={"turnover_chf": "Turnover CHF", "label": ""},
                      color_discrete_sequence=[CHART_COLORS[1]])
        fig2.update_layout(height=320, yaxis=dict(**YAXIS_STYLE, autorange="reversed"),
                           **PLOTLY_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

# --- Tabelle ---
st.markdown("---")
st.subheader("Handelsdaten")

display_cols = [c for c in [
    "file_date", "trade_date", "product_isin", "product_short_name",
    "product_symbol", "trade_price", "trade_size", "turnover_chf"
] if c in filtered.columns]

display_df = filtered[display_cols].copy()
if max_rows:
    display_df = display_df.head(max_rows)

st.dataframe(
    display_df.sort_values("file_date", ascending=False) if "file_date" in display_df.columns else display_df,
    use_container_width=True,
    height=400
)

st.download_button(
    "⬇️ CSV exportieren",
    data=filtered[display_cols].to_csv(index=False),
    file_name=f"delayed_trades_{start_date}_{end_date}.csv",
    mime="text/csv"
)
