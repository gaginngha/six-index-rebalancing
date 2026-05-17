"""
SIX Delayed Publication Viewer
Standalone Streamlit app - fetches OTC bond trade data directly from SIX Group.
No local index files required.
"""

import re
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
CHART_COLORS = ["#00d4aa", "#58a6ff", "#d4a017", "#ff6b6b", "#8b5cf6", "#06b6d4", "#f59e0b", "#ef4444"]


def parse_coupon_and_years(name: str) -> tuple[float | None, float | None]:
    """Extract coupon (%) and remaining years from SIX bond short names like '1.25 SSPITAL 18-26'."""
    if pd.isna(name):
        return None, None
    name = str(name)
    coupon_match = re.match(r"^([\d.]+)\s", name)
    coupon = float(coupon_match.group(1)) if coupon_match else None
    maturity_match = re.search(r"-(\d{2})$", name)
    if maturity_match:
        yy = int(maturity_match.group(1))
        maturity_year = 2000 + yy
        remaining = max((datetime(maturity_year, 7, 1) - datetime.now()).days / 365.25, 0)
    else:
        remaining = None
    return coupon, remaining


def calc_ytm(price: float, coupon: float, remaining_years: float) -> float | None:
    """Simplified YTM approximation: (C + (100-P)/N) / ((100+P)/2)."""
    if any(v is None or pd.isna(v) for v in [price, coupon, remaining_years]):
        return None
    if remaining_years <= 0 or price <= 0:
        return None
    return round((coupon + (100 - price) / remaining_years) / ((100 + price) / 2) * 100, 3)


def enrich_yield(df: pd.DataFrame) -> pd.DataFrame:
    if "product_short_name" not in df.columns or "trade_price" not in df.columns:
        return df
    parsed = df["product_short_name"].apply(parse_coupon_and_years)
    df = df.copy()
    df["coupon_pct"] = parsed.apply(lambda x: x[0])
    df["remaining_years"] = parsed.apply(lambda x: x[1])
    df["yield_pct"] = df.apply(
        lambda r: calc_ytm(r["trade_price"], r["coupon_pct"], r["remaining_years"]), axis=1
    )
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [
        col.encode("utf-8", "ignore").decode("utf-8")
        .replace("\ufeff", "").replace("ï»¿", "")
        .strip().lower().replace(" ", "_")
        for col in df.columns
    ]
    # Ensure consistent name for the bond name column
    for variant in ["product_short_name", "productshortname", "short_name"]:
        if variant in df.columns and variant != "product_short_name":
            df = df.rename(columns={variant: "product_short_name"})
    return df


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
        df = normalize_columns(df)
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
    for col in ["estimated_trade_pub_business_date", "trade_date"]:
        if col in combined.columns:
            combined[col] = pd.to_datetime(combined[col], errors="coerce")
    combined["file_date"] = pd.to_datetime(combined["file_date"])
    combined = enrich_yield(combined)
    return combined


def render_trade_table(data: pd.DataFrame, max_rows: int = 500):
    # Name first for visibility
    display_cols = [c for c in [
        "trade_date", "product_short_name", "product_isin",
        "product_symbol", "trade_price", "yield_pct", "trade_size",
    ] if c in data.columns]
    display_df = data[display_cols].copy()
    if "trade_date" in display_df.columns:
        display_df = display_df.sort_values("trade_date", ascending=False)
    if max_rows:
        display_df = display_df.head(max_rows)

    col_config = {}
    if "trade_size" in display_df.columns:
        col_config["trade_size"] = st.column_config.NumberColumn("Trade Size", format="%,.0f")
    if "trade_price" in display_df.columns:
        col_config["trade_price"] = st.column_config.NumberColumn("Preis", format="%.2f")
    if "yield_pct" in display_df.columns:
        col_config["yield_pct"] = st.column_config.NumberColumn("Yield %", format="%.3f")
    if "product_short_name" in display_df.columns:
        col_config["product_short_name"] = st.column_config.TextColumn("Name")
    if "product_isin" in display_df.columns:
        col_config["product_isin"] = st.column_config.TextColumn("ISIN")

    st.dataframe(display_df, use_container_width=True, height=400, column_config=col_config)
    st.download_button(
        "⬇️ CSV exportieren",
        data=display_df.to_csv(index=False),
        file_name=f"delayed_trades_export.csv",
        mime="text/csv",
        key=f"dl_{len(data)}"
    )


# ── App ──────────────────────────────────────────────────────────────────────

st.title("📈 SIX Delayed Publication Trades")
st.markdown("*OTC-Bond-Handelsdaten vom SIX Group — verzögert veröffentlicht*")

with st.sidebar:
    st.header("Zeitraum")
    default_end = datetime.now()
    default_start = default_end - timedelta(days=7)
    start_date = st.date_input("Von", value=default_start.date())
    end_date = st.date_input("Bis", value=default_end.date())
    load_clicked = st.button("📥 Daten laden", use_container_width=True, type="primary")
    if st.button("🔄 Cache leeren", use_container_width=True):
        fetch_date_range.clear()
        st.rerun()

if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

if load_clicked:
    st.session_state.data_loaded = True
    fetch_date_range.clear()

if not st.session_state.data_loaded:
    st.info("Zeitraum wählen und **Daten laden** klicken.")
    st.stop()

with st.spinner("Lade Handelsdaten..."):
    df = fetch_date_range(
        datetime.combine(start_date, datetime.min.time()),
        datetime.combine(end_date, datetime.min.time())
    )

if df.empty:
    st.warning("Keine Daten für den gewählten Zeitraum gefunden.")
    st.stop()

# --- Filter ---
with st.expander("🔍 Filter", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        isin_search = st.text_input("ISIN / Bond-Name suchen", placeholder="z.B. CH0550...")
    with col2:
        min_turnover = st.number_input("Min. Turnover CHF", min_value=0, value=0, step=100_000, format="%d")
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
    selected_date = None
    if "file_date" in filtered.columns and "turnover_chf" in filtered.columns:
        daily = filtered.groupby("file_date")["turnover_chf"].sum().reset_index()
        fig = px.bar(daily, x="file_date", y="turnover_chf",
                     labels={"file_date": "Datum", "turnover_chf": "Turnover CHF"},
                     color_discrete_sequence=[CHART_COLORS[0]])
        fig.update_layout(height=320, yaxis=YAXIS_STYLE, **PLOTLY_LAYOUT)
        event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="daily_chart")
        if event and event.selection and event.selection.get("points"):
            selected_date = event.selection["points"][0].get("x")

with chart2:
    st.subheader("Top 15 Bonds nach Turnover")
    if "turnover_chf" in filtered.columns:
        has_name = "product_short_name" in filtered.columns
        group_cols = ["product_isin"] + (["product_short_name"] if has_name else [])
        top = (filtered.groupby(group_cols)["turnover_chf"]
               .sum().reset_index().nlargest(15, "turnover_chf"))
        if has_name:
            top["label"] = top["product_short_name"].fillna(top["product_isin"]).str[:35]
        else:
            top["label"] = top["product_isin"]
        fig2 = px.bar(top, x="turnover_chf", y="label", orientation="h",
                      labels={"turnover_chf": "Turnover CHF", "label": ""},
                      color_discrete_sequence=[CHART_COLORS[1]])
        fig2.update_layout(height=320, yaxis=dict(**YAXIS_STYLE, autorange="reversed"), **PLOTLY_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

# --- Tagesdrill-down wenn Bar angeklickt ---
if selected_date:
    try:
        sel_dt = pd.to_datetime(selected_date)
        day_df = filtered[filtered["file_date"].dt.date == sel_dt.date()]
        st.markdown("---")
        st.subheader(f"Trades am {sel_dt.strftime('%d.%m.%Y')} — {len(day_df):,} Trades")
        render_trade_table(day_df, max_rows=max_rows)
    except Exception:
        pass

# --- Gesamttabelle ---
st.markdown("---")
st.subheader("Alle Handelsdaten")
render_trade_table(filtered, max_rows=max_rows)
