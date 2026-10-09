import io
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ======================================================
# CONFIG AND THEME
# ======================================================
st.set_page_config(page_title="Portfolio Value Analyser", layout="wide", initial_sidebar_state="collapsed")

DARK_BG = "#0B1F33"
CARD_BG = "#0f2a44"
BORDER = "#1f3d5c"
GREEN = "#2ECC71"
RED = "#E74C3C"
BLUE = "#4C82C3"
ORANGE = "#F39C12"
PURPLE = "#9B59B6"
CYAN = "#00A3A1"
WHITE = "#FFFFFF"
GREY = "#A7A8AA"
EQUINOR_RED = "#FF1243"
EQUINOR_DARK_RED = "#C4002F"
EQUINOR_LIGHT_RED = "#FF6B81"
EQUINOR_DARK_GREY = "#5F6369"
EQUINOR_BAR_PALETTE = [EQUINOR_RED, EQUINOR_DARK_RED, EQUINOR_LIGHT_RED, GREY, EQUINOR_DARK_GREY]
DELTA = "\u0394"

st.markdown(
    f"""
<style>
.stApp {{ background-color: {DARK_BG}; color: #FFFFFF; }}
h1,h2,h3,h4,h5,h6,p,label,span,div {{ color: #FFFFFF; }}
.block-container {{ padding-top: 1.4rem; padding-bottom: 2rem; max-width: 98%; }}
.metric-card {{ background: {CARD_BG}; border: 1px solid {BORDER}; border-radius: 12px; padding: 14px 16px; min-height: 98px; box-shadow: 0 2px 10px rgba(0,0,0,0.20); }}
.metric-label {{ color: #d0d8e0; font-size: 13px; margin-bottom: 5px; }}
.metric-value {{ color: #FFFFFF; font-size: 25px; font-weight: 800; }}
.metric-sub {{ color: #9aa6b2; font-size: 12px; margin-top: 4px; }}
.summary-box {{ background: {CARD_BG}; border: 1px solid {BORDER}; border-radius: 14px; padding: 18px 22px; margin-bottom: 16px; }}
[data-testid="stDataFrame"] {{ background-color: #FFFFFF; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 22px; background: {DARK_BG}; border-bottom: 1px solid rgba(255,255,255,0.10); padding-top: 6px; }}
.stTabs [data-baseweb="tab"] {{ height: 46px; white-space: nowrap; color: #FFFFFF; font-weight: 700; border-bottom: 3px solid transparent; padding-left: 0px; padding-right: 0px; }}
.stTabs [aria-selected="true"] {{ color: #FFFFFF !important; border-bottom: 3px solid {EQUINOR_RED} !important; }}
.stDownloadButton button {{ background-color: #007079; color: white; border-radius: 8px; border: 0px; }}
</style>
""",
    unsafe_allow_html=True,
)

# ======================================================
# CONSTANTS AND LOCATION LOOKUP
# ======================================================
REQUIRED_COLS = {"Load_Month_Start", "Purchase_LTC_Name", "Sales_LTC_Name", "freq", "f"}
SHIPPING_COLS = ["Total_Shipping_Cost", "Idle_Days", "Shipping_Days", "Shipping_Distance"]
EUROPE_TOKENS = ["NWE", "TTF", "UK", "NBP", "Zeebrugge", "France", "Spain", "Italy", "AE", "NAFTA", "Germany", "Poland", "Gate", "Klaipedos", "Europe", "Belgium", "Netherlands"]
ASIA_TOKENS = ["JKM", "Japan", "Korea", "KOGAS", "China", "CNOOC", "PetroChina", "India", "Osaka", "PTTT", "SE", "Dahej", "Taiwan", "Singapore", "Tokyo", "Higashiko", "Incheon", "Map Ta Phut", "Jakarta", "Asia", "Prism", "Achema", "Deepak", "Thailand", "SE Asia"]
SOUTH_AMERICA_TOKENS = ["Argentina", "Argentin", "Punta Colorada", "South America"]
US_TOKENS = ["US", "USA", "Sabine", "Corpus", "Corpus Christi", "Freeport", "Calcasieu", "Cove", "Cameron", "Elba", "Plaquemines", "Lake Charles", "Port Arthur", "Golden Pass", "Henry", "HH", "CC", "CC T1", "CC T2", "CC T3", "CC T4", "SP", "SP T1", "SP T2", "SP T3", "SP T4"]
SPOT_TOKENS = ["spot", "spt", "dummy", "local_spt"]
SHIPPING_TOKENS = ["empty", "idle", "ship", "shipping", "freight", "vessel", "ballast", "boil"]
BRENT_TOKENS = ["brent", "oil", "crude"]
HH_TOKENS = ["hh", "henry", "corpus", "corpus christi", "sabine", "cc", "sp"]
TTF_TOKENS = ["ttf", "nwe", "gate", "nbp", "zeebrugge", "klaipedos"]
JKM_TOKENS = ["jkm", "japan", "korea", "tokyo", "higashiko", "incheon", "osaka", "prism", "achema", "deepak", "thailand", "se asia"]
SNOHVIT_TOKENS = ["snohvit", "snøhvit", "melkoya", "melkøya", "hammerfest", "norway"]

_CHART_COUNTER = 0

LOCATION_COORDS: Dict[str, Tuple[float, float, str]] = {
    "jkm": (35.6762, 139.6503, "JKM / Japan"), "japan": (35.6762, 139.6503, "Japan"), "tokyo": (35.6762, 139.6503, "Tokyo"), "osaka": (34.6937, 135.5023, "Osaka"), "higashiko": (35.4437, 139.6380, "Higashiko"),
    "korea": (37.5665, 126.9780, "Korea"), "kogas": (37.5665, 126.9780, "Korea / KOGAS"), "incheon": (37.4563, 126.7052, "Incheon"),
    "china": (31.2304, 121.4737, "China"), "cnooc": (31.2304, 121.4737, "China / CNOOC"), "petrochina": (39.9042, 116.4074, "China / PetroChina"),
    "india": (21.1702, 72.8311, "India"), "deepak": (21.1702, 72.8311, "India / Deepak"), "dahej": (21.7000, 72.5500, "Dahej"),
    "thailand": (13.7563, 100.5018, "Thailand"), "pttt": (13.7563, 100.5018, "Thailand / PTTT"), "map ta phut": (12.6840, 101.1350, "Map Ta Phut"),
    "se asia": (1.3521, 103.8198, "SE Asia / Singapore"), "singapore": (1.3521, 103.8198, "Singapore"), "jakarta": (-6.2088, 106.8456, "Jakarta"), "taiwan": (25.0330, 121.5654, "Taiwan"),
    "prism": (35.6762, 139.6503, "Prism / Asia"), "achema": (55.1694, 23.8813, "Achema / Lithuania"),
    "ttf": (52.3676, 4.9041, "TTF / Netherlands"), "nwe": (51.9244, 4.4777, "NWE"), "nbp": (51.5074, -0.1278, "UK / NBP"), "uk": (51.5074, -0.1278, "UK"),
    "zeebrugge": (51.3290, 3.2070, "Zeebrugge"), "gate": (51.9444, 4.1528, "Gate"), "klaipedos": (55.7033, 21.1443, "Klaipedos"),
    "france": (48.8566, 2.3522, "France"), "spain": (40.4168, -3.7038, "Spain"), "italy": (41.9028, 12.4964, "Italy"), "germany": (52.5200, 13.4050, "Germany"), "poland": (52.2297, 21.0122, "Poland"), "belgium": (50.8503, 4.3517, "Belgium"), "netherlands": (52.3676, 4.9041, "Netherlands"),
    "argentina": (-34.6037, -58.3816, "Argentina"), "punta colorada": (-34.9000, -56.2000, "Punta Colorada"), "south america": (-23.5505, -46.6333, "South America"),
    "snohvit": (70.6634, 23.6821, "Snohvit / Melkoya"), "snøhvit": (70.6634, 23.6821, "Snohvit / Melkoya"), "melkoya": (70.6634, 23.6821, "Melkoya"), "melkøya": (70.6634, 23.6821, "Melkoya"), "hammerfest": (70.6634, 23.6821, "Hammerfest"),
    "sabine": (29.7350, -93.8700, "Sabine Pass"), "corpus": (27.8006, -97.3964, "Corpus Christi"), "corpus christi": (27.8006, -97.3964, "Corpus Christi"), "freeport": (28.9541, -95.3597, "Freeport"),
    "calcasieu": (30.2366, -93.3774, "Calcasieu"), "cameron": (29.7976, -93.3252, "Cameron"), "elba": (32.0870, -81.0090, "Elba Island"), "plaque": (29.3819, -89.6037, "Plaquemines"), "plaquemines": (29.3819, -89.6037, "Plaquemines"),
    "lake charles": (30.2266, -93.2174, "Lake Charles"), "port arthur": (29.8849, -93.9399, "Port Arthur"), "golden pass": (29.7604, -93.9299, "Golden Pass"), "hh": (29.7604, -95.3698, "Henry Hub / US Gulf"), "henry": (29.7604, -95.3698, "Henry Hub / US Gulf"), "cc": (27.8006, -97.3964, "Corpus Christi"), "sp": (29.7350, -93.8700, "Sabine Pass"),
}

FLOW_COLOUR_MAP = {
    "Argentina / South America": BLUE,
    "US Supply": RED,
    "Snohvit": WHITE,
    "Spot Sales": ORANGE,
    "Asia Sales": GREEN,
    "Europe Sales": PURPLE,
    "Other": GREY,
}

# ======================================================
# GENERAL HELPERS
# ======================================================
def next_chart_key(prefix="chart"):
    global _CHART_COUNTER
    _CHART_COUNTER += 1
    return f"{prefix}_{_CHART_COUNTER}"


def _contains_any_token(name: str, tokens: List[str]) -> bool:
    n = str(name)
    return any(re.search(rf"(?<![A-Za-z0-9_]){re.escape(tok)}(?![A-Za-z0-9_])", n, re.IGNORECASE) for tok in tokens)


def _coord_for_name(name: str):
    text = str(name).lower()
    for key, value in LOCATION_COORDS.items():
        if key in text:
            return value
    return None


def _strip_column_names(df: pl.DataFrame) -> pl.DataFrame:
    return df.rename({c: c.strip() for c in df.columns})


def _ensure_polars_date(df: pl.DataFrame, col="Load_Month_Start") -> pl.DataFrame:
    if col not in df.columns:
        return df
    dtype = df.schema.get(col)
    if dtype == pl.Utf8:
        return df.with_columns(pl.col(col).str.strptime(pl.Date, strict=False).alias(col))
    if dtype == pl.Datetime:
        return df.with_columns(pl.col(col).dt.date().alias(col))
    if dtype == pl.Date:
        return df
    return df.with_columns(pl.col(col).cast(pl.Date, strict=False).alias(col))


def _require_columns(df: pl.DataFrame, name: str):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error(f"File '{name}' is missing columns: {', '.join(missing)}. Required: Load_Month_Start, Purchase_LTC_Name, Sales_LTC_Name, freq, f.")
        st.stop()


def _canonical_base_common(name: str) -> str:
    s = str(name).strip()
    if not s:
        return s
    s = re.sub(r"\d.*$", "", s).strip()
    s = re.split(r"[\(\-\u2013\/\\,\|\:\;]", s)[0].strip()
    parts = s.split()
    return parts[0] if parts else s


def _word_boundary_mask(series: pd.Series, token: str) -> pd.Series:
    if not token:
        return pd.Series(False, index=series.index)
    pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])", re.IGNORECASE)
    return series.astype(str).apply(lambda s: bool(pattern.search(s)))


def fmt_signed(v, decimals=0):
    if pd.isna(v):
        return "n/a"
    sign = "+" if v > 0 else ""
    return f"{sign}{v:,.{decimals}f}"


def season_from_month(m) -> str:
    try:
        m = int(m)
    except Exception:
        return "Unknown"
    return "Winter Oct-Mar" if m in [10, 11, 12, 1, 2, 3] else "Summer Apr-Sep" if m in [4, 5, 6, 7, 8, 9] else "Unknown"


def quarter_from_month(m) -> str:
    try:
        m = int(m)
    except Exception:
        return "Unknown"
    return f"Q{((m - 1) // 3) + 1}"


def metric_card(label: str, value: str, sub: str = ""):
    st.markdown(f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div><div class='metric-sub'>{sub}</div></div>", unsafe_allow_html=True)


def metric_card_coloured(label: str, value: str, sub: str = "", colour: str = "#FFFFFF"):
    st.markdown(f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value' style='color:{colour};'>{value}</div><div class='metric-sub'>{sub}</div></div>", unsafe_allow_html=True)

# ======================================================
# CLASSIFICATION
# ======================================================
def classify_sales_region(name: str) -> str:
    if _contains_any_token(name, SOUTH_AMERICA_TOKENS):
        return "South America"
    if _contains_any_token(name, EUROPE_TOKENS):
        return "Europe"
    if _contains_any_token(name, ASIA_TOKENS):
        return "Asia"
    return "Other"


def is_purchase_us(name: str) -> bool:
    n = str(name).strip()
    return _contains_any_token(n, US_TOKENS) or bool(re.search(r"^(CC|SP)(\s|_|-|$)", n, re.IGNORECASE)) or bool(re.search(r"\b(CC|SP)\s*T\d*\b", n, re.IGNORECASE))


def is_snohvit(name: str) -> bool:
    return _contains_any_token(str(name), SNOHVIT_TOKENS)


def flow_category(purchase_name: str, sales_name: str) -> str:
    combined = f"{purchase_name} {sales_name}"
    if _contains_any_token(combined, SOUTH_AMERICA_TOKENS):
        return "Argentina / South America"
    if is_snohvit(purchase_name) or is_snohvit(sales_name):
        return "Snohvit"
    if is_purchase_us(purchase_name):
        return "US Supply"
    if _contains_any_token(sales_name, SPOT_TOKENS):
        return "Spot Sales"
    sales_region = classify_sales_region(sales_name)
    if sales_region == "Asia":
        return "Asia Sales"
    if sales_region == "Europe":
        return "Europe Sales"
    return "Other"


def classify_index_bucket(row: pd.Series) -> str:
    combined = f"{row.get('Purchase_LTC_Name', '')} {row.get('Sales_LTC_Name', '')}"
    if _contains_any_token(combined, JKM_TOKENS):
        return "JKM / Asia"
    if _contains_any_token(combined, TTF_TOKENS):
        return "TTF / Europe"
    if _contains_any_token(combined, HH_TOKENS):
        return "HH / US"
    if _contains_any_token(combined, BRENT_TOKENS):
        return "Brent / Oil"
    return "Other"


def classify_volume_bucket(row: pd.Series) -> str:
    purchase = str(row.get("Purchase_LTC_Name", ""))
    sales = str(row.get("Sales_LTC_Name", ""))
    combined = f"{purchase} {sales}"
    if _contains_any_token(combined, SOUTH_AMERICA_TOKENS):
        return "South America"
    if is_purchase_us(purchase):
        return "US Supply"
    region = classify_sales_region(sales)
    return region if region in {"Asia", "Europe"} else "Other"


def _tokens_from_df(df_pd: pd.DataFrame) -> set:
    names = []
    if "Purchase_LTC_Name" in df_pd.columns:
        names += df_pd["Purchase_LTC_Name"].astype(str).tolist()
    if "Sales_LTC_Name" in df_pd.columns:
        names += df_pd["Sales_LTC_Name"].astype(str).tolist()
    toks = set()
    for n in names:
        base = _canonical_base_common(n)
        b = base.strip().lower() if base else ""
        if b and b not in {"none", "empty"}:
            toks.add(b)
    return toks


def is_new_position_row(row: pd.Series, new_tokens_set: set) -> bool:
    p_base = _canonical_base_common(row.get("Purchase_LTC_Name", "")).strip().lower()
    s_base = _canonical_base_common(row.get("Sales_LTC_Name", "")).strip().lower()
    return p_base in new_tokens_set or s_base in new_tokens_set


def _pl_for_token_union(df_pd: pd.DataFrame, token: str) -> float:
    purchase_series = df_pd["Purchase_LTC_Name"].astype(str)
    sales_series = df_pd["Sales_LTC_Name"].astype(str)
    sales_is_none = sales_series.str.strip().str.casefold().eq("none")
    return float(pd.to_numeric(df_pd.loc[_word_boundary_mask(purchase_series, token) | ((~sales_is_none) & _word_boundary_mask(sales_series, token)), "profit_loss"], errors="coerce").fillna(0.0).sum())


def _freq_for_token_union(df_pd: pd.DataFrame, token: str) -> float:
    purchase_series = df_pd["Purchase_LTC_Name"].astype(str)
    sales_series = df_pd["Sales_LTC_Name"].astype(str)
    sales_is_none = sales_series.str.strip().str.casefold().eq("none")
    return float(pd.to_numeric(df_pd.loc[_word_boundary_mask(purchase_series, token) | ((~sales_is_none) & _word_boundary_mask(sales_series, token)), "freq"], errors="coerce").fillna(0.0).sum())


def classify_receiver(row: pd.Series, new_tokens_set: set) -> str:
    purchase = str(row.get("Purchase_LTC_Name", ""))
    sales = str(row.get("Sales_LTC_Name", ""))
    combined = f"{purchase} {sales}"
    sales_region = classify_sales_region(sales)
    sales_is_spot = _contains_any_token(sales, SPOT_TOKENS)
    if _contains_any_token(combined, SOUTH_AMERICA_TOKENS):
        return "South America supply"
    if is_new_position_row(row, new_tokens_set):
        return "New Position"
    if sales_is_spot:
        if sales_region == "Asia":
            return "Spot Optimisation Asia"
        if sales_region == "Europe":
            return "Spot Optimisation Europe"
        if sales_region == "South America":
            return "South America supply"
        return "Spot Optimisation Other"
    sales_norm = sales.strip().casefold()
    purchase_norm = purchase.strip().casefold()
    if sales_norm in {"empty", "none"} or purchase_norm == "empty" or _contains_any_token(combined, SHIPPING_TOKENS):
        return "Shipping / Idle / Empty"
    if is_purchase_us(purchase):
        return "Existing US Supply"
    if sales_region == "Asia":
        return "Existing Asia Sales"
    if sales_region == "Europe":
        return "Existing Europe Sales"
    if sales_region == "South America":
        return "South America supply"
    return "Other / Unclassified"

# ======================================================
# DATA PREP
# ======================================================
@st.cache_data(show_spinner=False)
def load_file(file_name: str, file_bytes: bytes) -> pl.DataFrame:
    if file_name.lower().endswith(".csv"):
        df = pl.read_csv(io.BytesIO(file_bytes), try_parse_dates=True)
    else:
        df = pl.read_excel(io.BytesIO(file_bytes))
    return _strip_column_names(df)


@st.cache_data(show_spinner=False)
def prepare_sankey(df: pl.DataFrame) -> pd.DataFrame:
    return (
        df.with_columns([
            (pl.col("freq").cast(pl.Float64, strict=False) / 100).alias("freq"),
            (pl.col("f").cast(pl.Float64, strict=False) / 100).alias("profit_loss"),
        ])
        .group_by(["Purchase_LTC_Name", "Sales_LTC_Name"])
        .agg([pl.col("freq").sum(), pl.col("profit_loss").sum()])
        .filter(pl.col("freq") != 0)
        .to_pandas()
    )


@st.cache_data(show_spinner=False)
def prepare_monthly_route(df: pl.DataFrame) -> pd.DataFrame:
    return (
        df.with_columns([
            (pl.col("freq").cast(pl.Float64, strict=False) / 100).alias("freq"),
            (pl.col("f").cast(pl.Float64, strict=False) / 100).alias("profit_loss"),
            pl.col("Load_Month_Start").dt.month().alias("month"),
            pl.col("Load_Month_Start").dt.year().alias("year"),
        ])
        .group_by(["Load_Month_Start", "year", "month", "Purchase_LTC_Name", "Sales_LTC_Name"])
        .agg([pl.col("freq").sum(), pl.col("profit_loss").sum()])
        .sort("Load_Month_Start")
        .to_pandas()
    )


@st.cache_data(show_spinner=False)
def prepare_risk_by_sim_full_file(file_name: str, file_bytes: bytes, label: str) -> pd.DataFrame:
    df = load_file(file_name, file_bytes)
    if "sim_id" not in df.columns or "f" not in df.columns:
        return pd.DataFrame()
    out = df.with_columns(pl.col("f").cast(pl.Float64, strict=False).alias("f_numeric")).group_by("sim_id").agg(pl.col("f_numeric").sum().alias(label)).sort("sim_id").to_pandas()
    out["sim_id"] = pd.to_numeric(out["sim_id"], errors="coerce")
    out[label] = pd.to_numeric(out[label], errors="coerce")
    return out.dropna()


def cvar_5(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return np.nan
    p5 = values.quantile(0.05)
    tail = values[values <= p5]
    return float(tail.mean()) if not tail.empty else np.nan


def risk_stats_table(values: pd.Series, label: str) -> pd.DataFrame:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "Metric": ["Mean", "Std dev", "P5", "P10", "P50", "P90", "P95", "CVaR 5%"],
        label: [values.mean(), values.std(ddof=0), values.quantile(0.05), values.quantile(0.10), values.quantile(0.50), values.quantile(0.90), values.quantile(0.95), cvar_5(values)],
    })


def volume_by_region(sankey_a: pd.DataFrame, sankey_b: pd.DataFrame) -> pd.DataFrame:
    def _vol(sankey, label):
        tmp = sankey.copy()
        tmp["Destination_Region"] = tmp["Sales_LTC_Name"].apply(classify_sales_region)
        return tmp.groupby("Destination_Region", as_index=False).agg(**{label: ("freq", "sum")})
    a = _vol(sankey_a, "Portfolio + new position")
    b = _vol(sankey_b, "Base portfolio")
    out = pd.merge(b, a, on="Destination_Region", how="outer").fillna(0)
    out["Change"] = out["Portfolio + new position"] - out["Base portfolio"]
    out["Abs_Change"] = out["Change"].abs()
    return out.sort_values("Abs_Change", ascending=False)


def build_shipping_existing_only(df_a: pl.DataFrame, df_b: pl.DataFrame, scale_shipping: bool, new_tokens_set: set) -> Dict[str, pd.DataFrame]:
    missing = [c for c in SHIPPING_COLS if c not in df_a.columns or c not in df_b.columns]
    if missing:
        return {"missing": pd.DataFrame({"Missing_Column": missing}), "total": pd.DataFrame(), "region": pd.DataFrame(), "raw_a": pd.DataFrame(), "raw_b": pd.DataFrame()}

    def _prep(df: pl.DataFrame, label: str) -> pd.DataFrame:
        scale = 100.0 if scale_shipping else 1.0
        tmp = df.select(["Purchase_LTC_Name", "Sales_LTC_Name"] + SHIPPING_COLS).to_pandas()
        tmp["Purchase_Base"] = tmp["Purchase_LTC_Name"].apply(lambda x: _canonical_base_common(x).strip().lower())
        tmp["Sales_Base"] = tmp["Sales_LTC_Name"].apply(lambda x: _canonical_base_common(x).strip().lower())
        tmp["Is_New_Position_Row"] = tmp.apply(lambda r: (r["Purchase_Base"] in new_tokens_set) or (r["Sales_Base"] in new_tokens_set), axis=1)
        tmp = tmp[~tmp["Is_New_Position_Row"]].copy()
        for c in SHIPPING_COLS:
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce").fillna(0.0) / scale
        tmp["Destination_Region"] = tmp["Sales_LTC_Name"].apply(classify_sales_region)
        tmp["Case"] = label
        return tmp

    a = _prep(df_a, "With New Position")
    b = _prep(df_b, "Original Portfolio")
    total = pd.DataFrame([{"Metric": c, "Original Portfolio": b[c].sum(), "With New Position": a[c].sum(), "Change": a[c].sum() - b[c].sum()} for c in SHIPPING_COLS])
    total["% Change"] = np.where(total["Original Portfolio"].abs() > 1e-9, total["Change"] / total["Original Portfolio"], np.nan)
    a_reg = a.groupby("Destination_Region", as_index=False)[SHIPPING_COLS].sum()
    b_reg = b.groupby("Destination_Region", as_index=False)[SHIPPING_COLS].sum()
    region = pd.merge(b_reg, a_reg, on="Destination_Region", how="outer", suffixes=("_Original", "_With")).fillna(0)
    for c in SHIPPING_COLS:
        region[f"{c}_Change"] = region[f"{c}_With"] - region[f"{c}_Original"]
    return {"missing": pd.DataFrame(), "total": total, "region": region, "raw_a": a, "raw_b": b}

# ======================================================
# CHART AND MAP HELPERS
# ======================================================
def apply_dark_layout(fig, height=450, title=None):
    fig.update_layout(
        title=title if title else fig.layout.title.text,
        font=dict(color="#FFFFFF"),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        height=height,
        legend=dict(font=dict(color="#FFFFFF")),
        margin=dict(l=40, r=30, t=70, b=40),
    )
    fig.update_xaxes(title_font=dict(color="#FFFFFF"), tickfont=dict(color="#FFFFFF"), color="#FFFFFF", gridcolor="rgba(255,255,255,0.10)")
    fig.update_yaxes(title_font=dict(color="#FFFFFF"), tickfont=dict(color="#FFFFFF"), color="#FFFFFF", gridcolor="rgba(255,255,255,0.10)")
    return fig


def dark_bar(df: pd.DataFrame, x: str, y: str, title: str, color: Optional[str] = None, horizontal: bool = False, height: int = 430):
    if df.empty:
        st.info("No data to display.")
        return
    fig = px.bar(df, x=y, y=x, color=color, orientation="h", title=title, color_discrete_sequence=EQUINOR_BAR_PALETTE) if horizontal else px.bar(df, x=x, y=y, color=color, title=title, color_discrete_sequence=EQUINOR_BAR_PALETTE)
    if horizontal:
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
    if color is None:
        fig.update_traces(marker_color=EQUINOR_RED)
    apply_dark_layout(fig, height=height)
    st.plotly_chart(fig, use_container_width=True, key=next_chart_key("bar"))


def plot_signed_donut(df, label_col, value_col, title):
    if df.empty or label_col not in df.columns or value_col not in df.columns:
        st.info("No data to display.")
        return
    tmp = df[[label_col, value_col]].copy()
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce").fillna(0.0)
    tmp = tmp[tmp[value_col].abs() > 1e-12].copy()
    if tmp.empty:
        st.info("No non-zero values to display.")
        return
    tmp["Abs_Value"] = tmp[value_col].abs()
    tmp["Signed_Value"] = tmp[value_col]
    fig = px.pie(tmp, names=label_col, values="Abs_Value", hole=0.45, color_discrete_sequence=EQUINOR_BAR_PALETTE, custom_data=["Signed_Value"], title=title)
    fig.update_traces(textposition="inside", texttemplate="%{label}<br>%{percent}<br>%{customdata[0]:,.0f}", hovertemplate="%{label}<br>Contribution: %{customdata[0]:,.0f}<br>Share of absolute total: %{percent}<extra></extra>")
    apply_dark_layout(fig, height=470)
    st.plotly_chart(fig, use_container_width=True, key=next_chart_key("donut"))


def filter_out_none_empty_sales(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    sales = df["Sales_LTC_Name"].astype(str).str.strip().str.casefold()
    out = df.loc[~sales.isin({"none", "empty"})].copy()
    if not out.empty:
        out = out.groupby(["Purchase_LTC_Name", "Sales_LTC_Name"], as_index=False)[["freq", "profit_loss"]].sum()
    return out


def render_sankey(df: pd.DataFrame, title: str, diff_mode: bool = False):
    if df.empty:
        return go.Figure()
    labels = pd.unique(df[["Purchase_LTC_Name", "Sales_LTC_Name"]].values.ravel()).tolist()
    label_map = {l: i for i, l in enumerate(labels)}
    values = (df["freq"].abs() if diff_mode else df["freq"]).tolist()
    colors = df["freq"].apply(lambda x: "rgba(46,204,113,0.70)" if x > 0 else "rgba(231,76,60,0.70)").tolist() if diff_mode else ["rgba(217,30,24,0.6)"] * len(df)
    src = df["Purchase_LTC_Name"].map(label_map).tolist()
    tgt = df["Sales_LTC_Name"].map(label_map).tolist()
    node_totals = [0] * len(labels)
    for s, t, v in zip(src, tgt, values):
        node_totals[s] += v
        node_totals[t] += v
    fig = go.Figure(go.Sankey(node=dict(label=labels, color=BLUE, pad=15, thickness=20, customdata=[int(round(x)) for x in node_totals], hovertemplate="%{label}<br>Total cargoes: %{customdata}<extra></extra>"), link=dict(source=src, target=tgt, value=values, color=colors)))
    fig.update_layout(title=title, font=dict(color="#FFFFFF"), paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG, height=420)
    return fig


def make_excel_download(frames: Dict[str, pd.DataFrame]) -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        for name, frame in frames.items():
            safe = re.sub(r"[^A-Za-z0-9_ ]", "", name)[:31] or "Sheet"
            frame.to_excel(writer, sheet_name=safe, index=False)
    return out.getvalue()


def _safe_metric_from_risk(risk_stats: pd.DataFrame, metric_name: str, column: str):
    if risk_stats.empty or "Metric" not in risk_stats.columns or column not in risk_stats.columns:
        return np.nan
    row = risk_stats[risk_stats["Metric"] == metric_name]
    return np.nan if row.empty else float(row[column].iloc[0])


def _pct_change(new_value, old_value):
    if pd.isna(new_value) or pd.isna(old_value) or abs(old_value) < 1e-12:
        return np.nan
    return (new_value - old_value) / abs(old_value)


def _fmt_value(v, decimals=0):
    if pd.isna(v):
        return "n/a"
    return f"{v:,.{decimals}f}"


def _fmt_delta(v, decimals=0):
    if pd.isna(v):
        return "n/a"
    sign = "+" if v > 0 else ""
    return f"{sign}{v:,.{decimals}f}"


def kpi_box(label, value, sub="", positive_is_good=True):
    if pd.isna(value):
        colour = "#9aa6b2"
        display = "n/a"
    else:
        colour = GREEN if (value >= 0 if positive_is_good else value <= 0) else RED
        display = _fmt_delta(value)
    st.markdown(f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value' style='color:{colour};'>{display}</div><div class='metric-sub'>{sub}</div></div>", unsafe_allow_html=True)


def value_box(label, value, sub=""):
    st.markdown(f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{_fmt_value(value)}</div><div class='metric-sub'>{sub}</div></div>", unsafe_allow_html=True)


def region_index_sankey(df):
    if df.empty:
        st.info("No region-index data to display.")
        return
    tmp = df.copy()
    tmp["Flow_Value"] = pd.to_numeric(tmp["Profit_Loss"], errors="coerce").fillna(0.0).abs()
    tmp = tmp[tmp["Flow_Value"] > 1e-12].copy()
    if tmp.empty:
        st.info("No non-zero region-index flows to display.")
        return
    labels = pd.unique(tmp[["Sales_Region", "Index_Bucket"]].values.ravel()).tolist()
    label_map = {label: i for i, label in enumerate(labels)}
    fig = go.Figure(go.Sankey(node=dict(label=labels, color=BLUE, pad=18, thickness=20), link=dict(source=tmp["Sales_Region"].map(label_map), target=tmp["Index_Bucket"].map(label_map), value=tmp["Flow_Value"], color="rgba(217,30,24,0.45)")))
    apply_dark_layout(fig, height=500, title="Region to index attribution")
    st.plotly_chart(fig, use_container_width=True, key=next_chart_key("region_index_sankey"))


def build_destination_map_data(df: pd.DataFrame, value_col="profit_loss", cargo_col="freq") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for _, r in df.iterrows():
        coord = _coord_for_name(r.get("Sales_LTC_Name", ""))
        if coord is None:
            continue
        lat, lon, label = coord
        rows.append({
            "Location": label,
            "Sales_LTC_Name": r.get("Sales_LTC_Name", ""),
            "lat": lat,
            "lon": lon,
            "Profit_Loss": float(pd.to_numeric(r.get(value_col, 0.0), errors="coerce") or 0.0),
            "Cargo": float(pd.to_numeric(r.get(cargo_col, 0.0), errors="coerce") or 0.0),
            "Region": classify_sales_region(r.get("Sales_LTC_Name", "")),
        })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).groupby(["Location", "lat", "lon", "Region"], as_index=False).agg(Profit_Loss=("Profit_Loss", "sum"), Cargo=("Cargo", "sum"))
    out["Signed"] = np.where(out["Profit_Loss"] >= 0, "Positive", "Negative")
    return out


def build_flow_map_data(df: pd.DataFrame, include_category: bool = False) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for _, r in df.iterrows():
        src = _coord_for_name(r.get("Purchase_LTC_Name", ""))
        dst = _coord_for_name(r.get("Sales_LTC_Name", ""))
        if src is None or dst is None:
            continue
        slat, slon, slabel = src
        dlat, dlon, dlabel = dst
        category = flow_category(str(r.get("Purchase_LTC_Name", "")), str(r.get("Sales_LTC_Name", ""))) if include_category else "Difference"
        rows.append({
            "Source": slabel,
            "Destination": dlabel,
            "Category": category,
            "src_lat": slat,
            "src_lon": slon,
            "dst_lat": dlat,
            "dst_lon": dlon,
            "freq": float(pd.to_numeric(r.get("freq", 0.0), errors="coerce") or 0.0),
            "profit_loss": float(pd.to_numeric(r.get("profit_loss", 0.0), errors="coerce") or 0.0),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).groupby(["Source", "Destination", "Category", "src_lat", "src_lon", "dst_lat", "dst_lon"], as_index=False).agg(freq=("freq", "sum"), profit_loss=("profit_loss", "sum"))


def apply_geo_layout(fig, title: str):
    fig.update_layout(
        title=title,
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font=dict(color="#FFFFFF"),
        height=640,
        margin=dict(l=0, r=0, t=60, b=0),
        geo=dict(
            projection_type="natural earth",
            bgcolor=DARK_BG,
            showland=True,
            landcolor="rgb(55, 60, 66)",
            showocean=True,
            oceancolor="rgb(20, 35, 50)",
            showcountries=True,
            countrycolor="rgba(255,255,255,0.18)",
            coastlinecolor="rgba(255,255,255,0.25)",
            showframe=False,
        ),
    )
    return fig


def map_bubble(df: pd.DataFrame, metric: str, title: str):
    if df.empty or metric not in df.columns:
        st.info("No map data available. Add more location keywords to LOCATION_COORDS if needed.")
        return
    tmp = df.copy()
    tmp["size"] = tmp[metric].abs()
    tmp["size"] = np.where(tmp["size"] <= 1e-12, 1.0, tmp["size"])
    fig = px.scatter_geo(
        tmp,
        lat="lat",
        lon="lon",
        size="size",
        color="Signed",
        color_discrete_map={"Positive": GREEN, "Negative": RED},
        hover_name="Location",
        hover_data={"Profit_Loss": ":,.0f", "Cargo": ":,.0f", "Region": True, "lat": False, "lon": False, "size": False},
    )
    apply_geo_layout(fig, title)
    st.plotly_chart(fig, use_container_width=True, key=next_chart_key("map"))


def difference_flow_map(df: pd.DataFrame, title: str):
    if df.empty:
        st.info("No difference flow map data available. Add more location keywords to LOCATION_COORDS if needed.")
        return
    fig = go.Figure()
    max_freq = max(float(df["freq"].abs().max()), 1.0)
    for _, r in df.iterrows():
        colour = GREEN if r["freq"] >= 0 else RED
        text_direction = "More cargoes" if r["freq"] >= 0 else "Fewer cargoes"
        width = 1.0 + 6.0 * abs(r["freq"]) / max_freq
        fig.add_trace(go.Scattergeo(
            lon=[r["src_lon"], r["dst_lon"]],
            lat=[r["src_lat"], r["dst_lat"]],
            mode="lines",
            line=dict(width=width, color=colour),
            opacity=0.72,
            hoverinfo="text",
            text=f"{r['Source']} -> {r['Destination']}<br>{text_direction}: {r['freq']:,.0f}<br>P/L: {r['profit_loss']:,.0f}",
            showlegend=False,
        ))
    add_flow_nodes(fig, df)
    apply_geo_layout(fig, title)
    st.plotly_chart(fig, use_container_width=True, key=next_chart_key("difference_flow_map"))


def optimised_flow_map(df: pd.DataFrame, title: str):
    if df.empty:
        st.info("No optimised flow map data available. Add more location keywords to LOCATION_COORDS if needed.")
        return
    fig = go.Figure()
    max_freq = max(float(df["freq"].abs().max()), 1.0)
    for _, r in df.iterrows():
        category = r.get("Category", "Other")
        colour = FLOW_COLOUR_MAP.get(category, GREY)
        width = 1.0 + 7.0 * abs(r["freq"]) / max_freq
        fig.add_trace(go.Scattergeo(
            lon=[r["src_lon"], r["dst_lon"]],
            lat=[r["src_lat"], r["dst_lat"]],
            mode="lines",
            line=dict(width=width, color=colour),
            opacity=0.76,
            hoverinfo="text",
            text=f"{r['Source']} -> {r['Destination']}<br>Category: {category}<br>Cargoes: {r['freq']:,.0f}<br>P/L: {r['profit_loss']:,.0f}",
            showlegend=False,
        ))
    add_flow_nodes(fig, df)
    apply_geo_layout(fig, title)
    st.plotly_chart(fig, use_container_width=True, key=next_chart_key("optimised_flow_map"))


def add_flow_nodes(fig, df: pd.DataFrame):
    node_rows = []
    for _, r in df.iterrows():
        node_rows.append({"Location": r["Source"], "lat": r["src_lat"], "lon": r["src_lon"]})
        node_rows.append({"Location": r["Destination"], "lat": r["dst_lat"], "lon": r["dst_lon"]})
    if not node_rows:
        return
    nodes = pd.DataFrame(node_rows).drop_duplicates()
    fig.add_trace(go.Scattergeo(
        lon=nodes["lon"],
        lat=nodes["lat"],
        text=nodes["Location"],
        mode="markers",
        marker=dict(size=7, color=BLUE, line=dict(width=1, color="#FFFFFF")),
        hoverinfo="text",
        showlegend=False,
    ))


def flow_legend_table(df: pd.DataFrame):
    if df.empty or "Category" not in df.columns:
        st.info("No flow categories available.")
        return
    legend = df.groupby("Category", as_index=False).agg(Cargoes=("freq", "sum"), Routes=("Destination", "count"))
    legend["Colour"] = legend["Category"].map(FLOW_COLOUR_MAP).fillna(GREY)
    legend = legend.sort_values("Cargoes", key=lambda s: s.abs(), ascending=False)
    st.markdown("### Flow colour legend")
    for _, r in legend.iterrows():
        st.markdown(
            f"<div style='display:flex;align-items:center;margin-bottom:8px;'>"
            f"<div style='width:22px;height:12px;background:{r['Colour']};border:1px solid #FFFFFF;margin-right:10px;'></div>"
            f"<div><b>{r['Category']}</b><br><span style='color:#9aa6b2;'>Cargoes: {r['Cargoes']:,.0f} | Routes: {int(r['Routes'])}</span></div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.dataframe(legend[["Category", "Cargoes", "Routes"]], use_container_width=True, hide_index=True)

# ======================================================
# APP BODY AND CALCULATIONS
# ======================================================
st.title("Portfolio Value Analyser")
st.markdown("**Please contact Tore Rødland (Trodla@equinor.com) for any questions**")
scale_shipping = st.sidebar.checkbox("Scale shipping/day/distance columns by /100", value=True, key="scale_shipping_checkbox")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Upload file with Portfolio + new positions**")
    file_a = st.file_uploader("", type=["csv", "xlsx"], key="file_a")
with c2:
    st.markdown("**Upload file with the original Portfolio only**")
    file_b = st.file_uploader("", type=["csv", "xlsx"], key="file_b")
if not file_a or not file_b:
    st.info("Upload both portfolio files to start the analysis.")
    st.stop()

file_a_bytes = file_a.getvalue()
file_b_bytes = file_b.getvalue()
df_a_raw = load_file(file_a.name, file_a_bytes)
df_b_raw = load_file(file_b.name, file_b_bytes)
_require_columns(df_a_raw, file_a.name)
_require_columns(df_b_raw, file_b.name)
df_a_raw = _ensure_polars_date(df_a_raw)
df_b_raw = _ensure_polars_date(df_b_raw)
if not (df_a_raw.schema.get("Load_Month_Start") == pl.Date and df_b_raw.schema.get("Load_Month_Start") == pl.Date):
    st.error("Could not normalise Load_Month_Start to date in both files.")
    st.stop()

all_periods = pl.concat([df_a_raw.select("Load_Month_Start"), df_b_raw.select("Load_Month_Start")]).unique().sort("Load_Month_Start").to_series().to_list()
st.markdown("**Select Period Range**")
period_from, period_to = st.select_slider("", options=all_periods, value=(all_periods[0], all_periods[-1]), key="period_range_slider")
df_a = df_a_raw.filter((pl.col("Load_Month_Start") >= period_from) & (pl.col("Load_Month_Start") <= period_to))
df_b = df_b_raw.filter((pl.col("Load_Month_Start") >= period_from) & (pl.col("Load_Month_Start") <= period_to))

sankey_a = prepare_sankey(df_a)
sankey_b = prepare_sankey(df_b)
diff = pd.merge(sankey_a, sankey_b, on=["Purchase_LTC_Name", "Sales_LTC_Name"], how="outer", suffixes=("_A", "_B")).fillna(0)
for col in ["freq_A", "freq_B", "profit_loss_A", "profit_loss_B"]:
    diff[col] = pd.to_numeric(diff[col], errors="coerce").fillna(0.0)
diff["freq"] = diff["freq_A"] - diff["freq_B"]
diff["profit_loss"] = diff["profit_loss_A"] - diff["profit_loss_B"]
diff = diff[diff["freq"] != 0].copy()

tokens_a = _tokens_from_df(sankey_a)
tokens_b = _tokens_from_df(sankey_b)
new_tokens = sorted(tokens_a - tokens_b)
new_tokens_set = set(new_tokens)
total_pl = float(diff["profit_loss"].sum())
new_pl_total = float(sum(_pl_for_token_union(sankey_a, tok) for tok in new_tokens))
existing_pl = total_pl - new_pl_total
total_cargo_a = float(sankey_a["freq"].sum())
total_cargo_b = float(sankey_b["freq"].sum())
total_cargo_delta = total_cargo_a - total_cargo_b
new_cargo_total = float(sum(_freq_for_token_union(sankey_a, tok) for tok in new_tokens))
existing_cargo_delta = total_cargo_delta - new_cargo_total
existing_rate = existing_pl / (3_800_000.0 * total_cargo_delta) if abs(total_cargo_delta) > 1e-9 else 0.0

diff_e = diff.copy()
diff_e["Is_New_Position_Row"] = diff_e.apply(lambda r: is_new_position_row(r, new_tokens_set), axis=1)
diff_e["Sales_Region"] = diff_e["Sales_LTC_Name"].apply(classify_sales_region)
diff_e["Purchase_Is_US"] = diff_e["Purchase_LTC_Name"].apply(is_purchase_us)
diff_e["Index_Bucket"] = diff_e.apply(classify_index_bucket, axis=1)
diff_e["Volume_Bucket"] = diff_e.apply(classify_volume_bucket, axis=1)
diff_e["Receiver_Group"] = diff_e.apply(lambda r: classify_receiver(r, new_tokens_set), axis=1)
diff_e["Route"] = diff_e["Purchase_LTC_Name"].astype(str) + " -> " + diff_e["Sales_LTC_Name"].astype(str)
diff_e["Cargo_Positive"] = diff_e["freq"].clip(lower=0)
diff_e["Cargo_Negative"] = diff_e["freq"].clip(upper=0)
diff_e["PL_Positive"] = diff_e["profit_loss"].clip(lower=0)
diff_e["PL_Negative"] = diff_e["profit_loss"].clip(upper=0)
existing_diff = diff_e[~diff_e["Is_New_Position_Row"]].copy()

shipping = build_shipping_existing_only(df_a, df_b, scale_shipping, new_tokens_set)
shipping_cost_change = np.nan
if shipping["missing"].empty and not shipping["total"].empty:
    row = shipping["total"][shipping["total"]["Metric"] == "Total_Shipping_Cost"]
    if not row.empty:
        shipping_cost_change = float(row["Change"].iloc[0])
shipping_savings_value = -shipping_cost_change if pd.notna(shipping_cost_change) else 0.0

receiver_attr = diff_e.groupby("Receiver_Group", as_index=False).agg(Profit_Loss=("profit_loss", "sum"), Positive_Profit_Loss=("PL_Positive", "sum"), Negative_Profit_Loss=("PL_Negative", "sum"), Net_Cargo_Change=("freq", "sum"), Positive_Cargo_Change=("Cargo_Positive", "sum"), Negative_Cargo_Change=("Cargo_Negative", "sum"), Routes=("Route", "count")).sort_values("Profit_Loss", key=lambda s: s.abs(), ascending=False)
receiver_attr["Share_of_Total"] = np.where(abs(total_pl) > 1e-9, receiver_attr["Profit_Loss"] / total_pl, 0.0)

commercial_raw_all = existing_diff.groupby("Receiver_Group", as_index=False).agg(Profit_Loss=("profit_loss", "sum"), Net_Cargo_Change=("freq", "sum"), Positive_Cargo_Change=("Cargo_Positive", "sum"), Negative_Cargo_Change=("Cargo_Negative", "sum"), Routes=("Route", "count")).sort_values("Profit_Loss", key=lambda s: s.abs(), ascending=False)
existing_receiver = commercial_raw_all.copy()
if pd.notna(shipping_cost_change):
    existing_receiver = pd.concat([existing_receiver, pd.DataFrame([{"Receiver_Group": "Total Shipping Cost", "Profit_Loss": shipping_cost_change, "Net_Cargo_Change": np.nan, "Positive_Cargo_Change": np.nan, "Negative_Cargo_Change": np.nan, "Routes": np.nan}])], ignore_index=True)

commercial_raw = commercial_raw_all[~commercial_raw_all["Receiver_Group"].eq("Shipping / Idle / Empty")].copy()
commercial_reallocation_value = existing_pl - shipping_savings_value
existing_reallocation_level1 = pd.DataFrame([
    {"Driver": "Shipping Savings", "Value": shipping_savings_value, "Level": "Main", "Note": "Change in Total_Shipping_Cost inverted because lower cost is value-positive."},
    {"Driver": "Commercial Reallocation", "Value": commercial_reallocation_value, "Level": "Main", "Note": "Residual existing portfolio value after shipping savings."},
])
commercial_receiver_display = commercial_raw.copy()
raw_commercial_total = float(pd.to_numeric(commercial_receiver_display["Profit_Loss"], errors="coerce").fillna(0.0).sum()) if not commercial_receiver_display.empty else 0.0
if not commercial_receiver_display.empty:
    if abs(raw_commercial_total) > 1e-9:
        scale_factor = commercial_reallocation_value / raw_commercial_total
        commercial_receiver_display["Raw_Profit_Loss"] = commercial_receiver_display["Profit_Loss"]
        commercial_receiver_display["Profit_Loss"] = pd.to_numeric(commercial_receiver_display["Profit_Loss"], errors="coerce").fillna(0.0) * scale_factor
        commercial_receiver_display["Scaling_Factor"] = scale_factor
    else:
        commercial_receiver_display["Raw_Profit_Loss"] = commercial_receiver_display["Profit_Loss"]
        commercial_receiver_display["Profit_Loss"] = 0.0
        commercial_receiver_display["Scaling_Factor"] = np.nan
else:
    commercial_receiver_display = pd.DataFrame(columns=["Receiver_Group", "Profit_Loss", "Raw_Profit_Loss", "Net_Cargo_Change", "Positive_Cargo_Change", "Negative_Cargo_Change", "Routes", "Scaling_Factor"])
commercial_receiver_display = commercial_receiver_display.sort_values("Profit_Loss", key=lambda s: s.abs(), ascending=False)
existing_receiver_display = pd.concat([
    pd.DataFrame([{"Receiver_Group": "Shipping Savings", "Profit_Loss": shipping_savings_value, "Raw_Profit_Loss": shipping_cost_change, "Net_Cargo_Change": np.nan, "Positive_Cargo_Change": np.nan, "Negative_Cargo_Change": np.nan, "Routes": np.nan, "Scaling_Factor": np.nan}]),
    commercial_receiver_display,
], ignore_index=True)

route_monthly_a = prepare_monthly_route(df_a)
route_monthly_b = prepare_monthly_route(df_b)
route_monthly_diff = pd.merge(route_monthly_a, route_monthly_b, on=["Load_Month_Start", "year", "month", "Purchase_LTC_Name", "Sales_LTC_Name"], how="outer", suffixes=("_A", "_B")).fillna(0)
for col in ["freq_A", "freq_B", "profit_loss_A", "profit_loss_B"]:
    route_monthly_diff[col] = pd.to_numeric(route_monthly_diff[col], errors="coerce").fillna(0.0)
route_monthly_diff["freq"] = route_monthly_diff["freq_A"] - route_monthly_diff["freq_B"]
route_monthly_diff["profit_loss"] = route_monthly_diff["profit_loss_A"] - route_monthly_diff["profit_loss_B"]
route_monthly_diff["Is_New_Position_Row"] = route_monthly_diff.apply(lambda r: is_new_position_row(r, new_tokens_set), axis=1)
route_monthly_existing = route_monthly_diff[~route_monthly_diff["Is_New_Position_Row"]].copy()
route_monthly_existing["season"] = route_monthly_existing["month"].apply(season_from_month)
route_monthly_existing["quarter"] = route_monthly_existing["month"].apply(quarter_from_month)
route_monthly_existing["period"] = pd.to_datetime(route_monthly_existing["Load_Month_Start"]).dt.strftime("%Y-%m")
monthly_existing = route_monthly_existing.groupby(["period", "season"], as_index=False).agg(Profit_Loss=("profit_loss", "sum"), Cargo_Change=("freq", "sum"))
seasonal_existing = route_monthly_existing.groupby("season", as_index=False).agg(Profit_Loss=("profit_loss", "sum"), Cargo_Change=("freq", "sum"))
quarter_existing = route_monthly_existing.groupby("quarter", as_index=False).agg(Profit_Loss=("profit_loss", "sum"), Cargo_Change=("freq", "sum")).sort_values("quarter")
year_existing = route_monthly_existing.groupby("year", as_index=False).agg(Profit_Loss=("profit_loss", "sum"), Cargo_Change=("freq", "sum")).sort_values("year")

risk_a = prepare_risk_by_sim_full_file(file_a.name, file_a_bytes, "Portfolio + new position")
risk_b = prepare_risk_by_sim_full_file(file_b.name, file_b_bytes, "Base portfolio")
if not risk_a.empty and not risk_b.empty:
    risk_merged = pd.merge(risk_b, risk_a, on="sim_id", how="inner")
    risk_merged["Change"] = risk_merged["Portfolio + new position"] - risk_merged["Base portfolio"]
    risk_stats = pd.merge(risk_stats_table(risk_merged["Base portfolio"], "Base portfolio"), risk_stats_table(risk_merged["Portfolio + new position"], "Portfolio + new position"), on="Metric", how="outer")
    risk_stats["Change"] = risk_stats["Portfolio + new position"] - risk_stats["Base portfolio"]
else:
    risk_merged = pd.DataFrame()
    risk_stats = pd.DataFrame()

p10_change = _safe_metric_from_risk(risk_stats, "P10", "Change")
p50_change = _safe_metric_from_risk(risk_stats, "P50", "Change")
p90_change = _safe_metric_from_risk(risk_stats, "P90", "Change")
regional_attr = existing_diff.groupby("Sales_Region", as_index=False).agg(Profit_Loss=("profit_loss", "sum"), Cargo_Change=("freq", "sum"), Routes=("Route", "count")).sort_values("Profit_Loss", key=lambda s: s.abs(), ascending=False)
index_attr = existing_diff.groupby("Index_Bucket", as_index=False).agg(Profit_Loss=("profit_loss", "sum"), Cargo_Change=("freq", "sum"), Routes=("Route", "count")).sort_values("Profit_Loss", key=lambda s: s.abs(), ascending=False)
region_index_attr = existing_diff.groupby(["Sales_Region", "Index_Bucket"], as_index=False).agg(Profit_Loss=("profit_loss", "sum"), Cargo_Change=("freq", "sum"), Routes=("Route", "count")).sort_values("Profit_Loss", key=lambda s: s.abs(), ascending=False)
volume_region = volume_by_region(sankey_a, sankey_b)
new_position_volume_rows = sankey_a[sankey_a.apply(lambda r: is_new_position_row(r, new_tokens_set), axis=1)].copy()
if not new_position_volume_rows.empty:
    new_position_volume_rows["Volume_Bucket"] = new_position_volume_rows.apply(classify_volume_bucket, axis=1)
    vol_new = new_position_volume_rows.groupby("Volume_Bucket", as_index=False).agg(Volume=("freq", "sum"))
    vol_new["Source"] = "New position volume"
else:
    vol_new = pd.DataFrame(columns=["Volume_Bucket", "Volume", "Source"])
vol_existing = existing_diff.groupby("Volume_Bucket", as_index=False).agg(Volume=("freq", "sum"))
vol_existing["Source"] = "Existing positions volume change"
volume_decomposition = pd.concat([vol_new, vol_existing], ignore_index=True)

shipping_total = shipping.get("total", pd.DataFrame()).copy()
summary_shipping_lines = []
if not shipping_total.empty:
    for metric in ["Total_Shipping_Cost", "Idle_Days", "Shipping_Days", "Shipping_Distance"]:
        row = shipping_total[shipping_total["Metric"] == metric]
        if not row.empty:
            change = float(row["Change"].iloc[0])
            direction = "decreases" if change < 0 else "increases" if change > 0 else "is unchanged"
            summary_shipping_lines.append(f"- **{metric.replace('_', ' ')}** {direction} by **{fmt_signed(change)}**.")


def _top_text(df, label_col, value_col, n=3, positive=True):
    if df.empty:
        return "No values available."
    tmp = df[[label_col, value_col]].copy()
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce").fillna(0.0)
    tmp = tmp[tmp[value_col] > 0] if positive else tmp[tmp[value_col] < 0]
    if tmp.empty:
        return "No positive contributors." if positive else "No negative contributors."
    tmp = tmp.sort_values(value_col, ascending=not positive).head(n)
    return ", ".join([f"{r[label_col]} ({fmt_signed(r[value_col])})" for _, r in tmp.iterrows()])

season_top = seasonal_existing.sort_values("Profit_Loss", ascending=False).head(1) if not seasonal_existing.empty else pd.DataFrame()
quarter_top = quarter_existing.sort_values("Profit_Loss", ascending=False).head(1) if not quarter_existing.empty else pd.DataFrame()
month_top = monthly_existing.sort_values("Profit_Loss", ascending=False).head(1) if not monthly_existing.empty else pd.DataFrame()
year_top = year_existing.sort_values("Profit_Loss", ascending=False).head(1) if not year_existing.empty else pd.DataFrame()
region_best = regional_attr.sort_values("Profit_Loss", ascending=False).head(1) if not regional_attr.empty else pd.DataFrame()
region_worst = regional_attr.sort_values("Profit_Loss", ascending=True).head(1) if not regional_attr.empty else pd.DataFrame()
index_best = index_attr.sort_values("Profit_Loss", ascending=False).head(1) if not index_attr.empty else pd.DataFrame()
index_worst = index_attr.sort_values("Profit_Loss", ascending=True).head(1) if not index_attr.empty else pd.DataFrame()

# Map data
value_map_df = build_destination_map_data(sankey_a, "profit_loss", "freq")
optimised_flow_map_df = build_flow_map_data(sankey_a, include_category=True)
difference_flow_map_df = build_flow_map_data(diff, include_category=False)

main_driver = pd.DataFrame([
    {"Level": "Main", "Parent": "Total Change", "Driver": "New Position Value", "Value": new_pl_total, "Cargo_Change": new_cargo_total, "Note": "Standalone contribution from nodes present only in A."},
    {"Level": "Main", "Parent": "Total Change", "Driver": "Existing Portfolio Reallocation", "Value": existing_pl, "Cargo_Change": existing_cargo_delta, "Note": "Residual after new position value."},
])
level1_driver = existing_reallocation_level1.copy()
level1_driver["Parent"] = "Existing Portfolio Reallocation"
level1_driver["Cargo_Change"] = np.nan
commercial_sub_driver = commercial_receiver_display.rename(columns={"Receiver_Group": "Driver", "Profit_Loss": "Value", "Net_Cargo_Change": "Cargo_Change"})
commercial_sub_driver.insert(0, "Level", "Commercial sub-bucket")
commercial_sub_driver.insert(1, "Parent", "Commercial Reallocation")
commercial_sub_driver["Note"] = "Commercial sub-bucket scaled to reconcile to Commercial Reallocation."
driver_hierarchy = pd.concat([main_driver, level1_driver[["Level", "Parent", "Driver", "Value", "Cargo_Change", "Note"]], commercial_sub_driver[["Level", "Parent", "Driver", "Value", "Cargo_Change", "Note"]]], ignore_index=True)

# ======================================================
# TABS
# ======================================================
st_tabs = st.tabs(["Summary", "Executive Summary", "Map", "Sankey comparison", "Receiver Attribution", "Driver Diagnostics", "Seasonal & Monthly", "Index & Region", "Risk", "Volume", "Shipping", "Data Export"])

with st_tabs[0]:
    st.markdown("## Summary")
    st.markdown("<div class='summary-box'>", unsafe_allow_html=True)
    st.markdown(f"""
### Portfolio impact
The total portfolio change is **{fmt_signed(total_pl)}**. Of this, **{fmt_signed(new_pl_total)}** comes from the new position value and **{fmt_signed(existing_pl)}** comes from Existing Portfolio Reallocation.

### Driver diagnostics
Existing Portfolio Reallocation contributes **{fmt_signed(existing_pl)}**. This is split into **Shipping Savings of {fmt_signed(shipping_savings_value)}** and **Commercial Reallocation of {fmt_signed(commercial_reallocation_value)}**.

The largest positive commercial contributors are: **{_top_text(commercial_receiver_display, 'Receiver_Group', 'Profit_Loss', 3, True)}**. The largest negative commercial contributors are: **{_top_text(commercial_receiver_display, 'Receiver_Group', 'Profit_Loss', 3, False)}**.

### Shipping
{chr(10).join(summary_shipping_lines) if summary_shipping_lines else 'Shipping diagnostics are not available because the required shipping columns are missing.'}

### Risk
P10 changes by **{fmt_signed(p10_change)}**, P50 changes by **{fmt_signed(p50_change)}**, and P90 changes by **{fmt_signed(p90_change)}**.
""")
    if not season_top.empty:
        st.markdown(f"### Seasonality\nThe strongest season is **{season_top.iloc[0]['season']}** with **{fmt_signed(season_top.iloc[0]['Profit_Loss'])}**.")
    if not quarter_top.empty:
        st.markdown(f"The strongest quarter is **{quarter_top.iloc[0]['quarter']}** with **{fmt_signed(quarter_top.iloc[0]['Profit_Loss'])}**.")
    if not month_top.empty:
        st.markdown(f"The strongest month is **{month_top.iloc[0]['period']}** with **{fmt_signed(month_top.iloc[0]['Profit_Loss'])}**.")
    if not year_top.empty:
        st.markdown(f"The strongest year is **{int(year_top.iloc[0]['year'])}** with **{fmt_signed(year_top.iloc[0]['Profit_Loss'])}**.")
    st.markdown("### Index and region")
    if not region_best.empty:
        st.markdown(f"The largest positive regional contribution comes from **{region_best.iloc[0]['Sales_Region']}** with **{fmt_signed(region_best.iloc[0]['Profit_Loss'])}**.")
    if not region_worst.empty:
        st.markdown(f"The largest negative regional contribution comes from **{region_worst.iloc[0]['Sales_Region']}** with **{fmt_signed(region_worst.iloc[0]['Profit_Loss'])}**.")
    if not index_best.empty:
        st.markdown(f"The strongest index bucket is **{index_best.iloc[0]['Index_Bucket']}** with **{fmt_signed(index_best.iloc[0]['Profit_Loss'])}**.")
    if not index_worst.empty:
        st.markdown(f"The weakest index bucket is **{index_worst.iloc[0]['Index_Bucket']}** with **{fmt_signed(index_worst.iloc[0]['Profit_Loss'])}**.")
    st.markdown("### Overall implications")
    if total_pl > 0:
        st.markdown("Overall, adding the new deal is value-accretive for the portfolio. The conclusion is positive, especially if the value is supported by shipping savings, commercial reallocation and stable risk metrics.")
    elif total_pl < 0:
        st.markdown("Overall, adding the new deal is value-negative for the portfolio. The deal would need other strategic or commercial benefits to offset the value loss shown here.")
    else:
        st.markdown("Overall, adding the new deal is broadly neutral on portfolio value based on the selected data.")
    st.markdown("</div>", unsafe_allow_html=True)

with st_tabs[1]:
    st.markdown("## Executive Summary")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        metric_card("Total portfolio change", fmt_signed(total_pl), "A minus B, validated diff")
    with k2:
        metric_card("New position value", fmt_signed(new_pl_total), "Nodes present only in A")
    with k3:
        metric_card("Existing portfolio reallocation", fmt_signed(existing_pl), "Residual portfolio effect")
    with k4:
        metric_card("Existing nodes $/MMBtu", f"{existing_rate:,.6f}", "Existing P/L / (3.8m x delta cargoes)")
    bridge_x = ["New position", "Existing positions", "Total change"]
    bridge_y = [new_pl_total, existing_pl, total_pl]
    bridge_base = [0, new_pl_total, 0]
    bridge_colours = [BLUE, GREEN if existing_pl >= 0 else RED, GREY]
    fig = go.Figure(go.Bar(x=bridge_x, y=bridge_y, base=bridge_base, marker_color=bridge_colours, text=[fmt_signed(new_pl_total), fmt_signed(existing_pl), fmt_signed(total_pl)], textposition="inside", name="Portfolio bridge"))
    apply_dark_layout(fig, height=430, title="Portfolio change bridge")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True, key=next_chart_key("exec_bridge"))

with st_tabs[2]:
    st.markdown("## Map")
    st.caption("Portfolio Value Map is based on the Portfolio + New Position file. Optimized Cargo Flow Map is based on all routes in the Portfolio + New Position file. Difference Cargo Flow Map is based on route differences between cases, like the Sankey difference chart.")
    map_tabs = st.tabs(["Portfolio Value Map", "Optimized Cargo Flow Map", "Difference Cargo Flow Map"])
    with map_tabs[0]:
        map_bubble(value_map_df, "Profit_Loss", "Portfolio value by destination - Portfolio + new positions")
    with map_tabs[1]:
        c_map, c_legend = st.columns([3, 1])
        with c_map:
            optimised_flow_map(optimised_flow_map_df, "Optimized cargo flow map - Portfolio + new positions")
        with c_legend:
            flow_legend_table(optimised_flow_map_df)
    with map_tabs[2]:
        difference_flow_map(difference_flow_map_df, "Difference cargo flow map - A minus B")

with st_tabs[3]:
    st.markdown("## Sankey comparison")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(render_sankey(filter_out_none_empty_sales(sankey_a), "Portfolio + new positions"), use_container_width=True, key="sankey_A")
    with c2:
        st.plotly_chart(render_sankey(filter_out_none_empty_sales(sankey_b), "Portfolio"), use_container_width=True, key="sankey_B")
    with c3:
        st.plotly_chart(render_sankey(filter_out_none_empty_sales(diff), "Difference", diff_mode=True), use_container_width=True, key="sankey_DIFF")

with st_tabs[4]:
    st.markdown("## Receiver Attribution")
    st.caption("Spot market sales in Asia are classified as Spot Optimisation Asia. Contracted Asian sales remain Existing Asia Sales.")
    c1, c2 = st.columns([2, 1])
    with c1:
        dark_bar(receiver_attr, "Receiver_Group", "Profit_Loss", "Receiver attribution by P/L", horizontal=True)
    with c2:
        plot_signed_donut(receiver_attr, "Receiver_Group", "Profit_Loss", "Receiver contribution split")

with st_tabs[5]:
    st.markdown("## Driver Diagnostics")
    st.caption("Existing reallocation is split into Shipping Savings and Commercial Reallocation. Commercial sub-buckets exclude Shipping / Idle / Empty and are scaled to reconcile to Commercial Reallocation.")
    dark_bar(driver_hierarchy[driver_hierarchy["Level"] == "Main"], "Driver", "Value", "Main value buckets", horizontal=True)
    st.markdown("### Existing Portfolio Reallocation: top-level split")
    c1, c2 = st.columns([2, 1])
    with c1:
        dark_bar(existing_reallocation_level1, "Driver", "Value", "Shipping savings vs commercial reallocation", horizontal=True)
    with c2:
        plot_signed_donut(existing_reallocation_level1, "Driver", "Value", "Top-level split")
    st.markdown("### Commercial Reallocation sub-buckets")
    c3, c4 = st.columns([2, 1])
    with c3:
        dark_bar(commercial_receiver_display, "Receiver_Group", "Profit_Loss", "Commercial Reallocation sub-buckets", horizontal=True)
    with c4:
        plot_signed_donut(commercial_receiver_display, "Receiver_Group", "Profit_Loss", "Commercial contribution split")
    st.markdown("### Additional diagnostic drivers")
    k1, k2, k3 = st.columns(3)
    with k1:
        kpi_box("Delta P10", p10_change, "Change in downside value")
    with k2:
        kpi_box("Delta P50", p50_change, "Change in median value")
    with k3:
        kpi_box("Delta P90", p90_change, "Change in upside value")

with st_tabs[6]:
    st.markdown("## Seasonal & Monthly")
    c1, c2 = st.columns(2)
    with c1:
        dark_bar(seasonal_existing, "season", "Profit_Loss", "Existing reallocation by season")
    with c2:
        dark_bar(quarter_existing, "quarter", "Profit_Loss", "Existing reallocation by quarter")
    dark_bar(year_existing, "year", "Profit_Loss", "Existing reallocation by year")
    if not monthly_existing.empty:
        fig = px.line(monthly_existing, x="period", y="Profit_Loss", markers=True, title="Monthly existing portfolio reallocation", color_discrete_sequence=[EQUINOR_RED])
        apply_dark_layout(fig, height=500)
        st.plotly_chart(fig, use_container_width=True, key=next_chart_key("monthly_trend"))

with st_tabs[7]:
    st.markdown("## Index & Region")
    c1, c2 = st.columns(2)
    with c1:
        dark_bar(regional_attr, "Sales_Region", "Profit_Loss", "Regional P/L change, existing rows only")
    with c2:
        dark_bar(index_attr, "Index_Bucket", "Profit_Loss", "Index bucket P/L change, existing rows only")
    region_index_sankey(region_index_attr)
    c3, c4 = st.columns(2)
    with c3:
        plot_signed_donut(regional_attr, "Sales_Region", "Profit_Loss", "Regional contribution split")
    with c4:
        plot_signed_donut(index_attr, "Index_Bucket", "Profit_Loss", "Index contribution split")

with st_tabs[8]:
    st.markdown("## Risk")
    if risk_stats.empty:
        st.info("Could not calculate risk statistics because sim_id was not found in the uploaded files.")
    else:
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            kpi_box("Delta P10", p10_change)
        with k2:
            kpi_box("Delta P50", p50_change)
        with k3:
            kpi_box("Delta P90", p90_change)
        with k4:
            value_box("Base P50", _safe_metric_from_risk(risk_stats, "P50", "Base portfolio"), "Before new position")
        with k5:
            value_box("New P50", _safe_metric_from_risk(risk_stats, "P50", "Portfolio + new position"), "Portfolio + new position")
        base_sorted = np.sort(risk_merged["Base portfolio"].values)
        with_sorted = np.sort(risk_merged["Portfolio + new position"].values)
        p_base = np.arange(1, len(base_sorted) + 1) / len(base_sorted)
        p_with = np.arange(1, len(with_sorted) + 1) / len(with_sorted)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=p_base, y=base_sorted, mode="lines", name="Base portfolio", line=dict(color=WHITE, width=3)))
        fig.add_trace(go.Scatter(x=p_with, y=with_sorted, mode="lines", name="Portfolio + new position", line=dict(color=EQUINOR_RED, width=3)))
        for _, q in [("P10", 0.10), ("P50", 0.50), ("P90", 0.90)]:
            fig.add_vline(x=q, line_dash="dot", line_color="rgba(255,255,255,0.35)")
        apply_dark_layout(fig, height=560, title="Portfolio value S-Curve by simulation")
        fig.update_xaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True, key=next_chart_key("risk"))

with st_tabs[9]:
    st.markdown("## Volume")
    if volume_decomposition.empty:
        st.info("No volume data to display.")
    else:
        fig = px.bar(volume_decomposition, x="Volume_Bucket", y="Volume", color="Source", barmode="group", title="New position volume vs existing-position volume change", color_discrete_sequence=EQUINOR_BAR_PALETTE)
        apply_dark_layout(fig, height=500)
        st.plotly_chart(fig, use_container_width=True, key=next_chart_key("volume_decomposition"))
        plot_signed_donut(volume_decomposition, "Volume_Bucket", "Volume", "Volume contribution split")

with st_tabs[10]:
    st.markdown("## Shipping")
    if not shipping["missing"].empty:
        st.info("Shipping diagnostics require these columns in both files: " + ", ".join(SHIPPING_COLS) + ". Missing: " + ", ".join(shipping["missing"]["Missing_Column"].tolist()))
    else:
        shipping_total = shipping["total"].copy()
        s_cols = st.columns(4)
        for idx, metric in enumerate(SHIPPING_COLS):
            row = shipping_total[shipping_total["Metric"] == metric]
            before = float(row["Original Portfolio"].iloc[0])
            after = float(row["With New Position"].iloc[0])
            change = float(row["Change"].iloc[0])
            colour = GREEN if change < 0 else RED if change > 0 else "#9aa6b2"
            with s_cols[idx % 4]:
                metric_card_coloured(metric.replace("_", " "), f"{DELTA} {change:,.0f}", f"Original: {before:,.0f}<br>With new position: {after:,.0f}", colour=colour)
        selected_metric = st.selectbox("Select shipping metric to inspect", SHIPPING_COLS, key="shipping_metric_select")
        metric_row = shipping_total[shipping_total["Metric"] == selected_metric].copy()
        long_metric = metric_row.melt(id_vars="Metric", value_vars=["Original Portfolio", "With New Position"], var_name="Case", value_name="Value")
        fig = px.bar(long_metric, x="Case", y="Value", color="Case", title=f"{selected_metric.replace('_',' ')}: original vs with new position", color_discrete_sequence=EQUINOR_BAR_PALETTE)
        apply_dark_layout(fig, height=430)
        st.plotly_chart(fig, use_container_width=True, key=next_chart_key("shipping_selected_metric"))
        dark_bar(shipping_total, "Metric", "% Change", "Relative change by shipping metric")

with st_tabs[11]:
    st.markdown("## Data Export")
    excel_bytes = make_excel_download({
        "Executive Summary": pd.DataFrame([
            {"Metric": "Total portfolio change", "Value": total_pl},
            {"Metric": "New position value", "Value": new_pl_total},
            {"Metric": "Existing portfolio reallocation", "Value": existing_pl},
            {"Metric": "Shipping savings", "Value": shipping_savings_value},
            {"Metric": "Commercial reallocation", "Value": commercial_reallocation_value},
        ]),
        "Receiver Attribution": receiver_attr,
        "Existing Sub Buckets Raw": existing_receiver,
        "Existing Top Level Visual": existing_reallocation_level1,
        "Commercial Buckets Visual": commercial_receiver_display,
        "Existing Combined Visual": existing_receiver_display,
        "Driver Hierarchy": driver_hierarchy,
        "Regional Attribution Existing": regional_attr,
        "Index Attribution Existing": index_attr,
        "Risk Stats": risk_stats,
        "Risk By Sim": risk_merged,
        "Volume Destination": volume_region,
        "Volume Decomposition": volume_decomposition,
        "Shipping Total ExistingOnly": shipping.get("total", pd.DataFrame()),
        "Shipping Region ExistingOnly": shipping.get("region", pd.DataFrame()),
        "Map Value": value_map_df,
        "Optimized Map Flows": optimised_flow_map_df,
        "Difference Map Flows": difference_flow_map_df,
        "Seasonal Existing": seasonal_existing,
        "Quarter Existing": quarter_existing,
        "Year Existing": year_existing,
        "Monthly Existing": monthly_existing,
        "Diff Enhanced": diff_e,
        "Sankey A": sankey_a,
        "Sankey B": sankey_b,
        "Diff Raw": diff,
    })
    st.download_button("Download analysis pack as Excel", data=excel_bytes, file_name="portfolio_comparison_analysis.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="download_analysis_pack")

