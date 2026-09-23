"""
================================================================================
RETAIL INTELLIGENCE PLATFORM
Sales & Customer Analytics — End-to-End Business Intelligence Dashboard
================================================================================
A single-file Streamlit application that takes a raw, messy retail transactions
dataset through validation, cleaning, feature engineering, multi-layer
analytics (KPIs -> Trends -> Drivers -> Risk/Opportunity -> Action) and
presents it as an interactive, futuristic business-intelligence product.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Author: Data Analytics Internship Final Project
Dataset: retail_sales_dataset.csv (order-level retail transactions, India)
================================================================================
"""

import re
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Retail Intelligence | Sales & Customer Analytics",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = "retail_sales_dataset.csv"

# ==============================================================================
# 2. DESIGN SYSTEM — COLOR TOKENS
# ==============================================================================
COLORS = {
    "bg_primary": "#05070f",
    "bg_secondary": "#0a0e24",
    "bg_card": "rgba(16, 20, 46, 0.72)",
    "bg_card_solid": "#10142e",
    "border": "rgba(120, 140, 255, 0.18)",
    "cyan": "#22d3ee",
    "purple": "#a855f7",
    "pink": "#ec4899",
    "blue": "#3b82f6",
    "gold": "#fbbf24",
    "green": "#34d399",
    "red": "#f87171",
    "orange": "#fb923c",
    "text_primary": "#eef1fb",
    "text_secondary": "#8b93c4",
    "text_muted": "#5b6291",
}

CHART_COLORWAY = [COLORS["cyan"], COLORS["purple"], COLORS["pink"], COLORS["gold"],
                  COLORS["green"], COLORS["blue"], COLORS["orange"], "#f472b6"]

PLOTLY_TEMPLATE = {
    "layout": go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Exo 2, sans-serif", color=COLORS["text_primary"], size=13),
        colorway=CHART_COLORWAY,
        xaxis=dict(gridcolor="rgba(139,147,196,0.12)", zerolinecolor="rgba(139,147,196,0.15)",
                    linecolor="rgba(139,147,196,0.2)", tickfont=dict(color=COLORS["text_secondary"])),
        yaxis=dict(gridcolor="rgba(139,147,196,0.12)", zerolinecolor="rgba(139,147,196,0.15)",
                    linecolor="rgba(139,147,196,0.2)", tickfont=dict(color=COLORS["text_secondary"])),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=COLORS["text_secondary"])),
        margin=dict(l=10, r=10, t=50, b=10),
        hoverlabel=dict(bgcolor="#151a3a", font=dict(family="Space Mono, monospace", color="#eef1fb"),
                         bordercolor="rgba(120,140,255,0.35)"),
    )
}

# ==============================================================================
# 3. GLOBAL CSS — FUTURISTIC THEME (Orbitron / Exo 2 / Space Mono)
# ==============================================================================
def inject_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;800;900&family=Exo+2:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=Space+Mono:wght@400;700&display=swap');

    :root {{
        --bg-primary: {COLORS['bg_primary']};
        --bg-secondary: {COLORS['bg_secondary']};
        --bg-card: {COLORS['bg_card']};
        --border: {COLORS['border']};
        --cyan: {COLORS['cyan']};
        --purple: {COLORS['purple']};
        --pink: {COLORS['pink']};
        --gold: {COLORS['gold']};
        --green: {COLORS['green']};
        --red: {COLORS['red']};
        --text-primary: {COLORS['text_primary']};
        --text-secondary: {COLORS['text_secondary']};
    }}

    html, body, [class*="css"] {{
        font-family: 'Exo 2', sans-serif;
    }}

    /* ---------------- APP BACKGROUND ---------------- */
    .stApp {{
        background:
            radial-gradient(ellipse 900px 600px at 12% -10%, rgba(168,85,247,0.20), transparent 60%),
            radial-gradient(ellipse 1000px 700px at 110% 10%, rgba(34,211,238,0.14), transparent 55%),
            radial-gradient(ellipse 800px 500px at 50% 120%, rgba(236,72,153,0.12), transparent 55%),
            linear-gradient(180deg, #05070f 0%, #060a1c 45%, #05070f 100%);
        background-attachment: fixed;
    }}

    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header[data-testid="stHeader"] {{background: rgba(0,0,0,0); height: 0;}}
    .block-container {{padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1500px;}}

    /* ---------------- SCROLLBAR ---------------- */
    ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
    ::-webkit-scrollbar-track {{ background: #05070f; }}
    ::-webkit-scrollbar-thumb {{ background: linear-gradient(180deg, var(--cyan), var(--purple)); border-radius: 10px; }}

    /* ---------------- TYPOGRAPHY ---------------- */
    h1, h2, h3 {{
        font-family: 'Orbitron', sans-serif !important;
        letter-spacing: 0.02em;
        color: var(--text-primary) !important;
    }}
    .stMarkdown p, .stMarkdown li, label, .stSelectbox, .stMultiSelect {{
        font-family: 'Exo 2', sans-serif;
        color: var(--text-secondary);
    }}
    code, .mono {{ font-family: 'Space Mono', monospace !important; }}

    /* ---------------- SIDEBAR ---------------- */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #070a1c 0%, #0a0e26 100%);
        border-right: 1px solid var(--border);
    }}
    section[data-testid="stSidebar"] .block-container {{ padding-top: 1.6rem; }}

    .brand-wrap {{
        padding: 0 0.2rem 1.1rem 0.2rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid var(--border);
    }}
    .brand-title {{
        font-family: 'Orbitron', sans-serif;
        font-weight: 800;
        font-size: 1.28rem;
        line-height: 1.15;
        background: linear-gradient(90deg, var(--cyan), var(--purple) 60%, var(--pink));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 26px rgba(168,85,247,0.35);
        margin: 0;
    }}
    .brand-sub {{
        font-family: 'Space Mono', monospace;
        font-size: 0.62rem;
        letter-spacing: 0.18em;
        color: var(--text-muted);
        margin-top: 0.35rem;
        text-transform: uppercase;
    }}

    /* Sidebar radio -> nav list */
    section[data-testid="stSidebar"] div[role="radiogroup"] {{ gap: 0.28rem; }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label {{
        background: rgba(255,255,255,0.02);
        border: 1px solid transparent;
        border-radius: 10px;
        padding: 0.55rem 0.8rem !important;
        transition: all 0.2s ease;
        width: 100%;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
        background: rgba(34,211,238,0.08);
        border-color: rgba(34,211,238,0.3);
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {{
        display: none;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] p {{
        font-family: 'Exo 2', sans-serif !important;
        font-weight: 600;
        font-size: 0.92rem;
        color: var(--text-secondary);
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) {{
        background: linear-gradient(90deg, rgba(34,211,238,0.16), rgba(168,85,247,0.16));
        border: 1px solid rgba(34,211,238,0.45);
        box-shadow: 0 0 18px rgba(34,211,238,0.12), inset 0 0 12px rgba(168,85,247,0.08);
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) p {{
        color: #eafeff;
        text-shadow: 0 0 12px rgba(34,211,238,0.5);
    }}

    /* ---------------- HERO HEADER ---------------- */
    .hero-eyebrow {{
        font-family: 'Space Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.28em;
        color: var(--cyan);
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }}
    .hero-title {{
        font-family: 'Orbitron', sans-serif;
        font-weight: 900;
        font-size: 2.5rem;
        line-height: 1.05;
        background: linear-gradient(100deg, #ffffff 10%, var(--cyan) 45%, var(--purple) 75%, var(--pink) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 0.4rem 0;
        filter: drop-shadow(0 0 22px rgba(34,211,238,0.18));
    }}
    .hero-desc {{
        color: var(--text-secondary);
        font-size: 0.98rem;
        max-width: 780px;
    }}

    /* ---------------- CARD (glass + glow border) ---------------- */
    .glass-card {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1.3rem 1.4rem;
        backdrop-filter: blur(14px);
        position: relative;
        box-shadow: 0 8px 32px rgba(0,0,0,0.35);
        margin-bottom: 1.1rem;
    }}
    .glass-card::before {{
        content: "";
        position: absolute; inset: 0;
        border-radius: 18px;
        padding: 1px;
        background: linear-gradient(135deg, rgba(34,211,238,0.35), rgba(168,85,247,0.1) 40%, rgba(236,72,153,0.25));
        -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        -webkit-mask-composite: xor;
        mask-composite: exclude;
        pointer-events: none;
    }}
    .card-title {{
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        font-size: 0.98rem;
        color: var(--text-primary);
        margin-bottom: 0.1rem;
        display: flex; align-items: center; gap: 0.5rem;
    }}
    .card-sub {{
        font-family: 'Exo 2', sans-serif;
        font-size: 0.78rem;
        color: var(--text-muted);
        margin-bottom: 0.9rem;
    }}

    /* ---------------- KPI CARD ---------------- */
    .kpi-card {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.05rem 1.2rem 0.95rem 1.2rem;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(14px);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        height: 100%;
        box-shadow: 0 6px 24px rgba(0,0,0,0.3);
    }}
    .kpi-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 12px 34px rgba(34,211,238,0.14);
    }}
    .kpi-card::after {{
        content: "";
        position: absolute; top: -40%; right: -25%;
        width: 140px; height: 140px; border-radius: 50%;
        background: radial-gradient(circle, var(--glow-color, rgba(34,211,238,0.35)), transparent 70%);
        filter: blur(6px);
        pointer-events: none;
    }}
    .kpi-icon {{
        width: 34px; height: 34px; border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.05rem;
        background: var(--icon-bg, rgba(34,211,238,0.14));
        border: 1px solid var(--icon-border, rgba(34,211,238,0.35));
        margin-bottom: 0.65rem;
    }}
    .kpi-label {{
        font-family: 'Space Mono', monospace;
        font-size: 0.66rem;
        letter-spacing: 0.13em;
        color: var(--text-muted);
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }}
    .kpi-value {{
        font-family: 'Orbitron', sans-serif;
        font-weight: 800;
        font-size: 1.55rem;
        color: var(--text-primary);
        line-height: 1.15;
    }}
    .kpi-delta {{
        font-family: 'Space Mono', monospace;
        font-size: 0.72rem;
        margin-top: 0.45rem;
        display: inline-block;
        padding: 0.14rem 0.5rem;
        border-radius: 20px;
    }}
    .kpi-delta.up {{ color: var(--green); background: rgba(52,211,153,0.12); }}
    .kpi-delta.down {{ color: var(--red); background: rgba(248,113,113,0.12); }}
    .kpi-delta.flat {{ color: var(--text-muted); background: rgba(139,147,196,0.1); }}

    /* ---------------- INSIGHT / ACTION CARDS ---------------- */
    .insight-card {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-left: 3px solid var(--accent, var(--cyan));
        border-radius: 12px;
        padding: 0.95rem 1.1rem;
        margin-bottom: 0.7rem;
        backdrop-filter: blur(10px);
    }}
    .insight-tag {{
        font-family: 'Space Mono', monospace;
        font-size: 0.62rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--accent, var(--cyan));
        margin-bottom: 0.25rem;
        display: block;
    }}
    .insight-title {{
        font-family: 'Exo 2', sans-serif;
        font-weight: 700;
        font-size: 0.95rem;
        color: var(--text-primary);
        margin-bottom: 0.2rem;
    }}
    .insight-body {{
        font-size: 0.85rem;
        color: var(--text-secondary);
        line-height: 1.5;
    }}

    /* ---------------- BADGES / PILLS ---------------- */
    .pill {{
        display: inline-block;
        font-family: 'Space Mono', monospace;
        font-size: 0.68rem;
        padding: 0.22rem 0.65rem;
        border-radius: 20px;
        border: 1px solid var(--border);
        color: var(--text-secondary);
        background: rgba(255,255,255,0.02);
        margin-right: 0.35rem;
    }}
    .pill.cyan {{ color: var(--cyan); border-color: rgba(34,211,238,0.4); background: rgba(34,211,238,0.08); }}
    .pill.purple {{ color: var(--purple); border-color: rgba(168,85,247,0.4); background: rgba(168,85,247,0.08); }}
    .pill.green {{ color: var(--green); border-color: rgba(52,211,153,0.4); background: rgba(52,211,153,0.08); }}
    .pill.red {{ color: var(--red); border-color: rgba(248,113,113,0.4); background: rgba(248,113,113,0.08); }}
    .pill.gold {{ color: var(--gold); border-color: rgba(251,191,36,0.4); background: rgba(251,191,36,0.08); }}

    /* ---------------- SECTION HEADER ---------------- */
    .section-head {{
        display: flex; align-items: baseline; gap: 0.6rem;
        margin: 1.6rem 0 0.9rem 0;
    }}
    .section-head h3 {{ margin: 0 !important; font-size: 1.15rem !important; }}
    .section-head .line {{
        flex: 1; height: 1px;
        background: linear-gradient(90deg, rgba(34,211,238,0.35), transparent);
    }}

    /* ---------------- STREAMLIT WIDGET OVERRIDES ---------------- */
    div[data-testid="stMetric"] {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 0.8rem 1rem;
    }}
    .stSelectbox div[data-baseweb="select"] > div, .stMultiSelect div[data-baseweb="select"] > div {{
        background: rgba(16,20,46,0.8) !important;
        border-color: var(--border) !important;
        border-radius: 10px !important;
    }}
    .stTextInput input, .stDateInput input {{
        background: rgba(16,20,46,0.8) !important;
        color: var(--text-primary) !important;
        border-color: var(--border) !important;
        border-radius: 10px !important;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        background: rgba(255,255,255,0.02);
        padding: 5px;
        border-radius: 12px;
        border: 1px solid var(--border);
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px;
        font-family: 'Exo 2', sans-serif;
        font-weight: 600;
        color: var(--text-secondary);
        padding: 0.5rem 1rem;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(90deg, rgba(34,211,238,0.18), rgba(168,85,247,0.18)) !important;
        color: #eafeff !important;
    }}
    .stButton button {{
        font-family: 'Exo 2', sans-serif;
        font-weight: 600;
        background: linear-gradient(90deg, rgba(34,211,238,0.16), rgba(168,85,247,0.16));
        border: 1px solid rgba(34,211,238,0.4);
        color: var(--text-primary);
        border-radius: 10px;
        transition: all 0.2s ease;
    }}
    .stButton button:hover {{
        border-color: var(--cyan);
        box-shadow: 0 0 16px rgba(34,211,238,0.25);
        transform: translateY(-1px);
    }}
    div[data-testid="stDataFrame"] {{
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }}
    div[data-testid="stExpander"] {{
        background: var(--bg-card);
        border: 1px solid var(--border) !important;
        border-radius: 14px !important;
    }}
    hr {{ border-color: var(--border); }}

    /* progress-style bar for quality metrics */
    .qbar-track {{
        width: 100%; height: 8px; border-radius: 6px;
        background: rgba(255,255,255,0.06);
        overflow: hidden; margin-top: 0.4rem;
    }}
    .qbar-fill {{ height: 100%; border-radius: 6px; }}

    /* fade-in animation for main content */
    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    .block-container > div {{ animation: fadeInUp 0.5s ease; }}

    </style>
    """, unsafe_allow_html=True)


# ==============================================================================
# 4. DATA LOADING
# ==============================================================================
@st.cache_data(show_spinner=False)
def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


# ==============================================================================
# 5. DATA QUALITY / VALIDATION REPORT  (computed on RAW data, before cleaning)
# ==============================================================================
@st.cache_data(show_spinner=False)
def compute_quality_report(raw: pd.DataFrame) -> dict:
    n_rows, n_cols = raw.shape
    missing_by_col = raw.isnull().sum()
    total_missing_cells = int(missing_by_col.sum())
    duplicate_rows = int(raw.duplicated().sum())
    duplicate_order_ids = int(raw["order_id"].duplicated().sum()) if "order_id" in raw.columns else 0

    # invalid-value detection (sentinel errors / out-of-range values), matches clean_data() logic
    invalid_mask = pd.Series(False, index=raw.index)
    invalid_counts = {}

    if "age" in raw.columns:
        m = raw["age"].notna() & ((raw["age"] < 5) | (raw["age"] > 100))
        invalid_counts["Age out of range (<5 or >100)"] = int(m.sum())
        invalid_mask |= m
    if "quantity" in raw.columns:
        m = raw["quantity"].notna() & ((raw["quantity"] <= 0) | (raw["quantity"] > 100))
        invalid_counts["Quantity invalid (<=0 or >100)"] = int(m.sum())
        invalid_mask |= m
    if "shipping_cost" in raw.columns:
        m = raw["shipping_cost"].notna() & (raw["shipping_cost"] < 0)
        invalid_counts["Negative shipping cost"] = int(m.sum())
        invalid_mask |= m
    if "days_to_ship" in raw.columns:
        m = raw["days_to_ship"].notna() & (raw["days_to_ship"] < 0)
        invalid_counts["Negative days-to-ship"] = int(m.sum())
        invalid_mask |= m
    if "discount_pct" in raw.columns:
        m = raw["discount_pct"].notna() & ((raw["discount_pct"] < 0) | (raw["discount_pct"] > 1))
        invalid_counts["Discount % out of [0,1] range"] = int(m.sum())
        invalid_mask |= m
    if "gender" in raw.columns:
        m = raw["gender"].notna() & (~raw["gender"].str.strip().str.lower().isin(
            ["male", "female", "other", "m", "f"]))
        invalid_counts["Unrecognized gender label"] = int(m.sum())
    if "customer_satisfaction" in raw.columns:
        m = raw["customer_satisfaction"].notna() & (
            (raw["customer_satisfaction"] < 1) | (raw["customer_satisfaction"] > 5))
        invalid_counts["Satisfaction score out of 1-5 range"] = int(m.sum())
        invalid_mask |= m

    total_invalid_cells = int(sum(invalid_counts.values()))
    rows_flagged_invalid = int(invalid_mask.sum())
    clean_record_estimate = n_rows - duplicate_rows - rows_flagged_invalid

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "missing_by_col": missing_by_col[missing_by_col > 0].sort_values(ascending=False),
        "total_missing_cells": total_missing_cells,
        "duplicate_rows": duplicate_rows,
        "duplicate_order_ids": duplicate_order_ids,
        "invalid_counts": invalid_counts,
        "total_invalid_cells": total_invalid_cells,
        "rows_flagged_invalid": rows_flagged_invalid,
        "clean_record_estimate": max(clean_record_estimate, 0),
        "completeness_pct": round(100 * (1 - total_missing_cells / (n_rows * n_cols)), 2),
    }


# ==============================================================================
# 6. DATA CLEANING PIPELINE
# ==============================================================================
def _parse_mixed_dates(series: pd.Series) -> pd.Series:
    """The raw file mixes three date formats: ISO (YYYY-MM-DD), long-form
    (Month DD, YYYY) and day-first slashes (DD/MM/YYYY). Try each in turn."""
    formats = ["%Y-%m-%d", "%B %d, %Y", "%d/%m/%Y"]
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    remaining = series.notna()
    for fmt in formats:
        if not remaining.any():
            break
        candidate = pd.to_datetime(series[remaining], format=fmt, errors="coerce")
        newly_parsed = candidate.notna()
        idx = series[remaining].index[newly_parsed]
        result.loc[idx] = candidate[newly_parsed].values
        remaining.loc[idx] = False
    return result


def _normalize_gender(series: pd.Series) -> pd.Series:
    mapping = {"m": "Male", "male": "Male", "f": "Female", "female": "Female", "other": "Other"}
    return series.str.strip().str.lower().map(mapping).fillna(series)


@st.cache_data(show_spinner=False)
def clean_data(raw: pd.DataFrame):
    df = raw.copy()
    log = []
    start_rows = len(df)

    # 1) Drop exact duplicate rows (same order captured twice)
    dupes = int(df.duplicated().sum())
    df = df.drop_duplicates()
    log.append(f"Removed {dupes} exact duplicate rows.")

    # 2) Drop rows with no order_id (cannot be uniquely identified / audited)
    missing_id = int(df["order_id"].isna().sum())
    df = df[df["order_id"].notna()]
    log.append(f"Removed {missing_id} rows with a missing order_id.")

    # 3) Parse the three mixed date formats into a single datetime column
    df["order_date"] = _parse_mixed_dates(df["order_date"].astype("string"))
    unparsed_dates = int(df["order_date"].isna().sum())
    log.append(f"Parsed order_date across 3 mixed formats (ISO / long-form / day-first); "
               f"{unparsed_dates} remain missing and are excluded from time-based charts.")

    # 4) Normalize text fields (whitespace + casing)
    for col in ["region", "city", "product_category", "product_name", "payment_method", "customer_name"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
    df["order_status"] = df["order_status"].astype("string").str.strip().str.title()
    df["gender"] = _normalize_gender(df["gender"].astype("string"))

    # 5) Fix sentinel / out-of-range numeric errors -> treat as missing (do NOT fabricate values)
    invalid_age = int((((df["age"] < 5) | (df["age"] > 100)) & df["age"].notna()).sum())
    df.loc[(df["age"] < 5) | (df["age"] > 100), "age"] = np.nan

    invalid_qty = int((((df["quantity"] <= 0) | (df["quantity"] > 100)) & df["quantity"].notna()).sum())
    df.loc[(df["quantity"] <= 0) | (df["quantity"] > 100), "quantity"] = np.nan

    invalid_ship_cost = int(((df["shipping_cost"] < 0) & df["shipping_cost"].notna()).sum())
    df.loc[df["shipping_cost"] < 0, "shipping_cost"] = np.nan

    invalid_days = int(((df["days_to_ship"] < 0) & df["days_to_ship"].notna()).sum())
    df.loc[df["days_to_ship"] < 0, "days_to_ship"] = np.nan

    log.append(f"Flagged sentinel/invalid numeric values as missing rather than guessing at them: "
               f"{invalid_age} age values, {invalid_qty} quantity values, "
               f"{invalid_ship_cost} shipping-cost values, {invalid_days} days-to-ship values.")

    # 6) discount_pct missing -> assume no discount was applied (0), a reasonable business default
    filled_discount = int(df["discount_pct"].isna().sum())
    df["discount_pct"] = df["discount_pct"].fillna(0.0)
    log.append(f"Filled {filled_discount} missing discount_pct values with 0 (no discount applied).")

    # 7) Rows critical to revenue analysis must have a valid sales_amount and category
    before = len(df)
    df = df[df["sales_amount"].notna() & df["product_category"].notna()]
    log.append(f"Removed {before - len(df)} rows missing sales_amount or product_category "
               f"(cannot be analyzed for revenue).")

    # 8) return_flag / order_status consistency
    df["return_flag"] = df["return_flag"].astype("boolean")
    df["return_flag"] = df["return_flag"].fillna(df["order_status"].str.lower().eq("returned"))

    final_rows = len(df)
    log.append(f"Final cleaned dataset: {final_rows:,} rows retained from {start_rows:,} raw rows "
               f"({final_rows/start_rows*100:.1f}% retention).")

    return df.reset_index(drop=True), log


# ==============================================================================
# 7. FEATURE ENGINEERING
# ==============================================================================
@st.cache_data(show_spinner=False)
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["year"] = df["order_date"].dt.year
    df["month"] = df["order_date"].dt.month
    df["month_name"] = df["order_date"].dt.strftime("%b")
    df["quarter"] = df["order_date"].dt.quarter
    df["year_month"] = df["order_date"].dt.to_period("M").astype(str)
    df["year_quarter"] = df["year"].astype("Int64").astype(str) + "-Q" + df["quarter"].astype("Int64").astype(str)
    df["weekday"] = df["order_date"].dt.day_name()

    df["profit_margin_pct"] = np.where(df["sales_amount"] > 0, df["profit"] / df["sales_amount"] * 100, np.nan)

    bins = [0, 18, 25, 35, 45, 55, 65, 200]
    labels = ["<18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)

    df["is_returned"] = df["return_flag"].fillna(False) | df["order_status"].eq("Returned")
    df["is_cancelled"] = df["order_status"].eq("Cancelled")
    df["is_completed"] = df["order_status"].isin(["Delivered", "Shipped"])

    df["customer_type"] = np.where(
        df.groupby("customer_id")["order_id"].transform("count") > 1, "Repeat", "One-Time"
    )

    def sat_band(x):
        if pd.isna(x):
            return "Not Rated"
        if x >= 4:
            return "Promoter (4-5)"
        if x >= 3:
            return "Passive (3)"
        return "Detractor (1-2)"
    df["satisfaction_band"] = df["customer_satisfaction"].apply(sat_band)

    return df


# ==============================================================================
# 8. UI HELPER COMPONENTS
# ==============================================================================
def fmt_inr(x, decimals=0):
    if pd.isna(x):
        return "—"
    if abs(x) >= 1e7:
        return f"₹{x/1e7:,.2f} Cr"
    if abs(x) >= 1e5:
        return f"₹{x/1e5:,.2f} L"
    if decimals == 0:
        return f"₹{x:,.0f}"
    return f"₹{x:,.{decimals}f}"


def fmt_num(x, decimals=0):
    if pd.isna(x):
        return "—"
    if decimals == 0:
        return f"{x:,.0f}"
    return f"{x:,.{decimals}f}"


def kpi_card(label, value, icon="◆", delta=None, delta_dir="flat", accent="cyan"):
    accent_map = {
        "cyan": ("rgba(34,211,238,0.35)", "rgba(34,211,238,0.14)", "rgba(34,211,238,0.4)"),
        "purple": ("rgba(168,85,247,0.35)", "rgba(168,85,247,0.14)", "rgba(168,85,247,0.4)"),
        "pink": ("rgba(236,72,153,0.35)", "rgba(236,72,153,0.14)", "rgba(236,72,153,0.4)"),
        "gold": ("rgba(251,191,36,0.35)", "rgba(251,191,36,0.14)", "rgba(251,191,36,0.4)"),
        "green": ("rgba(52,211,153,0.35)", "rgba(52,211,153,0.14)", "rgba(52,211,153,0.4)"),
        "red": ("rgba(248,113,113,0.35)", "rgba(248,113,113,0.14)", "rgba(248,113,113,0.4)"),
    }
    glow, icon_bg, icon_border = accent_map.get(accent, accent_map["cyan"])
    delta_html = ""
    if delta is not None:
        cls = {"up": "up", "down": "down", "flat": "flat"}[delta_dir]
        arrow = {"up": "▲", "down": "▼", "flat": "●"}[delta_dir]
        delta_html = f'<span class="kpi-delta {cls}">{arrow} {delta}</span>'
    st.markdown(f"""
    <div class="kpi-card" style="--glow-color:{glow}; --icon-bg:{icon_bg}; --icon-border:{icon_border};">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def section_head(title, icon=""):
    st.markdown(f"""
    <div class="section-head">
        <h3>{icon} {title}</h3>
        <div class="line"></div>
    </div>
    """, unsafe_allow_html=True)


def card_open(title, sub="", icon=""):
    st.markdown(f"""
    <div class="glass-card">
        <div class="card-title">{icon} {title}</div>
        <div class="card-sub">{sub}</div>
    """, unsafe_allow_html=True)


def card_close():
    st.markdown("</div>", unsafe_allow_html=True)


def insight_card(tag, title, body, accent="cyan"):
    accent_hex = {"cyan": COLORS["cyan"], "purple": COLORS["purple"], "pink": COLORS["pink"],
                  "gold": COLORS["gold"], "green": COLORS["green"], "red": COLORS["red"]}.get(accent, COLORS["cyan"])
    st.markdown(f"""
    <div class="insight-card" style="--accent:{accent_hex};">
        <span class="insight-tag">{tag}</span>
        <div class="insight-title">{title}</div>
        <div class="insight-body">{body}</div>
    </div>
    """, unsafe_allow_html=True)


def pill(text, color="cyan"):
    return f'<span class="pill {color}">{text}</span>'


def style_fig(fig, height=380, title=None):
    fig.update_layout(PLOTLY_TEMPLATE["layout"])
    fig.update_layout(height=height)
    if title:
        fig.update_layout(title=dict(text=title, font=dict(family="Exo 2, sans-serif", size=14,
                                                              color=COLORS["text_primary"])))
    return fig


# ==============================================================================
# 9. ANALYTICS FUNCTIONS
# ==============================================================================
def kpi_summary(df: pd.DataFrame) -> dict:
    # Total Sales/Profit/Orders are computed over ALL recorded transactions (gross,
    # matching the raw sales_amount column) so that AOV = Total Sales / Total Orders
    # stays internally consistent. Return rate and cancel rate are reported
    # separately as their own risk metrics rather than being netted out here.
    total_sales = df["sales_amount"].sum()
    total_profit = df["profit"].sum()
    total_orders = df["order_id"].nunique()
    total_customers = df["customer_id"].nunique()
    total_units = df["quantity"].sum()
    aov = total_sales / total_orders if total_orders else 0
    margin = (total_profit / total_sales * 100) if total_sales else 0
    avg_satisfaction = df["customer_satisfaction"].mean()
    return_rate = df["is_returned"].mean() * 100
    cancel_rate = df["is_cancelled"].mean() * 100
    avg_days_to_ship = df["days_to_ship"].mean()
    repeat_rate = (df.drop_duplicates("customer_id")["customer_type"].eq("Repeat")).mean() * 100
    return dict(total_sales=total_sales, total_profit=total_profit, total_orders=total_orders,
                total_customers=total_customers, total_units=total_units, aov=aov, margin=margin,
                avg_satisfaction=avg_satisfaction, return_rate=return_rate, cancel_rate=cancel_rate,
                avg_days_to_ship=avg_days_to_ship, repeat_rate=repeat_rate)


def period_over_period(df: pd.DataFrame):
    """Split data into the two most recent calendar years present for YoY comparison."""
    years = sorted(df["year"].dropna().unique())
    if len(years) < 2:
        return None, None, None
    cur_y, prev_y = years[-1], years[-2]
    cur = df[df["year"] == cur_y]
    prev = df[df["year"] == prev_y]
    return cur, prev, (cur_y, prev_y)


def pct_change(cur, prev):
    if prev in (0, None) or pd.isna(prev):
        return None
    return (cur - prev) / prev * 100


def sales_trend(df: pd.DataFrame, freq="M"):
    d = df.dropna(subset=["order_date"]).copy()
    if freq == "M":
        grp = d.groupby("year_month").agg(sales=("sales_amount", "sum"),
                                            profit=("profit", "sum"),
                                            orders=("order_id", "nunique")).reset_index()
        grp = grp.sort_values("year_month")
    else:
        grp = d.groupby("year_quarter").agg(sales=("sales_amount", "sum"),
                                              profit=("profit", "sum"),
                                              orders=("order_id", "nunique")).reset_index()
        grp = grp.sort_values("year_quarter")
    return grp


def category_analysis(df: pd.DataFrame):
    g = df.groupby("product_category").agg(
        revenue=("sales_amount", "sum"), profit=("profit", "sum"),
        units=("quantity", "sum"), orders=("order_id", "nunique"),
        avg_price=("unit_price", "mean"), avg_rating=("customer_satisfaction", "mean"),
        return_rate=("is_returned", "mean"),
    ).reset_index()
    g["margin_pct"] = np.where(g["revenue"] > 0, g["profit"] / g["revenue"] * 100, 0)
    g["revenue_share_pct"] = g["revenue"] / g["revenue"].sum() * 100
    g["return_rate"] = g["return_rate"] * 100
    return g.sort_values("revenue", ascending=False)


def product_analysis(df: pd.DataFrame, top_n=10):
    g = df.groupby(["product_name", "product_category"]).agg(
        revenue=("sales_amount", "sum"), profit=("profit", "sum"),
        units=("quantity", "sum"), orders=("order_id", "nunique"),
    ).reset_index()
    g["margin_pct"] = np.where(g["revenue"] > 0, g["profit"] / g["revenue"] * 100, 0)
    return g.sort_values("revenue", ascending=False).head(top_n)


def region_analysis(df: pd.DataFrame):
    g = df.groupby("region").agg(
        revenue=("sales_amount", "sum"), profit=("profit", "sum"),
        orders=("order_id", "nunique"), units=("quantity", "sum"),
        avg_order_value=("sales_amount", "mean"), avg_rating=("customer_satisfaction", "mean"),
        return_rate=("is_returned", "mean"),
    ).reset_index()
    g["margin_pct"] = np.where(g["revenue"] > 0, g["profit"] / g["revenue"] * 100, 0)
    g["return_rate"] = g["return_rate"] * 100
    return g.sort_values("revenue", ascending=False)


def city_analysis(df: pd.DataFrame, top_n=10):
    g = df.groupby(["city", "region"]).agg(
        revenue=("sales_amount", "sum"), orders=("order_id", "nunique"),
        avg_order_value=("sales_amount", "mean"),
    ).reset_index()
    return g.sort_values("revenue", ascending=False).head(top_n)


def customer_analysis(df: pd.DataFrame):
    by_gender = df.groupby("gender").agg(
        revenue=("sales_amount", "sum"), orders=("order_id", "nunique"),
        avg_order_value=("sales_amount", "mean"), avg_rating=("customer_satisfaction", "mean"),
    ).reset_index()
    by_age = df.dropna(subset=["age_group"]).groupby("age_group", observed=True).agg(
        revenue=("sales_amount", "sum"), orders=("order_id", "nunique"),
        avg_order_value=("sales_amount", "mean"),
    ).reset_index()
    by_type = df.groupby("customer_type").agg(
        revenue=("sales_amount", "sum"), orders=("order_id", "nunique"),
        customers=("customer_id", "nunique"), avg_order_value=("sales_amount", "mean"),
    ).reset_index()
    return by_gender, by_age, by_type


def payment_analysis(df: pd.DataFrame):
    g = df.groupby("payment_method").agg(
        revenue=("sales_amount", "sum"), orders=("order_id", "nunique"),
        avg_order_value=("sales_amount", "mean"),
    ).reset_index()
    g["txn_share_pct"] = g["orders"] / g["orders"].sum() * 100
    g["revenue_share_pct"] = g["revenue"] / g["revenue"].sum() * 100
    return g.sort_values("revenue", ascending=False)


def fulfillment_analysis(df: pd.DataFrame):
    status = df["order_status"].value_counts(normalize=True).mul(100).reset_index()
    status.columns = ["order_status", "pct"]
    by_region_ship = df.groupby("region").agg(
        avg_days_to_ship=("days_to_ship", "mean"), avg_shipping_cost=("shipping_cost", "mean"),
    ).reset_index()
    return status, by_region_ship


def satisfaction_analysis(df: pd.DataFrame):
    dist = df.dropna(subset=["customer_satisfaction"])["customer_satisfaction"].value_counts().sort_index()
    by_cat = df.dropna(subset=["customer_satisfaction"]).groupby("product_category")["customer_satisfaction"].mean().sort_values(ascending=False)
    by_region = df.dropna(subset=["customer_satisfaction"]).groupby("region")["customer_satisfaction"].mean().sort_values(ascending=False)
    band = df["satisfaction_band"].value_counts()
    return dist, by_cat, by_region, band


def correlation_matrix(df: pd.DataFrame):
    cols = ["quantity", "unit_price", "discount_pct", "sales_amount", "profit",
            "shipping_cost", "customer_satisfaction", "days_to_ship", "profit_margin_pct"]
    cols = [c for c in cols if c in df.columns]
    return df[cols].corr()


# ==============================================================================
# 10. SIDEBAR — NAVIGATION + GLOBAL FILTERS
# ==============================================================================
NAV_ITEMS = [
    ("◈  Executive Overview", "exec"),
    ("📈  Sales Performance", "sales"),
    ("📦  Product Analytics", "product"),
    ("🗺️  Regional Analytics", "region"),
    ("👥  Customer Analytics", "customer"),
    ("💳  Payment & Fulfillment", "payment"),
    ("⭐  Ratings & Returns", "satisfaction"),
    ("🧪  Statistical EDA", "eda"),
    ("🛡️  Data Quality", "quality"),
    ("💡  Business Insights", "insights"),
]


def render_sidebar(df: pd.DataFrame):
    with st.sidebar:
        st.markdown("""
        <div class="brand-wrap">
            <div class="brand-title">🛰️ RETAIL<br/>INTELLIGENCE</div>
            <div class="brand-sub">Sales &amp; Customer Analytics</div>
        </div>
        """, unsafe_allow_html=True)

        labels = [item[0] for item in NAV_ITEMS]
        keys = [item[1] for item in NAV_ITEMS]
        choice_label = st.radio("Navigate", labels, label_visibility="collapsed")
        page = keys[labels.index(choice_label)]

        st.markdown("<div style='margin:1.1rem 0; border-top:1px solid rgba(120,140,255,0.18);'></div>",
                    unsafe_allow_html=True)
        st.markdown("<div class='brand-sub' style='margin-bottom:0.6rem;'>▤ GLOBAL FILTERS</div>",
                    unsafe_allow_html=True)

        years = sorted(df["year"].dropna().unique().tolist())
        year_sel = st.multiselect("Year", years, default=years)

        regions = sorted(df["region"].dropna().unique().tolist())
        region_sel = st.multiselect("Region", regions, default=regions)

        categories = sorted(df["product_category"].dropna().unique().tolist())
        cat_sel = st.multiselect("Category", categories, default=categories)

        payments = sorted(df["payment_method"].dropna().unique().tolist())
        pay_sel = st.multiselect("Payment Method", payments, default=payments)

        st.markdown("<div style='margin:1.1rem 0; border-top:1px solid rgba(120,140,255,0.18);'></div>",
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div class='brand-sub'>DATASET</div>
        <div style='font-family:"Space Mono",monospace; font-size:0.72rem; color:#8b93c4; line-height:1.6; margin-top:0.4rem;'>
            {len(df):,} clean records<br/>
            {df['order_date'].dt.year.min():.0f}–{df['order_date'].dt.year.max():.0f}<br/>
            {df['city'].nunique()} cities · {df['region'].nunique()} regions
        </div>
        """, unsafe_allow_html=True)

    return page, dict(year=year_sel, region=region_sel, category=cat_sel, payment=pay_sel)


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    out = df.copy()
    if filters["year"]:
        out = out[out["year"].isin(filters["year"])]
    if filters["region"]:
        out = out[out["region"].isin(filters["region"])]
    if filters["category"]:
        out = out[out["product_category"].isin(filters["category"])]
    if filters["payment"]:
        out = out[out["payment_method"].isin(filters["payment"])]
    return out


def hero(title, desc, eyebrow="RETAIL SALES INTELLIGENCE PLATFORM"):
    st.markdown(f"""
    <div class="hero-eyebrow">{eyebrow}</div>
    <div class="hero-title">{title}</div>
    <div class="hero-desc">{desc}</div>
    """, unsafe_allow_html=True)
    st.write("")


# ==============================================================================
# 11. PAGE — EXECUTIVE OVERVIEW
# ==============================================================================
def page_executive_overview(df: pd.DataFrame):
    hero("Executive Overview", "Comprehensive at-a-glance performance across sales, profit, customers "
         "and fulfillment — the CEO-level snapshot of the business.")

    k = kpi_summary(df)
    cur, prev, yrs = period_over_period(df)
    deltas = {}
    if cur is not None:
        ck, pk = kpi_summary(cur), kpi_summary(prev)
        for key in ["total_sales", "total_profit", "aov", "avg_satisfaction", "return_rate"]:
            deltas[key] = pct_change(ck[key], pk[key])

    def dd(key):
        v = deltas.get(key)
        if v is None:
            return None, "flat"
        return f"{v:+.1f}% YoY", ("up" if v > 0 else ("down" if v < 0 else "flat"))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        d, dirn = dd("total_sales")
        kpi_card("Total Sales", fmt_inr(k["total_sales"]), "💰", d, dirn, "cyan")
    with c2:
        d, dirn = dd("total_profit")
        kpi_card("Total Profit", fmt_inr(k["total_profit"]), "📈", d, dirn, "green")
    with c3:
        kpi_card("Total Orders", fmt_num(k["total_orders"]), "🧾", None, "flat", "purple")
    with c4:
        kpi_card("Total Customers", fmt_num(k["total_customers"]), "👤", None, "flat", "pink")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        d, dirn = dd("aov")
        kpi_card("Avg Order Value", fmt_inr(k["aov"], 0), "🛒", d, dirn, "gold")
    with c6:
        kpi_card("Profit Margin", f"{k['margin']:.1f}%", "📊", None, "flat", "cyan")
    with c7:
        d, dirn = dd("avg_satisfaction")
        dirn = "up" if (deltas.get("avg_satisfaction") or 0) > 0 else "down"
        kpi_card("Avg Satisfaction", f"{k['avg_satisfaction']:.2f} / 5", "⭐", d, dirn, "purple")
    with c8:
        d, dirn = dd("return_rate")
        dirn = "down" if (deltas.get("return_rate") or 0) > 0 else "up"  # lower return rate is "good"
        kpi_card("Return Rate", f"{k['return_rate']:.1f}%", "↩️", d, dirn, "red")

    if yrs:
        units_txt = fmt_num(k["total_units"])
        pills_html = (
            pill(f"YoY comparison: {yrs[1]:.0f} \u2192 {yrs[0]:.0f}", "cyan")
            + pill(f"Units sold: {units_txt}", "purple")
            + pill(f"Repeat customer rate: {k['repeat_rate']:.1f}%", "gold")
            + pill(f"Avg days to ship: {k['avg_days_to_ship']:.1f}", "green")
        )
        st.markdown(f"<div style='margin-top:0.6rem;'>{pills_html}</div>", unsafe_allow_html=True)

    left, right = st.columns([2, 1])
    with left:
        card_open("Sales & Profit Trend", "Monthly revenue and profit over the full dataset window", "📈")
        trend = sales_trend(df, "M")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trend["year_month"], y=trend["sales"], name="Sales",
                                   mode="lines", fill="tozeroy",
                                   line=dict(color=COLORS["cyan"], width=2.5),
                                   fillcolor="rgba(34,211,238,0.12)"))
        fig.add_trace(go.Scatter(x=trend["year_month"], y=trend["profit"], name="Profit",
                                   mode="lines", line=dict(color=COLORS["purple"], width=2.5, dash="dot")))
        fig = style_fig(fig, 340)
        fig.update_layout(hovermode="x unified", legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True)
        card_close()

    with right:
        card_open("Category Contribution", "% share of total revenue", "🧩")
        cat = category_analysis(df)
        fig = go.Figure(go.Pie(labels=cat["product_category"], values=cat["revenue"], hole=0.62,
                                 marker=dict(colors=CHART_COLORWAY, line=dict(color="#05070f", width=2)),
                                 textinfo="none"))
        fig.update_layout(showlegend=True, legend=dict(orientation="v", font=dict(size=11)))
        fig.add_annotation(text=f"<b>{fmt_inr(cat['revenue'].sum())}</b><br><span style='font-size:10px'>Total Sales</span>",
                            showarrow=False, font=dict(color=COLORS["text_primary"], size=13))
        fig = style_fig(fig, 340)
        st.plotly_chart(fig, use_container_width=True)
        card_close()

    c1, c2, c3 = st.columns(3)
    with c1:
        card_open("Revenue by Region", "Regional performance comparison", "🗺️")
        reg = region_analysis(df)
        fig = px.bar(reg, x="region", y="revenue", color="region", color_discrete_sequence=CHART_COLORWAY,
                     text=reg["revenue"].apply(lambda v: fmt_inr(v)))
        fig.update_traces(textposition="outside", showlegend=False)
        fig = style_fig(fig, 300)
        fig.update_layout(xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        card_close()
    with c2:
        card_open("Customer Composition", "Repeat vs one-time buyers", "👥")
        _, _, by_type = customer_analysis(df)
        fig = go.Figure(go.Pie(labels=by_type["customer_type"], values=by_type["customers"], hole=0.58,
                                 marker=dict(colors=[COLORS["cyan"], COLORS["pink"]], line=dict(color="#05070f", width=2))))
        fig = style_fig(fig, 300)
        st.plotly_chart(fig, use_container_width=True)
        card_close()
    with c3:
        card_open("Payment Method Mix", "Share of transactions", "💳")
        pay = payment_analysis(df)
        fig = go.Figure(go.Pie(labels=pay["payment_method"], values=pay["orders"], hole=0.58,
                                 marker=dict(colors=CHART_COLORWAY, line=dict(color="#05070f", width=2))))
        fig = style_fig(fig, 300)
        st.plotly_chart(fig, use_container_width=True)
        card_close()

    section_head("Quick Insights", "💡")
    cat_sorted = category_analysis(df)
    reg_sorted = region_analysis(df)
    pay_sorted = payment_analysis(df)
    top_cat = cat_sorted.iloc[0]
    top_region = reg_sorted.iloc[0]
    top_pay = pay_sorted.iloc[0]
    ic1, ic2, ic3, ic4 = st.columns(4)
    with ic1:
        insight_card("Top Category", top_cat["product_category"],
                     f"Generated {fmt_inr(top_cat['revenue'])} ({top_cat['revenue_share_pct']:.1f}% of sales).", "cyan")
    with ic2:
        insight_card("Top Region", top_region["region"],
                     f"Leads with {fmt_inr(top_region['revenue'])} in revenue "
                     f"at a {top_region['margin_pct']:.1f}% margin.", "purple")
    with ic3:
        insight_card("Top Payment Method", top_pay["payment_method"],
                     f"{fmt_num(top_pay['orders'])} transactions "
                     f"({top_pay['txn_share_pct']:.1f}% of volume).", "gold")
    with ic4:
        insight_card("Customer Satisfaction", f"{k['avg_satisfaction']:.2f} / 5",
                     f"Return rate stands at {k['return_rate']:.1f}% across all completed and returned orders.", "pink")


# ==============================================================================
# 12. PAGE — SALES PERFORMANCE
# ==============================================================================
def page_sales_performance(df: pd.DataFrame):
    hero("Sales Performance", "Trend and driver analysis — where revenue is moving, and why.")

    tabs = st.tabs(["📈 Trend", "🔎 Drivers", "📅 Seasonality"])

    with tabs[0]:
        gran = st.radio("Granularity", ["Monthly", "Quarterly"], horizontal=True, label_visibility="collapsed")
        card_open("Revenue & Profit Over Time", "Line chart is the correct choice here: it is the clearest way "
                  "to show a continuous trend across ordered time periods.", "📈")
        trend = sales_trend(df, "M" if gran == "Monthly" else "Q")
        xcol = "year_month" if gran == "Monthly" else "year_quarter"
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trend[xcol], y=trend["sales"], name="Sales", mode="lines+markers",
                                   line=dict(color=COLORS["cyan"], width=2.5), fill="tozeroy",
                                   fillcolor="rgba(34,211,238,0.10)", marker=dict(size=5)))
        fig.add_trace(go.Scatter(x=trend[xcol], y=trend["profit"], name="Profit", mode="lines+markers",
                                   line=dict(color=COLORS["green"], width=2, dash="dot"), marker=dict(size=4)))
        fig = style_fig(fig, 380)
        fig.update_layout(hovermode="x unified", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)
        card_close()

        c1, c2 = st.columns(2)
        with c1:
            card_open("Orders Over Time", "Bar chart — discrete counts per period compare more clearly as bars than lines.", "🧾")
            fig = px.bar(trend, x=xcol, y="orders", color_discrete_sequence=[COLORS["purple"]])
            fig = style_fig(fig, 300)
            fig.update_layout(xaxis_title="", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
            card_close()
        with c2:
            card_open("Profit Margin % Over Time", "Line chart tracks a ratio's direction over time most clearly.", "📊")
            trend["margin_pct"] = np.where(trend["sales"] > 0, trend["profit"] / trend["sales"] * 100, 0)
            fig = px.line(trend, x=xcol, y="margin_pct", markers=True, color_discrete_sequence=[COLORS["gold"]])
            fig = style_fig(fig, 300)
            fig.update_layout(xaxis_title="", yaxis_title="Margin %")
            st.plotly_chart(fig, use_container_width=True)
            card_close()

    with tabs[1]:
        card_open("What Is Driving Revenue?", "Category × Region cross-tab — a heatmap is the right chart for "
                  "spotting which combinations concentrate revenue.", "🔥")
        pivot = df.pivot_table(index="product_category", columns="region", values="sales_amount", aggfunc="sum", fill_value=0)
        fig = go.Figure(go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index,
                                     colorscale=[[0, "#0a0e24"], [0.5, "#3b1e6e"], [1, COLORS["cyan"]]],
                                     colorbar=dict(title="Revenue")))
        fig = style_fig(fig, 420)
        st.plotly_chart(fig, use_container_width=True)
        card_close()

        c1, c2 = st.columns(2)
        with c1:
            card_open("Discount vs Profit Margin", "Scatter plot is the right choice to expose the relationship "
                      "between two continuous variables.", "💸")
            sample = df.dropna(subset=["discount_pct", "profit_margin_pct"]).sample(
                min(1500, len(df)), random_state=1)
            fig = px.scatter(sample, x="discount_pct", y="profit_margin_pct", color="product_category",
                             color_discrete_sequence=CHART_COLORWAY, opacity=0.55)
            fig = style_fig(fig, 340)
            fig.update_layout(xaxis_title="Discount %", yaxis_title="Profit Margin %")
            st.plotly_chart(fig, use_container_width=True)
            card_close()
        with c2:
            corr_val = df[["discount_pct", "profit_margin_pct"]].dropna().corr().iloc[0, 1]
            direction = "a weak negative" if corr_val < -0.05 else ("a weak positive" if corr_val > 0.05 else "almost no")
            insight_card("Driver Finding", "Discounting Impact on Margin",
                         f"Correlation between discount % and profit margin is r = {corr_val:.2f} — "
                         f"indicating {direction} relationship in this dataset. Deep discounting is not "
                         f"strongly eroding margins here, but should still be monitored by category.", "purple")
            best_cat = category_analysis(df).sort_values("margin_pct", ascending=False).iloc[0]
            insight_card("Driver Finding", "Highest-Margin Category",
                         f"{best_cat['product_category']} converts revenue to profit most efficiently at "
                         f"{best_cat['margin_pct']:.1f}% margin — a candidate for expanded marketing spend.", "green")

    with tabs[2]:
        card_open("Sales by Weekday", "Bar chart for comparing discrete weekday buckets.", "📅")
        wd = df.groupby("weekday")["sales_amount"].sum().reindex(
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        fig = px.bar(x=wd.index, y=wd.values, color=wd.values, color_continuous_scale=[COLORS["purple"], COLORS["cyan"]])
        fig = style_fig(fig, 320)
        fig.update_layout(xaxis_title="", yaxis_title="Sales", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        card_close()


# ==============================================================================
# 13. PAGE — PRODUCT ANALYTICS
# ==============================================================================
def page_product_analytics(df: pd.DataFrame):
    hero("Product Analytics", "Category and product-level performance — revenue, margin, and volume drivers.")

    cat = category_analysis(df)

    c1, c2, c3, c4 = st.columns(4)
    top_rev = cat.sort_values("revenue", ascending=False).iloc[0]
    top_margin = cat.sort_values("margin_pct", ascending=False).iloc[0]
    top_units = df.groupby("product_category")["quantity"].sum().idxmax()
    worst_return = cat.sort_values("return_rate", ascending=False).iloc[0]
    with c1:
        kpi_card("Top Revenue Category", top_rev["product_category"], "🏆", fmt_inr(top_rev["revenue"]), "flat", "cyan")
    with c2:
        kpi_card("Highest Margin Category", top_margin["product_category"], "📈", f"{top_margin['margin_pct']:.1f}%", "up", "green")
    with c3:
        kpi_card("Most Units Sold", top_units, "📦", None, "flat", "purple")
    with c4:
        kpi_card("Highest Return Rate", worst_return["product_category"], "⚠️", f"{worst_return['return_rate']:.1f}%", "down", "red")

    left, right = st.columns([1.3, 1])
    with left:
        card_open("Category Revenue vs Profit Margin", "Combo bar+line — compares an absolute value against "
                  "a ratio without misleading dual-axis distortion.", "🧩")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=cat["product_category"], y=cat["revenue"], name="Revenue",
                             marker_color=COLORS["cyan"], yaxis="y"))
        fig.add_trace(go.Scatter(x=cat["product_category"], y=cat["margin_pct"], name="Margin %",
                                   mode="lines+markers", line=dict(color=COLORS["gold"], width=2.5), yaxis="y2"))
        fig.update_layout(yaxis=dict(title="Revenue"), yaxis2=dict(title="Margin %", overlaying="y", side="right",
                                                                     showgrid=False))
        fig = style_fig(fig, 400)
        fig.update_layout(legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True)
        card_close()
    with right:
        card_open("Revenue Share by Category", "Donut chart for part-to-whole composition.", "🧁")
        fig = go.Figure(go.Pie(labels=cat["product_category"], values=cat["revenue"], hole=0.55,
                                 marker=dict(colors=CHART_COLORWAY, line=dict(color="#05070f", width=2))))
        fig = style_fig(fig, 400)
        st.plotly_chart(fig, use_container_width=True)
        card_close()

    card_open("Top 10 Products by Revenue", "Horizontal bar — best for ranking many labeled items legibly.", "🏅")
    top_products = product_analysis(df, 10).sort_values("revenue")
    fig = px.bar(top_products, x="revenue", y="product_name", orientation="h", color="product_category",
                color_discrete_sequence=CHART_COLORWAY, text=top_products["revenue"].apply(lambda v: fmt_inr(v)))
    fig.update_traces(textposition="outside")
    fig = style_fig(fig, 420)
    fig.update_layout(yaxis_title="", xaxis_title="Revenue")
    st.plotly_chart(fig, use_container_width=True)
    card_close()

    c1, c2 = st.columns(2)
    with c1:
        card_open("Revenue vs Quantity — Product Positioning", "Scatter plot reveals products that sell a lot "
                  "but generate little revenue vs low-volume, high-value items.", "🎯")
        pdt = product_analysis(df, 25)
        fig = px.scatter(pdt, x="units", y="revenue", size="revenue", color="product_category",
                         hover_name="product_name", color_discrete_sequence=CHART_COLORWAY, size_max=32)
        fig = style_fig(fig, 380)
        fig.update_layout(xaxis_title="Units Sold", yaxis_title="Revenue")
        st.plotly_chart(fig, use_container_width=True)
        card_close()
    with c2:
        card_open("Category Detail Table", "Full metrics for analyst review.", "📋")
        show = cat[["product_category", "revenue", "profit", "margin_pct", "units", "orders", "return_rate"]].copy()
        show.columns = ["Category", "Revenue", "Profit", "Margin %", "Units", "Orders", "Return %"]
        show["Revenue"] = show["Revenue"].apply(lambda v: fmt_inr(v))
        show["Profit"] = show["Profit"].apply(lambda v: fmt_inr(v))
        show["Margin %"] = show["Margin %"].round(1)
        show["Return %"] = show["Return %"].round(1)
        st.dataframe(show, use_container_width=True, hide_index=True, height=380)
        card_close()


# ==============================================================================
# 14. PAGE — REGIONAL ANALYTICS
# ==============================================================================
def page_regional_analytics(df: pd.DataFrame):
    hero("Regional Analytics", "Geographic performance across India's 5 regions and 20 cities.")

    reg = region_analysis(df)
    c1, c2, c3, c4 = st.columns(4)
    top_region = reg.sort_values("revenue", ascending=False).iloc[0]
    top_aov_region = reg.sort_values("avg_order_value", ascending=False).iloc[0]
    top_margin_region = reg.sort_values("margin_pct", ascending=False).iloc[0]
    worst_return_region = reg.sort_values("return_rate", ascending=False).iloc[0]
    with c1:
        kpi_card("Leading Region", top_region["region"], "🏆", fmt_inr(top_region["revenue"]), "flat", "cyan")
    with c2:
        kpi_card("Highest AOV", top_aov_region["region"], "🛒", fmt_inr(top_aov_region["avg_order_value"]), "up", "gold")
    with c3:
        kpi_card("Highest Margin", top_margin_region["region"], "📈", f"{top_margin_region['margin_pct']:.1f}%", "up", "green")
    with c4:
        kpi_card("Highest Return Rate", worst_return_region["region"], "⚠️", f"{worst_return_region['return_rate']:.1f}%", "down", "red")

    left, right = st.columns([1.3, 1])
    with left:
        card_open("Region Performance Comparison", "Grouped bar for comparing several metrics per region side by side.", "🗺️")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=reg["region"], y=reg["revenue"], name="Revenue", marker_color=COLORS["cyan"]))
        fig.add_trace(go.Bar(x=reg["region"], y=reg["profit"], name="Profit", marker_color=COLORS["purple"]))
        fig.update_layout(barmode="group")
        fig = style_fig(fig, 380)
        fig.update_layout(legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True)
        card_close()
    with right:
        card_open("Avg Order Value by Region", "Bar chart ranks a single derived metric clearly.", "💵")
        r2 = reg.sort_values("avg_order_value")
        fig = px.bar(r2, x="avg_order_value", y="region", orientation="h", color="region",
                    color_discrete_sequence=CHART_COLORWAY)
        fig.update_traces(showlegend=False)
        fig = style_fig(fig, 380)
        fig.update_layout(xaxis_title="Avg Order Value", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        card_close()

    card_open("Top 10 Cities by Revenue", "Horizontal bar keeps 10 city labels fully legible.", "🏙️")
    city = city_analysis(df, 10).sort_values("revenue")
    fig = px.bar(city, x="revenue", y="city", orientation="h", color="region",
                color_discrete_sequence=CHART_COLORWAY, text=city["revenue"].apply(lambda v: fmt_inr(v)))
    fig.update_traces(textposition="outside")
    fig = style_fig(fig, 440)
    fig.update_layout(yaxis_title="", xaxis_title="Revenue")
    st.plotly_chart(fig, use_container_width=True)
    card_close()

    card_open("Regional Detail Table", "Full metrics for analyst review.", "📋")
    show = reg[["region", "revenue", "profit", "margin_pct", "orders", "avg_order_value", "avg_rating", "return_rate"]].copy()
    show.columns = ["Region", "Revenue", "Profit", "Margin %", "Orders", "Avg Order Value", "Avg Rating", "Return %"]
    for col in ["Revenue", "Profit", "Avg Order Value"]:
        show[col] = show[col].apply(lambda v: fmt_inr(v))
    show["Margin %"] = show["Margin %"].round(1)
    show["Avg Rating"] = show["Avg Rating"].round(2)
    show["Return %"] = show["Return %"].round(1)
    st.dataframe(show, use_container_width=True, hide_index=True)
    card_close()


# ==============================================================================
# 15. PAGE — CUSTOMER ANALYTICS
# ==============================================================================
def page_customer_analytics(df: pd.DataFrame):
    hero("Customer Analytics", "Who buys, how often, and what drives their spending.")

    by_gender, by_age, by_type = customer_analysis(df)
    k = kpi_summary(df)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total Customers", fmt_num(k["total_customers"]), "👤", None, "flat", "cyan")
    with c2:
        kpi_card("Repeat Customer Rate", f"{k['repeat_rate']:.1f}%", "🔁", None, "flat", "purple")
    with c3:
        top_age = by_age.sort_values("revenue", ascending=False).iloc[0]["age_group"]
        kpi_card("Top Age Segment", str(top_age), "🎂", None, "flat", "gold")
    with c4:
        best_gender_aov = by_gender.sort_values("avg_order_value", ascending=False).iloc[0]
        kpi_card("Highest AOV Segment", best_gender_aov["gender"], "💎", fmt_inr(best_gender_aov["avg_order_value"]), "flat", "pink")

    c1, c2 = st.columns(2)
    with c1:
        card_open("Revenue by Age Group", "Bar chart — age bands are discrete, ordered categories.", "🎂")
        fig = px.bar(by_age, x="age_group", y="revenue", color="age_group", color_discrete_sequence=CHART_COLORWAY)
        fig.update_traces(showlegend=False)
        fig = style_fig(fig, 340)
        fig.update_layout(xaxis_title="", yaxis_title="Revenue")
        st.plotly_chart(fig, use_container_width=True)
        card_close()
    with c2:
        card_open("Gender Comparison", "Grouped bar — compares revenue and AOV across a small set of categories.", "⚖️")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=by_gender["gender"], y=by_gender["avg_order_value"], name="Avg Order Value",
                             marker_color=COLORS["cyan"]))
        fig = style_fig(fig, 340)
        fig.update_layout(yaxis_title="Avg Order Value", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        card_close()

    c1, c2 = st.columns(2)
    with c1:
        card_open("Repeat vs One-Time Customers", "Donut — simple two-part composition.", "🔁")
        fig = go.Figure(go.Pie(labels=by_type["customer_type"], values=by_type["customers"], hole=0.6,
                                 marker=dict(colors=[COLORS["cyan"], COLORS["pink"]], line=dict(color="#05070f", width=2))))
        fig = style_fig(fig, 320)
        st.plotly_chart(fig, use_container_width=True)
        card_close()
    with c2:
        card_open("Repeat vs One-Time — Spending Behavior", "Bar chart compares average spend across the two segments.", "💰")
        fig = px.bar(by_type, x="customer_type", y="avg_order_value", color="customer_type",
                    color_discrete_sequence=[COLORS["cyan"], COLORS["pink"]])
        fig.update_traces(showlegend=False)
        fig = style_fig(fig, 320)
        fig.update_layout(xaxis_title="", yaxis_title="Avg Order Value")
        st.plotly_chart(fig, use_container_width=True)
        card_close()

    card_open("Category Preference by Gender", "Heatmap surfaces which categories each gender segment favors.", "🔥")
    pivot = df.pivot_table(index="gender", columns="product_category", values="sales_amount", aggfunc="sum", fill_value=0)
    fig = go.Figure(go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index,
                                 colorscale=[[0, "#0a0e24"], [0.5, "#6d28d9"], [1, COLORS["pink"]]],
                                 colorbar=dict(title="Revenue")))
    fig = style_fig(fig, 260)
    st.plotly_chart(fig, use_container_width=True)
    card_close()


# ==============================================================================
# 16. PAGE — PAYMENT & FULFILLMENT
# ==============================================================================
def page_payment_fulfillment(df: pd.DataFrame):
    hero("Payment & Fulfillment", "How customers pay, and how efficiently orders are delivered.")

    pay = payment_analysis(df)
    status, ship_region = fulfillment_analysis(df)
    k = kpi_summary(df)

    c1, c2, c3, c4 = st.columns(4)
    top_pay_vol = pay.sort_values("orders", ascending=False).iloc[0]
    top_pay_rev = pay.sort_values("revenue", ascending=False).iloc[0]
    with c1:
        kpi_card("Most-Used Payment", top_pay_vol["payment_method"], "💳", f"{top_pay_vol['txn_share_pct']:.1f}% of orders", "flat", "cyan")
    with c2:
        kpi_card("Highest-Revenue Method", top_pay_rev["payment_method"], "💰", f"{top_pay_rev['revenue_share_pct']:.1f}% of revenue", "flat", "gold")
    with c3:
        kpi_card("Avg Days to Ship", f"{k['avg_days_to_ship']:.1f} days", "🚚", None, "flat", "purple")
    with c4:
        kpi_card("Cancellation Rate", f"{k['cancel_rate']:.1f}%", "❌", None, "flat", "red")

    left, right = st.columns([1.2, 1])
    with left:
        card_open("Transaction Share vs Revenue Share by Payment Method", "Grouped bar — reveals when the most-used "
                  "method isn't the most valuable by revenue.", "💳")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=pay["payment_method"], y=pay["txn_share_pct"], name="Txn Share %", marker_color=COLORS["cyan"]))
        fig.add_trace(go.Bar(x=pay["payment_method"], y=pay["revenue_share_pct"], name="Revenue Share %", marker_color=COLORS["purple"]))
        fig.update_layout(barmode="group")
        fig = style_fig(fig, 380)
        fig.update_layout(legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True)
        card_close()
    with right:
        card_open("Order Status Breakdown", "Donut for part-to-whole status composition.", "📦")
        fig = go.Figure(go.Pie(labels=status["order_status"], values=status["pct"], hole=0.55,
                                 marker=dict(colors=CHART_COLORWAY, line=dict(color="#05070f", width=2))))
        fig = style_fig(fig, 380)
        st.plotly_chart(fig, use_container_width=True)
        card_close()

    c1, c2 = st.columns(2)
    with c1:
        card_open("Avg Days to Ship by Region", "Bar chart ranks fulfillment speed across regions.", "🚚")
        s2 = ship_region.sort_values("avg_days_to_ship")
        fig = px.bar(s2, x="region", y="avg_days_to_ship", color="region", color_discrete_sequence=CHART_COLORWAY)
        fig.update_traces(showlegend=False)
        fig = style_fig(fig, 320)
        fig.update_layout(xaxis_title="", yaxis_title="Avg Days")
        st.plotly_chart(fig, use_container_width=True)
        card_close()
    with c2:
        card_open("Avg Order Value by Payment Method", "Bar chart ranks a single metric per category.", "💵")
        p2 = pay.sort_values("avg_order_value")
        fig = px.bar(p2, x="avg_order_value", y="payment_method", orientation="h", color="payment_method",
                    color_discrete_sequence=CHART_COLORWAY)
        fig.update_traces(showlegend=False)
        fig = style_fig(fig, 320)
        fig.update_layout(yaxis_title="", xaxis_title="Avg Order Value")
        st.plotly_chart(fig, use_container_width=True)
        card_close()


# ==============================================================================
# 17. PAGE — RATINGS & RETURNS
# ==============================================================================
def page_satisfaction(df: pd.DataFrame):
    hero("Ratings & Returns", "Customer experience analysis — satisfaction drivers and return risk.")

    dist, by_cat, by_region, band = satisfaction_analysis(df)
    k = kpi_summary(df)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Avg Satisfaction", f"{k['avg_satisfaction']:.2f} / 5", "⭐", None, "flat", "gold")
    with c2:
        promoters_pct = band.get("Promoter (4-5)", 0) / band.sum() * 100
        kpi_card("Promoters (4-5★)", f"{promoters_pct:.1f}%", "😊", None, "flat", "green")
    with c3:
        detractors_pct = band.get("Detractor (1-2)", 0) / band.sum() * 100
        kpi_card("Detractors (1-2★)", f"{detractors_pct:.1f}%", "😞", None, "flat", "red")
    with c4:
        kpi_card("Overall Return Rate", f"{k['return_rate']:.1f}%", "↩️", None, "flat", "purple")

    c1, c2 = st.columns(2)
    with c1:
        card_open("Rating Distribution", "Bar chart — the standard way to show a discrete 1-5 rating histogram.", "⭐")
        fig = px.bar(x=dist.index, y=dist.values, color=dist.values,
                    color_continuous_scale=[COLORS["pink"], COLORS["cyan"]])
        fig.update_layout(coloraxis_showscale=False)
        fig = style_fig(fig, 320)
        fig.update_layout(xaxis_title="Rating", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
        card_close()
    with c2:
        card_open("Avg Rating by Category", "Horizontal bar ranks categories by customer experience.", "🧩")
        fig = px.bar(x=by_cat.values, y=by_cat.index, orientation="h", color=by_cat.values,
                    color_continuous_scale=[COLORS["purple"], COLORS["cyan"]])
        fig.update_layout(coloraxis_showscale=False)
        fig = style_fig(fig, 320)
        fig.update_layout(xaxis_title="Avg Rating", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        card_close()

    card_open("Return Rate by Category", "Bar chart flags categories carrying disproportionate return risk.", "⚠️")
    cat = category_analysis(df).sort_values("return_rate", ascending=False)
    avg_return = cat["return_rate"].mean()
    fig = px.bar(cat, x="product_category", y="return_rate",
                color=cat["return_rate"] > avg_return,
                color_discrete_map={True: COLORS["red"], False: COLORS["cyan"]})
    fig.add_hline(y=avg_return, line_dash="dot", line_color=COLORS["gold"],
                  annotation_text=f"Avg: {avg_return:.1f}%", annotation_font_color=COLORS["gold"])
    fig.update_traces(showlegend=False)
    fig = style_fig(fig, 340)
    fig.update_layout(xaxis_title="", yaxis_title="Return Rate %")
    st.plotly_chart(fig, use_container_width=True)
    card_close()

    c1, c2 = st.columns(2)
    with c1:
        corr = df[["customer_satisfaction", "sales_amount"]].dropna().corr().iloc[0, 1]
        rel = "no meaningful" if abs(corr) < 0.05 else ("a weak positive" if corr > 0 else "a weak negative")
        insight_card("Rating vs Spend", "Does satisfaction track order value?",
                     f"Correlation between customer satisfaction and order value is r = {corr:.2f}, "
                     f"indicating {rel} relationship in this dataset — satisfaction here is driven more by "
                     f"product/fulfillment experience than transaction size.", "cyan")
    with c2:
        worst_cat_return = cat.iloc[0]
        insight_card("Risk Flag", f"{worst_cat_return['product_category']} — Elevated Return Rate",
                     f"Return rate of {worst_cat_return['return_rate']:.1f}% is the highest among all categories, "
                     f"versus a {avg_return:.1f}% average — worth a quality/fit investigation.", "red")


# ==============================================================================
# 18. PAGE — STATISTICAL EDA
# ==============================================================================
def page_eda(df: pd.DataFrame):
    hero("Statistical EDA", "Distribution shapes, descriptive statistics, and correlation structure of the "
         "underlying numeric variables.")

    metric = st.selectbox("Choose a variable to inspect",
                          ["sales_amount", "profit", "quantity", "unit_price", "discount_pct",
                           "customer_satisfaction", "days_to_ship", "profit_margin_pct"])

    c1, c2 = st.columns([1.4, 1])
    with c1:
        card_open(f"Distribution of {metric}", "Histogram — the correct chart for visualizing the shape "
                  "(skew, spread, outliers) of a single continuous variable.", "📊")
        series = df[metric].dropna()
        fig = px.histogram(series, nbins=40, color_discrete_sequence=[COLORS["cyan"]])
        fig.add_vline(x=series.mean(), line_color=COLORS["gold"], line_dash="dash",
                     annotation_text="mean", annotation_font_color=COLORS["gold"])
        fig.add_vline(x=series.median(), line_color=COLORS["pink"], line_dash="dot",
                     annotation_text="median", annotation_font_color=COLORS["pink"])
        fig.update_traces(showlegend=False)
        fig = style_fig(fig, 380)
        fig.update_layout(xaxis_title=metric, yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)
        card_close()
    with c2:
        card_open("Descriptive Statistics", "Precise summary values for the analyst.", "🧮")
        desc = df[metric].describe()
        stat_rows = [
            ("Count", f"{desc['count']:,.0f}"), ("Mean", f"{desc['mean']:,.2f}"),
            ("Std Dev", f"{desc['std']:,.2f}"), ("Min", f"{desc['min']:,.2f}"),
            ("Q1 (25%)", f"{desc['25%']:,.2f}"), ("Median", f"{desc['50%']:,.2f}"),
            ("Q3 (75%)", f"{desc['75%']:,.2f}"), ("Max", f"{desc['max']:,.2f}"),
        ]
        rows_html = "".join(
            f"<div style='display:flex; justify-content:space-between; padding:0.4rem 0; "
            f"border-bottom:1px solid rgba(120,140,255,0.1); font-family:\"Space Mono\",monospace; font-size:0.85rem;'>"
            f"<span style='color:#8b93c4;'>{k}</span><span style='color:#eef1fb; font-weight:700;'>{v}</span></div>"
            for k, v in stat_rows
        )
        st.markdown(rows_html, unsafe_allow_html=True)
        card_close()

    card_open("Correlation Matrix", "Heatmap — the standard visualization for a full correlation structure "
              "across multiple numeric variables at once.", "🔗")
    corr = correlation_matrix(df)
    fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns,
                                 colorscale=[[0, COLORS["pink"]], [0.5, "#0a0e24"], [1, COLORS["cyan"]]],
                                 zmid=0, text=np.round(corr.values, 2), texttemplate="%{text}",
                                 textfont=dict(size=10), colorbar=dict(title="r")))
    fig = style_fig(fig, 460)
    st.plotly_chart(fig, use_container_width=True)
    card_close()

    st.markdown("""
    <div class="insight-body" style="font-size:0.82rem; padding:0 0.2rem;">
    ⚠️ <b>Correlation is not causation.</b> These coefficients describe association strength and direction
    only — any relationship observed here should be validated with domain knowledge before it informs a
    business decision.
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# 19. PAGE — DATA QUALITY
# ==============================================================================
def page_data_quality(raw: pd.DataFrame, quality: dict, log: list, cleaned_rows: int):
    hero("Data Quality Report", "Full transparency on what was wrong with the raw data, and exactly how it "
         "was handled. Raw data is never silently modified.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Raw Rows", fmt_num(quality["n_rows"]), "📥", None, "flat", "cyan")
    with c2:
        kpi_card("Missing Cells", fmt_num(quality["total_missing_cells"]), "❓", None, "flat", "gold")
    with c3:
        kpi_card("Duplicate Rows", fmt_num(quality["duplicate_rows"]), "🧬", None, "flat", "red")
    with c4:
        kpi_card("Clean Records Retained", fmt_num(cleaned_rows), "✅",
                f"{cleaned_rows/quality['n_rows']*100:.1f}% of raw", "up", "green")

    c1, c2 = st.columns(2)
    with c1:
        card_open("Missing Values by Column", "Horizontal bar ranks columns by data-completeness risk.", "❓")
        m = quality["missing_by_col"]
        fig = px.bar(x=m.values, y=m.index, orientation="h", color_discrete_sequence=[COLORS["gold"]])
        fig = style_fig(fig, 420)
        fig.update_layout(xaxis_title="Missing Count", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        card_close()
    with c2:
        card_open("Invalid / Sentinel Values Detected", "Bar chart of out-of-range and impossible values found "
                  "in the raw file (e.g. age = 999, negative quantity).", "🚨")
        inv = pd.Series(quality["invalid_counts"])
        inv = inv[inv > 0].sort_values()
        fig = px.bar(x=inv.values, y=inv.index, orientation="h", color_discrete_sequence=[COLORS["red"]])
        fig = style_fig(fig, 420)
        fig.update_layout(xaxis_title="Count", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        card_close()

    section_head("Raw → Validated → Cleaned → Processed", "🛡️")
    stages = ["Raw Data", "Validation", "Cleaning", "Processed Data"]
    stage_counts = [quality["n_rows"], quality["n_rows"] - quality["duplicate_rows"],
                    quality["n_rows"] - quality["duplicate_rows"] - quality["rows_flagged_invalid"], cleaned_rows]
    fig = go.Figure(go.Funnel(y=stages, x=stage_counts, marker=dict(color=CHART_COLORWAY[:4]),
                                textinfo="value+percent initial"))
    fig = style_fig(fig, 340)
    st.plotly_chart(fig, use_container_width=True)

    section_head("Cleaning Log", "📝")
    card_open("Every Transformation Applied", "In order — nothing is changed without being logged here.", "🧾")
    for i, entry in enumerate(log, 1):
        st.markdown(f"""
        <div style="display:flex; gap:0.7rem; padding:0.5rem 0; border-bottom:1px solid rgba(120,140,255,0.08);">
            <span style="font-family:'Space Mono',monospace; color:{COLORS['cyan']}; font-weight:700; min-width:24px;">{i:02d}</span>
            <span style="color:{COLORS['text_secondary']}; font-size:0.87rem; line-height:1.5;">{entry}</span>
        </div>
        """, unsafe_allow_html=True)
    card_close()

    with st.expander("🔍 Preview Raw Data (first 20 rows, unmodified)"):
        st.dataframe(raw.head(20), use_container_width=True)


# ==============================================================================
# 20. PAGE — BUSINESS INSIGHTS  (Insight Hierarchy: KPI -> Trend -> Driver -> Risk/Opportunity -> Action)
# ==============================================================================
def page_business_insights(df: pd.DataFrame):
    hero("Business Insights", "Structured through the BI Insight Hierarchy — from raw numbers to a "
         "recommended action.")

    k = kpi_summary(df)
    cat = category_analysis(df)
    reg = region_analysis(df)
    pay = payment_analysis(df)
    trend = sales_trend(df, "Q")
    _, _, by_type = customer_analysis(df)

    # ---------- LEVEL 1: KPIs ----------
    section_head("Level 1 — KPIs: What Is Happening?", "①")
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Total Sales", fmt_inr(k["total_sales"]), "💰", None, "flat", "cyan")
    with c2: kpi_card("Total Profit", fmt_inr(k["total_profit"]), "📈", None, "flat", "green")
    with c3: kpi_card("Profit Margin", f"{k['margin']:.1f}%", "📊", None, "flat", "purple")
    with c4: kpi_card("Avg Order Value", fmt_inr(k["aov"]), "🛒", None, "flat", "gold")

    # ---------- LEVEL 2: TRENDS ----------
    section_head("Level 2 — Trends: Which Direction Is It Going?", "②")
    q_first, q_last = trend.iloc[0], trend.iloc[-1]
    sales_growth = pct_change(q_last["sales"], q_first["sales"])
    trend_dir = "risen" if (sales_growth or 0) > 0 else "declined"
    insight_card("Trend", f"Quarterly Sales Have {trend_dir.title()}",
                 f"From {q_first['year_quarter']} to {q_last['year_quarter']}, quarterly sales moved from "
                 f"{fmt_inr(q_first['sales'])} to {fmt_inr(q_last['sales'])} "
                 f"({sales_growth:+.1f}% change across the full window).", "cyan" if trend_dir == "risen" else "red")
    fig = px.line(trend, x="year_quarter", y="sales", markers=True, color_discrete_sequence=[COLORS["cyan"]])
    fig = style_fig(fig, 260)
    fig.update_layout(xaxis_title="", yaxis_title="Sales")
    st.plotly_chart(fig, use_container_width=True)

    # ---------- LEVEL 3: DRIVERS ----------
    section_head("Level 3 — Drivers: Why Is It Happening?", "③")
    top_cat = cat.sort_values("revenue", ascending=False).iloc[0]
    top_region = reg.sort_values("revenue", ascending=False).iloc[0]
    c1, c2 = st.columns(2)
    with c1:
        insight_card("Driver", f"{top_cat['product_category']} Leads Category Revenue",
                     f"Contributes {fmt_inr(top_cat['revenue'])} ({top_cat['revenue_share_pct']:.1f}% of total "
                     f"sales) at a {top_cat['margin_pct']:.1f}% margin — the single biggest driver of overall revenue.", "purple")
    with c2:
        insight_card("Driver", f"{top_region['region']} Region Outperforms",
                     f"{top_region['region']} generates {fmt_inr(top_region['revenue'])}, the highest of any "
                     f"region, with a {top_region['margin_pct']:.1f}% profit margin.", "gold")

    # ---------- LEVEL 4: RISK & OPPORTUNITY ----------
    section_head("Level 4 — Risk & Opportunity", "④")
    high_return_cat = cat.sort_values("return_rate", ascending=False).iloc[0]
    low_margin_cat = cat.sort_values("margin_pct", ascending=True).iloc[0]
    best_margin_cat = cat.sort_values("margin_pct", ascending=False).iloc[0]
    onetime_share = by_type.set_index("customer_type").loc["One-Time", "customers"] / by_type["customers"].sum() * 100
    c1, c2 = st.columns(2)
    with c1:
        insight_card("⚠ Risk", f"{high_return_cat['product_category']} — High Return Rate",
                     f"Return rate of {high_return_cat['return_rate']:.1f}% is the highest across categories, "
                     f"tying up margin in reverse logistics and refunds.", "red")
        insight_card("⚠ Risk", "Low Repeat-Purchase Base",
                     f"{onetime_share:.1f}% of customers have purchased only once — the business is heavily "
                     f"dependent on new-customer acquisition rather than retention.", "red")
    with c2:
        insight_card("✓ Opportunity", f"{best_margin_cat['product_category']} — Margin Leader",
                     f"Converts revenue to profit most efficiently at {best_margin_cat['margin_pct']:.1f}% margin "
                     f"— scaling this category's marketing spend has the best profit leverage.", "green")
        insight_card("✓ Opportunity", f"{top_region['region']} — Regional Expansion",
                     f"Already the top-performing region by revenue and margin — a strong candidate for "
                     f"further inventory investment or a new fulfillment hub.", "green")

    # ---------- LEVEL 5: ACTION ----------
    section_head("Level 5 — Recommended Actions", "⑤")
    actions = [
        ("cyan", "Investigate Returns", f"Audit product quality, sizing/fit information and listing accuracy for "
         f"{high_return_cat['product_category']} — its {high_return_cat['return_rate']:.1f}% return rate is "
         f"eroding realized margin."),
        ("green", "Scale the Margin Leader", f"Increase marketing and inventory allocation toward "
         f"{best_margin_cat['product_category']}, which converts revenue to profit most efficiently in this dataset."),
        ("gold", "Launch a Retention Program", f"With {onetime_share:.1f}% of customers buying only once, a "
         f"loyalty or win-back campaign targeted at one-time buyers could materially lift repeat-purchase rate."),
        ("purple", f"Double Down on {top_region['region']}", f"Regional revenue leader with strong margin — "
         f"prioritize stock availability and delivery speed here ahead of lower-performing regions."),
    ]
    for accent, title, body in actions:
        insight_card("Action", title, body, accent)


# ==============================================================================
# 21. MAIN
# ==============================================================================
def main():
    inject_css()

    try:
        raw = load_raw_data(DATA_PATH)
    except FileNotFoundError:
        st.markdown(f"""
        <div class="hero-eyebrow">SYSTEM ERROR</div>
        <div class="hero-title" style="font-size:1.8rem;">Dataset Not Found</div>
        <div class="hero-desc">Place <code>{DATA_PATH}</code> in the same folder as this script and reload.
        See README.md for setup instructions.</div>
        """, unsafe_allow_html=True)
        st.stop()

    quality = compute_quality_report(raw)
    cleaned, log = clean_data(raw)
    featured = engineer_features(cleaned)

    page, filters = render_sidebar(featured)
    filtered = apply_filters(featured, filters)

    if len(filtered) == 0:
        st.warning("No records match the current filter selection. Adjust filters in the sidebar.")
        st.stop()

    if page == "exec":
        page_executive_overview(filtered)
    elif page == "sales":
        page_sales_performance(filtered)
    elif page == "product":
        page_product_analytics(filtered)
    elif page == "region":
        page_regional_analytics(filtered)
    elif page == "customer":
        page_customer_analytics(filtered)
    elif page == "payment":
        page_payment_fulfillment(filtered)
    elif page == "satisfaction":
        page_satisfaction(filtered)
    elif page == "eda":
        page_eda(filtered)
    elif page == "quality":
        page_data_quality(raw, quality, log, len(featured))
    elif page == "insights":
        page_business_insights(filtered)

    st.markdown(f"""
    <div style="margin-top:2.5rem; padding-top:1.2rem; border-top:1px solid rgba(120,140,255,0.15);
                text-align:center; font-family:'Space Mono',monospace; font-size:0.7rem; color:#5b6291;">
        RETAIL INTELLIGENCE PLATFORM &nbsp;·&nbsp; Built with Streamlit + Plotly &nbsp;·&nbsp;
        Data as of {featured['order_date'].max():%b %Y}
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
