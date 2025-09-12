# app.py
# Indian Stock Analyzer — NSE (NIFTY Smallcap 250, Midcap 150, NIFTY 500) + BSE SmallCap
# Fetch-on-demand only; all filters are local. Supports multi-MA, RSI ranges, early breakout, golden cross, sector filter, charts & CSV export.

import io
import re
import time
import math
import requests
import datetime as dt
import os
import pickle
import shutil
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
from bs4 import BeautifulSoup

st.set_page_config(
    page_title="Indian Stock Analyzer — NSE & BSE Small/Mid",
    layout="wide",
    initial_sidebar_state="expanded"
)

SS = {
    "universe": "universe_df",
    "prices": "prices_store",
    "last_fetch": "last_fetch_ts",
}

# -----------------------
# Data sources
# -----------------------
CSV_SOURCES = {
    "NIFTY Smallcap 250": [
        "https://niftyindices.com/IndexConstituent/ind_niftysmallcap250list.csv",
        "https://nsearchives.nseindia.com/content/indices/ind_niftysmallcap250list.csv",
    ],
    "NIFTY Midcap 150": [
        "https://niftyindices.com/IndexConstituent/ind_niftymidcap150list.csv",
        "https://nsearchives.nseindia.com/content/indices/ind_niftymidcap150list.csv",
    ],
    "NIFTY 500": [
        "https://niftyindices.com/IndexConstituent/ind_nifty500list.csv",
        "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
    ],
}

# BSE SmallCap (primary: Screener static page; fallback: BSE Indices Watch)
SCREENER_BSE_SMALLCAP_URL = "https://www.screener.in/company/1128/"
BSE_SMALLCAP_URL = "https://www.bseindia.com/sensex/IndicesWatch_Weight.aspx?iname=SMLCAP&index_Code=82"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Referer": "https://www.bseindia.com/",
}

# -----------------------
# Helpers
# -----------------------
def fetch_constituents_nse(index_name: str) -> pd.DataFrame:
    errors = []
    for url in CSV_SOURCES[index_name]:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=25)
            resp.raise_for_status()
            df = pd.read_csv(io.BytesIO(resp.content))
            df.columns = [c.strip() for c in df.columns]

            if "CompanyName" in df.columns and "Company Name" not in df.columns:
                df["Company Name"] = df["CompanyName"]
            if "SYMBOL" in df.columns and "Symbol" not in df.columns:
                df["Symbol"] = df["SYMBOL"]
            if "Industry Name" in df.columns and "Industry" not in df.columns:
                df["Industry"] = df["Industry Name"]

            keep_cols = [c for c in ["Company Name", "Industry", "Symbol", "Series", "ISIN Code"] if c in df.columns]
            df = df[keep_cols].copy()
            df["Index"] = index_name
            df["Exchange"] = "NSE"

            df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
            if "Series" in df.columns:
                df = df[df["Series"].astype(str).str.upper().eq("EQ") | df["Series"].isna()]

            return df.reset_index(drop=True)
        except Exception as e:
            errors.append(f"{url}: {e}")
    raise RuntimeError("Failed to fetch NIFTY constituents:\n - " + "\n - ".join(errors))


def _clean_bse_code(s: str) -> str:
    s = str(s).strip()
    m = re.search(r"\b(\d{5,6})\b", s)
    return m.group(1) if m else ""


def fetch_bse_smallcap_constituents() -> pd.DataFrame:
    """
    Primary: Screener.in static page.
    Fallback: BSE Indices Watch (best-effort).
    Returns columns: Company Name, Industry (NaN), Symbol (BSE code), Series, ISIN Code, Index, Exchange, YFTicker.
    """
    # --- Try Screener first ---
    try:
        r = requests.get(SCREENER_BSE_SMALLCAP_URL, headers=HEADERS, timeout=25)
        r.raise_for_status()
        tables = pd.read_html(r.text)
        target = None
        for t in tables:
            cols = [str(c).strip().lower() for c in t.columns]
            if any("company" in c for c in cols) and len(t) > 20:
                target = t
                break
        if target is not None:
            target.columns = [str(c).strip() for c in target.columns]
            # Try to detect a code column; otherwise map from anchors
            code_col = None
            for c in target.columns:
                if re.search(r"bse|code|scrip", c, flags=re.I):
                    code_col = c
                    break
            if code_col is None:
                soup = BeautifulSoup(r.text, "lxml")
                code_map = {}
                for a in soup.select("table a[href]"):
                    txt = a.get_text(" ", strip=True)
                    href = a.get("href", "")
                    code = _clean_bse_code(txt) or _clean_bse_code(href)
                    if code:
                        code_map[txt.upper()] = code
                target["_BSE_CODE_"] = target.iloc[:, 0].astype(str).str.upper().map(code_map).fillna("")
                code_col = "_BSE_CODE_"

            out = pd.DataFrame({
                "Company Name": target.iloc[:, 0].astype(str).str.strip(),
                "Industry": np.nan,
                "Symbol": target[code_col].astype(str).apply(_clean_bse_code),
                "Series": np.nan,
                "ISIN Code": np.nan,
                "Index": "BSE SmallCap",
                "Exchange": "BSE",
            })
            out = out[out["Symbol"] != ""].drop_duplicates(subset=["Symbol"]).reset_index(drop=True)
            out["YFTicker"] = out["Symbol"].apply(lambda s: f"{s}.BO")
            if not out.empty:
                return out
    except Exception:
        pass

    # --- Fallback: BSE page (JS-filled; best-effort) ---
    r = requests.get(BSE_SMALLCAP_URL, headers=HEADERS, timeout=25)
    r.raise_for_status()
    try:
        tables = pd.read_html(r.text)
        candidate = None
        for t in tables:
            cols = [str(c).lower() for c in t.columns]
            if any("scrip" in c for c in cols) and any("company" in c for c in cols):
                candidate = t
                break
        if candidate is not None:
            ren = {}
            for c in candidate.columns:
                s = str(c)
                if re.search(r"scrip", s, re.I): ren[c] = "Scrip Code"
                elif re.search(r"company", s, re.I): ren[c] = "Company Name"
                elif re.search(r"isin", s, re.I): ren[c] = "ISIN Code"
            df = candidate.rename(columns=ren)
        else:
            text = BeautifulSoup(r.text, "lxml").get_text(" ", strip=True)
            rows = re.findall(r"(\d{5,6})\s+([A-Z0-9&.\- ]{3,})", text)
            df = pd.DataFrame(rows, columns=["Scrip Code", "Company Name"])

        df["Scrip Code"] = df["Scrip Code"].astype(str).str.strip()
        df = df[df["Scrip Code"].str.fullmatch(r"\d{5,6}")].copy()
        out = pd.DataFrame({
            "Company Name": df["Company Name"].astype(str).str.strip(),
            "Industry": np.nan,
            "Symbol": df["Scrip Code"],
            "Series": np.nan,
            "ISIN Code": df.get("ISIN Code", np.nan),
            "Index": "BSE SmallCap",
            "Exchange": "BSE",
        }).drop_duplicates(subset=["Symbol"]).reset_index(drop=True)
        out["YFTicker"] = out["Symbol"].apply(lambda s: f"{s}.BO")
        if out.empty:
            raise RuntimeError("BSE SmallCap parse produced an empty table")
        return out
    except Exception as e:
        raise RuntimeError(f"Could not load BSE SmallCap constituents from any source: {e}")


def to_yf_ticker(symbol: str, exchange: str) -> str:
    s = str(symbol).strip().upper()
    if s.endswith(".NS") or s.endswith(".BO"):
        return s
    if exchange == "NSE":
        return f"{s}.NS"
    if exchange == "BSE":
        return f"{s}.BO"
    return s


def build_universe(selected_indices: List[str]) -> pd.DataFrame:
    frames = []
    for idx in selected_indices:
        if idx in CSV_SOURCES:
            frames.append(fetch_constituents_nse(idx))
        elif idx == "BSE SmallCap":
            frames.append(fetch_bse_smallcap_constituents())

    if not frames:
        return pd.DataFrame(columns=["Company Name","Industry","Symbol","Series","ISIN Code","Index","Exchange","YFTicker"])

    uni = pd.concat(frames, ignore_index=True)

    # Add Yahoo Finance tickers for merge/downloads
    if "YFTicker" not in uni.columns:
        uni["YFTicker"] = uni.apply(lambda r: to_yf_ticker(r["Symbol"], r["Exchange"]), axis=1)

    # Deduplicate by instrument
    uni = uni.drop_duplicates(subset=["YFTicker"]).reset_index(drop=True)
    return uni


def chunked(iterable, size):
    it = list(iterable)
    for i in range(0, len(it), size):
        yield it[i:i + size]


def get_price_history(
    tickers: List[str],
    lookback_days: int,
    interval: str,
    max_batch: int = 50,
    pause_seconds: float = 0.5
) -> Dict[str, pd.DataFrame]:
    period = f"{lookback_days}d"
    out: Dict[str, pd.DataFrame] = {}

    for batch in chunked(tickers, max_batch):
        data = None
        for attempt in range(1, 3):
            try:
                data = yf.download(
                    tickers=batch,
                    period=period,
                    interval=interval,
                    group_by="ticker",
                    auto_adjust=True,
                    threads=True,
                    progress=False
                )
                break
            except Exception:
                data = None
                time.sleep(1.0)

        if data is None:
            continue

        if isinstance(data.columns, pd.MultiIndex):
            for t in data.columns.levels[0]:
                try:
                    df_t = data[t].dropna(how="all").copy()
                    if not df_t.empty and "Close" in df_t.columns:
                        df_t.index.name = "Date"
                        out[t] = df_t
                except Exception:
                    pass
        else:
            if "Close" in data.columns:
                data.index.name = "Date"
                out[batch[0]] = data

        time.sleep(pause_seconds)

    out = {k: v for k, v in out.items() if len(v) >= max(30, math.ceil(lookback_days * 0.5))}
    return out


def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0.0)


def last_cross_up(fast: pd.Series, slow: pd.Series) -> Optional[pd.Timestamp]:
    diff = fast - slow
    cross = (diff > 0) & (diff.shift(1) <= 0)
    if cross.any():
        return cross[cross].index[-1]
    return None


def compute_indicators_for_ticker(
    df: pd.DataFrame,
    ma_windows: List[int],
    rsi_period: int,
    breakout_lookback: int,
    near_breakout_pct: float,
    golden_fast: int,
    golden_slow: int,
    golden_within_days: int
) -> Dict:
    out = {}
    c = df["Close"].copy()

    for w in ma_windows:
        df[f"MA{w}"] = c.rolling(w).mean()

    df["RSI"] = rsi_wilder(c, period=rsi_period)

    rolling_high = c.rolling(breakout_lookback, min_periods=2).max()
    prev_high = rolling_high.shift(1)
    latest_close = c.iloc[-1]
    latest_prev_high = prev_high.iloc[-1]
    is_breakout = bool(latest_close > latest_prev_high) if not np.isnan(latest_prev_high) else False
    near_breakout = bool(latest_close >= (latest_prev_high * (1 - near_breakout_pct / 100.0))) if not np.isnan(latest_prev_high) else False

    fast_ma = df[f"MA{golden_fast}"] if f"MA{golden_fast}" in df.columns else c.rolling(golden_fast).mean()
    slow_ma = df[f"MA{golden_slow}"] if f"MA{golden_slow}" in df.columns else c.rolling(golden_slow).mean()
    cross_date = last_cross_up(fast_ma, slow_ma)
    has_recent_golden = False
    if cross_date is not None:
        has_recent_golden = (df.index[-1] - cross_date).days <= golden_within_days

    out["Close"] = float(latest_close)
    out["RSI"] = float(df["RSI"].iloc[-1])
    out["Breakout"] = is_breakout
    out["NearBreakout"] = near_breakout
    out["GoldenCrossDate"] = cross_date
    out["GoldenCrossRecent"] = has_recent_golden
    for w in ma_windows:
        out[f"MA{w}"] = float(df[f"MA{w}"].iloc[-1]) if f"MA{w}" in df.columns else np.nan

    out["PriceAboveAllMAs"] = all(out.get(f"MA{w}", np.nan) < latest_close for w in ma_windows if not np.isnan(out.get(f"MA{w}", np.nan)))
    sorted_windows = sorted([w for w in ma_windows if not np.isnan(out.get(f"MA{w}", np.nan))])
    aligned = True
    for i in range(len(sorted_windows)-1):
        if not (out[f"MA{sorted_windows[i]}"] > out[f"MA{sorted_windows[i+1]}"]):
            aligned = False
            break
    out["MAAligned"] = aligned if len(sorted_windows) >= 2 else False

    return out


def build_indicator_table(
    prices_store: Dict[str, pd.DataFrame],
    ma_windows: List[int],
    rsi_period: int,
    breakout_lookback: int,
    near_breakout_pct: float,
    golden_fast: int,
    golden_slow: int,
    golden_within_days: int
) -> pd.DataFrame:
    rows = []
    for yfticker, df in prices_store.items():
        try:
            snap = compute_indicators_for_ticker(
                df=df.copy(),
                ma_windows=ma_windows,
                rsi_period=int(rsi_period),
                breakout_lookback=int(breakout_lookback),
                near_breakout_pct=float(near_breakout_pct),
                golden_fast=int(golden_fast),
                golden_slow=int(golden_slow),
                golden_within_days=int(golden_within_days)
            )
            snap["yfticker"] = yfticker
            rows.append(snap)
        except Exception:
            continue
    return pd.DataFrame(rows)


def merge_with_universe(ind_table: pd.DataFrame, universe_df: pd.DataFrame) -> pd.DataFrame:
    if ind_table.empty:
        return ind_table
    merged = ind_table.merge(
        universe_df[["YFTicker", "Symbol", "Company Name", "Industry", "Index", "Exchange"]],
        left_on="yfticker", right_on="YFTicker", how="left"
    ).drop(columns=["YFTicker"])
    return merged.sort_values(by=["Exchange", "Symbol"]).reset_index(drop=True)


def plot_stock(df: pd.DataFrame, title: str, ma_windows: List[int], rsi_period: int):
    c = df["Close"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=c, mode="lines", name="Close"))
    for w in sorted(set(ma_windows)):
        fig.add_trace(go.Scatter(x=df.index, y=c.rolling(w).mean(), mode="lines", name=f"MA{w}"))
    fig.update_layout(
        title=title,
        xaxis_title="Date", yaxis_title="Price", height=420, margin=dict(l=40, r=10, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )

    rsi = rsi_wilder(c, period=rsi_period)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df.index, y=rsi, mode="lines", name=f"RSI({rsi_period})"))
    fig2.add_hrect(y0=30, y1=70, line_width=0, fillcolor="lightgray", opacity=0.2)
    fig2.update_layout(
        title=f"RSI({rsi_period})",
        xaxis_title="Date", yaxis_title="RSI", height=240, margin=dict(l=40, r=10, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        yaxis=dict(range=[0, 100])
    )
    return fig, fig2

# -----------------------
# Sidebar — Fetch controls
# -----------------------
st.sidebar.header("Stocks Data Fetch")

indices_choice = st.sidebar.multiselect(
    "Indices",
    options=["NIFTY Smallcap 250", "NIFTY Midcap 150", "NIFTY 500", "BSE SmallCap"],
    default=["NIFTY Smallcap 250", "NIFTY Midcap 150"],
    help="Add BSE SmallCap and/or NIFTY 500 to expand the scan universe."
)

col_a, col_b = st.sidebar.columns(2)
lookback_days = col_a.number_input("Lookback (days)", min_value=30, max_value=800, value=400, step=10)
interval = col_b.selectbox("Interval", options=["1d", "1h", "30m", "15m", "5m", "1m"], index=0)

CACHE_DIR = "cache"
UNIVERSE_CACHE = os.path.join(CACHE_DIR, "universe.pkl")
PRICES_CACHE = os.path.join(CACHE_DIR, "prices.pkl")

def save_cache(universe_df, prices):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(UNIVERSE_CACHE, "wb") as f:
        pickle.dump(universe_df, f)
    with open(PRICES_CACHE, "wb") as f:
        pickle.dump(prices, f)

def load_cache():
    if os.path.exists(UNIVERSE_CACHE) and os.path.exists(PRICES_CACHE):
        with open(UNIVERSE_CACHE, "rb") as f:
            universe_df = pickle.load(f)
        with open(PRICES_CACHE, "rb") as f:
            prices = pickle.load(f)
        return universe_df, prices
    return None, None

def clear_cache():
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
        
# -----------------------
# Sidebar controls
# -----------------------
fetch_btn = st.sidebar.button("📥 Fetch latest (clear & reload)")
clear_cache_btn = st.sidebar.button("🗑️ Clear cache")
auto_fetch = st.sidebar.checkbox("Auto-fetch on page load", value=True)

def do_fetch():
    with st.spinner("Fetching constituents…"):
        universe_df = build_universe(indices_choice)
    yf_tickers = universe_df["YFTicker"].dropna().unique().tolist()
    with st.spinner(f"Downloading {len(yf_tickers)} tickers price history…"):
        prices = get_price_history(yf_tickers, lookback_days=int(lookback_days), interval=interval)
    st.session_state[SS["universe"]] = universe_df
    st.session_state[SS["prices"]] = prices
    st.session_state[SS["last_fetch"]] = dt.datetime.now()
    save_cache(universe_df, prices) 

# -----------------------
# Init session state keys safely
# -----------------------
for k in [SS["universe"], SS["prices"], SS["last_fetch"]]:
    if k not in st.session_state:
        st.session_state[k] = None

# -----------------------
# Cache + fetch logic
# -----------------------
if clear_cache_btn:
    clear_cache()
    for k in [SS["universe"], SS["prices"], SS["last_fetch"]]:
        st.session_state[k] = None
    st.success("Cache cleared. Click *Fetch latest* to reload fresh data.")

elif fetch_btn:
    do_fetch()

elif st.session_state.get(SS["universe"]) is None:
    # Try loading cache first
    universe_df, prices = load_cache()
    if universe_df is not None and prices is not None:
        st.session_state[SS["universe"]] = universe_df
        st.session_state[SS["prices"]] = prices
    elif auto_fetch:
        do_fetch()



universe_df = st.session_state[SS["universe"]]
prices_store = st.session_state.get(SS["prices"], {})

if not prices_store:
    st.warning("No price data available yet (fetch may have failed). Try *Fetch latest* again.")
    st.stop()

# -----------------------
# Filters (local only)
# -----------------------
st.sidebar.header("Technical Filters (local)")

# Multi-MA selection
default_mas = [20, 50, 200]
ma_string = st.sidebar.text_input("MAs (comma-sep)", value="20,50,200", help="e.g., 10,20,50,100,200")
try:
    ma_windows = sorted({int(x.strip()) for x in ma_string.split(",") if x.strip()})
    if not ma_windows:
        ma_windows = default_mas
except Exception:
    ma_windows = default_mas

require_price_above_all_ma = st.sidebar.checkbox("Price above all selected MAs", value=True)
require_ma_alignment = st.sidebar.checkbox("MA alignment (short > long)", value=False)

# RSI
rsi_period = st.sidebar.number_input("RSI period (days)", min_value=5, max_value=50, value=14, step=1)
rsi_min, rsi_max = st.sidebar.slider("RSI range", min_value=0, max_value=100, value=(40, 80))

# Breakout
st.sidebar.subheader("Early Breakout")
breakout_lookback = st.sidebar.number_input("Breakout lookback (days)", min_value=5, max_value=100, value=20, step=1)
near_breakout_pct = st.sidebar.number_input("Within X% of breakout", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
require_breakout = st.sidebar.checkbox("Require fresh breakout (close > prior N-day high)", value=False)
allow_near_breakout = st.sidebar.checkbox("Or near breakout (within X%)", value=True)

# Golden Cross
st.sidebar.subheader("Golden Cross")
golden_fast = st.sidebar.number_input("Fast MA", min_value=5, max_value=100, value=50, step=1)
golden_slow = st.sidebar.number_input("Slow MA", min_value=20, max_value=400, value=200, step=5)
golden_window = st.sidebar.number_input("Cross happened within N days", min_value=1, max_value=100, value=20, step=1)
require_golden = st.sidebar.checkbox("Require recent Golden Cross", value=False)

# Sector filter
all_sectors = sorted([s for s in universe_df["Industry"].dropna().unique()])
selected_sectors = st.sidebar.multiselect("Industry (optional)", options=all_sectors, default=[])

# Compute indicators locally
with st.spinner("Computing indicators locally…"):
    ind_table = build_indicator_table(
        prices_store=prices_store,
        ma_windows=ma_windows,
        rsi_period=int(rsi_period),
        breakout_lookback=int(breakout_lookback),
        near_breakout_pct=float(near_breakout_pct),
        golden_fast=int(golden_fast),
        golden_slow=int(golden_slow),
        golden_within_days=int(golden_window)
    )
    merged = merge_with_universe(ind_table, universe_df)

# Apply filters
df = merged.copy()
if selected_sectors:
    df = df[df["Industry"].isin(selected_sectors)]

df = df[(df["RSI"] >= rsi_min) & (df["RSI"] <= rsi_max)]

if require_price_above_all_ma and "PriceAboveAllMAs" in df.columns:
    df = df[df["PriceAboveAllMAs"] == True]

if require_ma_alignment and "MAAligned" in df.columns:
    df = df[df["MAAligned"] == True]

if require_golden and "GoldenCrossRecent" in df.columns:
    df = df[df["GoldenCrossRecent"] == True]

if require_breakout and "Breakout" in df.columns:
    df = df[df["Breakout"] == True]
elif allow_near_breakout and "NearBreakout" in df.columns:
    df = df[df["Breakout"] | df["NearBreakout"]]

# -----------------------
# Main UI
# -----------------------
colL, colR = st.columns([2, 1])
with colL:
    st.subheader("Technically strong candidates")
    st.caption("Filters apply to the locally cached dataset only. Use *Fetch latest* to refresh.")
    if SS["last_fetch"] in st.session_state and st.session_state[SS["last_fetch"]]:
        st.text(f"Data last fetched: {st.session_state[SS['last_fetch']].strftime('%Y-%m-%d %H:%M:%S')}")

    show_cols = ["Symbol", "Company Name", "Industry", "Index", "Exchange", "Close"]
    show_cols += [c for c in df.columns if c.startswith("MA")]
    show_cols += ["RSI", "Breakout", "NearBreakout", "GoldenCrossRecent", "GoldenCrossDate"]
    show_cols = [c for c in show_cols if c in df.columns]
    st.dataframe(df[show_cols].sort_values(["Exchange","Symbol"]), use_container_width=True)

    csv_bytes = df[show_cols].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download filtered list (CSV)", data=csv_bytes, file_name="filtered_candidates.csv", mime="text/csv")

with colR:
    st.subheader("Chart a stock")
    choices_df = (df if not df.empty else merged)[["yfticker","Symbol","Company Name","Exchange"]].dropna()
    if choices_df.empty:
        st.info("No symbols match the current filters. Adjust filters or refetch.")
    else:
        labels = {row["yfticker"]: f"{row['Company Name']} — [{row['Symbol']}] ({row['Exchange']})" for _, row in choices_df.iterrows()}
        choice = st.selectbox("Pick a company", options=list(labels.keys()), format_func=lambda k: labels.get(k, k))
        if choice in prices_store:
            title = labels.get(choice, choice)
            fig_price, fig_rsi = plot_stock(prices_store[choice], title, ma_windows=ma_windows, rsi_period=int(rsi_period))
            st.plotly_chart(fig_price, use_container_width=True)
            st.plotly_chart(fig_rsi, use_container_width=True)
        else:
            st.info("No chart data for this symbol in the cache (try refetch).")

st.markdown("---")
st.caption("""
Notes:
- NIFTY constituents from Nifty Indices CSVs (Smallcap 250, Midcap 150, NIFTY 500); BSE SmallCap via Screener (static) with BSE page as fallback.
- Yahoo tickers use .NS (NSE) and .BO (BSE scrip codes, e.g., 500325.BO).
- Prices via yfinance are delayed; plug your real-time provider into get_price_history() if needed.
- Fetch happens only when you click *Fetch latest*; all filters are local & instant.
""")
