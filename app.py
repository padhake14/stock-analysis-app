# app.py
# Indian Stock Analyzer — NSE (NIFTY Smallcap 250, Midcap 150, NIFTY 500) + BSE SmallCap
# Fetch-on-demand only; all filters are local. Supports multi-MA, RSI ranges, early breakout, golden cross, sector filter, charts & CSV export.

import io
import re
import time
import math
import json
import requests
import datetime as dt
import os
import pickle
import shutil
import urllib.parse
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from bs4 import BeautifulSoup

st.set_page_config(
    page_title="Indian Stock Analyzer — NSE & BSE Small/Mid",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root {
  --ink: #e2e8f0;
  --muted: #94a3b8;
  --panel: rgba(255, 255, 255, 0.06);
  --card: rgba(255, 255, 255, 0.1);
  --border: rgba(255, 255, 255, 0.18);
  --accent: #0ea5e9;
  --accent-2: #16a34a;
  --accent-3: #f97316;
}
[data-testid="stAppViewContainer"] {
  background: radial-gradient(circle at 18% 20%, rgba(14, 165, 233, 0.18), transparent 28%),
              radial-gradient(circle at 82% 14%, rgba(22, 163, 74, 0.18), transparent 24%),
              linear-gradient(135deg, #0b1220 0%, #0c182c 40%, #0b1220 100%);
  color: var(--ink);
  font-family: 'Space Grotesk', 'Inter', system-ui, -apple-system, sans-serif;
}
[data-testid="stSidebar"] > div:first-child {
  background: rgba(7, 11, 22, 0.78);
  backdrop-filter: blur(14px);
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {
  color: var(--ink);
}
.block-container {
  padding: 1.6rem 2.3rem 2rem;
}
h1, h2, h3, h4 {
  color: #f8fafc;
  letter-spacing: -0.02em;
}
p, label, span, .stCaption, .stText, .stMarkdown {
  color: var(--ink);
}
.hero {
  background: linear-gradient(135deg, rgba(14,165,233,0.18), rgba(22,163,74,0.15));
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 1rem 1.4rem;
  box-shadow: 0 12px 35px rgba(0,0,0,0.25);
  display: grid;
  grid-template-columns: minmax(240px, 1.15fr) minmax(280px, 1fr);
  gap: 1.2rem;
  align-items: center;
}
.hero h1 {
  margin: 0.2rem 0 0.4rem 0;
  font-size: 1.9rem;
}
.hero .eyebrow {
  text-transform: uppercase;
  letter-spacing: 0.22em;
  font-size: 0.74rem;
  color: var(--muted);
  margin-bottom: 0.2rem;
}
.pill-row { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.35rem; }
.pill {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.3rem 0.65rem;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid var(--border);
  font-size: 0.82rem;
  color: var(--ink);
}
.hero p { color: var(--ink); }
.hero .stat-grid { margin-top: 0; }
.hero .stat-card { background: rgba(7, 11, 22, 0.6); }
.stat-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 0.75rem;
  margin-top: 0.8rem;
}
.stat-card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 0.85rem 1rem;
  box-shadow: 0 8px 24px rgba(0,0,0,0.22);
}
.stat-label { color: var(--muted); font-size: 0.85rem; margin-bottom: 0.2rem; }
.stat-value { color: #f8fafc; font-size: 1.4rem; font-weight: 700; }
.stat-note { color: var(--muted); font-size: 0.85rem; }
.glass-panel {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 0.9rem 1rem;
  box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}
div[data-testid="stDataFrame"], div[data-testid="stDataEditor"] {
  border: 1px solid #1f2a3d;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 10px 30px rgba(0,0,0,0.2);
}
button[kind="primary"], .stButton button {
  border-radius: 999px;
  border: 1px solid transparent;
  background: linear-gradient(135deg, #0ea5e9, #22c55e);
  color: #0b1220;
  font-weight: 700;
  box-shadow: 0 12px 24px rgba(14,165,233,0.25);
}
.stDownloadButton button {
  border-radius: 10px;
  background: #0f172a;
  color: #f8fafc;
  border: 1px solid var(--border);
}
div[data-baseweb="select"] > div {
  border-radius: 12px;
  border: 1px solid #1f2a3d;
}
div[data-baseweb="input"] > div {
  border-radius: 12px;
  border: 1px solid #1f2a3d;
}
.stSlider > div > div > div {
  color: #f8fafc;
}
@media (max-width: 900px) {
  .hero {
    grid-template-columns: 1fr;
  }
}
</style>
"""
st.markdown(CUSTOM_STYLE, unsafe_allow_html=True)

SS = {
    "universe": "universe_df",
    "prices": "prices_store",
    "last_fetch": "last_fetch_ts",
    "fetch_errors": "last_fetch_errors",
}

# -----------------------
# Data sources
# -----------------------
def nse_csv_urls(slug: str) -> List[str]:
    safe_slug = urllib.parse.quote(str(slug))
    return [
        f"https://niftyindices.com/IndexConstituent/ind_{slug}list.csv",
        f"https://nsearchives.nseindia.com/content/indices/ind_{slug}list.csv",
        f"https://www.nseindia.com/api/equity-stockIndices?csv=true&index={safe_slug}&selectValFormat=crores",
    ]

# Broad + sectoral NSE indices (expandable). Values allow multiple slug fallbacks.
NSE_INDEX_SLUGS = {
    "NIFTY 100": ["nifty100"],
    "NIFTY 200": ["nifty200"],
    "NIFTY 500": ["nifty500"],
    "NIFTY Midcap 50": ["niftymidcap50"],
    "NIFTY Midcap 100": ["niftymidcap100"],
    "NIFTY Midcap 150": ["niftymidcap150"],
    "NIFTY Smallcap 50": ["niftysmallcap50"],
    "NIFTY Smallcap 100": ["niftysmallcap100"],
    "NIFTY Smallcap 250": ["niftysmallcap250"],
    "NIFTY MidSmallcap 400": ["niftymidsmallcap400"],
    "NIFTY500 Multicap 50:25:25": ["nifty500multicap502525", "NIFTY500 Multicap 50:25:25"],
    "NIFTY Financial Services 25/50": ["niftyfinserv25_50", "NIFTY Financial Services 25/50"],
    "NIFTY Financial Services Ex-Bank": ["niftyfinservexbank", "Financial Services Ex-Bank"],
    "NIFTY MidSmall Healthcare": ["MidSmall Healthcare"],
    "NIFTY MidSmall Financial Services": ["niftymidsmallfinancialservices", "MidSmall Financial Services"],
    "NIFTY MidSmall IT & Telecom": ["niftymidsmallittelecom", "MidSmall IT & Telecom"],
    "NIFTY CAPITAL MARKETS": ["niftycapitalmarkets", "NIFTY CAPITAL MARKETS"],
    "NIFTY Chemicals": ["niftychemicals"],
    "NIFTY500 Healthcare": ["NIFTY500 HEALTHCARE"],
    "NIFTY Auto": ["niftyauto"],
    "NIFTY FMCG": ["niftyfmcg"],
    "NIFTY IT": ["niftyit"],
    "NIFTY Media": ["niftymedia"],
    "NIFTY Metal": ["niftymetal"],
    "NIFTY Pharma": ["niftypharma"],
    "NIFTY PSU Bank": ["niftypsubank"],
    "NIFTY Private Bank": ["niftyprivatebank"],
    "NIFTY Realty": ["niftyrealty"],
    "NIFTY Healthcare Index": ["niftyhealthcare", "niftyhealthcareindex"],
    "NIFTY Consumer Durables": ["niftyconsumerdurables"],
    "NIFTY Oil & Gas": ["niftyoilgas"],
}

CSV_SOURCES = {}
for name, slugs in NSE_INDEX_SLUGS.items():
    urls = []
    for slug in slugs:
        urls.extend(nse_csv_urls(slug))
    CSV_SOURCES[name] = urls

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
            df = pd.read_csv(io.BytesIO(resp.content), on_bad_lines="skip")
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

            df = df.reset_index(drop=True)
            if df.empty:
                raise ValueError("Empty table after parsing")
            return df
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


REQUIRED_UNIVERSE_COLS = [
    "Company Name",
    "Industry",
    "Symbol",
    "Series",
    "ISIN Code",
    "Index",
    "Exchange",
    "YFTicker",
]


def ensure_universe_columns(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Guarantee downstream columns exist even if a fallback source omits them."""
    if df is None or not isinstance(df, pd.DataFrame):
        return df
    df = df.copy()
    for col in REQUIRED_UNIVERSE_COLS:
        if col not in df.columns:
            df[col] = np.nan

    if "Symbol" in df.columns:
        df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
    if "Exchange" in df.columns:
        df["Exchange"] = df["Exchange"].astype(str).str.strip().str.upper()
    if "Company Name" in df.columns and "Symbol" in df.columns:
        df["Company Name"] = df["Company Name"].fillna(df["Symbol"])

    def _maybe_yf(row):
        if pd.isna(row.get("Symbol")) or pd.isna(row.get("Exchange")):
            return np.nan
        return to_yf_ticker(row["Symbol"], row["Exchange"])

    if "YFTicker" not in df.columns:
        df["YFTicker"] = df.apply(_maybe_yf, axis=1)
    else:
        df["YFTicker"] = df["YFTicker"].where(df["YFTicker"].notna(), df.apply(_maybe_yf, axis=1))
    return df


@st.cache_data(show_spinner=False, ttl=900)
def fetch_delivery_snapshot(symbols: List[str]) -> pd.DataFrame:
    """
    Fetch latest NSE delivery stats (delivery volume and delivery %) for given symbols.
    Best-effort; returns empty DataFrame if NSE blocks requests.
    """
    if not symbols:
        return pd.DataFrame()
    session = requests.Session()
    session.headers.update(HEADERS)
    try:
        # Prime cookies
        session.get("https://www.nseindia.com", timeout=10)
    except Exception:
        pass

    rows = []
    for sym in symbols:
        try:
            url = f"https://www.nseindia.com/api/quote-equity?symbol={urllib.parse.quote(sym)}"
            r = session.get(url, timeout=12)
            r.raise_for_status()
            data = r.json()
            swdp = data.get("securityWiseDP", [])
            if not swdp:
                continue
            dp = swdp[0]
            rows.append({
                "Symbol": sym,
                "DeliveryVolume": float(dp.get("deliveryQuantity", np.nan)),
                "TradedVolume": float(dp.get("quantityTraded", np.nan)),
                "DeliveryPct": float(dp.get("deliveryToTradedQuantity", np.nan)),
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def build_universe(selected_indices: List[str]) -> pd.DataFrame:
    frames = []
    errors = []
    for idx in selected_indices:
        try:
            if idx in CSV_SOURCES:
                frames.append(fetch_constituents_nse(idx))
            elif idx == "BSE SmallCap":
                frames.append(fetch_bse_smallcap_constituents())
        except Exception as e:
            errors.append(f"{idx}: {e}")

    st.session_state[SS["fetch_errors"]] = errors

    if not frames:
        msg = "No constituents loaded."
        if errors:
            msg += " Errors:\n - " + "\n - ".join(errors)
        raise RuntimeError(msg)

    uni = pd.concat(frames, ignore_index=True)

    # Add Yahoo Finance tickers for merge/downloads
    if "YFTicker" not in uni.columns:
        uni["YFTicker"] = uni.apply(lambda r: to_yf_ticker(r["Symbol"], r["Exchange"]), axis=1)

    uni = ensure_universe_columns(uni)
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
    use_max: bool = False,
    max_batch: int = 50,
    pause_seconds: float = 0.5
) -> Dict[str, pd.DataFrame]:
    period = "max" if use_max else f"{lookback_days}d"
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

    min_len = 30 if use_max else max(30, math.ceil(lookback_days * 0.5))
    out = {k: v for k, v in out.items() if len(v) >= min_len}
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

    if "Volume" in df.columns:
        latest_vol = float(df["Volume"].iloc[-1])
        avg_vol_20 = float(df["Volume"].tail(20).mean()) if len(df) >= 1 else np.nan
        out["LatestVolume"] = latest_vol
        out["AvgVolume20"] = avg_vol_20
        out["VolumeVsAvg20"] = (latest_vol / avg_vol_20) if avg_vol_20 and not np.isnan(avg_vol_20) and avg_vol_20 > 0 else np.nan
    else:
        out["LatestVolume"] = np.nan
        out["AvgVolume20"] = np.nan
        out["VolumeVsAvg20"] = np.nan

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


def _to_lw_series(series: pd.Series) -> List[Dict]:
    data = []
    if series is None:
        return data
    for ts, val in series.items():
        if pd.isna(val):
            continue
        try:
            epoch = int(pd.Timestamp(ts).timestamp())
        except Exception:
            continue
        data.append({"time": epoch, "value": float(val)})
    return data


def _to_lw_ohlc(df: pd.DataFrame) -> List[Dict]:
    data = []
    if df is None or df.empty:
        return data
    required = ["Open", "High", "Low", "Close"]
    if not all(col in df.columns for col in required):
        return data
    for ts, row in df.iterrows():
        if any(pd.isna(row.get(c)) for c in required):
            continue
        try:
            epoch = int(pd.Timestamp(ts).timestamp())
        except Exception:
            continue
        data.append({
            "time": epoch,
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
        })
    return data


def _to_lw_volume(df: pd.DataFrame) -> List[Dict]:
    if df is None or df.empty or "Volume" not in df.columns:
        return []
    out = []
    has_ohlc = all(col in df.columns for col in ["Open", "Close"])
    for ts, row in df.iterrows():
        vol = row.get("Volume", np.nan)
        if pd.isna(vol):
            continue
        try:
            epoch = int(pd.Timestamp(ts).timestamp())
        except Exception:
            continue
        color = "#94a3b8"
        if has_ohlc:
            try:
                color = "#22c55e" if float(row["Close"]) >= float(row["Open"]) else "#ef4444"
            except Exception:
                pass
        out.append({"time": epoch, "value": float(vol), "color": color})
    return out


def _infer_base_minutes(df: pd.DataFrame) -> Optional[float]:
    if df is None or df.empty:
        return None
    idx = pd.to_datetime(df.index)
    diffs = idx.to_series().diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        return None
    mins = diffs.dt.total_seconds() / 60.0
    return float(mins.min()) if not mins.empty else None


def _resample_prices(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    return df.resample(rule, label="right", closed="right").agg(agg).dropna(how="any")


def prepare_chart_timeframe(df: pd.DataFrame, timeframe: str) -> (pd.DataFrame, Optional[str]):
    """Return data adjusted to requested timeframe and an optional note."""
    if df is None or df.empty:
        return df, None
    timeframe = (timeframe or "1d").lower()
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    base_minutes = _infer_base_minutes(df)
    note = None

    try:
        if timeframe == "15m":
            if base_minutes and base_minutes <= 15:
                if base_minutes < 15:
                    df = _resample_prices(df, "15T")
            else:
                note = "15m view needs intraday fetch (<=15m interval). Showing original data instead."
        elif timeframe == "1d":
            if base_minutes and base_minutes < 1440:
                df = _resample_prices(df, "1D")
        elif timeframe in ("1w", "1wk", "1week", "weekly"):
            if base_minutes and base_minutes <= 1440:
                df = _resample_prices(df, "1W")
            else:
                note = "Weekly view needs at least daily data. Showing original data."
        elif timeframe in ("1mo", "1m", "monthly"):
            if base_minutes and base_minutes <= 1440:
                df = _resample_prices(df, "1M")
            else:
                note = "Monthly view needs at least daily data. Showing original data."
    except Exception:
        note = "Could not resample to requested timeframe; showing original data."

    return df, note


def render_lightweight_charts(
    df: pd.DataFrame,
    title: str,
    ma_windows: List[int],
    rsi_period: int,
    price_chart_type: str = "Area",
    chart_timeframe: str = "1d"
):
    """Render price + RSI using TradingView Lightweight Charts via Streamlit components."""
    if df is None or df.empty:
        st.info("No data available for this symbol.")
        return

    df, note = prepare_chart_timeframe(df, chart_timeframe)
    if note:
        st.caption(note)

    ma_windows = sorted(set(ma_windows))

    for w in ma_windows:
        df[f"MA{w}"] = df["Close"].rolling(w).mean()

    rsi_series = rsi_wilder(df["Close"], period=rsi_period)

    price_chart_type = (price_chart_type or "Area").strip().title()
    price_data_line = _to_lw_series(df["Close"])
    price_data_ohlc = _to_lw_ohlc(df)
    volume_data = _to_lw_volume(df)

    ma_data = []
    ma_colors = ["#22c55e", "#0ea5e9", "#f97316", "#a855f7", "#ef4444", "#14b8a6"]
    for idx, w in enumerate(ma_windows):
        series_data = _to_lw_series(df[f"MA{w}"])
        if not series_data:
            continue
        ma_data.append({
            "name": f"MA{w}",
            "color": ma_colors[idx % len(ma_colors)],
            "data": series_data
        })

    rsi_data = _to_lw_series(pd.Series(rsi_series, index=df.index))
    chart_id = f"lw-{abs(hash(title + chart_timeframe)) % 10_000_000}"

    html = f"""
    <div id="{chart_id}-wrap" style="width: 100%; background: #0b1220; border: 1px solid #1f2a3d; border-radius: 12px; padding: 10px; position: relative;">
      <div style="display: flex; justify-content: flex-end; gap: 8px; margin-bottom: 6px;">
        <button id="{chart_id}-fs-btn" style="background: #111827; color: #e2e8f0; border: 1px solid #1f2a3d; border-radius: 8px; padding: 6px 10px; cursor: pointer;">⤢ Fullscreen</button>
      </div>
      <div id="{chart_id}-price" style="height: 380px;"></div>
      <div id="{chart_id}-volume" style="height: 140px; margin-top: 6px;"></div>
      <div id="{chart_id}-rsi" style="height: 200px; margin-top: 12px;"></div>
    </div>
    <script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
    <script>
      (function() {{
        const priceContainer = document.getElementById("{chart_id}-price");
        const volumeContainer = document.getElementById("{chart_id}-volume");
        const rsiContainer = document.getElementById("{chart_id}-rsi");
        const wrap = document.getElementById("{chart_id}-wrap");
        const fsBtn = document.getElementById("{chart_id}-fs-btn");

        const baseLayout = {{
          layout: {{ background: {{ type: 'solid', color: '#0b1220' }}, textColor: '#e2e8f0' }},
          grid: {{ vertLines: {{ color: '#1f2937' }}, horzLines: {{ color: '#1f2937' }} }},
          crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
          rightPriceScale: {{ borderColor: '#1f2937' }},
          timeScale: {{ borderColor: '#1f2937', timeVisible: true, secondsVisible: false }},
        }};

        const priceChart = LightweightCharts.createChart(priceContainer, Object.assign({{ height: 380 }}, baseLayout));
        const priceType = "{price_chart_type}";
        if (priceType === "Candles" && {json.dumps(bool(price_data_ohlc))}) {{
          const s = priceChart.addCandlestickSeries({{
            upColor: '#22c55e',
            downColor: '#ef4444',
            borderUpColor: '#22c55e',
            borderDownColor: '#ef4444',
            wickUpColor: '#22c55e',
            wickDownColor: '#ef4444',
          }});
          s.setData({json.dumps(price_data_ohlc)});
        }} else if (priceType === "Bars" && {json.dumps(bool(price_data_ohlc))}) {{
          const s = priceChart.addBarSeries({{
            upColor: '#22c55e',
            downColor: '#ef4444',
            thinBars: false,
          }});
          s.setData({json.dumps(price_data_ohlc)});
        }} else if (priceType === "Line") {{
          const s = priceChart.addLineSeries({{
            color: '#22c55e',
            lineWidth: 2,
          }});
          s.setData({json.dumps(price_data_line)});
        }} else {{
          const s = priceChart.addAreaSeries({{
            lineColor: '#22c55e',
            topColor: 'rgba(34,197,94,0.35)',
            bottomColor: 'rgba(34,197,94,0.05)',
            priceFormat: {{ type: 'price', precision: 2, minMove: 0.01 }},
          }});
          s.setData({json.dumps(price_data_line)});
        }}

        const maPayload = {json.dumps(ma_data)};
        maPayload.forEach(cfg => {{
          const s = priceChart.addLineSeries({{
            color: cfg.color,
            lineWidth: 2,
            title: cfg.name,
          }});
          s.setData(cfg.data);
        }});

        let volumeChart = null;
        if ({json.dumps(bool(volume_data))}) {{
          volumeChart = LightweightCharts.createChart(volumeContainer, Object.assign({{ height: 140 }}, baseLayout));
          const volSeries = volumeChart.addHistogramSeries({{
            priceFormat: {{ type: 'volume' }},
            color: '#94a3b8',
          }});
          volSeries.setData({json.dumps(volume_data)});
        }} else {{
          volumeContainer.style.display = "none";
        }}

        const rsiChart = LightweightCharts.createChart(rsiContainer, Object.assign({{ height: 200 }}, baseLayout));
        const rsiLine = rsiChart.addLineSeries({{ color: '#0ea5e9', lineWidth: 2 }});
        rsiLine.setData({json.dumps(rsi_data)});
        rsiChart.addHistogramSeries({{
          color: 'rgba(226,232,240,0.08)',
          priceFormat: {{ type: 'volume' }},
        }}).setData([]);
        rsiChart.applyOptions({{
          leftPriceScale: {{ visible: false }},
          rightPriceScale: {{ visible: true }},
          timeScale: {{ timeVisible: true, secondsVisible: false }},
          grid: {{ vertLines: {{ color: '#1f2937' }}, horzLines: {{ color: '#1f2937' }} }},
        }});
        rsiChart.addBaselineSeries({{
          baseValue: 50,
          topLineColor: 'rgba(34,197,94,0.4)',
          bottomLineColor: 'rgba(239,68,68,0.4)',
          baseLineColor: 'rgba(148,163,184,0.5)',
          lineWidth: 1,
        }}).setData([]);

        const rsiBand = rsiChart.addLineSeries({{ color: 'rgba(148,163,184,0.4)', lineWidth: 1, lineStyle: 2 }});
        const bandData = [];
        const rsiSrc = {json.dumps(rsi_data)};
        rsiSrc.forEach(p => bandData.push({{ time: p.time, value: 70 }}));
        rsiBand.setData(bandData);
        const rsiBandLow = rsiChart.addLineSeries({{ color: 'rgba(148,163,184,0.4)', lineWidth: 1, lineStyle: 2 }});
        const bandDataLow = [];
        rsiSrc.forEach(p => bandDataLow.push({{ time: p.time, value: 30 }}));
        rsiBandLow.setData(bandDataLow);

        const resizeObserver = new ResizeObserver(entries => {{
          const width = entries[0].contentRect.width;
          priceChart.applyOptions({{ width }});
          if (volumeChart) volumeChart.applyOptions({{ width }});
          rsiChart.applyOptions({{ width }});
        }});
        resizeObserver.observe(wrap);

        const syncResize = () => {{
          const {{ left, right }} = priceChart.timeScale().getVisibleRange() || {{}};
          if (left && right) {{
            rsiChart.timeScale().setVisibleRange({{ from: left, to: right }});
            if (volumeChart) volumeChart.timeScale().setVisibleRange({{ from: left, to: right }});
          }}
        }};
        priceChart.timeScale().subscribeVisibleTimeRangeChange(syncResize);

        const toggleFullscreen = () => {{
          if (!document.fullscreenElement) {{
            if (wrap.requestFullscreen) wrap.requestFullscreen();
          }} else {{
            if (document.exitFullscreen) document.exitFullscreen();
          }}
        }};
        const updateFsText = () => {{
          fsBtn.textContent = document.fullscreenElement ? "⤢ Exit Fullscreen" : "⤢ Fullscreen";
        }};
        fsBtn.addEventListener("click", toggleFullscreen);
        document.addEventListener("fullscreenchange", () => {{
          updateFsText();
          resizeObserver.observe(wrap);
        }});
        updateFsText();
      }})();
    </script>
    """
    st.components.v1.html(html, height=620, scrolling=False)

# -----------------------
# Sidebar — Fetch controls
# -----------------------
st.sidebar.header("Stocks Data Fetch")

index_options = list(NSE_INDEX_SLUGS.keys()) + ["BSE SmallCap"]
default_indices = ["NIFTY Smallcap 250", "NIFTY Midcap 150"]

# Restore indices from URL params if present; otherwise fall back to defaults
query_params = st.query_params
indices_from_url: List[str] = []
if "indices" in query_params:
    for val in query_params.get("indices", []):
        indices_from_url.extend([v.strip() for v in val.split(",") if v.strip()])

def _sanitize_indices(raw_list: List[str], fallback: List[str]) -> List[str]:
    seen = set()
    ordered = []
    for item in raw_list:
        if item in index_options and item not in seen:
            ordered.append(item)
            seen.add(item)
    if not ordered:
        ordered = [i for i in fallback if i in index_options]
    return ordered

preselected_indices = _sanitize_indices(indices_from_url, fallback=default_indices)

if "indices_choice" not in st.session_state:
    st.session_state["indices_choice"] = preselected_indices
else:
    st.session_state["indices_choice"] = _sanitize_indices(st.session_state["indices_choice"], fallback=default_indices)

indices_choice = st.sidebar.multiselect(
    "Indices",
    options=index_options,
    default=st.session_state["indices_choice"],
    key="indices_choice",
    help="Add sectoral and broad NIFTY indices (plus BSE SmallCap) to expand the scan universe."
)

# Persist current selection into the URL so page refresh retains it
if indices_choice:
    st.query_params["indices"] = ",".join(indices_choice)
elif "indices" in st.query_params:
    st.query_params.pop("indices", None)

use_max_history = st.sidebar.checkbox("Use max available history", value=False, help="Fetch the full history Yahoo Finance allows for the chosen interval.")
col_a, col_b = st.sidebar.columns(2)
lookback_days = col_a.number_input("Lookback (days)", min_value=30, max_value=800, value=400, step=10, disabled=use_max_history)
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
        return ensure_universe_columns(universe_df), prices
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
    lookback_display = "max available" if use_max_history else f"{int(lookback_days)}d"
    with st.spinner(f"Downloading {len(yf_tickers)} tickers price history ({lookback_display})…"):
        prices = get_price_history(
            yf_tickers,
            lookback_days=int(lookback_days) if not use_max_history else 0,
            interval=interval,
            use_max=use_max_history
        )
    st.session_state[SS["universe"]] = universe_df
    st.session_state[SS["prices"]] = prices
    st.session_state[SS["last_fetch"]] = dt.datetime.now()
    if SS["fetch_errors"] not in st.session_state:
        st.session_state[SS["fetch_errors"]] = []
    save_cache(universe_df, prices) 

# -----------------------
# Init session state keys safely
# -----------------------
for k in [SS["universe"], SS["prices"], SS["last_fetch"], SS["fetch_errors"]]:
    if k not in st.session_state:
        st.session_state[k] = None

# -----------------------
# Cache + fetch logic
# -----------------------
if clear_cache_btn:
    clear_cache()
    for k in [SS["universe"], SS["prices"], SS["last_fetch"], SS["fetch_errors"]]:
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
        st.session_state[SS["fetch_errors"]] = []
    elif auto_fetch:
        do_fetch()



universe_df = st.session_state[SS["universe"]]
universe_df = ensure_universe_columns(universe_df)
st.session_state[SS["universe"]] = universe_df
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
select_all_sectors = st.sidebar.checkbox("Select all sectors", value=False)
selected_sectors = st.sidebar.multiselect("Industry (optional)", options=all_sectors, default=[])
if select_all_sectors:
    selected_sectors = all_sectors

# Company filter (lets you zero-in on specific names)
company_query = st.sidebar.text_input(
    "Company name contains (optional)",
    value="",
    help="Case-insensitive match; e.g., typing 'bank' will show all banking names."
)
company_picks = st.sidebar.multiselect(
    "Or pick companies",
    options=sorted(universe_df["Company Name"].dropna().unique()),
    default=[]
)

st.sidebar.subheader("Volume / Delivery")
min_avg_vol = st.sidebar.number_input("Min 20-day avg volume", min_value=0, value=0, step=10000)
min_latest_vol = st.sidebar.number_input("Min latest volume", min_value=0, value=0, step=10000)
min_vol_vs_avg = st.sidebar.number_input("Min volume vs avg (x)", min_value=0.0, value=0.0, step=0.1, format="%.1f")
delivery_filter = st.sidebar.checkbox("Enable delivery filter (NSE only)", value=False)
min_delivery_pct = st.sidebar.slider("Min delivery %", min_value=0, max_value=100, value=0, step=5, disabled=not delivery_filter)
min_delivery_qty = st.sidebar.number_input("Min delivery quantity", min_value=0, value=0, step=10000, disabled=not delivery_filter)

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

# Delivery snapshot (NSE only) if requested
if delivery_filter:
    nse_symbols = merged.loc[merged["Exchange"] == "NSE", "Symbol"].dropna().unique().tolist()
    if nse_symbols:
        with st.spinner("Fetching NSE delivery snapshots…"):
            delivery_df = fetch_delivery_snapshot(nse_symbols)
        if delivery_df is not None and not delivery_df.empty:
            merged = merged.merge(delivery_df, on="Symbol", how="left")

# Apply filters
df = merged.copy()
if selected_sectors and not select_all_sectors:
    df = df[df["Industry"].isin(selected_sectors)]

if company_query.strip():
    needle = company_query.strip()
    df = df[df["Company Name"].fillna("").str.contains(needle, case=False, na=False)]

if company_picks:
    df = df[df["Company Name"].isin(company_picks)]

if min_avg_vol > 0 and "AvgVolume20" in df.columns:
    df = df[df["AvgVolume20"] >= min_avg_vol]

if min_latest_vol > 0 and "LatestVolume" in df.columns:
    df = df[df["LatestVolume"] >= min_latest_vol]

if min_vol_vs_avg > 0 and "VolumeVsAvg20" in df.columns:
    df = df[df["VolumeVsAvg20"] >= min_vol_vs_avg]

if delivery_filter:
    if min_delivery_pct > 0 and "DeliveryPct" in df.columns:
        df = df[df["DeliveryPct"] >= min_delivery_pct]
    if min_delivery_qty > 0 and "DeliveryVolume" in df.columns:
        df = df[df["DeliveryVolume"] >= min_delivery_qty]

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
# Hero + quick stats
# -----------------------
last_fetch_ts = st.session_state.get(SS["last_fetch"])
last_fetch_text = last_fetch_ts.strftime("%b %d, %H:%M") if last_fetch_ts else "Not fetched"
filtered_count = len(df)
universe_count = len(universe_df) if isinstance(universe_df, pd.DataFrame) else 0
indices_label = ", ".join(indices_choice) if indices_choice else "No indices selected"
active_indices = ", ".join(sorted(universe_df["Index"].dropna().unique())) if universe_df is not None and not universe_df.empty else "—"
active_exchanges = " / ".join(sorted(df["Exchange"].dropna().unique())) if not df.empty else "—"
lookback_label = "max available" if use_max_history else f"{int(lookback_days)}d"

st.markdown(f"""
<div class="hero">
  <div>
    <div class="eyebrow">NSE + BSE Technical Scanner</div>
    <h1>Technically strong stocks, at a glance</h1>
    <p>Scan smallcap & midcap universes with MA alignment, RSI bounds, early breakouts, and golden cross detections — entirely local after fetching.</p>
    <div class="pill-row">
      <span class="pill">Multi-MA & RSI</span>
      <span class="pill">Breakouts + Golden Cross</span>
      <span class="pill">{indices_label}</span>
      <span class="pill">{lookback_label} · {interval} data</span>
    </div>
  </div>
  <div class="stat-grid">
    <div class="stat-card">
      <div class="stat-label">Matches now</div>
      <div class="stat-value">{filtered_count}</div>
      <div class="stat-note">after current filters</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Universe loaded</div>
      <div class="stat-value">{universe_count}</div>
      <div class="stat-note">{active_indices}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Exchanges seen</div>
      <div class="stat-value">{active_exchanges}</div>
      <div class="stat-note">NSE / BSE coverage</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Last fetch</div>
      <div class="stat-value">{last_fetch_text}</div>
      <div class="stat-note">manual or cache</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

if st.session_state.get(SS["fetch_errors"]):
    st.warning("Some indices failed to load:\n- " + "\n- ".join(st.session_state[SS["fetch_errors"]]))

# -----------------------
# Main UI
# -----------------------
colL, colR = st.columns([2, 1])
with colL:
    st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
    st.subheader("Technically strong stocks")
    st.caption("Filters apply to the locally cached dataset only. Use *Fetch latest* to refresh. Check a row to preview its chart.")
    if SS["last_fetch"] in st.session_state and st.session_state[SS["last_fetch"]]:
        st.caption(f"Data last fetched: {st.session_state[SS['last_fetch']].strftime('%Y-%m-%d %H:%M:%S')}")

    show_cols = ["Symbol", "Company Name", "Industry", "Index", "Exchange", "Close"]
    show_cols += [c for c in df.columns if c.startswith("MA")]
    show_cols += [
        "RSI",
        "Breakout",
        "NearBreakout",
        "GoldenCrossRecent",
        "GoldenCrossDate",
        "LatestVolume",
        "AvgVolume20",
        "VolumeVsAvg20",
        "DeliveryPct",
        "DeliveryVolume",
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    table_source = df.sort_values(["Exchange", "Symbol"]).reset_index(drop=True)
    table_base = table_source[show_cols].reset_index(drop=True)
    table_cols = ["Select"] + show_cols + (["yfticker"] if "yfticker" in table_source.columns else [])
    table_for_ui = table_base.copy()
    table_for_ui.insert(0, "Select", False)
    if "yfticker" in table_source.columns:
        table_for_ui["yfticker"] = table_source["yfticker"].values

    column_config = {
        "Select": st.column_config.CheckboxColumn("Select", help="Check a row to load its chart")
    }
    if "yfticker" in table_cols:
        column_config["yfticker"] = st.column_config.TextColumn("YF Ticker", help="Used for charting")

    edited_table = st.data_editor(
        table_for_ui[table_cols],
        hide_index=True,
        use_container_width=True,
        column_config=column_config,
        key="main_table_editor"
    )

    selected_row = edited_table[edited_table["Select"]]
    if len(selected_row) > 1:
        st.caption("Multiple rows selected; showing the first checked row.")
    selected_from_table = selected_row.iloc[0] if not selected_row.empty else None

    csv_bytes = df[show_cols].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download filtered list (CSV)", data=csv_bytes, file_name="filtered_candidates.csv", mime="text/csv")
    st.markdown("</div>", unsafe_allow_html=True)

with colR:
    st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
    st.subheader("Chart a stock")
    choices_df = (df if not df.empty else merged)[["yfticker","Symbol","Company Name","Exchange"]].dropna()
    preselected = None
    preselected_label = None
    if selected_from_table is not None:
        preselected = selected_from_table.get("yfticker")
        if pd.notna(preselected):
            preselected_label = f"{selected_from_table.get('Company Name','')} — [{selected_from_table.get('Symbol','')}] ({selected_from_table.get('Exchange','')})"

    if choices_df.empty:
        st.info("No symbols match the current filters. Adjust filters or refetch.")
    else:
        labels = {row["yfticker"]: f"{row['Company Name']} — [{row['Symbol']}] ({row['Exchange']})" for _, row in choices_df.iterrows()}
        options = list(labels.keys())
        default_index = options.index(preselected) if preselected in options else 0
        if preselected_label:
            st.caption(f"Loaded from table: {preselected_label}")
        choice = st.selectbox(
            "Pick a company (or click a row in the table)",
            options=options,
            index=default_index,
            format_func=lambda k: labels.get(k, k)
        )
        if choice in prices_store:
            title = labels.get(choice, choice)
            chart_timeframe = st.radio(
                "Chart timeframe",
                options=["15m", "1d", "1w", "1mo"],
                index=1,
                horizontal=True,
                help="Shows price/RSI at the selected timeframe. Weekly/Monthly are built from daily+ data."
            )
            chart_style = st.radio(
                "Price chart style",
                options=["Area", "Line", "Candles", "Bars"],
                index=0,
                horizontal=True,
                help="Switch between area/line or OHLC-based candles/bars (needs OHLC data)."
            )
            render_lightweight_charts(
                prices_store[choice],
                f"{title} — {chart_timeframe}",
                ma_windows=ma_windows,
                rsi_period=int(rsi_period),
                price_chart_type=chart_style,
                chart_timeframe=chart_timeframe
            )
        else:
            st.info("No chart data for this symbol in the cache (try refetch).")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption("""
Disclaimer:
The information, charts, and analysis shared are for educational purposes only and should not be considered as financial or investment advice. 
Stock market investments are subject to risks, and you should consult with a qualified financial advisor before making any investment decisions. 
We are not responsible for any losses incurred based on the information provided
""")
