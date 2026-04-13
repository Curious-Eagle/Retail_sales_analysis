"""
fetch_live_data.py
──────────────────
Fetches real, live data from three public APIs and loads it into SQLite
so that analyse.py can run its SQL-based analysis.

Data sources (all free, no API keys required):
────────────────────────────────────────────────
  1. ONS Beta API  — UK Retail Sales Index, monthly, all sub-sectors
     https://api.beta.ons.gov.uk/v1/datasets/retail-sales-index/
     Real government statistics updated monthly.

  2. Yahoo Finance (via yfinance) — UK-listed retailer stock prices
     Major UK retailers: Tesco, M&S, Next, Sainsbury's, JD Sports,
     Ocado, Frasers Group, B&M, Greggs, WH Smith.
     Five years of daily OHLCV data + company info (market cap, P/E, etc.)

  3. Frankfurter API — GBP exchange rates (historical + live)
     https://api.frankfurter.app
     Free, no key. Used to show GBP strength vs EUR, USD, JPY.

Usage:
    pip install yfinance requests pandas
    python scripts/fetch_live_data.py
"""

import io
import json
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

DATA_DIR = Path("data")
DB_PATH  = DATA_DIR / "retail_live.db"
DATA_DIR.mkdir(exist_ok=True)

# Latest RSI CSV — the ONS keeps the URL stable; the version number in
# the path bumps with each release. We resolve it dynamically.
ONS_EDITIONS_URL = (
    "https://api.beta.ons.gov.uk/v1/datasets/"
    "retail-sales-index/editions/time-series/versions"
)

UK_RETAILERS = {
    "TSCO.L": "Tesco",
    "MKS.L":  "Marks & Spencer",
    "NXT.L":  "Next",
    "SBRY.L": "Sainsbury's",
    "JD.L":   "JD Sports",
    "OCDO.L": "Ocado",
    "FRAS.L": "Frasers Group",
    "BME.L":  "B&M European Value",
    "GRG.L":  "Greggs",
    "SMWH.L": "WH Smith",
}

FX_TARGETS = ["EUR", "USD", "JPY", "AUD", "CAD"]
FX_HISTORY_YEARS = 5


# ── HELPERS ───────────────────────────────────────────────────────────────────

def get(url: str, **kwargs) -> requests.Response:
    resp = requests.get(url, timeout=30, **kwargs)
    resp.raise_for_status()
    return resp


def stamp() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %Human:%M:%S")


# ── 1. ONS RETAIL SALES INDEX ─────────────────────────────────────────────────

def fetch_ons_rsi() -> pd.DataFrame:
    """
    Downloads the ONS Retail Sales Index CSV directly from the ONS
    download server. The CSV uses a v4 format with columns:
        v4_0, data_marking, mmm-yy, Time, countries, Geography,
        sic-unofficial, StandardIndustrialClassification,
        type-of-prices, Prices, seasonal-adjustment, SeasonalAdjustment
    """
    print("  [ONS] Resolving latest RSI version...")
    versions_resp = get(ONS_EDITIONS_URL)
    versions = versions_resp.json().get("items", [])
    if not versions:
        raise RuntimeError("No ONS RSI versions found.")
    # Sort by version number descending to get the latest
    versions.sort(key=lambda v: v.get("version", 0), reverse=True)
    latest_version = versions[0]["version"]
    csv_url = (
        f"https://download.ons.gov.uk/downloads/datasets/"
        f"retail-sales-index/editions/time-series/versions/{latest_version}.csv"
    )
    print(f"  [ONS] Downloading RSI CSV (version {latest_version}) from:")
    print(f"        {csv_url}")

    resp = get(csv_url)
    df   = pd.read_csv(io.StringIO(resp.text), low_memory=False)
    print(f"  [ONS] Downloaded {len(df):,} rows  |  columns: {list(df.columns)}")
    return df, latest_version


def clean_ons_rsi(df: pd.DataFrame, version: int) -> pd.DataFrame:
    """
    Cleans and normalises the raw ONS RSI CSV.

    The ONS v4 CSV has a somewhat irregular header. We rename to
    human-friendly names and filter to seasonally adjusted, index
    (not value-in-£), Great Britain geography.
    """
    # Normalise column names
    df.columns = [c.strip() for c in df.columns]

    # Typical v4 RSI columns (may vary slightly across versions):
    #   v4_0, data_marking, mmm-yy, Time,
    #   countries, Geography,
    #   sic-unofficial, StandardIndustrialClassification,
    #   type-of-prices, Prices,
    #   seasonal-adjustment, SeasonalAdjustment
    # The ONS v4 CSV value column is named v4_0, v4_1, v4_2 etc. depending on
    # how many footnote columns precede it. Find it dynamically.
    v4_col = next((c for c in df.columns if c.lower().startswith("v4_")), None)
    if v4_col:
        df = df.rename(columns={v4_col: "value"})

    sic_col = next(
        (c for c in df.columns
         if "standardindustrial" in c.lower() or "unofficialstandard" in c.lower()),
        None,
    )

    rename = {
        "Data Marking":      "data_marking",
        "data_marking":      "data_marking",
        "mmm-yy":            "period_code",
        "Time":              "time_label",
        "countries":         "geo_code",
        "Geography":         "geography",
        "sic-unofficial":    "sic_code",
        "type-of-prices":    "price_type",
        "Prices":            "prices_label",
        "seasonal-adjustment":"sa_code",
        "SeasonalAdjustment":"sa_label",
    }
    if sic_col:
        rename[sic_col] = "sector"
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # SA filter: the sa_code column contains the dimension code value,
    # which is "seasonal-adjustment" for SA rows (not the string "SA").
    if "sa_code" in df.columns:
        sa_vals = df["sa_code"].str.lower()
        sa_mask = sa_vals.str.contains("seasonal-adjustment", na=False)
        df_sa   = df[sa_mask]
        if len(df_sa) > 0:
            df = df_sa

    # Price type filter: pick the most informative single measure.
    # Prefer "chained-volume" (volume index) — closest to a real
    # quantity measure rather than a nominal value inflated by price.
    if "price_type" in df.columns:
        preferred = [
            "chained-volume-of-retail-sales",
            "value-of-retail-sales-at-current-prices",
        ]
        for pref in preferred:
            df_pref = df[df["price_type"] == pref]
            if len(df_pref) > 0:
                df = df_pref
                break

    # Keep Great Britain (K03000001) — only geography in this dataset
    if "geo_code" in df.columns:
        uk_codes = {"K02000001", "K03000001", "K04000001"}
        df_uk = df[df["geo_code"].isin(uk_codes)]
        if len(df_uk) > 0:
            df = df_uk

    # Parse period — ONS uses "Jan-26" format
    if "period_code" in df.columns:
        df["period"] = pd.to_datetime(df["period_code"], format="%b-%y", errors="coerce")
        df = df.dropna(subset=["period"])
        df["year"]  = df["period"].dt.year
        df["month"] = df["period"].dt.month

    # Coerce value
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])

    df["ons_version"]  = version
    df["fetched_at"]   = datetime.utcnow().isoformat()

    keep = ["period", "year", "month", "sector", "value",
            "sa_code", "price_type", "ons_version", "fetched_at"]
    keep = [c for c in keep if c in df.columns]

    df = df[keep].copy()
    print(f"  [ONS] After cleaning: {len(df):,} rows | "
          f"{df['sector'].nunique() if 'sector' in df.columns else '?'} sectors | "
          f"period {df['period'].min().date()} → {df['period'].max().date()}")
    return df


# ── 2. YFINANCE — UK RETAILER STOCKS ─────────────────────────────────────────

def fetch_stocks() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetches five years of daily prices and company info for each retailer.
    Returns (prices_df, info_df).
    """
    print("\n  [yfinance] Fetching UK retailer stock data...")
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=FX_HISTORY_YEARS * 365)

    all_prices = []
    info_rows  = []

    for ticker_sym, company in UK_RETAILERS.items():
        try:
            print(f"    {company} ({ticker_sym})...", end="", flush=True)
            tk   = yf.Ticker(ticker_sym)
            hist = tk.history(start=start_date, end=end_date, auto_adjust=True)
            if hist.empty:
                print(" ✗ no price data")
                continue

            hist = hist.reset_index()
            hist["ticker"]  = ticker_sym
            hist["company"] = company
            hist["date"]    = pd.to_datetime(hist["Date"]).dt.date.astype(str)

            price_cols = ["date", "ticker", "company",
                          "Open", "High", "Low", "Close", "Volume"]
            price_cols = [c for c in price_cols if c in hist.columns]
            prices     = hist[price_cols].copy()
            prices.columns = [c.lower() for c in prices.columns]
            all_prices.append(prices)

            # Company info
            info = tk.info or {}
            info_rows.append({
                "ticker":          ticker_sym,
                "company":         company,
                "market_cap":      info.get("marketCap"),
                "enterprise_value":info.get("enterpriseValue"),
                "pe_ratio":        info.get("trailingPE"),
                "forward_pe":      info.get("forwardPE"),
                "revenue":         info.get("totalRevenue"),
                "profit_margin":   info.get("profitMargins"),
                "ebitda_margin":   info.get("ebitdaMargins"),
                "roe":             info.get("returnOnEquity"),
                "debt_to_equity":  info.get("debtToEquity"),
                "currency":        info.get("currency", "GBP"),
                "sector":          info.get("sector", "Consumer Defensive"),
                "employees":       info.get("fullTimeEmployees"),
                "fetched_at":      datetime.utcnow().isoformat(),
            })
            print(f" ✓ ({len(prices):,} days)")

        except Exception as exc:
            print(f" ✗ {exc}")

    prices_df = pd.concat(all_prices, ignore_index=True) if all_prices else pd.DataFrame()
    info_df   = pd.DataFrame(info_rows)
    return prices_df, info_df


# ── 3. FRANKFURTER — GBP FX RATES ────────────────────────────────────────────

def fetch_fx() -> pd.DataFrame:
    """
    Downloads historical GBP FX rates from the Frankfurter API.
    Returns a long-format DataFrame: date, currency, rate.
    """
    print("\n  [FX] Fetching GBP exchange rates from Frankfurter API...")
    end_date   = datetime.today().date()
    start_date = end_date - timedelta(days=FX_HISTORY_YEARS * 365)
    targets    = ",".join(FX_TARGETS)

    url  = (f"https://api.frankfurter.app/"
            f"{start_date}..{end_date}?from=GBP&to={targets}")
    print(f"    {url}")
    data = get(url).json()

    rows = []
    for date_str, rates in data.get("rates", {}).items():
        for currency, rate in rates.items():
            rows.append({
                "date":     date_str,
                "base":     "GBP",
                "currency": currency,
                "rate":     rate,
            })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    print(f"  [FX] {len(df):,} rows | {df['currency'].nunique()} currencies | "
          f"{df['date'].min().date()} → {df['date'].max().date()}")
    return df


# ── 4. SAVE TO SQLITE ─────────────────────────────────────────────────────────

def save_to_db(ons_df, prices_df, info_df, fx_df):
    print(f"\n  [DB] Saving to {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)

    # ONS retail sales index
    if not ons_df.empty:
        ons_df.to_sql("ons_retail_monthly", conn, if_exists="replace", index=False)
        n = conn.execute("SELECT COUNT(*) FROM ons_retail_monthly").fetchone()[0]
        print(f"    ons_retail_monthly  : {n:>8,} rows")

    # Stock prices
    if not prices_df.empty:
        prices_df.to_sql("stocks_daily", conn, if_exists="replace", index=False)
        n = conn.execute("SELECT COUNT(*) FROM stocks_daily").fetchone()[0]
        print(f"    stocks_daily        : {n:>8,} rows")

    # Company info
    if not info_df.empty:
        info_df.to_sql("stocks_info", conn, if_exists="replace", index=False)
        n = conn.execute("SELECT COUNT(*) FROM stocks_info").fetchone()[0]
        print(f"    stocks_info         : {n:>8,} rows")

    # FX rates
    if not fx_df.empty:
        fx_df.to_sql("fx_rates", conn, if_exists="replace", index=False)
        n = conn.execute("SELECT COUNT(*) FROM fx_rates").fetchone()[0]
        print(f"    fx_rates            : {n:>8,} rows")

    conn.close()


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  RETAIL LIVE DATA FETCHER")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)
    print()

    errors = []

    # 1. ONS
    print("[1/3] ONS UK Retail Sales Index")
    print("-" * 40)
    try:
        raw_ons, version = fetch_ons_rsi()
        ons_df = clean_ons_rsi(raw_ons, version)
    except Exception as e:
        print(f"  ✗ ONS fetch failed: {e}")
        errors.append(f"ONS: {e}")
        ons_df = pd.DataFrame()

    # 2. yfinance
    print("\n[2/3] UK Retailer Stock Data (Yahoo Finance)")
    print("-" * 40)
    try:
        prices_df, info_df = fetch_stocks()
    except Exception as e:
        print(f"  ✗ yfinance fetch failed: {e}")
        errors.append(f"yfinance: {e}")
        prices_df, info_df = pd.DataFrame(), pd.DataFrame()

    # 3. FX rates
    print("\n[3/3] GBP Exchange Rates (Frankfurter API)")
    print("-" * 40)
    try:
        fx_df = fetch_fx()
    except Exception as e:
        print(f"  ✗ FX fetch failed: {e}")
        errors.append(f"FX: {e}")
        fx_df = pd.DataFrame()

    # Save
    save_to_db(ons_df, prices_df, info_df, fx_df)

    print()
    print("=" * 65)
    if errors:
        print("  Completed with warnings:")
        for err in errors:
            print(f"    ⚠  {err}")
    else:
        print("  All data fetched successfully.")
    print("  Ready to run: python scripts/analyse.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
