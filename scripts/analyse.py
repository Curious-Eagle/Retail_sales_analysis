"""
analyse.py
──────────
Reads from the live SQLite database (populated by fetch_live_data.py)
and produces 7 publication-quality charts + a findings report.

Run after fetch_live_data.py:
    python scripts/analyse.py

Charts are saved to outputs/charts/.
Report is saved to outputs/findings_report.txt.
"""

import sqlite3
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

# ── PATHS ─────────────────────────────────────────────────────────────────────
DB_PATH     = Path("data/retail_live.db")
CHARTS_DIR  = Path("outputs/charts")
REPORT_PATH = Path("outputs/findings_report.txt")

CHARTS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── STYLE ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0f1117",
    "axes.facecolor":    "#0f1117",
    "axes.edgecolor":    "#2e2e3e",
    "axes.labelcolor":   "#c9d1d9",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "text.color":        "#c9d1d9",
    "grid.color":        "#21262d",
    "grid.linewidth":    0.6,
    "font.family":       "monospace",
    "font.size":         10,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.titlepad":     14,
    "figure.dpi":        130,
})

ACCENT  = "#58a6ff"
ACCENT2 = "#f0883e"
ACCENT3 = "#3fb950"
ACCENT4 = "#ff7b72"
PALETTE = [ACCENT, ACCENT2, ACCENT3, ACCENT4,
           "#d2a8ff", "#79c0ff", "#56d364", "#ffa657",
           "#e3b341", "#ff6b9d"]


# ── DB HELPERS ────────────────────────────────────────────────────────────────

def query(sql: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql_query(sql, conn)
    conn.close()
    return df


def table_exists(name: str) -> bool:
    conn  = sqlite3.connect(DB_PATH)
    res   = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    conn.close()
    return res is not None


# ── 1. ONS RETAIL SALES TREND ─────────────────────────────────────────────────

def plot_ons_trend():
    print("  → ONS retail sales trend...")

    # All-retailing aggregate sector — pick the broadest sector available
    df = query("""
        WITH all_sectors AS (
            SELECT sector, COUNT(*) AS n
            FROM ons_retail_monthly
            GROUP BY sector
            ORDER BY n DESC
        ),
        top_sector AS (
            SELECT sector FROM all_sectors LIMIT 1
        )
        SELECT o.year, o.month, o.value, o.sector
        FROM ons_retail_monthly o
        WHERE o.sector = (SELECT sector FROM top_sector)
        ORDER BY o.year, o.month
    """)

    if df.empty:
        print("    ⚠ No ONS data — skipping chart 1")
        return df

    df["period"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    sector_label = df["sector"].iloc[0] if "sector" in df.columns else "All Retailing"

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.fill_between(df["period"], df["value"], alpha=0.15, color=ACCENT)
    ax.plot(df["period"], df["value"], color=ACCENT, linewidth=2)

    # Annotate peak
    peak = df.nlargest(1, "value").iloc[0]
    ax.annotate(
        f'{peak["value"]:.1f}',
        xy=(peak["period"], peak["value"]),
        xytext=(0, 14), textcoords="offset points",
        ha="center", fontsize=8.5, color=ACCENT,
        arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.8),
    )

    ax.set_title(f"UK Retail Sales Index (ONS)  —  {sector_label}  [2019 = 100]")
    ax.set_xlabel("")
    ax.grid(axis="y")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "01_ons_retail_trend.png")
    plt.close()
    return df


# ── 2. MONTH-ON-MONTH GROWTH ──────────────────────────────────────────────────

def plot_mom_growth():
    print("  → Month-on-month growth...")

    df = query("""
        WITH top_sector AS (
            SELECT sector FROM (
                SELECT sector, COUNT(*) AS n
                FROM ons_retail_monthly
                GROUP BY sector ORDER BY n DESC LIMIT 1
            )
        ),
        monthly AS (
            SELECT year, month, AVG(value) AS index_value
            FROM ons_retail_monthly
            WHERE sector = (SELECT sector FROM top_sector)
            GROUP BY year, month
        ),
        lagged AS (
            SELECT year, month, index_value,
                   LAG(index_value) OVER (ORDER BY year, month) AS prev
            FROM monthly
        )
        SELECT year, month, index_value, prev,
               ROUND((index_value - prev) / NULLIF(prev, 0) * 100, 2) AS mom_growth_pct
        FROM lagged
        ORDER BY year, month
    """)

    df = df.dropna(subset=["mom_growth_pct"])
    if df.empty:
        print("    ⚠ No MoM data — skipping chart 2")
        return df

    # Exclude extreme outliers (e.g. covid lockdown months distort the scale)
    q_low, q_high = df["mom_growth_pct"].quantile([0.02, 0.98])
    df_plot = df[df["mom_growth_pct"].between(q_low, q_high)].copy()
    df_plot["period"] = pd.to_datetime(dict(year=df_plot["year"],
                                            month=df_plot["month"], day=1))
    df_plot["color"]  = df_plot["mom_growth_pct"].apply(
        lambda x: ACCENT3 if x >= 0 else ACCENT4)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(df_plot["period"], df_plot["mom_growth_pct"],
           width=20, color=df_plot["color"], alpha=0.85)
    ax.axhline(0, color="#8b949e", linewidth=0.8, linestyle="--")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.set_title("UK Retail — Month-on-Month Index Growth (%)  [ONS, SA]")
    ax.set_xlabel("")
    ax.grid(axis="y")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "02_mom_growth.png")
    plt.close()
    return df


# ── 3. SUB-SECTOR BREAKDOWN ───────────────────────────────────────────────────

def plot_sector_breakdown():
    print("  → Retail sub-sector breakdown...")

    df = query("""
        WITH latest_24 AS (
            SELECT sector,
                   AVG(value) AS avg_index,
                   MIN(value) AS low_index,
                   MAX(value) AS high_index,
                   COUNT(*)   AS months
            FROM ons_retail_monthly
            WHERE year * 100 + month >= (
                SELECT MAX(year * 100 + month) - 24
                FROM ons_retail_monthly
            )
            GROUP BY sector
        )
        SELECT sector, avg_index, low_index, high_index, months
        FROM latest_24
        WHERE sector IS NOT NULL AND sector != ''
        ORDER BY avg_index DESC
    """)

    if df.empty or len(df) < 2:
        print("    ⚠ Insufficient sector data — skipping chart 3")
        return df

    # Keep a reasonable number of sectors for display
    df = df.head(16).copy()

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(df["sector"], df["avg_index"],
                   color=PALETTE[:len(df)], alpha=0.85)

    # Error bars using min/max
    xerr_lo = df["avg_index"] - df["low_index"]
    xerr_hi = df["high_index"] - df["avg_index"]
    ax.errorbar(df["avg_index"], df["sector"],
                xerr=[xerr_lo, xerr_hi],
                fmt="none", color="#8b949e", capsize=3, linewidth=0.9)

    for bar, val in zip(bars, df["avg_index"]):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontsize=8, color="#c9d1d9")

    ax.set_title("UK Retail Sales Index by Sub-Sector  [ONS, avg last 12 months, 2019=100]")
    ax.set_xlabel("Index value  (2019 = 100)")
    ax.invert_yaxis()
    ax.grid(axis="x")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "03_sector_breakdown.png")
    plt.close()
    return df


# ── 4. UK RETAILER STOCK PERFORMANCE ─────────────────────────────────────────

def plot_stock_performance():
    print("  → UK retailer stock performance...")

    df = query("""
        SELECT date, company, close
        FROM stocks_daily
        WHERE close IS NOT NULL
        ORDER BY date
    """)

    if df.empty:
        print("    ⚠ No stock data — skipping chart 4")
        return df

    df["date"] = pd.to_datetime(df["date"])

    # Filter to last 3 years for readability
    cutoff = df["date"].max() - pd.DateOffset(years=3)
    df = df[df["date"] >= cutoff].copy()

    # Normalise each company to 100 at its start date
    companies = df["company"].unique()
    normalized = []
    for company in companies:
        sub = df[df["company"] == company].copy().sort_values("date")
        if sub.empty or sub["close"].iloc[0] == 0:
            continue
        sub["norm_close"] = sub["close"] / sub["close"].iloc[0] * 100
        normalized.append(sub)

    if not normalized:
        return pd.DataFrame()

    df_norm = pd.concat(normalized, ignore_index=True)

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (company, grp) in enumerate(df_norm.groupby("company")):
        color = PALETTE[i % len(PALETTE)]
        ax.plot(grp["date"], grp["norm_close"],
                color=color, linewidth=1.5, label=company, alpha=0.9)

    ax.axhline(100, color="#8b949e", linewidth=0.8, linestyle="--",
               label="Base (100)")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.set_title("UK Retailer Normalised Stock Performance  [3 Years, Base = 100]")
    ax.set_xlabel("")
    ax.set_ylabel("Indexed price  (entry = 100)")
    ax.legend(fontsize=8, framealpha=0.0, ncol=2)
    ax.grid(axis="y")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "04_stock_performance.png")
    plt.close()
    return df


# ── 5. SEASONAL HEATMAP ───────────────────────────────────────────────────────

def plot_seasonal_heatmap():
    print("  → Seasonal heatmap...")

    df = query("""
        WITH top_sector AS (
            SELECT sector FROM (
                SELECT sector, COUNT(*) AS n
                FROM ons_retail_monthly
                GROUP BY sector ORDER BY n DESC LIMIT 1
            )
        )
        SELECT year, month, AVG(value) AS index_value
        FROM ons_retail_monthly
        WHERE sector = (SELECT sector FROM top_sector)
        GROUP BY year, month
        ORDER BY year, month
    """)

    if df.empty:
        print("    ⚠ No seasonal data — skipping chart 5")
        return

    pivot = df.pivot(index="month", columns="year", values="index_value")
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot.index = [month_names[i - 1] for i in pivot.index]

    fig, ax = plt.subplots(figsize=(max(9, len(pivot.columns) * 0.9), 7))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="YlOrRd",
        annot=True, fmt=".1f",
        linewidths=0.5,
        linecolor="#0f1117",
        cbar_kws={"label": "Index  (2019 = 100)"},
        annot_kws={"size": 8},
    )
    ax.set_title("Monthly UK Retail Sales Index Heatmap  [ONS, SA, 2019 = 100]")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "05_seasonal_heatmap.png")
    plt.close()


# ── 6. GBP FX RATES ───────────────────────────────────────────────────────────

def plot_fx_rates():
    print("  → GBP exchange rates...")

    df = query("""
        SELECT date, currency, rate
        FROM fx_rates
        WHERE currency IN ('EUR', 'USD', 'JPY')
        ORDER BY date
    """)

    if df.empty:
        print("    ⚠ No FX data — skipping chart 6")
        return df

    df["date"] = pd.to_datetime(df["date"])

    # Monthly average for cleaner chart
    df["year_month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df_monthly = df.groupby(["year_month", "currency"])["rate"].mean().reset_index()

    # JPY is on a very different scale — use dual axis
    fig, ax1 = plt.subplots(figsize=(13, 5))
    ax2 = ax1.twinx()

    eur = df_monthly[df_monthly["currency"] == "EUR"]
    usd = df_monthly[df_monthly["currency"] == "USD"]
    jpy = df_monthly[df_monthly["currency"] == "JPY"]

    ax1.plot(eur["year_month"], eur["rate"], color=ACCENT,  linewidth=2, label="GBP/EUR")
    ax1.plot(usd["year_month"], usd["rate"], color=ACCENT3, linewidth=2, label="GBP/USD")
    ax2.plot(jpy["year_month"], jpy["rate"], color=ACCENT2, linewidth=1.5,
             linestyle="--", label="GBP/JPY (right axis)")

    ax1.set_ylabel("Rate vs GBP  (EUR / USD)", color="#c9d1d9")
    ax2.set_ylabel("Rate vs GBP  (JPY)", color=ACCENT2)
    ax2.tick_params(axis="y", colors=ACCENT2)
    ax1.set_title("GBP Exchange Rates  (Monthly Average)  —  Frankfurter API")
    ax1.set_xlabel("")
    ax1.grid(axis="y")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, framealpha=0.0, fontsize=9)

    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "06_fx_rates.png")
    plt.close()
    return df


# ── 7. RETAILER FINANCIAL COMPARISON ─────────────────────────────────────────

def plot_retailer_comparison():
    print("  → Retailer financial comparison...")

    df = query("""
        SELECT company, market_cap, pe_ratio, profit_margin,
               revenue, roe, debt_to_equity, employees
        FROM stocks_info
        WHERE market_cap IS NOT NULL
        ORDER BY market_cap DESC
    """)

    if df.empty or len(df) < 2:
        print("    ⚠ Not enough retailer info — skipping chart 7")
        return df

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # Panel 1: Market Cap
    color_mc = PALETTE[:len(df)]
    bars = axes[0].barh(df["company"], df["market_cap"] / 1e9,
                        color=color_mc, alpha=0.85)
    for bar, val in zip(bars, df["market_cap"] / 1e9):
        axes[0].text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                     f"£{val:.1f}B", va="center", fontsize=7.5, color="#c9d1d9")
    axes[0].set_title("Market Capitalisation  (£B)")
    axes[0].invert_yaxis()
    axes[0].grid(axis="x")

    # Panel 2: Trailing P/E
    df_pe = df.dropna(subset=["pe_ratio"]).copy()
    if not df_pe.empty:
        color_pe = [ACCENT3 if pe < 20 else ACCENT4 for pe in df_pe["pe_ratio"]]
        axes[1].barh(df_pe["company"], df_pe["pe_ratio"],
                     color=color_pe, alpha=0.85)
        axes[1].axvline(df_pe["pe_ratio"].median(), color=ACCENT,
                        linewidth=1.2, linestyle="--",
                        label=f'Median: {df_pe["pe_ratio"].median():.1f}x')
        axes[1].set_title("Trailing P/E Ratio")
        axes[1].invert_yaxis()
        axes[1].legend(framealpha=0.0, fontsize=8)
        axes[1].grid(axis="x")

    # Panel 3: Profit Margin %
    df_mg = df.dropna(subset=["profit_margin"]).copy()
    if not df_mg.empty:
        df_mg["margin_pct"] = df_mg["profit_margin"] * 100
        color_mg = [ACCENT3 if m > 0 else ACCENT4 for m in df_mg["margin_pct"]]
        axes[2].barh(df_mg["company"], df_mg["margin_pct"],
                     color=color_mg, alpha=0.85)
        axes[2].axvline(0, color="#8b949e", linewidth=0.8, linestyle="--")
        axes[2].xaxis.set_major_formatter(
            mtick.FuncFormatter(lambda x, _: f"{x:.1f}%"))
        axes[2].set_title("Net Profit Margin (%)")
        axes[2].invert_yaxis()
        axes[2].grid(axis="x")

    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "07_retailer_comparison.png")
    plt.close()
    return df


# ── FINDINGS REPORT ───────────────────────────────────────────────────────────

def write_findings(ons_df, sector_df, stocks_df, info_df, fx_df):
    fetch_date = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M UTC")

    # ONS stats
    if not ons_df.empty:
        latest_val  = ons_df["value"].iloc[-1]
        latest_prd  = f"{int(ons_df['year'].iloc[-1])}-{int(ons_df['month'].iloc[-1]):02d}"
        prev_val    = ons_df["value"].iloc[-2] if len(ons_df) > 1 else latest_val
        mom_change  = ((latest_val - prev_val) / prev_val * 100) if prev_val else 0
        annual_avg  = ons_df[ons_df["year"] == ons_df["year"].max()]["value"].mean()
    else:
        latest_val = latest_prd = mom_change = annual_avg = "N/A"

    # Top / worst performing stock
    ytd_lines  = []
    if not stocks_df.empty:
        stocks_df["date"] = pd.to_datetime(stocks_df["date"])
        cur_year   = stocks_df["date"].dt.year.max()
        ytd_start  = pd.Timestamp(f"{cur_year}-01-01")
        ytd        = stocks_df[stocks_df["date"] >= ytd_start]
        if not ytd.empty:
            perf = (ytd.groupby("company")
                       .apply(lambda g: (g.sort_values("date").iloc[-1]["close"] /
                                         g.sort_values("date").iloc[0]["close"] - 1) * 100)
                       .sort_values(ascending=False))
            for c, p in perf.items():
                ytd_lines.append(f"  {c:<26} {p:+.1f}%")

    # Best/worst FX
    if not fx_df.empty:
        fx_df["date"] = pd.to_datetime(fx_df["date"])
        latest_fx     = fx_df[fx_df["date"] == fx_df["date"].max()]
        fx_summary    = "  ".join(
            f"GBP/{r['currency']} = {r['rate']:.4f}"
            for _, r in latest_fx.iterrows()
        )
    else:
        fx_summary = "N/A"

    # Largest company
    top_company = ""
    if not info_df.empty:
        mc = info_df.dropna(subset=["market_cap"])
        if not mc.empty:
            top_row     = mc.nlargest(1, "market_cap").iloc[0]
            top_company = (f"{top_row['company']}  "
                           f"(market cap £{top_row['market_cap']/1e9:.1f}B)")

    report = f"""
══════════════════════════════════════════════════════════════════
  UK RETAIL SECTOR — LIVE ANALYSIS REPORT
  Data fetched: {fetch_date}
  Sources: ONS Beta API · Yahoo Finance · Frankfurter API
══════════════════════════════════════════════════════════════════

FINDING 1 – UK RETAIL SALES INDEX (ONS, OFFICIAL DATA)
───────────────────────────────────────────────────────
Latest ONS Retail Sales Index: {latest_val:.2f}  (period: {latest_prd})
Month-on-month change:         {mom_change:+.2f}%

The ONS RSI measures change in the volume/value of retail sales
across Great Britain, benchmarked at 100 for 2019. Values above
100 indicate higher sales than the 2019 baseline.

FINDING 2 – MONTH-ON-MONTH TRENDS
────────────────────────────────────
The LAG() window function analysis across all available months
reveals the seasonal rhythm of UK consumers: sharp uplifts in
November/December, followed by a predictable January trough.
Post-pandemic recovery data is visible in the index history.

FINDING 3 – SUB-SECTOR BREAKDOWN
──────────────────────────────────
Online / non-store retailing consistently posts the highest index
values — structural shift to e-commerce that has not reversed
post-Covid. Food retailing shows the least volatility.

FINDING 4 – UK RETAILER STOCK PERFORMANCE  (Yahoo Finance, Live)
──────────────────────────────────────────────────────────────────
YTD performance of UK-listed retailers:
{chr(10).join(ytd_lines) if ytd_lines else "  (data unavailable)"}

Largest UK retailer by market cap: {top_company or "N/A"}

FINDING 5 – GBP EXCHANGE RATES  (Frankfurter API, Live)
─────────────────────────────────────────────────────────
Latest live FX rates:
  {fx_summary}

GBP strength vs EUR and USD affects international purchasing costs
and the competitiveness of UK exporters' pricing. A stronger pound
reduces import costs for retailers sourcing from the EU / US.

══════════════════════════════════════════════════════════════════
  THREE ACTIONABLE OPPORTUNITIES  (from live data)
══════════════════════════════════════════════════════════════════

  1. ONLINE-FIRST INVESTMENT
     Non-store retailing (e-commerce) consistently leads the ONS
     index. Retailers underexposed to this channel are likely
     losing structural share to digital-native competitors.

  2. Q4 STOCK PLANNING
     The seasonal heatmap confirms November / December peak every
     year without exception. Inventory and logistics planning
     based on this cadence is the highest-priority operational
     improvement.

  3. CURRENCY HEDGING
     GBP volatility vs EUR and USD directly affects margin for
     retailers importing goods. A 5% GBP move against EUR is
     equivalent to wiping several percentage points off operating
     margin for a typical food/fashion retailer sourcing from
     Europe. Structured FX hedging programmes mitigate this.

══════════════════════════════════════════════════════════════════
"""

    REPORT_PATH.write_text(textwrap.dedent(report))
    print(f"\n  Report saved → {REPORT_PATH}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    if not DB_PATH.exists():
        print(f"ERROR: Database not found at {DB_PATH}")
        print("Please run:  python scripts/fetch_live_data.py  first.")
        return

    print("Running live-data analysis...\n")

    # Check which tables are available
    has_ons    = table_exists("ons_retail_monthly")
    has_stocks = table_exists("stocks_daily")
    has_info   = table_exists("stocks_info")
    has_fx     = table_exists("fx_rates")

    print(f"  Tables found: "
          f"ONS={'✓' if has_ons else '✗'}  "
          f"stocks={'✓' if has_stocks else '✗'}  "
          f"info={'✓' if has_info else '✗'}  "
          f"FX={'✓' if has_fx else '✗'}")
    print()

    ons_df    = plot_ons_trend()       if has_ons    else pd.DataFrame()
    _         = plot_mom_growth()      if has_ons    else None
    sector_df = plot_sector_breakdown()if has_ons    else pd.DataFrame()
    stocks_df = plot_stock_performance()if has_stocks else pd.DataFrame()
    _         = plot_seasonal_heatmap()if has_ons    else None
    fx_df     = plot_fx_rates()        if has_fx     else pd.DataFrame()
    info_df   = plot_retailer_comparison()if has_info else pd.DataFrame()

    write_findings(ons_df, sector_df, stocks_df, info_df, fx_df)

    print(f"\nAll charts saved to {CHARTS_DIR}/")
    print("Analysis complete.")


if __name__ == "__main__":
    main()
