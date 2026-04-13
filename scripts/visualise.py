"""
visualise.py
────────────
Generates 12 additional charts from the live SQLite database,
complementing the 7 produced by analyse.py.

Chart types included:
  08  Donut chart          — retail sector share of latest index
  09  Stacked area         — sector trends over time (ONS)
  10  Box plot             — monthly index distribution by sector
  11  Bubble chart         — market cap vs YTD return vs margin
  12  Correlation heatmap  — pairwise stock return correlations
  13  Candlestick OHLC     — Tesco daily price (last 6 months)
  14  Return distribution  — KDE of daily returns per retailer
  15  Drawdown chart       — rolling max-drawdown per retailer
  16  Rolling volatility   — 30-day annualised vol per retailer
  17  Dual-axis overlay    — ONS retail index + GBP/EUR on one plot
  18  52-week range        — current price vs 52-wk high/low band
  19  Polar radar          — sector performance radar
  20  Volume heatmap       — trading volume calendar heatmap (Tesco)

Run after fetch_live_data.py:
    python scripts/visualise.py
"""

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde

# ── PATHS ─────────────────────────────────────────────────────────────────────
DB_PATH    = Path("data/retail_live.db")
CHARTS_DIR = Path("outputs/charts")
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

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
PALETTE = [
    "#58a6ff","#f0883e","#3fb950","#ff7b72",
    "#d2a8ff","#79c0ff","#56d364","#ffa657",
    "#e3b341","#ff6b9d","#a5d6ff","#85e89d",
]


def query(sql: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql_query(sql, conn)
    conn.close()
    return df


def table_exists(name: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    res  = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    conn.close()
    return res is not None


def save(fig, name: str):
    path = CHARTS_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {name}")


# ══════════════════════════════════════════════════════════════════════════════
# 08  DONUT CHART — sector share of latest average index
# ══════════════════════════════════════════════════════════════════════════════

def chart_sector_donut():
    df = query("""
        SELECT sector, AVG(value) AS avg_idx
        FROM ons_retail_monthly
        WHERE year * 100 + month >= (
            SELECT MAX(year * 100 + month) - 24
            FROM ons_retail_monthly
        )
          AND sector IS NOT NULL AND sector != ''
        GROUP BY sector
        ORDER BY avg_idx DESC
    """)

    if df.empty:
        print("  ✗  08_sector_donut.png  (no data)")
        return

    fig, (ax_pie, ax_legend) = plt.subplots(1, 2, figsize=(13, 7),
                                             gridspec_kw={"width_ratios": [1.4, 1]})

    wedges, texts, autotexts = ax_pie.pie(
        df["avg_idx"],
        labels=None,
        autopct="%1.1f%%",
        colors=PALETTE[:len(df)],
        startangle=90,
        pctdistance=0.78,
        wedgeprops=dict(width=0.52, edgecolor="#0f1117", linewidth=2),
        textprops=dict(color="#c9d1d9"),
    )
    for at in autotexts:
        at.set_fontsize(8.5)
        at.set_fontweight("bold")

    # Centre label
    ax_pie.text(0, 0, "UK Retail\nSector Mix", ha="center", va="center",
                fontsize=11, color="#c9d1d9", fontweight="bold",
                linespacing=1.6)

    ax_pie.set_title("Sector Share of Retail Sales Index\n[ONS, avg last 24 months]")

    # Legend panel
    ax_legend.axis("off")
    for i, (_, row) in enumerate(df.iterrows()):
        y = 0.93 - i * 0.083
        ax_legend.add_patch(plt.Rectangle((0, y - 0.025), 0.07, 0.05,
                                          color=PALETTE[i % len(PALETTE)],
                                          transform=ax_legend.transAxes))
        ax_legend.text(0.11, y, f"{row['sector'].replace('-', ' ').title()}",
                       va="center", fontsize=9, transform=ax_legend.transAxes,
                       color="#c9d1d9")
        ax_legend.text(0.95, y, f"{row['avg_idx']:.1f}",
                       va="center", ha="right", fontsize=9,
                       transform=ax_legend.transAxes, color="#8b949e")

    ax_legend.text(0.11, 0.98, "Sector", va="top", fontsize=9,
                   transform=ax_legend.transAxes, color="#8b949e")
    ax_legend.text(0.95, 0.98, "Avg Index", va="top", ha="right", fontsize=9,
                   transform=ax_legend.transAxes, color="#8b949e")

    fig.tight_layout()
    save(fig, "08_sector_donut.png")


# ══════════════════════════════════════════════════════════════════════════════
# 09  STACKED AREA — sector trends over time
# ══════════════════════════════════════════════════════════════════════════════

def chart_sector_stacked_area():
    df = query("""
        SELECT year, month, sector, value
        FROM ons_retail_monthly
        WHERE sector IS NOT NULL AND sector != ''
          AND year >= 2010
        ORDER BY year, month
    """)

    if df.empty:
        print("  ✗  09_sector_stacked_area.png  (no data)")
        return

    df["period"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    pivot = df.pivot_table(index="period", columns="sector",
                           values="value", aggfunc="mean")
    pivot = pivot.fillna(method="ffill").fillna(method="bfill")

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.stackplot(pivot.index, pivot.T.values,
                 labels=[c.replace("-", " ").title() for c in pivot.columns],
                 colors=PALETTE[:len(pivot.columns)],
                 alpha=0.82)

    ax.set_title("UK Retail Sales Index — Stacked by Sector  [ONS, 2010–present]")
    ax.set_xlabel("")
    ax.set_ylabel("Cumulative Index")
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.0,
              ncol=2, labelcolor="#c9d1d9")
    ax.grid(axis="y")
    fig.tight_layout()
    save(fig, "09_sector_stacked_area.png")


# ══════════════════════════════════════════════════════════════════════════════
# 10  BOX PLOT — monthly index distribution per sector
# ══════════════════════════════════════════════════════════════════════════════

def chart_sector_boxplot():
    df = query("""
        SELECT sector, value
        FROM ons_retail_monthly
        WHERE sector IS NOT NULL AND sector != ''
          AND year >= 2010
        ORDER BY sector
    """)

    if df.empty:
        print("  ✗  10_sector_boxplot.png  (no data)")
        return

    sectors = (df.groupby("sector")["value"]
                 .median()
                 .sort_values(ascending=False)
                 .index.tolist())

    data_by_sector = [df[df["sector"] == s]["value"].dropna().values
                      for s in sectors]
    labels = [s.replace("-", " ").title() for s in sectors]

    fig, ax = plt.subplots(figsize=(13, 6))
    bp = ax.boxplot(
        data_by_sector,
        vert=True,
        patch_artist=True,
        labels=labels,
        medianprops=dict(color="#0f1117", linewidth=2),
        whiskerprops=dict(color="#8b949e"),
        capprops=dict(color="#8b949e"),
        flierprops=dict(marker="o", markersize=3,
                        markerfacecolor="#8b949e", alpha=0.5),
    )
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.82)

    ax.set_title("Distribution of Monthly Retail Sales Index by Sector  [ONS, 2010–present]")
    ax.set_ylabel("Index value  (2019 = 100)")
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8.5)
    ax.grid(axis="y")
    fig.tight_layout()
    save(fig, "10_sector_boxplot.png")


# ══════════════════════════════════════════════════════════════════════════════
# 11  BUBBLE CHART — market cap vs YTD return vs profit margin
# ══════════════════════════════════════════════════════════════════════════════

def chart_bubble():
    info = query("""
        SELECT company, market_cap, pe_ratio, profit_margin, revenue
        FROM stocks_info
        WHERE market_cap IS NOT NULL
    """)
    prices = query("""
        SELECT company, date, close
        FROM stocks_daily
        WHERE date >= date('now', 'start of year')
        ORDER BY date
    """)

    if info.empty or prices.empty:
        print("  ✗  11_bubble_chart.png  (no data)")
        return

    prices["date"] = pd.to_datetime(prices["date"])
    ytd = (prices.groupby("company")
                 .apply(lambda g: (g.sort_values("date").iloc[-1]["close"] /
                                   g.sort_values("date").iloc[0]["close"] - 1) * 100,
                        include_groups=False)
                 .reset_index(name="ytd_return"))

    merged = info.merge(ytd, on="company")
    merged = merged.dropna(subset=["profit_margin"])
    merged["margin_pct"] = merged["profit_margin"] * 100

    # Bubble size: proportional to market cap
    size_scale = (merged["market_cap"] / merged["market_cap"].max() * 1800 + 80)
    colors = [ACCENT3 if r >= 0 else ACCENT4 for r in merged["ytd_return"]]

    fig, ax = plt.subplots(figsize=(12, 7))
    sc = ax.scatter(
        merged["ytd_return"], merged["margin_pct"],
        s=size_scale, c=colors, alpha=0.80,
        edgecolors="#2e2e3e", linewidths=1.2,
    )

    for _, row in merged.iterrows():
        ax.annotate(
            row["company"],
            xy=(row["ytd_return"], row["margin_pct"]),
            xytext=(6, 4), textcoords="offset points",
            fontsize=8.5, color="#c9d1d9",
        )

    ax.axvline(0, color="#8b949e", linewidth=0.8, linestyle="--")
    ax.axhline(0, color="#8b949e", linewidth=0.8, linestyle="--")
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:+.1f}%"))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.set_xlabel("YTD Return  (%)")
    ax.set_ylabel("Net Profit Margin  (%)")
    ax.set_title("UK Retailers — YTD Return vs Profit Margin\n"
                 "[bubble size = market cap, green = positive YTD]")
    ax.grid(axis="both")

    # Size legend
    for cap, label in [(5e9, "£5B"), (15e9, "£15B"), (30e9, "£30B")]:
        if cap <= merged["market_cap"].max():
            ax.scatter([], [], s=cap / merged["market_cap"].max() * 1800 + 80,
                       c="#8b949e", alpha=0.6, label=label, edgecolors="#2e2e3e")
    ax.legend(title="Market cap", framealpha=0.0, fontsize=8.5)

    fig.tight_layout()
    save(fig, "11_bubble_chart.png")


# ══════════════════════════════════════════════════════════════════════════════
# 12  CORRELATION HEATMAP — pairwise daily return correlations
# ══════════════════════════════════════════════════════════════════════════════

def chart_correlation_heatmap():
    df = query("""
        SELECT date, company, close
        FROM stocks_daily
        ORDER BY date
    """)

    if df.empty:
        print("  ✗  12_correlation_heatmap.png  (no data)")
        return

    df["date"] = pd.to_datetime(df["date"])
    pivot  = df.pivot(index="date", columns="company", values="close")
    returns = pivot.pct_change().dropna()
    corr   = returns.corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(
        corr, ax=ax, mask=mask,
        cmap=cmap, vmin=-1, vmax=1, center=0,
        annot=True, fmt=".2f", annot_kws={"size": 8.5},
        linewidths=0.5, linecolor="#0f1117",
        square=True,
        cbar_kws={"label": "Pearson r", "shrink": 0.8},
    )
    ax.set_title("UK Retailer Stock — Daily Return Correlation Matrix  [5 Years]")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    save(fig, "12_correlation_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# 13  CANDLESTICK OHLC — Tesco, last 120 trading days
# ══════════════════════════════════════════════════════════════════════════════

def chart_candlestick():
    df = query("""
        SELECT date, open, high, low, close, volume
        FROM stocks_daily
        WHERE ticker = 'TSCO.L'
        ORDER BY date DESC
        LIMIT 120
    """)

    if df.empty:
        print("  ✗  13_candlestick.png  (no data)")
        return

    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    df["x"]    = range(len(df))

    fig = plt.figure(figsize=(14, 7))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    for _, row in df.iterrows():
        bull = row["close"] >= row["open"]
        color  = ACCENT3 if bull else ACCENT4
        body_lo = min(row["open"], row["close"])
        body_hi = max(row["open"], row["close"])
        body_h  = max(body_hi - body_lo, 0.1)   # ensure visible width

        # Candle body
        ax1.bar(row["x"], body_h, bottom=body_lo,
                width=0.6, color=color, alpha=0.85)
        # Wick
        ax1.plot([row["x"], row["x"]], [row["low"], row["high"]],
                 color=color, linewidth=0.8, alpha=0.9)

    # 20-day SMA
    df["sma20"] = df["close"].rolling(20).mean()
    ax1.plot(df["x"], df["sma20"], color=ACCENT, linewidth=1.5,
             linestyle="--", label="20-day SMA", alpha=0.9)

    ax1.set_title("Tesco (TSCO.L) — Candlestick OHLC + 20-day SMA  [last 120 days]")
    ax1.set_ylabel("Price  (GBp)")
    ax1.legend(framealpha=0.0, fontsize=9)
    ax1.grid(axis="y")
    plt.setp(ax1.get_xticklabels(), visible=False)

    # Volume
    vol_colors = [ACCENT3 if df["close"].iloc[i] >= df["open"].iloc[i] else ACCENT4
                  for i in range(len(df))]
    ax2.bar(df["x"], df["volume"] / 1e6, color=vol_colors, alpha=0.70, width=0.6)
    ax2.set_ylabel("Vol (M)", fontsize=8)
    ax2.grid(axis="y")
    ax2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.0f}M"))

    # X-axis labels: show every 20th date
    tick_pos   = df["x"][::20]
    tick_labels= df["date"][::20].dt.strftime("%b %y")
    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels(tick_labels, fontsize=8)

    fig.tight_layout()
    save(fig, "13_candlestick.png")


# ══════════════════════════════════════════════════════════════════════════════
# 14  RETURN DISTRIBUTION — KDE of daily returns per retailer
# ══════════════════════════════════════════════════════════════════════════════

def chart_return_distribution():
    df = query("""
        SELECT date, company, close
        FROM stocks_daily
        ORDER BY date
    """)

    if df.empty:
        print("  ✗  14_return_distribution.png  (no data)")
        return

    df["date"] = pd.to_datetime(df["date"])
    pivot   = df.pivot(index="date", columns="company", values="close")
    returns = pivot.pct_change().dropna() * 100

    companies = returns.columns.tolist()
    fig, ax   = plt.subplots(figsize=(13, 6))

    x_grid = np.linspace(-6, 6, 400)
    for i, company in enumerate(companies):
        data = returns[company].dropna().values
        if len(data) < 5:
            continue
        kde  = gaussian_kde(data, bw_method=0.4)
        ax.plot(x_grid, kde(x_grid), color=PALETTE[i % len(PALETTE)],
                linewidth=2, label=company, alpha=0.9)
        ax.fill_between(x_grid, kde(x_grid), alpha=0.06,
                        color=PALETTE[i % len(PALETTE)])

    ax.axvline(0, color="#8b949e", linewidth=1, linestyle="--")
    ax.set_title("Daily Return Distribution — UK Retailers  [KDE, 5 Years]")
    ax.set_xlabel("Daily Return  (%)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8.5, framealpha=0.0, ncol=2)
    ax.grid(axis="y")
    fig.tight_layout()
    save(fig, "14_return_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# 15  DRAWDOWN CHART — rolling maximum drawdown per retailer (3 years)
# ══════════════════════════════════════════════════════════════════════════════

def chart_drawdown():
    df = query("""
        SELECT date, company, close
        FROM stocks_daily
        WHERE date >= date('now', '-3 years')
        ORDER BY date
    """)

    if df.empty:
        print("  ✗  15_drawdown.png  (no data)")
        return

    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot(index="date", columns="company", values="close")

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, company in enumerate(pivot.columns):
        prices   = pivot[company].dropna()
        cum_max  = prices.cummax()
        drawdown = (prices - cum_max) / cum_max * 100
        ax.plot(drawdown.index, drawdown.values,
                color=PALETTE[i % len(PALETTE)],
                linewidth=1.5, label=company, alpha=0.85)
        ax.fill_between(drawdown.index, drawdown.values,
                        alpha=0.04, color=PALETTE[i % len(PALETTE)])

    ax.axhline(0, color="#8b949e", linewidth=0.8, linestyle="--")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.set_title("Rolling Drawdown from All-Time High — UK Retailers  [3 Years]")
    ax.set_xlabel("")
    ax.set_ylabel("Drawdown from Peak  (%)")
    ax.legend(fontsize=8, framealpha=0.0, ncol=2)
    ax.grid(axis="y")
    fig.tight_layout()
    save(fig, "15_drawdown.png")


# ══════════════════════════════════════════════════════════════════════════════
# 16  ROLLING VOLATILITY — 30-day annualised vol per retailer
# ══════════════════════════════════════════════════════════════════════════════

def chart_rolling_volatility():
    df = query("""
        SELECT date, company, close
        FROM stocks_daily
        WHERE date >= date('now', '-3 years')
        ORDER BY date
    """)

    if df.empty:
        print("  ✗  16_rolling_volatility.png  (no data)")
        return

    df["date"] = pd.to_datetime(df["date"])
    pivot   = df.pivot(index="date", columns="company", values="close")
    returns = pivot.pct_change()

    WINDOW = 30
    vol    = returns.rolling(WINDOW).std() * np.sqrt(252) * 100   # annualised %

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, company in enumerate(vol.columns):
        series = vol[company].dropna()
        ax.plot(series.index, series.values,
                color=PALETTE[i % len(PALETTE)],
                linewidth=1.6, label=company, alpha=0.88)

    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.set_title(f"30-Day Rolling Annualised Volatility — UK Retailers  [3 Years]")
    ax.set_xlabel("")
    ax.set_ylabel("Annualised Volatility  (%)")
    ax.legend(fontsize=8.5, framealpha=0.0, ncol=2)
    ax.grid(axis="y")
    fig.tight_layout()
    save(fig, "16_rolling_volatility.png")


# ══════════════════════════════════════════════════════════════════════════════
# 17  DUAL-AXIS OVERLAY — ONS retail index + GBP/EUR
# ══════════════════════════════════════════════════════════════════════════════

def chart_dual_axis_overlay():
    ons = query("""
        WITH top_sector AS (
            SELECT sector FROM (
                SELECT sector, COUNT(*) AS n
                FROM ons_retail_monthly
                GROUP BY sector ORDER BY n DESC LIMIT 1
            )
        )
        SELECT year, month, AVG(value) AS idx
        FROM ons_retail_monthly
        WHERE sector = (SELECT sector FROM top_sector)
          AND year >= 2015
        GROUP BY year, month
        ORDER BY year, month
    """)
    fx = query("""
        SELECT date, AVG(rate) AS rate
        FROM fx_rates
        WHERE currency = 'EUR'
        GROUP BY date
        ORDER BY date
    """)

    if ons.empty or fx.empty:
        print("  ✗  17_dual_axis_overlay.png  (no data)")
        return

    ons["period"] = pd.to_datetime(dict(year=ons["year"], month=ons["month"], day=1))
    fx["date"]    = pd.to_datetime(fx["date"])

    # Monthly avg FX
    fx["month_period"] = fx["date"].dt.to_period("M").dt.to_timestamp()
    fx_monthly = fx.groupby("month_period")["rate"].mean().reset_index()
    fx_monthly.columns = ["period", "rate"]

    merged = ons.merge(fx_monthly, on="period", how="inner")

    fig, ax1 = plt.subplots(figsize=(13, 5))
    ax2 = ax1.twinx()

    ax1.fill_between(merged["period"], merged["idx"],
                     alpha=0.12, color=ACCENT)
    ax1.plot(merged["period"], merged["idx"],
             color=ACCENT, linewidth=2, label="ONS Retail Index (left)")
    ax2.plot(merged["period"], merged["rate"],
             color=ACCENT2, linewidth=2, linestyle="--",
             label="GBP/EUR (right)")

    ax1.set_ylabel("ONS Retail Sales Index  (2019 = 100)", color=ACCENT)
    ax2.set_ylabel("GBP / EUR Rate", color=ACCENT2)
    ax2.tick_params(axis="y", colors=ACCENT2)
    ax1.set_title("UK Retail Sales Index vs GBP/EUR Exchange Rate  [2015–present]")
    ax1.set_xlabel("")
    ax1.grid(axis="y")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               framealpha=0.0, fontsize=9)

    fig.tight_layout()
    save(fig, "17_dual_axis_overlay.png")


# ══════════════════════════════════════════════════════════════════════════════
# 18  52-WEEK RANGE — price position within high/low band
# ══════════════════════════════════════════════════════════════════════════════

def chart_52week_range():
    df = query("""
        SELECT company, ticker, close, date
        FROM stocks_daily
        WHERE date >= date('now', '-365 days')
        ORDER BY date
    """)

    if df.empty:
        print("  ✗  18_52week_range.png  (no data)")
        return

    df["date"] = pd.to_datetime(df["date"])
    summary = df.groupby("company").agg(
        low_52w=("close", "min"),
        high_52w=("close", "max"),
        current=("close", "last"),
    ).reset_index()
    summary["range"] = summary["high_52w"] - summary["low_52w"]
    summary["pct_from_low"] = ((summary["current"] - summary["low_52w"])
                               / summary["range"] * 100)
    summary = summary.sort_values("pct_from_low", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    y_pos = range(len(summary))

    # Grey background bar (full 52-wk range)
    ax.barh(y_pos, [100] * len(summary), left=0,
            color="#21262d", height=0.55)

    # Coloured fill: how far current price is from 52-wk low
    bar_colors = [ACCENT3 if v >= 50 else ACCENT4
                  for v in summary["pct_from_low"]]
    ax.barh(y_pos, summary["pct_from_low"],
            color=bar_colors, height=0.55, alpha=0.88)

    # Current price dot
    ax.scatter(summary["pct_from_low"], y_pos,
               color="white", s=60, zorder=5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(summary["company"], fontsize=9)
    ax.set_xlabel("Position within 52-week range  (0% = 52-wk low, 100% = 52-wk high)")
    ax.axvline(50, color="#8b949e", linewidth=1, linestyle="--")
    ax.set_xlim(0, 105)
    ax.set_title("52-Week Price Range — UK Retailers\n"
                 "[dot = current price position; green = above midpoint]")
    ax.grid(axis="x")

    # Annotate pct
    for i, (_, row) in enumerate(summary.iterrows()):
        ax.text(row["pct_from_low"] + 2, i,
                f"{row['pct_from_low']:.0f}%",
                va="center", fontsize=8.5, color="#c9d1d9")

    fig.tight_layout()
    save(fig, "18_52week_range.png")


# ══════════════════════════════════════════════════════════════════════════════
# 19  POLAR RADAR — sector performance across multiple metrics
# ══════════════════════════════════════════════════════════════════════════════

def chart_polar_radar():
    df = query("""
        SELECT sector,
               AVG(value)  AS avg_idx,
               MAX(value)  AS peak_idx,
               MIN(value)  AS trough_idx,
               COUNT(*)    AS n_months,
               STDEV(value) AS vol_idx
        FROM ons_retail_monthly
        WHERE year >= 2010
          AND sector IS NOT NULL AND sector != ''
        GROUP BY sector
    """)

    # SQLite doesn't have STDEV — compute manually if needed
    if "vol_idx" not in df.columns or df["vol_idx"].isna().all():
        all_df = query("""
            SELECT sector, value
            FROM ons_retail_monthly
            WHERE year >= 2010 AND sector IS NOT NULL
        """)
        vol = all_df.groupby("sector")["value"].std().reset_index()
        vol.columns = ["sector", "vol_idx"]
        df = df.merge(vol, on="sector", how="left")

    if df.empty or len(df) < 3:
        print("  ✗  19_polar_radar.png  (no data)")
        return

    # Normalise each metric 0–1 for the radar
    metrics = ["avg_idx", "peak_idx", "n_months"]
    # vol = lower is better, so invert
    df["stability"] = 1 / (df["vol_idx"] + 0.01)

    metrics = ["avg_idx", "peak_idx", "stability"]
    metric_labels = ["Avg Index", "Peak Index", "Stability\n(low vol)"]

    # Normalize
    for m in metrics:
        df[f"{m}_norm"] = (df[m] - df[m].min()) / (df[m].max() - df[m].min() + 1e-9)

    norm_cols = [f"{m}_norm" for m in metrics]
    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10),
                           subplot_kw=dict(polar=True))
    ax.set_facecolor("#0f1117")

    for i, (_, row) in enumerate(df.iterrows()):
        values = [row[c] for c in norm_cols] + [row[norm_cols[0]]]
        ax.plot(angles, values, color=PALETTE[i % len(PALETTE)],
                linewidth=2, alpha=0.9,
                label=row["sector"].replace("-", " ").title())
        ax.fill(angles, values, alpha=0.08,
                color=PALETTE[i % len(PALETTE)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=10, color="#c9d1d9")
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)
    ax.spines["polar"].set_color("#2e2e3e")
    ax.grid(color="#2e2e3e", linewidth=0.8)

    ax.set_title("Retail Sector Performance Radar\n[ONS, normalised 0–1, 2010–present]",
                 pad=30)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
              fontsize=8.5, framealpha=0.0)

    fig.tight_layout()
    save(fig, "19_polar_radar.png")


# ══════════════════════════════════════════════════════════════════════════════
# 20  VOLUME CALENDAR HEATMAP — Tesco monthly avg volume
# ══════════════════════════════════════════════════════════════════════════════

def chart_volume_heatmap():
    df = query("""
        SELECT date, volume
        FROM stocks_daily
        WHERE ticker = 'TSCO.L'
        ORDER BY date
    """)

    if df.empty:
        print("  ✗  20_volume_heatmap.png  (no data)")
        return

    df["date"]  = pd.to_datetime(df["date"])
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month

    pivot = df.groupby(["year", "month"])["volume"].mean().reset_index()
    pivot = pivot.pivot(index="month", columns="year", values="volume")
    pivot = pivot / 1e6   # convert to millions
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot.index = [month_names[i - 1] for i in pivot.index]

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.2), 7))
    sns.heatmap(
        pivot, ax=ax,
        cmap="rocket_r",
        annot=True, fmt=".0f",
        linewidths=0.5, linecolor="#0f1117",
        cbar_kws={"label": "Avg Daily Volume (M shares)"},
        annot_kws={"size": 9},
    )
    ax.set_title("Tesco (TSCO.L) — Monthly Average Trading Volume  (M shares)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    save(fig, "20_volume_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# PATCH: SQLite doesn't have STDEV — add it as a custom function
# ══════════════════════════════════════════════════════════════════════════════

import math

class _StdevFunc:
    def __init__(self):
        self.M = 0.0
        self.S = 0.0
        self.k = 0

    def step(self, value):
        if value is None:
            return
        self.k += 1
        tM = self.M + (value - self.M) / self.k
        self.S += (value - self.M) * (value - tM)
        self.M = tM

    def finalize(self):
        if self.k < 2:
            return None
        return math.sqrt(self.S / (self.k - 1))


def _patched_query(sql: str) -> pd.DataFrame:
    """query() with STDEV user-defined aggregate."""
    conn = sqlite3.connect(DB_PATH)
    conn.create_aggregate("STDEV", 1, _StdevFunc)
    df   = pd.read_sql_query(sql, conn)
    conn.close()
    return df


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    if not DB_PATH.exists():
        print(f"ERROR: {DB_PATH} not found.")
        print("Run:  python scripts/fetch_live_data.py  first.")
        return

    has_ons    = table_exists("ons_retail_monthly")
    has_stocks = table_exists("stocks_daily")
    has_info   = table_exists("stocks_info")
    has_fx     = table_exists("fx_rates")

    print("Generating additional charts...\n")
    print(f"  Data  ONS={'✓' if has_ons else '✗'}  "
          f"stocks={'✓' if has_stocks else '✗'}  "
          f"info={'✓' if has_info else '✗'}  "
          f"FX={'✓' if has_fx else '✗'}\n")

    # Override query for STDEV support
    global query
    query = _patched_query

    if has_ons:
        chart_sector_donut()
        chart_sector_stacked_area()
        chart_sector_boxplot()

    if has_stocks and has_info:
        chart_bubble()

    if has_stocks:
        chart_correlation_heatmap()
        chart_candlestick()
        chart_return_distribution()
        chart_drawdown()
        chart_rolling_volatility()
        chart_volume_heatmap()
        chart_52week_range()

    if has_ons and has_fx:
        chart_dual_axis_overlay()

    if has_ons:
        chart_polar_radar()

    print(f"\nDone. All charts saved to {CHARTS_DIR}/")


if __name__ == "__main__":
    main()
