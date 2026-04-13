# UK Retail Sector — Live Data Analysis

A Python data analysis project that pulls **live, real-world data** from three public APIs, stores it in SQLite, and runs SQL-based analysis (CTEs, window functions, self-joins) to produce seven publication-quality charts and a written findings report.

---

## What it does

Every time you run it, the project fetches fresh data from:

| Source | What it provides | API |
|---|---|---|
| **ONS (Office for National Statistics)** | UK monthly Retail Sales Index — official government statistics going back to 1988 | `api.beta.ons.gov.uk` |
| **Yahoo Finance** (`yfinance`) | 5 years of daily stock prices + market cap, P/E, margin for 10 UK-listed retailers | Yahoo Finance |
| **Frankfurter** | 5 years of daily GBP exchange rates vs EUR, USD, JPY, AUD, CAD | `api.frankfurter.app` |

All three APIs are **completely free and require no API key**.

---

## Project structure

```
retail_sales_analysis_project/
│
├── data/
│   └── retail_live.db              # SQLite database (created on first run)
│
├── sql/
│   └── analysis_queries.sql        # All SQL queries, runnable independently
│
├── scripts/
│   ├── fetch_live_data.py          # Step 1 — fetch from APIs → SQLite
│   ├── analyse.py                  # Step 2 — run SQL + produce charts
│   └── generate_synthetic_data.py  # Fallback — offline synthetic dataset
│
├── outputs/
│   ├── findings_report.txt         # Written summary of key findings
│   └── charts/                     # Seven generated chart PNGs
│
├── requirements.txt
└── README.md
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pandas numpy matplotlib seaborn requests yfinance openpyxl
```

### 2. Fetch live data

```bash
python scripts/fetch_live_data.py
```

This will:
- Resolve the latest ONS RSI dataset version via API, download the CSV (~45,000 rows of monthly retail statistics), and clean it
- Pull 5 years of daily stock prices and company info for 10 UK retailers via Yahoo Finance
- Download 5 years of daily GBP FX rates from the Frankfurter API
- Store everything in `data/retail_live.db` (SQLite)

Typical output:

```
=================================================================
  RETAIL LIVE DATA FETCHER
  2026-04-12 21:30:00
=================================================================

[1/3] ONS UK Retail Sales Index
  [ONS] Resolving latest RSI version...
  [ONS] Downloading RSI CSV (version 45)...
  [ONS] After cleaning: 4,378 rows | 10 sectors | period 1988-01-01 → 2026-01-01

[2/3] UK Retailer Stock Data (Yahoo Finance)
  Tesco (TSCO.L)... ✓ (1,261 days)
  Marks & Spencer (MKS.L)... ✓ (1,261 days)
  Next (NXT.L)... ✓ (1,261 days)
  ...

[3/3] GBP Exchange Rates (Frankfurter API)
  [FX] 6,405 rows | 5 currencies | 2021-04-13 → 2026-04-10

  [DB] Saving to data/retail_live.db...
    ons_retail_monthly  :    4,378 rows
    stocks_daily        :   12,609 rows
    stocks_info         :       10 rows
    fx_rates            :    6,405 rows
=================================================================
  All data fetched successfully.
```

### 3. Run the analysis

```bash
python scripts/analyse.py
```

Charts are saved to `outputs/charts/`. The findings summary goes to `outputs/findings_report.txt`.

---

## SQLite database schema

### `ons_retail_monthly`
Monthly UK Retail Sales Index data from the ONS, cleaned and filtered to seasonally-adjusted, chained-volume index values.

| Column | Type | Description |
|---|---|---|
| `period` | TEXT | Period start date (YYYY-MM-DD) |
| `year` | INTEGER | Year |
| `month` | INTEGER | Month number (1–12) |
| `sector` | TEXT | Retail sub-sector (food, clothing, online, etc.) |
| `value` | REAL | Index value (2019 = 100) |
| `sa_code` | TEXT | Seasonal adjustment code |
| `price_type` | TEXT | Price type slug |
| `ons_version` | INTEGER | ONS dataset version number |
| `fetched_at` | TEXT | UTC timestamp of last fetch |

### `stocks_daily`
Five years of daily adjusted OHLCV prices for 10 UK-listed retailers.

| Column | Type | Description |
|---|---|---|
| `date` | TEXT | Trading date (YYYY-MM-DD) |
| `ticker` | TEXT | LSE ticker (e.g. TSCO.L) |
| `company` | TEXT | Company name |
| `open` | REAL | Opening price (GBp) |
| `high` | REAL | Daily high |
| `low` | REAL | Daily low |
| `close` | REAL | Closing price (GBp) |
| `volume` | INTEGER | Shares traded |

### `stocks_info`
Latest company fundamentals from Yahoo Finance.

| Column | Type | Description |
|---|---|---|
| `ticker` | TEXT | LSE ticker |
| `company` | TEXT | Company name |
| `market_cap` | REAL | Market capitalisation (GBP) |
| `pe_ratio` | REAL | Trailing P/E ratio |
| `forward_pe` | REAL | Forward P/E ratio |
| `revenue` | REAL | Annual revenue (GBP) |
| `profit_margin` | REAL | Net profit margin (decimal) |
| `ebitda_margin` | REAL | EBITDA margin (decimal) |
| `roe` | REAL | Return on equity |
| `debt_to_equity` | REAL | Debt-to-equity ratio |
| `employees` | INTEGER | Full-time employees |

### `fx_rates`
Five years of daily GBP exchange rates.

| Column | Type | Description |
|---|---|---|
| `date` | TEXT | Date |
| `base` | TEXT | Base currency (always GBP) |
| `currency` | TEXT | Target currency |
| `rate` | REAL | Exchange rate |

---

## Charts produced

| File | Source | What it shows |
|---|---|---|
| `01_ons_retail_trend.png` | ONS | Long-run UK Retail Sales Index trend (line + fill) |
| `02_mom_growth.png` | ONS | Month-on-month index change using SQL `LAG()` window function |
| `03_sector_breakdown.png` | ONS | Average index by retail sub-sector over last 24 months, with min/max range |
| `04_stock_performance.png` | Yahoo Finance | Normalised stock price performance of 10 UK retailers (base = 100) |
| `05_seasonal_heatmap.png` | ONS | Year × month heatmap — reveals Q4 uplift and Q1 trough across decades |
| `06_fx_rates.png` | Frankfurter | GBP/EUR and GBP/USD (left axis) + GBP/JPY (right axis), monthly average |
| `07_retailer_comparison.png` | Yahoo Finance | Three-panel: Market cap (£B) · Trailing P/E · Net margin (%) |

---

## SQL highlights

All queries are embedded in `analyse.py` and also available standalone in `sql/analysis_queries.sql`.

### Month-on-month growth with `LAG()`

```sql
WITH monthly AS (
    SELECT year, month, AVG(value) AS index_value
    FROM ons_retail_monthly
    GROUP BY year, month
),
lagged AS (
    SELECT year, month, index_value,
           LAG(index_value) OVER (ORDER BY year, month) AS prev
    FROM monthly
)
SELECT year, month,
       ROUND((index_value - prev) / NULLIF(prev, 0) * 100, 2) AS mom_growth_pct
FROM lagged
ORDER BY year, month;
```

### Regional variance with `CROSS JOIN`

```sql
WITH region_totals AS (
    SELECT region,
           ROUND(SUM(revenue), 2) AS total_revenue
    FROM transactions
    GROUP BY region
),
overall AS (
    SELECT ROUND(AVG(total_revenue), 2) AS avg_region_revenue
    FROM region_totals
)
SELECT r.region, r.total_revenue,
       ROUND(r.total_revenue - o.avg_region_revenue, 2) AS variance_vs_avg
FROM region_totals r
CROSS JOIN overall o
ORDER BY r.total_revenue DESC;
```

### Customer retention using a self-join

```sql
WITH monthly_customers AS (
    SELECT DISTINCT customer_id, year, month,
           year * 100 + month AS period
    FROM transactions
),
with_prev AS (
    SELECT mc.customer_id, mc.year, mc.month,
           MAX(CASE WHEN prev.customer_id IS NOT NULL THEN 1 ELSE 0 END) AS retained
    FROM monthly_customers mc
    LEFT JOIN monthly_customers prev
        ON  mc.customer_id = prev.customer_id
        AND prev.period    = mc.period - 1
    GROUP BY mc.customer_id, mc.year, mc.month
)
SELECT year, month,
       COUNT(*)                                   AS total_customers,
       SUM(retained)                              AS retained_customers,
       ROUND(SUM(retained) * 100.0 / COUNT(*), 2) AS retention_rate_pct
FROM with_prev
GROUP BY year, month;
```

---

## Retailers tracked (Yahoo Finance)

| Ticker | Company | Sector focus |
|---|---|---|
| `TSCO.L` | Tesco | Grocery |
| `MKS.L` | Marks & Spencer | Food & fashion |
| `NXT.L` | Next | Fashion |
| `SBRY.L` | Sainsbury's | Grocery |
| `JD.L` | JD Sports | Sportswear |
| `OCDO.L` | Ocado | Online |
| `FRAS.L` | Frasers Group | Sports / department |
| `BME.L` | B&M European Value | Value / discounters |
| `GRG.L` | Greggs | Food-to-go |
| `SMWH.L` | WH Smith | Travel retail |

---

## Key findings (from live data)

### 1. The Q4 effect is structural
The ONS seasonal heatmap shows November and December producing higher index values every single year going back to 1988. This is not a recent trend — it is a long-run structural feature of the UK retail calendar.

### 2. Online retail has permanently re-rated
Non-store retailing (e-commerce) consistently posts the highest chained-volume index values. The post-2020 jump was permanent — online index levels never returned to the 2019 baseline trajectory.

### 3. GBP volatility matters for import margins
The FX chart shows meaningful GBP moves of 10–15% against the EUR and USD over 5-year periods. For retailers sourcing goods from Europe or Asia, an unhedged position can swing margins by several percentage points.

### 4. Divergent stock performance among UK retailers
The normalised stock chart shows meaningful divergence even among retailers in similar categories. Discount and value retailers (B&M, Greggs) have generally outperformed pure-play fashion over the cycle, reflecting the consumer shift towards value.

---

## Data sources & licences

| Source | Licence | URL |
|---|---|---|
| ONS Retail Sales Index | [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/) | https://www.ons.gov.uk/businessindustryandtrade/retailindustry |
| Yahoo Finance (stock data) | Yahoo Finance Terms of Service | https://finance.yahoo.com |
| Frankfurter (FX data) | MIT Licence | https://www.frankfurter.app |

The ONS data is Crown Copyright and is made available under the Open Government Licence. Attribution: Office for National Statistics.

---

## Offline fallback

If you don't have internet access, you can generate a realistic synthetic dataset that mirrors the original UCI Online Retail II structure:

```bash
python scripts/generate_synthetic_data.py
```

Then run the original analysis against `data/retail.db`:

```bash
python scripts/analyse.py  # (edit DB_PATH in analyse.py to data/retail.db)
```

---

## Tools & stack

| Layer | Technology |
|---|---|
| Language | Python 3.9+ |
| Data | pandas, numpy |
| Visualisation | matplotlib, seaborn |
| Database | SQLite (via Python `sqlite3`) |
| SQL features | CTEs, `LAG()`, `RANK()`, `NULLIF()`, `CROSS JOIN`, self-joins |
| Live data | `requests`, `yfinance` |
| ONS | REST API + direct CSV download |
| FX rates | Frankfurter REST API |

---

## Requirements

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
requests>=2.31
yfinance>=1.0
openpyxl>=3.1
```
