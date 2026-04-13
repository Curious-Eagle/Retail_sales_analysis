-- sql/analysis_queries.sql
--
-- All the core SQL queries used in the analysis.
-- These run against the `transactions` table in retail.db.
-- You can run them individually in DB Browser for SQLite or any SQL client
-- that supports SQLite, or just let analyse.py execute them for you.
--
-- I've grouped them into five themes that mirror the questions
-- I was trying to answer:
--   1. Revenue trends
--   2. Month-on-month growth
--   3. Customer retention
--   4. Regional variance
--   5. Underperforming categories
--
-- All queries use CTEs and window functions throughout.
-- ─────────────────────────────────────────────────────────────────────────────


-- ── 1. MONTHLY REVENUE TREND ─────────────────────────────────────────────────
-- Simple monthly rollup to see the overall shape of the business over time.

SELECT
    year,
    month,
    COUNT(*)                          AS transactions,
    ROUND(SUM(revenue), 2)            AS total_revenue,
    ROUND(SUM(profit), 2)             AS total_profit,
    ROUND(AVG(revenue), 2)            AS avg_order_value,
    ROUND(SUM(profit) / NULLIF(SUM(revenue), 0) * 100, 2) AS margin_pct
FROM transactions
GROUP BY year, month
ORDER BY year, month;


-- ── 2. MONTH-ON-MONTH REVENUE GROWTH (window function) ───────────────────────
-- Uses LAG() to calculate how each month compares to the one before it.
-- Negative growth months are the ones worth digging into.

WITH monthly AS (
    SELECT
        year,
        month,
        ROUND(SUM(revenue), 2) AS revenue
    FROM transactions
    GROUP BY year, month
),
with_lag AS (
    SELECT
        year,
        month,
        revenue,
        LAG(revenue) OVER (ORDER BY year, month) AS prev_month_revenue
    FROM monthly
)
SELECT
    year,
    month,
    revenue,
    prev_month_revenue,
    ROUND(
        (revenue - prev_month_revenue) / NULLIF(prev_month_revenue, 0) * 100,
    2) AS mom_growth_pct
FROM with_lag
ORDER BY year, month;


-- ── 3. CUSTOMER RETENTION RATE ────────────────────────────────────────────────
-- Looks at what % of customers in each month had also purchased in the
-- previous month. Rough proxy for retention since we don't have account data.

WITH monthly_customers AS (
    SELECT DISTINCT
        customer_id,
        year,
        month,
        -- build a sortable period number so we can do the LAG easily
        year * 100 + month AS period
    FROM transactions
),
with_prev AS (
    SELECT
        mc.customer_id,
        mc.period,
        mc.year,
        mc.month,
        -- did this customer appear in the immediately prior month?
        MAX(CASE WHEN prev.customer_id IS NOT NULL THEN 1 ELSE 0 END) AS retained
    FROM monthly_customers mc
    LEFT JOIN monthly_customers prev
        ON  mc.customer_id = prev.customer_id
        AND prev.period     = mc.period - 1   -- previous calendar month
    GROUP BY mc.customer_id, mc.period, mc.year, mc.month
)
SELECT
    year,
    month,
    COUNT(*)                                    AS unique_customers,
    SUM(retained)                               AS retained_customers,
    ROUND(SUM(retained) * 100.0 / COUNT(*), 2) AS retention_rate_pct
FROM with_prev
GROUP BY year, month
ORDER BY year, month;


-- ── 4. REGIONAL SALES VARIANCE ────────────────────────────────────────────────
-- Compares each region's revenue against the overall average.
-- Regions sitting below average are candidates for investigation.

WITH region_totals AS (
    SELECT
        region,
        ROUND(SUM(revenue), 2)  AS total_revenue,
        ROUND(AVG(revenue), 2)  AS avg_order_value,
        COUNT(*)                AS transactions,
        COUNT(DISTINCT customer_id) AS unique_customers
    FROM transactions
    GROUP BY region
),
overall AS (
    SELECT ROUND(AVG(total_revenue), 2) AS avg_region_revenue
    FROM region_totals
)
SELECT
    r.region,
    r.total_revenue,
    r.avg_order_value,
    r.transactions,
    r.unique_customers,
    o.avg_region_revenue,
    ROUND(r.total_revenue - o.avg_region_revenue, 2) AS variance_vs_avg,
    ROUND((r.total_revenue - o.avg_region_revenue) / o.avg_region_revenue * 100, 2) AS variance_pct
FROM region_totals r
CROSS JOIN overall o
ORDER BY r.total_revenue DESC;


-- ── 5. CATEGORY PERFORMANCE + UNDERPERFORMERS ────────────────────────────────
-- Looks at each category's revenue, margin, and average order value.
-- Anything sitting below the median margin is worth flagging.

WITH cat_stats AS (
    SELECT
        category,
        ROUND(SUM(revenue), 2)      AS total_revenue,
        ROUND(SUM(profit), 2)       AS total_profit,
        ROUND(AVG(revenue), 2)      AS avg_order_value,
        ROUND(AVG(quantity), 2)     AS avg_quantity,
        COUNT(*)                    AS transactions,
        ROUND(SUM(profit) / NULLIF(SUM(revenue), 0) * 100, 2) AS margin_pct,
        ROUND(AVG(discount_pct), 2) AS avg_discount
    FROM transactions
    GROUP BY category
),
ranked AS (
    SELECT *,
        RANK() OVER (ORDER BY total_revenue DESC) AS revenue_rank,
        RANK() OVER (ORDER BY margin_pct   DESC) AS margin_rank
    FROM cat_stats
)
SELECT *
FROM ranked
ORDER BY total_revenue DESC;


-- ── 6. CHANNEL BREAKDOWN ──────────────────────────────────────────────────────
-- How do Online, In-Store, and Mobile App compare on revenue and AOV?

SELECT
    channel,
    COUNT(*)                        AS transactions,
    ROUND(SUM(revenue), 2)          AS total_revenue,
    ROUND(AVG(revenue), 2)          AS avg_order_value,
    ROUND(SUM(profit), 2)           AS total_profit,
    ROUND(AVG(discount_pct), 2)     AS avg_discount_pct
FROM transactions
GROUP BY channel
ORDER BY total_revenue DESC;


-- ── 7. TOP 10 HIGHEST REVENUE MONTHS ────────────────────────────────────────

SELECT
    year,
    month,
    ROUND(SUM(revenue), 2) AS revenue
FROM transactions
GROUP BY year, month
ORDER BY revenue DESC
LIMIT 10;
