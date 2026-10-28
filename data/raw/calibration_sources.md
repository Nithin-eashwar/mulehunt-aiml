# Calibration sources

Every `status: sourced` value in `simulator/config.yaml` traces to a row here.
Accessed 2026-10-27 unless stated. Secondary sources are labelled as such and
never cited as NPCI.

## Primary

| # | Source | What we took | Where in config |
|---|---|---|---|
| 1 | **NPCI UPI Product Statistics, monthly** — https://www.npci.org.in/product/upi/product-statistics (FY 2025-26 table) | Aug 2025: 20,008.31 Mn / ₹24,85,472.91 Cr; Sep: 19,633.43 / 24,89,736.54; Oct: 20,700.92 / 27,27,790.68; Nov: 20,466.98 / 26,31,632.63. Average ticket ₹1,242 / ₹1,268 / ₹1,318 / ₹1,286 | cross-check of #2 |
| 2 | **NPCI UPI Product Statistics, daily** (same page, "Daily Statistics" tab, Aug–Nov 2025) — saved as `data/raw/npci/upi_daily_2025-08_to_2025-11.csv`. Daily rows sum exactly to the monthly totals in #1 | Window average ticket ₹1,280; weekend volume within 1.2% of mean; Sunday value 0.83× mean; days 1–5 value 1.17× other days; 16–20 Oct volume +4.7% and 21–30 Oct −11.3% vs 6–15 Oct; 2 Oct +0.3% | `ticket_size.target_mean_inr`, `weekend.*`, `month_start.*`, `festivals` |
| 3 | **Lok Sabha Unstarred Question 2481**, Ministry of Finance, 15 Dec 2025 — https://eparlib.sansad.in/bitstream/123456789/3018042/1/AU2481_5bIzKk.pdf | FY2024-25: 12.64 lakh UPI fraud incidents, ₹981 crore → ₹7,761 per incident | `scenarios.avg_loss_check_inr` |
| 4 | **Diwali 2025 date** — Tribune India, https://www.tribuneindia.com/news/india/diwali-2025-date-confusion-cleared-which-day-will-india-celebrate-october-20-or-21/ ; five-day calendar (Dhanteras 18 Oct) — https://www.radhakrishnatemple.net/blog/diwali-2025-dates-5-days-festival-muhurat-timings-guide/ | Diwali 20 Oct 2025; Dhanteras 18 Oct 2025 (also NPCI's busiest day in the window) | `festivals` |
| 5 | **Project handoff** (`docs/specs/MuleHunt_PersonA_Handoff.md`) | 90-day window, split days, ₹10k/25k/50k thresholds, 1–2% label noise | `window.days`, `splits.*`, `fraud.thresholds`, `label_noise.rate` |
| 6 | **IBM AML benchmark, HI-Small laundering patterns** — https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml (CDLA-Sharing-1.0), saved as `data/raw/ibm_aml/HI-Small_Patterns.txt` | 370 attempts in 8 pattern types; structure check only | `simulator/validate.py` IBM section |

## Secondary (labelled `assumed` in config)

| Source | Figure | Used for |
|---|---|---|
| Outlook Money, "UPI transactions cross 16.99 billion in January 2025" — https://www.outlookmoney.com/banking/upi-transactions-cross-1699-billion-in-january-2025-marks-highest-growth-in-one-month | 62.35% P2M / 37.65% P2P by volume (FY2024-25 to Jan 2025); 86% of P2M payments ≤ ₹500 | `channel_mix`, `ticket_size.p2m` |
| Meetanshi, "UPI statistics" — https://meetanshi.com/blog/upi-statistics/ | Feb 2026 breakdown citing NPCI: 62.4% P2M | cross-check of `channel_mix` |
| Kaggle "UPI Transactions 2024 Dataset" (synthetic) | bank and state lists, market-share weighting idea | `bank_mix`, `state_mix` (design idea only, not data) |

## Not available

- **NPCI P2P vs P2M split:** NPCI's Ecosystem Statistics page (https://www.npci.org.in/product/ecosystem-statistics/upi) failed on 2026-10-27 (its data API returned 503 in the browser, 403 to direct requests). Retry before the final report; if it loads, replace the secondary figure and mark `channel_mix` as sourced.
- **Hour-of-day pattern:** NPCI publishes none. Assumed.
- **Dussehra 2025 date (2 Oct):** from general knowledge, coincides with Gandhi Jayanti; confirm against the official central government holiday list.
