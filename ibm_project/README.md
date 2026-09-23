# 🛰️ Retail Intelligence Platform
### Sales & Customer Analytics — End-to-End Business Intelligence Dashboard

A single-file, production-styled Streamlit application that turns a raw, messy
retail-transactions dataset (4,310 order-level records) into an interactive
Business Intelligence product — following the full BI flow taught in the
masterclass:

```
Data → Information → Insight → Decision → Action
```

---

## 1. What this project does

The dataset arrives with realistic real-world data problems: **3 mixed date
formats**, **109 duplicate order rows**, **missing values across 14 of 21
columns**, and **sentinel/impossible values** (age = 999, negative quantity,
negative shipping cost). The application:

1. **Validates** the raw data and produces a full Data Quality Report (nothing
   is silently changed).
2. **Cleans** it with a transparent, logged pipeline (deduplication, mixed-date
   parsing, text normalization, invalid-value handling).
3. **Engineers features** (time periods, profit margin, age bands, customer
   type, satisfaction bands).
4. **Analyzes** it across 8 dimensions — Sales, Product, Regional, Customer,
   Payment/Fulfillment, Ratings/Returns, and full Statistical EDA.
5. **Presents insights** through the masterclass's 5-level Insight Hierarchy:
   **KPIs → Trends → Drivers → Risk & Opportunity → Action** — ending every
   analysis in a concrete, data-derived business recommendation rather than
   just a chart.

No prediction/ML model is included by design — the masterclass explicitly
states a model is optional, and this dataset's strongest, most honest story is
told through rigorous descriptive + diagnostic analytics.

## 2. Tech stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Data processing | Pandas, NumPy |
| Visualization | Plotly (Graph Objects + Express) |
| Front-end / App framework | Streamlit (single-page app, custom CSS) |
| Styling | Custom futuristic design system — Orbitron / Exo 2 / Space Mono (Google Fonts), glassmorphism cards, neon glow accents |
| Machine learning model | None — pure analytical BI platform (by design; see Project Report) |

There is no separate "backend" service — Streamlit's script-rerun model serves
as both the data/analytics layer and the rendering layer in this one file.

## 3. Project structure

```
├── app.py                     # The entire application (data pipeline + analytics + UI) — ONE file
├── requirements.txt           # Exact dependency versions
├── retail_sales_dataset.csv   # Source dataset (place in the same folder as app.py)
├── README.md                  # This file
└── Retail_Intelligence_Project_Report.docx   # Written concept note / project report
```

## 4. Quick start — run it locally

```bash
# 1. Clone this repository
git clone <your-repo-url>
cd <your-repo-folder>

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Make sure retail_sales_dataset.csv is in the same folder as app.py

# 5. Run the app
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

## 5. Dataset

- **File:** `retail_sales_dataset.csv`
- **Grain:** one row per order line item (4,310 raw rows → 4,200 after cleaning)
- **Fields:** order id/date, customer demographics, region/city, product
  category & name, quantity, unit price, discount %, sales amount, profit,
  shipping cost, payment method, customer satisfaction (1–5), return flag,
  order status, days to ship
- The dataset used is **not** the dataset used in the masterclass sessions.

## 6. Dashboard pages

| Page | What it answers |
|---|---|
| Executive Overview | High-level KPIs for a CEO-level audience |
| Sales Performance | Revenue/profit trend, seasonality, and drivers |
| Product Analytics | Category & product-level revenue, margin, ranking |
| Regional Analytics | Region and city performance comparison |
| Customer Analytics | Demographics, repeat vs one-time behavior |
| Payment & Fulfillment | Payment method mix, shipping speed, order status |
| Ratings & Returns | Customer satisfaction and return-rate risk |
| Statistical EDA | Distributions, descriptive stats, correlation matrix |
| Data Quality | Full before/after validation & cleaning transparency |
| Business Insights | KPI → Trend → Driver → Risk/Opportunity → Action |

## 7. Notes for evaluators

- All figures on every page are computed live from the cleaned dataset —
  nothing is hard-coded.
- Global filters (Year, Region, Category, Payment Method) in the sidebar
  recompute every chart and KPI on the page.
- See the **Data Quality** page inside the app for the complete, numbered
  cleaning log.
