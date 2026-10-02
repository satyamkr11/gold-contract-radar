# 🪙 Gold Contract Radar — MCX Commodity Derivatives Intelligence

**Hack in Hills '26 — Problem 3: Commodity Derivatives Intelligence**  
*Built for production-grade cross-contract relative value analysis, normalization, liquidity filtering, and walk-forward quantitative backtesting on Multi Commodity Exchange of India (MCX) Gold derivatives.*

---

## 📌 Project Overview

Gold derivatives on MCX are traded across multiple contract variants designed for different market participants (institutional, retail, jewellers, and systematic prop desks):
1. **GOLDM (Gold Mini)**
2. **GOLDTEN (Gold 10)**
3. **GOLDGUINEA (Gold Guinea)**
4. **GOLDPETAL (Gold Petal)**

Each contract differs in **lot size**, **quotation unit**, **purity/fineness**, and **expiry schedule**. Because quotes are not on an identical physical basis, raw prices cannot be compared directly without miscalculating spreads.

**Gold Contract Radar** solves this by:
- Ingesting official MCX Bhavcopy data with full schema validation and sanitization.
- Normalizing all quotes to a common standard: **₹ per gram of 100% fine gold equivalent**.
- Computing cross-contract pairwise spreads and strictly no-lookahead rolling Z-scores.
- Modeling realistic statutory exchange fees, taxes (CTT, Stamp Duty, GST), brokerage, and slippage.
- Executing a walk-forward backtester that respects physical contract expiries and avoids synthetic data.
- Providing an interactive Streamlit intelligence dashboard.

---

## 📐 MCX Gold Contract Specifications

| Symbol | Contract Name | Physical Lot Size | Quotation Unit | Purity (Fineness) | Fine Gold / Lot | Tick Size | Divisor ($U_{quote} \times Purity$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **GOLDM** | Gold Mini | 100 grams | per 10 grams | 995 (99.5% pure) | 99.50 grams | ₹1.00 | **9.9500** |
| **GOLDTEN** | Gold 10 | 10 grams | per 10 grams | 999 (99.9% pure) | 9.99 grams | ₹1.00 | **9.9900** |
| **GOLDGUINEA**| Gold Guinea | 8 grams | per 8 grams | 999 (99.9% pure) | 7.992 grams | ₹1.00 | **7.9920** |
| **GOLDPETAL** | Gold Petal | 1 gram | per 1 gram | 999 (99.9% pure) | 0.999 grams | ₹1.00 | **0.9990** |

---

## 🧮 Mathematical Normalization Methodology

The raw exchange quotation $Q$ is quoted in rupees per Quotation Unit ($U_{quote}$ grams of alloy at purity $P_{fineness}$).

### 1. Fine Gold Normalization Formula
$$\text{Fine Gold per Quote Unit} = U_{quote} \times \left(\frac{P_{fineness}}{1000}\right)$$

$$P_{normalized} = \frac{Q}{\text{Fine Gold per Quote Unit}} = \frac{Q}{U_{quote} \times (P_{fineness} / 1000)} \quad (\text{₹ / gram of 100\% fine gold})$$

- **GOLDM**: $P_{norm} = \frac{Q}{10 \times 0.995} = \frac{Q}{9.95}$
- **GOLDTEN**: $P_{norm} = \frac{Q}{10 \times 0.999} = \frac{Q}{9.99}$
- **GOLDGUINEA**: $P_{norm} = \frac{Q}{8 \times 0.999} = \frac{Q}{7.992}$
- **GOLDPETAL**: $P_{norm} = \frac{Q}{1 \times 0.999} = \frac{Q}{0.999}$

### 2. Relative Value Pair Spreads
For Leg A and Leg B on trade date $t$:
$$\text{Spread}_{A-B}(t) = P_{norm, A}(t) - P_{norm, B}(t) \quad (\text{₹ / gram fine gold})$$

$$\text{Spread}_{\%}(t) = \left(\frac{P_{norm, A}(t) - P_{norm, B}(t)}{P_{norm, B}(t)}\right) \times 100$$

### 3. Strict No-Lookahead Rolling Z-Score
To prevent lookahead bias, rolling mean ($\mu$) and sample standard deviation ($\sigma$) at day $t$ are calculated **strictly from historical observations preceding day $t$** ($t-W$ to $t-1$):
$$\mu_{t-1} = \frac{1}{W} \sum_{k=1}^{W} \text{Spread}(t-k), \quad \sigma_{t-1} = \sqrt{\frac{1}{W-1} \sum_{k=1}^{W} (\text{Spread}(t-k) - \mu_{t-1})^2}$$

$$Z(t) = \frac{\text{Spread}(t) - \mu_{t-1}}{\sigma_{t-1}}$$

- **Overbought / Upper Threshold ($Z > Z_{entry}$):** Leg A overpriced relative to Leg B $\rightarrow$ **Short A / Long B**.
- **Oversold / Lower Threshold ($Z < -Z_{entry}$):** Leg A underpriced relative to Leg B $\rightarrow$ **Long A / Short B**.
- **Mean Reversion Target ($|Z| \le Z_{exit}$):** Dislocation resolved $\rightarrow$ **Exit / Close Spread**.
- **Risk Stop ($|Z| \ge Z_{stop}$):** Spread diverges beyond tolerance $\rightarrow$ **Stop Loss Triggered**.

---

## 💸 Cost & Liquidity Friction Model

Relative value spreads are only tradable if expected convergence exceeds execution friction. The engine implements:
1. **Exchange Turnover Fees:** 0.0021% on total traded turnover.
2. **CTT (Commodity Transaction Tax):** 0.01% on non-agri futures sell side turnover.
3. **Stamp Duty:** 0.002% on buy side turnover.
4. **SEBI Regulatory Fee:** 0.0001% on turnover.
5. **GST:** 18% applied on (Brokerage + Exchange Fees + SEBI Fees).
6. **Brokerage:** Configurable (default: flat ₹20/order).
7. **Slippage:** Configurable per leg (default: 2.0 basis points).
8. **Liquidity Filters:** Minimum Volume (default: 5 lots), Minimum Open Interest (default: 10 lots), Minimum Traded Value (₹1.0 Lakh).

---

## 🧪 Walk-Forward Backtesting Rules

- **Signal Generation:** Generated at close of Day $t$.
- **Execution Timing:** Executed no earlier than Day $t+1$ at Open/Close prices (zero lookahead).
- **Contract Identity:** Positions are explicitly bound to `(Symbol, ExpiryDate)`.
- **Physical Expiry Protection:** Positions are systematically closed/rolled $N$ days before expiry (default: 2 days) to avoid physical delivery tender cycles.
- **Statistical Sufficiency Check:** If the dataset contains insufficient daily files ($< 15$ trade dates), the system explicitly reports **"Insufficient historical data"** rather than fabricating fake fills or synthetic history.

---

## 📂 Project Architecture

```
gold-contract-radar/
├── app.py                     # Interactive Streamlit intelligence dashboard
├── pytest.ini                 # Pytest configuration
├── README.md                  # System documentation and quantitative specifications
├── data/
│   ├── raw/                   # Raw unmodified MCX Bhavcopy CSV files
│   │   └── BhavCopyDateWise_01102026.csv
│   └── processed/             # Cleaned and normalized datasets
│       ├── gold_master.csv
│       └── gold_master.parquet
├── src/
│   ├── __init__.py
│   ├── cleaner.py             # Data inspection, whitespace normalization, schema validation
│   ├── normalization.py       # Contract specs and ₹/g fine gold normalization
│   ├── relative_value.py      # Cross-contract spreads and no-lookahead rolling Z-scores
│   ├── filters.py             # Cost model, statutory taxes, and liquidity filters
│   ├── pipeline.py            # Multi-file Bhavcopy ingestion and MCX downloader interface
│   └── backtest.py            # Walk-forward backtester and statistical sufficiency auditor
└── tests/
    └── test_radar.py          # Complete unit test suite (9 passing tests)
```

---

## 🚀 How to Run

### 1. Run Unit Tests
```bash
python -m pytest
```

### 2. Run Data Pipeline & Export Processed Data
```bash
python -c "from src.pipeline import MCXDataPipeline; p = MCXDataPipeline(); df, res = p.ingest_all(); print('Ingested', len(df), 'rows into', res.parquet_path)"
```

### 3. Launch Streamlit Intelligence Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8502`.

---

## 📊 Dashboard Modules

1. **📍 TODAY:** Snapshot of active contract prices, normalized ₹/g fine values, volume, open interest, liquidity badges, and instant pairwise spreads.
2. **📈 SPREADS:** Interactive spread visualizer with exact expiry date, nearest cycle ($M_1, M_2\dots$), and all-pairs pairing modes, rolling Z-score indicator, trade signals, and breakeven cost estimator.
3. **🧪 BACKTEST:** Strict walk-forward quantitative simulator with configurable lookback, Z-thresholds, slippage, and statistical sufficiency checks. Disaggregates Gross Profit Factor vs Net Profit Factor after friction.
4. **📅 CALENDAR:** Full contract lifecycle matrix, delivery units, tick sizes, and listing dates.
5. **🩺 DATA HEALTH:** Automated audit checklist of schema conformance, authoritative internal MCX date verification, deduplication, and data integrity.

---

## ⚠️ Key Assumptions & Quantitative Methodology

- **Relative Value vs Prediction:** This system is strictly an MCX Commodity Derivatives Intelligence engine for cross-contract relative value dislocation analysis; it does not attempt unconstrained directional price prediction.
- **Settlement Prices vs Executable Fills:** Daily Bhavcopy settlement prices represent official exchange mark-to-market prices. Real-time intraday bid/ask spreads may vary.
- **Volume & Open Interest as Liquidity Proxies:** High volume does not represent order-book market depth; the slippage and friction model should be calibrated accordingly.
- **Contract Availability:** GOLDTEN only exists across its actual listed horizon; historical data prior to listing is never fabricated.
- **Gross vs Net Friction Reality:** Raw statistical spread convergence may produce positive gross gains, but statutory taxes (CTT, Stamp Duty, GST, Exchange charges) and slippage establish a high hurdle rate. Transparent modeling exposes whether an opportunity is genuinely executable.
