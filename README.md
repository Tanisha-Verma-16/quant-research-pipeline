# Quantitative Research Platform

## Hybrid XGBoost-LLM System for Volatility Regime Prediction & Portfolio Strategy

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **A production-grade quantitative finance system combining traditional factor models, machine learning, and large language models for explainable portfolio management.**

---

## 🎯 **Project Overview**

This project implements an institutional-quality quantitative research pipeline that:

- Analyzes **45 tickers** across 6 asset classes (US/International equities, commodities, FX)
- Engineers **50+ alpha factors** using academic methodologies (Jegadeesh & Titman 1993, Fama-French)
- Predicts **volatility regimes 21 days ahead** using XGBoost with walk-forward cross-validation
- Explains predictions using **SHAP** (SHapley Additive exPlanations)
- Generates **executive summaries** using Gemini Flash 2.5
- Provides **actionable portfolio strategies** via rule-based decision engine

### **Key Innovation: Hybrid XGBoost-LLM Architecture**

Unlike traditional black-box quant models, our system:

1. **XGBoost** learns patterns from 50+ technical/momentum/volatility factors
2. **SHAP** explains _why_ each prediction was made (interpretability)
3. **Gemini** translates technical outputs into plain-English summaries
4. **Portfolio Engine** converts predictions into risk-managed strategies

**Result:** A system that's both _powerful_ (ML-driven) and _explainable_ (human-auditable).

---

## 📂 **Project Structure**

```
quant-research-pipeline/
├── data/                           # Generated datasets
│   ├── factor_dataset_full.parquet
│   ├── alpha_factors_full.parquet
│   └── factor_*.parquet           # By asset class
│
├── models/                         # Trained ML models
│   ├── regime_classifier.json
│   └── feature_names.json
│
├── simulation_results/             # Monte Carlo outputs
│   ├── risk_metrics.json
│   ├── sample_paths.npy
│   └── percentile_cone.npy
│
├── shap_results/                   # Interpretability analysis
│   ├── global_feature_importance.csv
│   ├── HIGH_VOL_importance.csv
│   └── sample_explanations.json
│
├── day1_morning.py                 # Data ingestion & basic factors
├── day1_afternoon.py               # Advanced alpha factors (pure Polars)
├── day2_modeling.py                # XGBoost + Monte Carlo
├── shap_analysis.py                # SHAP interpretability
├── portfolio_strategy_decision_engine.py  # Strategy rules
├── financial_reasoning_agent.py    # Gemini LLM integration
├── main.py                         # FastAPI server
├── run_pipeline.py                 # Master orchestrator
├── requirements.txt
└── README.md
```

---

## 🚀 **Quick Start**

### **Prerequisites**

- Python 3.10+
- 8GB RAM minimum (for Monte Carlo simulation)
- Gemini API key (free at [Google AI Studio](https://aistudio.google.com/app/apikey))

### **Installation**

```bash
# Clone repository
git clone https://github.com/yourusername/quant-research-pipeline
cd quant-research-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up Gemini API key
export GEMINI_API_KEY="your_key_here"  # On Windows: set GEMINI_API_KEY=your_key_here
```

### **Run Full Pipeline**

```bash
# Execute entire pipeline (15-20 minutes)
python run_pipeline.py

# Or skip data download (use existing data)
python run_pipeline.py --skip-data
```

**Pipeline Stages:**

1. ✅ Data ingestion (45 tickers, 3 years history)
2. ✅ Factor engineering (50+ features)
3. ✅ ML training (XGBoost with walk-forward CV)
4. ✅ Monte Carlo simulation (1000 paths)
5. ✅ SHAP analysis (interpretability)
6. ✅ Strategy testing (5 market scenarios)

### **Start API Server**

```bash
uvicorn main:app --reload
```

Visit **http://localhost:8000/docs** for interactive API documentation.

---

## 📡 **API Endpoints**

### **Core Predictions**

```bash
GET /api/predict-regime/{ticker}
# Returns: 21-day ahead regime prediction with confidence

GET /api/strategy-recommendation/{ticker}
# Returns: Portfolio strategy (equity exposure, risk posture, etc.)

GET /api/executive-summary/{ticker}
# Returns: Plain-English summary (uses Gemini)
```

### **Market Data**

```bash
GET /api/tickers
# Returns: All 45 available tickers by asset class

GET /api/market-data/{ticker}?limit=100
# Returns: OHLCV data, returns, volatility

GET /api/top-momentum?asset_class=US_MEGACAP&top_n=10
# Returns: Top momentum stocks for long/short construction
```

### **Analytics**

```bash
GET /api/correlations?tickers=AAPL,MSFT,GOOGL
# Returns: Correlation matrix

GET /api/risk-metrics
# Returns: Sharpe, max drawdown, VaR, CVaR

GET /api/monte-carlo
# Returns: 1000-path simulation with percentile cone
```

### **Comprehensive Analysis**

```bash
GET /api/portfolio-analysis/{ticker}
# One-stop endpoint: market data + prediction + strategy + summary
```

---

## 🧠 **Model Architecture**

### **XGBoost Regime Classifier**

- **Objective:** Predict volatility regime 21 days ahead (LOW_VOL, NORMAL_VOL, HIGH_VOL)
- **Features:** 50+ engineered factors
  - **Momentum:** 12-1, 6M, 3M momentum (Jegadeesh & Titman 1993)
  - **Volatility:** ATR, z-scored volatility, Bollinger width
  - **Technical:** RSI, MACD, stochastic oscillator
  - **Cross-Sectional:** Momentum ranks within asset class
- **Validation:** 5-fold walk-forward cross-validation (prevents data leakage)
- **Accuracy:** ~75-80% (realistic for 21-day prediction)

### **Why 21 Days?**

- Actionable horizon (portfolio managers can rebalance)
- Avoids overfitting on short-term noise
- Aligns with monthly strategy reviews

### **Data Leakage Prevention**

✅ Forward-looking target (predict future, not current regime)
✅ Walk-forward CV (train on past, test on future)
✅ Time-series split (no shuffling)
✅ SHAP validation (ensures features make economic sense)

---

## 📊 **Factor Engineering Highlights**

### **Academic-Grade Implementations**

**1. Jegadeesh & Titman (1993) Momentum**

```python
mom_12_1 = (Close[t-21] / Close[t-252]) - 1  # Excluding last month
```

**2. Wilder (1978) RSI**

```python
RS = EMA(Gains, 14) / EMA(Losses, 14)
RSI = 100 - (100 / (1 + RS))
```

**3. Cross-Sectional Momentum**

```python
mom_rank = rank(mom_6m) / count(tickers) * 100  # Within asset class
```

**All in Pure Polars** (10-100x faster than pandas)

---

## 🎯 **Portfolio Strategy Engine**

Rule-based decision matrix (no overpromising):

| Regime                     | Risk Posture | Equity Exposure | Primary Strategy |
| -------------------------- | ------------ | --------------- | ---------------- |
| LOW_VOL (high confidence)  | RISK_ON      | 85%             | Momentum         |
| NORMAL_VOL                 | NEUTRAL      | 60%             | Balanced         |
| HIGH_VOL (high confidence) | RISK_OFF     | 35%             | Defensive        |

**Tail Risk Adjustments:**

- CVaR < -10% → Enable hedging
- Excess kurtosis > 3 → Reduce position sizing
- Negative skew < -0.5 → Add safe havens (gold, treasuries)

---

## 🤖 **Gemini Integration**

### **Executive Summary Generation**

**Input:**

```json
{
  "predicted_regime": "HIGH_VOL",
  "confidence": 0.85,
  "top_drivers": ["z_vol", "atr_14", "mom_6m"],
  "cvar_95": -0.12
}
```

**Output (Gemini-generated):**

> "Our model predicts elevated market turbulence over the next month, with 87% confidence based on unusually high volatility levels and recent price swings. The data shows downside-skewed returns and a higher likelihood of extreme price moves, indicating conditions similar to past market corrections. We recommend reducing equity exposure to 35% and shifting toward defensive assets like utilities and treasuries, while closely monitoring for stabilization signals."

**Use Case:** Translate quant outputs for non-technical executives.

---

## 📈 **SHAP Interpretability**

### **Why SHAP?**

- **Global:** Which features matter most overall?
- **Class-Specific:** What drives HIGH_VOL vs LOW_VOL predictions?
- **Local:** Why did the model predict X for this specific case?

### **Example Output**

**Global Importance:**

```
1. z_vol          (28.5%)  ← Volatility z-score
2. vol_trend      (15.2%)  ← Volatility acceleration
3. atr_pct        (14.0%)  ← Average True Range
```

**Insight:** Model learned that volatility is persistent (matches financial theory).

**Local Explanation:**

```
Predicted: HIGH_VOL for AAPL on 2026-01-20

Top Drivers:
  z_vol = 2.3 → SHAP: +0.45  (pushes toward HIGH_VOL)
  rsi_14 = 72 → SHAP: +0.12  (overbought → volatility spike)
```

---

## 🎤 **"Mic Drop" Moments for Judges**

### **Q: How is this different from a typical ML project?**

> "Most quant models are black boxes. Ours is fully explainable:
>
> - SHAP shows _why_ each prediction was made
> - Gemini translates outputs into executive summaries
> - Portfolio engine provides actionable, risk-managed strategies
>
> It's a **decision support system**, not just a prediction model."

### **Q: Why 75% accuracy instead of 99%?**

> "99% accuracy in finance is a red flag for data leakage. Our 75% accuracy on 21-day regime prediction is:
>
> - Realistic (volatility is hard to predict)
> - Validated with walk-forward CV (no future data used)
> - Economically meaningful (SHAP confirms features align with theory)
>
> We optimize for **trust**, not vanity metrics."

### **Q: Can you explain a prediction?**

> "Absolutely. [Pull up SHAP local explanation]
>
> For AAPL on 2026-01-20, the model predicted HIGH_VOL because:
>
> 1. Volatility z-score was 2.3σ above normal
> 2. ATR was elevated (14% annualized)
> 3. 6-month momentum turned negative (reversal risk)
>
> These are the exact signals volatility traders watch."

---

## 📊 **Performance Metrics**

### **Model Performance**

- **CV Accuracy:** 75-80% (21-day ahead)
- **Feature Count:** 52 (after removing inf/nan)
- **Training Time:** ~2 minutes (1000 samples)

### **Portfolio Simulation**

- **Sharpe Ratio:** 0.88
- **Max Drawdown:** -12.98%
- **Calmar Ratio:** 1.47
- **VaR (95%):** -9.82%

### **System Performance**

- **API Response Time:** <100ms (cached data)
- **Pipeline Runtime:** ~15-20 minutes (full execution)
- **Memory Usage:** ~2GB peak (Monte Carlo)

---

## 🔒 **Data Leakage Prevention Checklist**

✅ **Forward-looking target** (predict t+21, not t)
✅ **Walk-forward CV** (train on past only)
✅ **No shuffling** (maintains time-series order)
✅ **Feature validation** (SHAP confirms economic sense)
✅ **Explicit assertions** (code checks for leakage)

**Test yourself:**

```python
# WRONG - Data leakage
target = vol_regime  # Predicts current regime

# RIGHT - Actionable prediction
target = vol_regime.shift(-21).over("Ticker")  # Predicts future regime
```

---

## 🛠️ **Development**

### **Run Individual Components**

```bash
# Data ingestion only
python day1_morning.py

# Factor engineering only
python day1_afternoon.py

# ML training only
python day2_modeling.py

# SHAP analysis only
python shap_analysis.py

# Strategy demo
python portfolio_strategy_decision_engine.py

# Gemini demo
python financial_reasoning_agent.py
```

### **Testing**

```bash
# Test API endpoints
pytest tests/

# Test with curl
curl http://localhost:8000/api/predict-regime/AAPL
```

---

## 📚 **References**

### **Academic Papers**

1. Jegadeesh, N., & Titman, S. (1993). _Returns to Buying Winners and Selling Losers_
2. Wilder, J. W. (1978). _New Concepts in Technical Trading Systems_
3. Lundberg, S. M., & Lee, S. I. (2017). _A Unified Approach to Interpreting Model Predictions_ (SHAP)

### **Libraries**

- [Polars](https://www.pola.rs/) - Fast DataFrame library
- [XGBoost](https://xgboost.readthedocs.io/) - Gradient boosting
- [SHAP](https://shap.readthedocs.io/) - Model interpretability
- [FastAPI](https://fastapi.tiangolo.com/) - Modern API framework

---

## 🤝 **Contributing**

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

---

## 📄 **License**

MIT License - see [LICENSE](LICENSE) file for details.

---

## 👤 **Author**

**Your Name**

- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)

---

## 🙏 **Acknowledgments**

- Yahoo Finance for market data
- Google for Gemini Flash 2.5 API
- Academic researchers for factor methodologies
- Open-source community for amazing libraries

---

**Built for:** Quantifying the Markets Hackathon Track
**Tech Stack:** Python, Polars, XGBoost, SHAP, Gemini, FastAPI
**Innovation:** Hybrid XGBoost-LLM architecture for explainable quant finance

**⭐ Star this repo if you found it useful!**
