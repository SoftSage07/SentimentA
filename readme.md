# 📊 EquiSense — AI-Powered Market Sentiment Analysis

> **Real-time financial sentiment intelligence for retail investors** — democratizing the market analysis tools that institutions pay lakhs for.

---

## 🎯 Problem Statement

**90 million retail investors in India make investment decisions based on incomplete information.**

While institutional traders have access to Bloomberg terminals, quant teams, and real-time sentiment feeds costing ₹10-15 lakhs annually, retail investors rely on fragmented news sources, social media noise, and guesswork.

**EquiSense bridges this gap** — providing institutional-grade sentiment analysis powered by FinBERT NLP and Gemini AI, accessible to anyone with an internet connection.

---

## ✨ Key Features

### 🧠 **Financial NLP with FinBERT**
- Sentiment classification trained specifically on financial language
- Understands context (e.g., "rate cut" is bullish, not bearish)
- 50-article batch processing in seconds

### 📰 **Multi-Source News Aggregation**
- GNews, NewsAPI, Finnhub, Bloomberg RSS, CNBC RSS
- Real-time data from 5+ sources in parallel
- Automatic deduplication and recency weighting

### 📊 **Visual-First Dashboard**
- **Sentiment Gauge**: Interactive Fear ↔ Greed indicator (-100 to +100)
- **Donut Chart**: Fear/Neutral/Greed distribution breakdown
- **Source Analysis**: Horizontal bar chart with article counts
- **Signal Cards**: Color-coded bullish/bearish/neutral alerts
- **SRI Progress Bar**: Sentiment Reliability Index (0-100)

### 🎯 **Advanced Analytics**

#### **Sentiment Reliability Index (SRI)**
Weighted composite score combining:
- Source credibility (Bloomberg 1.0 → Random aggregator 0.6)
- Article recency (6hr decay curve)
- Model confidence (FinBERT score)

#### **Market Regime Detection**
Classifies current environment using VIX + sentiment dispersion:
- **Risk-On**: Low volatility + consensus sentiment
- **Risk-Off**: High VIX + high dispersion
- **Event-Driven**: Mixed signals

#### **Smart Money Divergence Alerts**
Flags when sentiment and price action decouple:
- Bullish news + falling prices → Distribution risk
- Bearish news + rising prices → Capitulation signal

### 🤖 **AI Analysis with Gemini 2.5 Flash**
- Structured analyst-style summaries in 250 words
- Sentiment classification (Strongly Bullish → Strongly Bearish)
- Key insights + suggested action (Accumulate/Hold/Book Profits/Avoid)

---

## 🚀 Live Demo

**Try it here:** [equisense.streamlit.app](https://sentimenta-sa.streamlit.app/)


## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Frontend** | Streamlit | Interactive web dashboard |
| **NLP Model** | FinBERT (ProsusAI) | Financial sentiment classification |
| **AI Analysis** | Gemini 2.5 Flash | Natural language synthesis |
| **Visualization** | Plotly | Interactive charts (gauge, donut, bars) |
| **Market Data** | yfinance | NIFTY 50, India VIX, real-time prices |
| **News APIs** | GNews, NewsAPI, Finnhub | Multi-source news aggregation |
| **RSS Feeds** | feedparser | Bloomberg, CNBC financial news |

---

## 📖 How It Works

### **Workflow**

1. **Enter a search term** (e.g., "NIFTY 50", "Reliance Industries", "IT Sector")
2. **Multi-source data aggregation**: Fetches 30-50 articles from 5 news sources in parallel
3. **FinBERT sentiment classification**: Analyzes each headline for positive/negative/neutral sentiment
4. **Advanced analytics**: Calculates SRI, detects market regime, flags divergences
5. **AI synthesis**: Gemini 2.5 Flash generates a structured analyst summary
6. **Visual dashboard**: Presents insights through interactive charts and color-coded signals

### **Reading the Dashboard**

#### **Overview Tab**
- **Sentiment Gauge**: Net sentiment from -100 (extreme fear) to +100 (extreme greed)
- **Donut Chart**: Percentage breakdown (Fear / Neutral / Greed)
- **Signal Card**: Current market signal with color coding
- **Key Metrics**: NIFTY 50 price, India VIX, Put-Call Ratio
- **SRI Bar**: Confidence level (High/Medium/Low)
- **AI Analysis**: Gemini-generated summary

#### **Analysis Tab**
- **Source Breakdown**: Which news outlets were analyzed
- **Top Contributors**: Most influential positive/negative headlines
- **Directional Alignment**: Does sentiment match price action?

#### **Data Tab**
- **Articles**: Full list of analyzed headlines with links
- **Sentiment Data**: Raw FinBERT classification results
- **Limitations**: Known issues and disclaimers

### **Signal Interpretation**

| Signal | Meaning | Implication |
|--------|---------|-------------|
| 🔥 **Extreme Panic** | Fear > 70%, VIX > 20 | Potential reversal zone |
| 📉 **High Fear** | Fear 55-70% | Cautious buying opportunity |
| ⚖️ **Neutral Zone** | Balanced sentiment | Hold positions |
| 📈 **High Greed** | Greed 55-70% | Consider profit booking |
| ⚠️ **Extreme Euphoria** | Greed > 70% | Distribution risk |

---

## 🧪 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface (Streamlit)              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Overview │  │ Analysis │  │   Data   │  │ Sidebar  │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                     Data Aggregation Layer                   │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │
│  │  GNews  │  │ NewsAPI │  │ Finnhub │  │   RSS   │       │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘       │
└───────────────────────────┬─────────────────────────────────┘
                            │ (50 articles)
┌───────────────────────────▼─────────────────────────────────┐
│                    Sentiment Analysis Engine                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  FinBERT (ProsusAI/finbert)                          │  │
│  │  - Batch classification (positive/negative/neutral)  │  │
│  │  - Confidence scores (0-1)                           │  │
│  │  - Keyword fallback (if model unavailable)           │  │
│  └──────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                   Advanced Analytics Layer                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │     SRI     │  │   Regime    │  │ Divergence  │        │
│  │  Scoring    │  │  Detection  │  │    Alert    │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                  AI Synthesis (Gemini 2.5 Flash)             │
│  - Structured prompt (250 words)                            │
│  - Sentiment summary + Key insights + Outlook + Action      │
└─────────────────────────────────────────────────────────────┘
```

### **Key Algorithms**

#### **Sentiment Reliability Index (SRI)**
```python
SRI = (source_credibility × 0.4) + (recency_weight × 0.3) + (model_confidence × 0.3)

where:
  source_credibility ∈ [0.6, 1.0]  (Bloomberg = 1.0, GNews = 0.65)
  recency_weight     ∈ [0.3, 1.0]  (6h = 1.0, 72h = 0.5)
  model_confidence   ∈ [0.0, 1.0]  (FinBERT output score)
```

#### **Market Regime Classification**
```python
if VIX > 18 and dispersion > 0.35:
    return "Risk-Off"
elif dispersion > 0.40 and 12 < VIX <= 18:
    return "Event-Driven"
elif VIX < 12 and dispersion < 0.25:
    return "Risk-On"
else:
    return "Event-Driven"
```

---

## 📊 Performance Benchmarks

| Metric | Value | Notes |
|--------|-------|-------|
| **Analysis Time** | 10-15s | 50 articles, full pipeline |
| **API Calls** | 5-7 | Parallel execution |
| **FinBERT Accuracy** | ~85% | On financial headlines |
| **SRI Calibration** | ±8 points | Across different market regimes |
| **Cache Hit Rate** | ~60% | 15min TTL on news data |

---

## 🔮 Future Roadmap

Potential enhancements being considered:

- Portfolio-level sentiment tracking
- WhatsApp/Telegram alert integration
- Real-time options flow data (NSE)
- Historical backtesting with real data
- Mobile app (React Native / Flutter)
- Multi-language support (Hindi, regional languages)

---

## ⚠️ Known Limitations

### **Model Limitations**
- FinBERT may misclassify sarcasm or domain-specific jargon
- Headline-only analysis misses context from full articles
- Keyword fallback has significantly lower accuracy (~60%)

### **Data Limitations**
- RSS feeds can be stale or contain duplicates
- API rate limits reduce sample size during high-traffic periods
- Source credibility weights are heuristic, not empirically validated

### **Market Indicator Limitations**
- **PCR is simulated** — real-time NSE data not available via free APIs
- NIFTY/VIX data depend on exchange hours and yfinance availability
- Sentiment-price correlation breaks during regime changes

### **Backtesting Caveats**
- Simulated accuracy is illustrative, not predictive
- No transaction costs, slippage, or market impact modeled
- Limited lookback period (does not capture black swan events)

---

## 📜 Disclaimer

**This tool is for educational and research purposes only.**

EquiSense is **NOT**:
- Financial advice
- An investment recommendation
- A trading signal
- A substitute for professional financial consultation

**Important Notes:**
- Past sentiment patterns do not guarantee future performance
- Market conditions can change rapidly, invalidating prior analysis
- Always consult a registered financial advisor before making investment decisions
- The developers assume no liability for financial losses incurred

---

## 🙏 Acknowledgments

- **ProsusAI** — FinBERT model
- **Google** — Gemini 2.5 Flash API
- **Streamlit** — Rapid prototyping framework
- **Plotly** — Interactive visualizations
- **yfinance** — Market data access
- **HuggingFace** — Transformers library

---

## 📧 Contact

**For inquiries or collaboration opportunities:**

- **Email**: harshprabha95@gmail.com
- **LinkedIn**: [MY Profile](https://www.linkedin.com/in/harshprabha-30b87b376/)
- **Portfolio**: [My Website](https://softsage07.github.io/Portfolio/)

---

<div align="center">

**Built with ❤️ using Python, Streamlit, and AI**

*© 2026 [Harshprabha]. All rights reserved.*

</div>