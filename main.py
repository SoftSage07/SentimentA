import streamlit as st
import requests
import pandas as pd
from transformers import pipeline
from datetime import datetime, timedelta
import numpy as np
import google.generativeai as genai
import time

# Try to import yfinance, provide fallback if not available
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except Exception as e:
    YFINANCE_AVAILABLE = False
    yf_error = str(e)

# Try to import feedparser for RSS
try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False
    feedparser_error = "Module not installed"

st.set_page_config(page_title="Market Mood Radar", layout="wide", initial_sidebar_state="expanded")

GNEWS_API_KEY = st.secrets["GNEWS_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
NEWSAPI_KEY = st.secrets["NEWSAPI_KEY"]
FINNHUB_KEY = st.secrets["FINNHUB_KEY"]


# Load FinBERT with better error handling
@st.cache_resource
def load_model():
    try:
        model = pipeline("sentiment-analysis", model="ProsusAI/finbert", max_length=512, truncation=True)
        return model
    except Exception as e:
        st.error(f"❌ Error loading FinBERT model: {e}")
        st.info("💡 Try restarting the app or check your internet connection")
        return None

sentiment_model = load_model()

# Fetch news from multiple premium sources
@st.cache_data(ttl=900)  # Cache for 15 minutes
def fetch_gnews(api_key, query):
    if not api_key:
        return []
    
    try:
        url = f"https://gnews.io/api/v4/search?q={query}&lang=en&max=30&apikey={api_key}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if "articles" not in data or len(data["articles"]) == 0:
            return []
        
        articles = []
        for article in data["articles"]:
            text = article.get("title", "")
            desc = article.get("description", "")
            if desc:
                text += " " + desc
            articles.append({
                'text': text,
                'source': article.get('source', {}).get('name', 'GNews'),
                'url': article.get('url', ''),
                'publishedAt': article.get('publishedAt', '')
            })
        
        return articles
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            st.warning("⚠️ GNews API rate limit reached")
        return []
    except Exception as e:
        st.warning(f"⚠️ GNews error: {e}")
        return []

@st.cache_data(ttl=900)
def fetch_newsapi(api_key, query):
    if not api_key:
        return []
    
    try:
        # NewsAPI.org endpoint
        url = f"https://newsapi.org/v2/everything"
        params = {
            'q': query,
            'apiKey': api_key,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 30,
            'sources': 'bloomberg,cnbc,financial-times,the-wall-street-journal,fortune,business-insider'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') != 'ok' or not data.get('articles'):
            return []
        
        articles = []
        for article in data['articles']:
            text = article.get('title', '')
            desc = article.get('description', '')
            if desc:
                text += " " + desc
            articles.append({
                'text': text,
                'source': article.get('source', {}).get('name', 'NewsAPI'),
                'url': article.get('url', ''),
                'publishedAt': article.get('publishedAt', '')
            })
        
        return articles
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 426:
            st.warning("⚠️ NewsAPI: Upgrade required for premium sources")
        elif e.response.status_code == 429:
            st.warning("⚠️ NewsAPI rate limit reached")
        return []
    except Exception as e:
        st.warning(f"⚠️ NewsAPI error: {e}")
        return []

@st.cache_data(ttl=900)
def fetch_finnhub(api_key, query):
    if not api_key:
        return []
    
    try:
        # Finnhub company news endpoint
        url = f"https://finnhub.io/api/v1/news"
        params = {
            'category': 'general',
            'token': api_key
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            return []
        
        articles = []
        for article in data[:30]:
            # Filter by query relevance
            headline = article.get('headline', '')
            summary = article.get('summary', '')
            combined = (headline + " " + summary).lower()
            
            if query.lower() in combined or any(word in combined for word in query.lower().split()):
                text = headline
                if summary:
                    text += " " + summary
                articles.append({
                    'text': text,
                    'source': article.get('source', 'Finnhub'),
                    'url': article.get('url', ''),
                    'publishedAt': datetime.fromtimestamp(article.get('datetime', 0)).isoformat()
                })
        
        return articles
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            st.warning("⚠️ Finnhub API rate limit reached")
        return []
    except Exception as e:
        st.warning(f"⚠️ Finnhub error: {e}")
        return []

@st.cache_data(ttl=900)
def fetch_cnbc_rss(query):
    """Fetch from CNBC RSS feed (no API key needed)"""
    if not FEEDPARSER_AVAILABLE:
        return []
    
    try:
        # CNBC RSS feeds
        feeds = [
            'https://www.cnbc.com/id/100003114/device/rss/rss.html',  # Top News
            'https://www.cnbc.com/id/10001147/device/rss/rss.html',   # Finance
        ]
        
        articles = []
        for feed_url in feeds:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:15]:
                title = entry.get('title', '')
                summary = entry.get('summary', '')
                combined = (title + " " + summary).lower()
                
                # Filter by relevance
                if query.lower() in combined or any(word in combined for word in query.lower().split()):
                    text = title
                    if summary:
                        # Clean HTML tags from summary
                        import re
                        summary = re.sub('<[^<]+?>', '', summary)
                        text += " " + summary[:200]
                    
                    articles.append({
                        'text': text,
                        'source': 'CNBC',
                        'url': entry.get('link', ''),
                        'publishedAt': entry.get('published', '')
                    })
        
        return articles
    except Exception as e:
        return []

@st.cache_data(ttl=900)
def fetch_bloomberg_rss(query):
    """Fetch from Bloomberg RSS feed (no API key needed)"""
    if not FEEDPARSER_AVAILABLE:
        return []
    
    try:
        feeds = [
            'https://feeds.bloomberg.com/markets/news.rss',
            'https://feeds.bloomberg.com/economics/news.rss',
        ]
        
        articles = []
        for feed_url in feeds:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:15]:
                title = entry.get('title', '')
                summary = entry.get('summary', '')
                combined = (title + " " + summary).lower()
                
                if query.lower() in combined or any(word in combined for word in query.lower().split()):
                    text = title
                    if summary:
                        import re
                        summary = re.sub('<[^<]+?>', '', summary)
                        text += " " + summary[:200]
                    
                    articles.append({
                        'text': text,
                        'source': 'Bloomberg',
                        'url': entry.get('link', ''),
                        'publishedAt': entry.get('published', '')
                    })
        
        return articles
    except Exception as e:
        return []

# Market Data Functions
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_nifty():
    if not YFINANCE_AVAILABLE:
        return None, None
    
    try:
        nifty = yf.Ticker("^NSEI")
        hist = nifty.history(period="5d", interval="1d")
        
        if hist.empty:
            return None, None
        
        close_price = hist["Close"].iloc[-1]
        if len(hist) >= 2:
            prev_close = hist["Close"].iloc[-2]
            pct_change = ((close_price - prev_close) / prev_close) * 100
        else:
            pct_change = 0
        
        return round(close_price, 2), round(pct_change, 2)
    except Exception as e:
        return None, None

@st.cache_data(ttl=300)
def get_vix():
    if not YFINANCE_AVAILABLE:
        return None
    
    try:
        vix = yf.Ticker("^INDIAVIX")
        hist = vix.history(period="5d", interval="1d")
        
        if hist.empty:
            return None
        
        return round(hist["Close"].iloc[-1], 2)
    except Exception as e:
        return None

def get_pcr():
    # Demo PCR - in production use NSE options data
    return round(np.random.uniform(0.6, 1.4), 2)

# Improved sentiment analysis
def analyze_sentiment(articles):
    if not articles or sentiment_model is None:
        return 0, 0, 0, pd.DataFrame()
    
    try:
        texts = [article['text'][:512] for article in articles[:50]]
        
        results = sentiment_model(texts)
        
        # Create detailed dataframe
        df = pd.DataFrame({
            'text': [article['text'][:100] + '...' for article in articles[:50]],
            'source': [article['source'] for article in articles[:50]],
            'label': [r['label'] for r in results],
            'score': [round(r['score'], 3) for r in results],
            'url': [article.get('url', '') for article in articles[:50]]
        })
        
        total = len(df)
        if total == 0:
            return 0, 0, 0, pd.DataFrame()
        
        fear = round((len(df[df['label'] == 'negative']) / total) * 100, 2)
        greed = round((len(df[df['label'] == 'positive']) / total) * 100, 2)
        neutral = round((len(df[df['label'] == 'neutral']) / total) * 100, 2)
        
        return fear, greed, neutral, df
    except Exception as e:
        st.error(f"❌ Sentiment analysis error: {e}")
        return 0, 0, 0, pd.DataFrame()

# Enhanced signal generation
def generate_signal(fear, greed, neutral, vix, pcr, nifty_change):
    if vix is None:
        return "⏳ INSUFFICIENT DATA", "neutral"
    
    score = 0
    
    # Sentiment scoring
    if fear > 70:
        score -= 3
    elif fear > 55:
        score -= 2
    elif fear > 40:
        score -= 1
    
    if greed > 70:
        score += 3
    elif greed > 55:
        score += 2
    elif greed > 40:
        score += 1
    
    # VIX scoring
    if vix > 20:
        score -= 2
    elif vix > 15:
        score -= 1
    elif vix < 10:
        score += 2
    elif vix < 12:
        score += 1
    
    # PCR scoring
    if pcr > 1.3:
        score -= 2
    elif pcr > 1.1:
        score -= 1
    elif pcr < 0.7:
        score += 2
    elif pcr < 0.9:
        score += 1
    
    # Price action
    if nifty_change and nifty_change < -2:
        score -= 1
    elif nifty_change and nifty_change > 2:
        score += 1
    
    # Generate signal
    if score <= -5:
        return "🔥 EXTREME PANIC: Strong Reversal Potential", "danger"
    elif score <= -3:
        return "📉 HIGH FEAR: Cautious Buying Opportunity", "warning"
    elif score <= -1:
        return "😰 MODERATE FEAR: Wait & Watch", "info"
    elif score <= 1:
        return "⚖️ NEUTRAL ZONE: Market in Balance", "neutral"
    elif score <= 3:
        return "😊 MODERATE GREED: Stay Alert", "info"
    elif score <= 5:
        return "📈 HIGH GREED: Consider Profit Booking", "warning"
    else:
        return "⚠️ EXTREME EUPHORIA: Distribution Risk High", "danger"

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def explain_market_gemini(api_key, signal, fear, greed, neutral, vix, pcr, articles, query):
    if not api_key:
        return None
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name="gemini-pro")
        
        # Prepare article headlines with full context
        headlines_text = "\n".join([f"{i+1}. [{a['source']}] {a['text'][:200]}" for i, a in enumerate(articles[:15])])
        
        prompt = f"""You are a professional financial research analyst.

User Search Term: "{query}"

Task:
Only analyze, summarize, and reason about information that is DIRECTLY related to "{query}".
Ignore any news, macro events, or companies that are not clearly connected.

Market Sentiment Data:
- Overall Sentiment Signal: {signal}
- Fear Index: {fear}%
- Greed Index: {greed}%
- Neutral: {neutral}%
- VIX (Volatility): {vix if vix else 'N/A'}
- Put-Call Ratio: {pcr:.2f}

Recent News Articles:
{headlines_text}

From the provided news articles and data, do the following:

1. Filter:
   - Keep only articles where "{query}" is the primary subject
   - Discard generic market news unless it clearly impacts "{query}"

2. Classify Sentiment:
   - For each relevant article, label it as Bullish, Bearish, or Neutral for "{query}"
   - Briefly explain why in one line

3. Generate Insights:
   - Write 3 concise insights explaining how current news sentiment may affect "{query}"'s outlook

4. Market Outlook:
   - Give a one-line outlook: Strongly Bullish / Mildly Bullish / Neutral / Mildly Bearish / Strongly Bearish

5. Action Bias:
   - Based on sentiment and momentum, suggest one of:
     • Accumulate
     • Hold
     • Book Profits
     • Avoid
   - Add a short risk note

Rules:
- Do not talk about the overall market unless it directly influences "{query}"
- Do not mention unrelated stocks or sectors
- Be factual, crisp, and data-driven
- No fluff. No generic motivational talk
- Format your response clearly with headers for each section

Maximum 250 words."""
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        # More detailed error handling
        error_msg = str(e)
        if "models/gemini-pro" in error_msg or "not found" in error_msg.lower():
            # Try alternative model names
            try:
                model = genai.GenerativeModel(model_name="gemini-1.5-flash")
                response = model.generate_content(prompt)
                return response.text
            except:
                st.warning(f"⚠️ AI analysis unavailable: Gemini model access issue. Using free tier may have model restrictions.")
                return None
        else:
            st.warning(f"⚠️ AI analysis error: {error_msg}")
            return None

# --- UI ---
st.title("📊 Smart Money Market Sentiment Analyzer")
st.caption("Real-time sentiment analysis from CNBC, Bloomberg, Finnhub & NewsAPI")

# Sidebar Configuration
with st.sidebar:
    st.header("🔑 API Configuration")
    st.success("✅ APIs securely loaded from server (hidden)")

    
    st.markdown("---")
    st.header("🔍 Search Settings")
    
    search_query = st.text_input(
        "Target Stock/Topic", 
        value="NIFTY 50",
        help="Enter stock name, sector, or market event"
    )
    
    use_rss = st.checkbox("Include CNBC & Bloomberg RSS", value=FEEDPARSER_AVAILABLE, 
                          help="Requires feedparser: pip install feedparser")
    
    st.markdown("---")
    
    # System status
    st.subheader("🔧 System Status")
    if YFINANCE_AVAILABLE:
        st.success("✅ Market Data (yfinance)")
    else:
        st.error("❌ Market Data (yfinance)")
        with st.expander("View Error & Fix"):
            st.code(yf_error if 'yf_error' in globals() else "Unknown error")
            st.info("**Fix:** `pip install --upgrade yfinance`")
    
    if FEEDPARSER_AVAILABLE:
        st.success("✅ RSS Feeds (feedparser)")
    else:
        st.warning("⚠️ RSS Feeds (feedparser) - Optional")
        with st.expander("Install RSS Support"):
            st.info("**Install:** `pip install feedparser`")
    
    st.markdown("---")
    st.info("💡 **Sources**: GNews (required) + NewsAPI, Finnhub, CNBC RSS, Bloomberg RSS (optional)")

# Check for critical dependencies
if not YFINANCE_AVAILABLE:
    st.warning("⚠️ **yfinance not available** - Market indicators (NIFTY, VIX) will be disabled.")
    st.info("""
    **For Python 3.14.x users:**
    ```bash
    pip install --upgrade yfinance
    ```
    
    If that doesn't work:
    ```bash
    pip uninstall yfinance
    pip install yfinance --no-cache-dir
    ```
    """)

if not FEEDPARSER_AVAILABLE and use_rss:
    st.warning("⚠️ RSS feeds disabled. Install feedparser to enable CNBC & Bloomberg RSS.")
    use_rss = False

# Validation
if not GNEWS_API_KEY:
    st.error("❌ GNews API key not configured in the code!")
    st.stop()

# Main Analysis
if st.button("🔄 Analyze Market Sentiment", type="primary", use_container_width=True):
    
    with st.spinner(f"🔍 Analyzing '{search_query}' across multiple sources..."):
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_articles = []
        
        # Fetch from GNews
        status_text.text("📰 Fetching from GNews...")
        progress_bar.progress(15)
        gnews_articles = fetch_gnews(GNEWS_API_KEY, search_query)
        all_articles.extend(gnews_articles)
        
        # Fetch from NewsAPI
        if NEWSAPI_KEY:
            status_text.text("📊 Fetching from NewsAPI (Bloomberg, CNBC, WSJ)...")
            progress_bar.progress(30)
            newsapi_articles = fetch_newsapi(NEWSAPI_KEY, search_query)
            all_articles.extend(newsapi_articles)
        
        # Fetch from Finnhub
        if FINNHUB_KEY:
            status_text.text("📈 Fetching from Finnhub...")
            progress_bar.progress(45)
            finnhub_articles = fetch_finnhub(FINNHUB_KEY, search_query)
            all_articles.extend(finnhub_articles)
        
        # Fetch from RSS feeds
        if use_rss:
            status_text.text("📡 Fetching from CNBC RSS...")
            progress_bar.progress(55)
            cnbc_articles = fetch_cnbc_rss(search_query)
            all_articles.extend(cnbc_articles)
            
            status_text.text("📡 Fetching from Bloomberg RSS...")
            progress_bar.progress(65)
            bloomberg_articles = fetch_bloomberg_rss(search_query)
            all_articles.extend(bloomberg_articles)
        
        if not all_articles:
            st.error(f"❌ No articles found for '{search_query}'. Try enabling more sources or use a different search term.")
            st.stop()
        
        # Sentiment analysis
        status_text.text("🧠 Analyzing sentiment with FinBERT...")
        progress_bar.progress(75)
        fear, greed, neutral, sentiment_df = analyze_sentiment(all_articles)
        
        # Market data
        status_text.text("📈 Fetching market indicators...")
        progress_bar.progress(85)
        nifty_price, nifty_change = get_nifty()
        vix_value = get_vix()
        pcr_value = get_pcr()
        
        # Generate signal
        signal, signal_type = generate_signal(fear, greed, neutral, vix_value, pcr_value, nifty_change)
        
        # AI explanation
        gemini_explanation = None
        if GEMINI_API_KEY:
            status_text.text("🤖 Generating AI insights...")
            progress_bar.progress(95)
            gemini_explanation = explain_market_gemini(
                GEMINI_API_KEY, signal, fear, greed, neutral, vix_value, pcr_value, all_articles, search_query
            )
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        # Count sources
        source_counts = {}
        for article in all_articles:
            source = article['source']
            source_counts[source] = source_counts.get(source, 0) + 1
        
        # Store in session
        st.session_state.update({
            'query': search_query,
            'fear': fear,
            'greed': greed,
            'neutral': neutral,
            'sentiment_df': sentiment_df,
            'all_articles': all_articles,
            'source_counts': source_counts,
            'total_sources': len(all_articles),
            'nifty_price': nifty_price,
            'nifty_change': nifty_change,
            'vix': vix_value,
            'pcr': pcr_value,
            'signal': signal,
            'signal_type': signal_type,
            'gemini_explanation': gemini_explanation,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

# Display Results
if 'fear' in st.session_state:
    st.markdown("---")
    st.header(f"📊 Results: {st.session_state.query}")
    st.caption(f"Last updated: {st.session_state.timestamp}")
    
    # Key Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("😨 Fear Index", f"{st.session_state.fear}%")
    with col2:
        st.metric("🤑 Greed Index", f"{st.session_state.greed}%")
    with col3:
        st.metric("😐 Neutral", f"{st.session_state.neutral}%")
    with col4:
        st.metric("📚 Articles", st.session_state.total_sources)
    with col5:
        if st.session_state.nifty_price:
            st.metric("NIFTY 50", 
                     f"{st.session_state.nifty_price:,.0f}",
                     f"{st.session_state.nifty_change:+.2f}%")
    
    # Source breakdown
    st.subheader("📡 Data Sources")
    source_cols = st.columns(len(st.session_state.source_counts))
    for i, (source, count) in enumerate(st.session_state.source_counts.items()):
        with source_cols[i]:
            st.metric(source, count)
    
    # Market Indicators
    st.subheader("📈 Market Indicators")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        vix_color = "🔴" if st.session_state.vix and st.session_state.vix > 15 else "🟢"
        st.metric(f"{vix_color} VIX (Volatility)", 
                 f"{st.session_state.vix}" if st.session_state.vix else "N/A")
    
    with col2:
        pcr_color = "🔴" if st.session_state.pcr > 1.2 else "🟢" if st.session_state.pcr < 0.8 else "🟡"
        st.metric(f"{pcr_color} Put-Call Ratio", f"{st.session_state.pcr:.2f}")
    
    # Signal
    st.subheader("🚨 Market Signal")
    if st.session_state.signal_type == "danger":
        st.error(st.session_state.signal)
    elif st.session_state.signal_type == "warning":
        st.warning(st.session_state.signal)
    else:
        st.info(st.session_state.signal)
    
    # AI Analysis
    if st.session_state.gemini_explanation:
        st.subheader("🤖 AI Financial Analysis")
        st.markdown(st.session_state.gemini_explanation)
    else:
        st.info("💡 Configure Gemini API for detailed financial analysis")
    
    # Sentiment Distribution
    st.subheader("📊 Sentiment Distribution")
    chart_data = pd.DataFrame({
        'Sentiment': ['Fear', 'Neutral', 'Greed'],
        'Percentage': [st.session_state.fear, st.session_state.neutral, st.session_state.greed]
    })
    st.bar_chart(chart_data.set_index('Sentiment'))
    
    # Detailed Data
    tab1, tab2, tab3 = st.tabs(["📰 All Articles", "📊 Sentiment Analysis", "🔗 Source Links"])
    
    with tab1:
        if st.session_state.all_articles:
            articles_df = pd.DataFrame([
                {
                    'Source': a['source'],
                    'Headline': a['text'][:150] + '...' if len(a['text']) > 150 else a['text'],
                    'Published': a.get('publishedAt', 'N/A')
                }
                for a in st.session_state.all_articles
            ])
            st.dataframe(articles_df, use_container_width=True, height=400)
        else:
            st.info("No articles found")
    
    with tab2:
        if not st.session_state.sentiment_df.empty:
            st.dataframe(
                st.session_state.sentiment_df[['source', 'label', 'score', 'text']],
                use_container_width=True,
                height=400
            )
        else:
            st.info("No sentiment data available")
    
    with tab3:
        if st.session_state.all_articles:
            for article in st.session_state.all_articles[:30]:
                if article.get('url'):
                    st.markdown(f"**[{article['source']}]** [{article['text'][:100]}...]({article['url']})")
        else:
            st.info("No article links available")

# Footer
st.markdown("---")
st.caption("⚠️ **Disclaimer**: This tool is for educational purposes only. Not financial advice. Always do your own research.")

st.caption("Built with Streamlit • FinBERT • Gemini AI • GNews • NewsAPI • Finnhub • CNBC • Bloomberg")
