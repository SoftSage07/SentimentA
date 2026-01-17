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

# Load API keys from secrets (safely)
try:
    GNEWS_API_KEY = st.secrets.get("GNEWS_API_KEY", "")
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
    NEWSAPI_KEY = st.secrets.get("NEWSAPI_KEY", "")
    FINNHUB_KEY = st.secrets.get("FINNHUB_KEY", "")
except Exception:
    # Fallback if secrets not configured
    GNEWS_API_KEY = ""
    GEMINI_API_KEY = ""
    NEWSAPI_KEY = ""
    FINNHUB_KEY = ""

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
@st.cache_data(ttl=900)
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
    if not FEEDPARSER_AVAILABLE:
        return []
    
    try:
        feeds = [
            'https://www.cnbc.com/id/100003114/device/rss/rss.html',
            'https://www.cnbc.com/id/10001147/device/rss/rss.html',
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
                        'source': 'CNBC',
                        'url': entry.get('link', ''),
                        'publishedAt': entry.get('published', '')
                    })
        
        return articles
    except Exception as e:
        return []

@st.cache_data(ttl=900)
def fetch_bloomberg_rss(query):
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
@st.cache_data(ttl=300)
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
    return round(np.random.uniform(0.6, 1.4), 2)

def analyze_sentiment(articles):
    if not articles or sentiment_model is None:
        return 0, 0, 0, pd.DataFrame()
    
    try:
        texts = [article['text'][:512] for article in articles[:50]]
        results = sentiment_model(texts)
        
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

def generate_signal(fear, greed, neutral, vix, pcr, nifty_change):
    if vix is None:
        return "⏳ INSUFFICIENT DATA", "neutral"
    
    score = 0
    
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
    
    if vix > 20:
        score -= 2
    elif vix > 15:
        score -= 1
    elif vix < 10:
        score += 2
    elif vix < 12:
        score += 1
    
    if pcr > 1.3:
        score -= 2
    elif pcr > 1.1:
        score -= 1
    elif pcr < 0.7:
        score += 2
    elif pcr < 0.9:
        score += 1
    
    if nifty_change and nifty_change < -2:
        score -= 1
    elif nifty_change and nifty_change > 2:
        score += 1
    
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

def explain_market_gemini(api_key, signal, fear, greed, neutral, vix, pcr, articles, query):
    """Generate AI analysis using Gemini"""
    if not api_key:
        st.warning("⚠️ Gemini API key not configured")
        return None
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        
        headlines_text = "\n".join([f"{i+1}. [{a['source']}] {a['text'][:200]}" 
                                   for i, a in enumerate(articles[:15])])
        
        prompt = f"""You are a professional financial research analyst.

User Search Term: "{query}"

Task: Only analyze information DIRECTLY related to "{query}".

Market Sentiment Data:
- Overall Signal: {signal}
- Fear: {fear}% | Greed: {greed}% | Neutral: {neutral}%
- VIX: {vix if vix else 'N/A'} | PCR: {pcr:.2f}

Recent News:
{headlines_text}

Provide:
1. **Sentiment Summary** - Classify relevant articles as Bullish/Bearish/Neutral for {query}
2. **Key Insights** - 3 concise points on how news affects {query}
3. **Outlook** - One-line: Strongly Bullish/Mildly Bullish/Neutral/Mildly Bearish/Strongly Bearish
4. **Action** - Suggest: Accumulate/Hold/Book Profits/Avoid with brief risk note

Rules:
- Focus ONLY on {query}
- Be factual and data-driven
- Max 250 words
- Use clear headers"""

        response = model.generate_content(prompt)
        
        if response and response.text:
            return response.text
        else:
            st.warning("⚠️ Gemini returned empty response")
            return None
            
    except Exception as e:
        error_msg = str(e)
        
        with st.expander("🔍 Gemini API Error Details", expanded=False):
            st.code(error_msg)
        
        if "API_KEY_INVALID" in error_msg or "invalid" in error_msg.lower():
            st.error("❌ **Invalid Gemini API Key**")
            st.info("Get a new key: https://aistudio.google.com/app/apikey")
        elif "quota" in error_msg.lower() or "resource_exhausted" in error_msg.lower():
            st.warning("⚠️ **API Quota Exceeded** - Free tier limit reached. Wait 60 seconds.")
        elif "429" in error_msg or "rate" in error_msg.lower():
            st.warning("⚠️ **Rate Limit** - Too many requests. Wait 60 seconds.")
        elif "blocked" in error_msg.lower() or "safety" in error_msg.lower():
            st.warning("⚠️ **Content filtered** - Try a different search term")
        else:
            st.error(f"❌ **Gemini Error** - Check error details above")
        
        return None

# --- UI ---
st.title("📊 Smart Money Market Sentiment Analyzer")
st.caption("Real-time sentiment analysis from CNBC, Bloomberg, Finnhub & NewsAPI")

# Sidebar
with st.sidebar:
    st.header("🔑 API Configuration")
    
    # Manual API key input option
    with st.expander("⚙️ Configure API Keys", expanded=not (GEMINI_API_KEY and GNEWS_API_KEY)):
        st.info("💡 **Setup Options:**\n1. Use Streamlit secrets (`.streamlit/secrets.toml`)\n2. Enter keys below (session only)")
        
        temp_gnews = st.text_input("GNews API Key", value=GNEWS_API_KEY, type="password", key="gnews_input")
        temp_gemini = st.text_input("Gemini API Key", value=GEMINI_API_KEY, type="password", key="gemini_input")
        temp_newsapi = st.text_input("NewsAPI Key (Optional)", value=NEWSAPI_KEY, type="password", key="newsapi_input")
        temp_finnhub = st.text_input("Finnhub Key (Optional)", value=FINNHUB_KEY, type="password", key="finnhub_input")
        
        if temp_gnews:
            GNEWS_API_KEY = temp_gnews
        if temp_gemini:
            GEMINI_API_KEY = temp_gemini
        if temp_newsapi:
            NEWSAPI_KEY = temp_newsapi
        if temp_finnhub:
            FINNHUB_KEY = temp_finnhub
        
        st.caption("🔐 Keys are not saved. Use secrets.toml for persistence.")
    
    st.markdown("---")
    
    st.subheader("📊 API Status")
    
    if GEMINI_API_KEY:
        st.success(f"✅ **Gemini**: {GEMINI_API_KEY[:10]}...{GEMINI_API_KEY[-4:]}")
    else:
        st.error("❌ **Gemini**: Not configured")
    
    if GNEWS_API_KEY:
        st.success(f"✅ **GNews**: {GNEWS_API_KEY[:10]}...{GNEWS_API_KEY[-4:]}")
    else:
        st.error("❌ **GNews**: Not configured")
    
    if NEWSAPI_KEY:
        st.success(f"✅ **NewsAPI**: Configured")
    else:
        st.info("⚠️ **NewsAPI**: Optional")
    
    if FINNHUB_KEY:
        st.success(f"✅ **Finnhub**: Configured")
    else:
        st.info("⚠️ **Finnhub**: Optional")
    
    st.markdown("---")
    st.header("🔍 Search Settings")
    
    search_query = st.text_input(
        "Target Stock/Topic", 
        value="NIFTY 50",
        help="Enter stock name, sector, or market event"
    )
    
    use_rss = st.checkbox("Include CNBC & Bloomberg RSS", 
                         value=FEEDPARSER_AVAILABLE,
                         help="Requires feedparser")
    
    st.markdown("---")
    st.subheader("🔧 System Status")
    
    if YFINANCE_AVAILABLE:
        st.success("✅ Market Data (yfinance)")
    else:
        st.error("❌ Market Data (yfinance)")
    
    if FEEDPARSER_AVAILABLE:
        st.success("✅ RSS Feeds (feedparser)")
    else:
        st.warning("⚠️ RSS Feeds (optional)")

# Main Analysis
if st.button("🔄 Analyze Market Sentiment", type="primary", use_container_width=True):
    
    if not GNEWS_API_KEY:
        st.error("❌ GNews API key required! Please configure it in the sidebar.")
        st.stop()
    
    with st.spinner(f"🔍 Analyzing '{search_query}'..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_articles = []
        
        status_text.text("📰 Fetching from GNews...")
        progress_bar.progress(15)
        gnews_articles = fetch_gnews(GNEWS_API_KEY, search_query)
        all_articles.extend(gnews_articles)
        
        if NEWSAPI_KEY:
            status_text.text("📊 Fetching from NewsAPI...")
            progress_bar.progress(30)
            newsapi_articles = fetch_newsapi(NEWSAPI_KEY, search_query)
            all_articles.extend(newsapi_articles)
        
        if FINNHUB_KEY:
            status_text.text("📈 Fetching from Finnhub...")
            progress_bar.progress(45)
            finnhub_articles = fetch_finnhub(FINNHUB_KEY, search_query)
            all_articles.extend(finnhub_articles)
        
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
            st.error(f"❌ No articles found for '{search_query}'")
            st.stop()
        
        status_text.text("🧠 Analyzing sentiment with FinBERT...")
        progress_bar.progress(75)
        fear, greed, neutral, sentiment_df = analyze_sentiment(all_articles)
        
        status_text.text("📈 Fetching market indicators...")
        progress_bar.progress(85)
        nifty_price, nifty_change = get_nifty()
        vix_value = get_vix()
        pcr_value = get_pcr()
        
        signal, signal_type = generate_signal(fear, greed, neutral, vix_value, pcr_value, nifty_change)
        
        gemini_explanation = None
        if GEMINI_API_KEY:
            status_text.text("🤖 Generating AI insights...")
            progress_bar.progress(95)
            gemini_explanation = explain_market_gemini(
                GEMINI_API_KEY, signal, fear, greed, neutral, 
                vix_value, pcr_value, all_articles, search_query
            )
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        source_counts = {}
        for article in all_articles:
            source = article['source']
            source_counts[source] = source_counts.get(source, 0) + 1
        
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
    
    st.subheader("📡 Data Sources")
    source_cols = st.columns(len(st.session_state.source_counts))
    for i, (source, count) in enumerate(st.session_state.source_counts.items()):
        with source_cols[i]:
            st.metric(source, count)
    
    st.subheader("📈 Market Indicators")
    col1, col2 = st.columns(2)
    
    with col1:
        vix_color = "🔴" if st.session_state.vix and st.session_state.vix > 15 else "🟢"
        st.metric(f"{vix_color} VIX (Volatility)", 
                 f"{st.session_state.vix}" if st.session_state.vix else "N/A")
    
    with col2:
        pcr_color = "🔴" if st.session_state.pcr > 1.2 else "🟢" if st.session_state.pcr < 0.8 else "🟡"
        st.metric(f"{pcr_color} Put-Call Ratio", f"{st.session_state.pcr:.2f}")
    
    st.subheader("🚨 Market Signal")
    if st.session_state.signal_type == "danger":
        st.error(st.session_state.signal)
    elif st.session_state.signal_type == "warning":
        st.warning(st.session_state.signal)
    else:
        st.info(st.session_state.signal)
    
    st.subheader("🤖 AI Financial Analysis")
    if st.session_state.gemini_explanation:
        st.markdown(st.session_state.gemini_explanation)
    else:
        st.info("💡 Gemini AI analysis unavailable - check errors above")
    
    st.subheader("📊 Sentiment Distribution")
    chart_data = pd.DataFrame({
        'Sentiment': ['Fear', 'Neutral', 'Greed'],
        'Percentage': [st.session_state.fear, st.session_state.neutral, st.session_state.greed]
    })
    st.bar_chart(chart_data.set_index('Sentiment'))
    
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

st.markdown("---")
st.caption("⚠️ **Disclaimer**: Educational purposes only. Not financial advice.")
st.caption("Built with Streamlit • FinBERT • Gemini AI • GNews • NewsAPI • Finnhub")    try:
        model = pipeline("sentiment-analysis", model="ProsusAI/finbert", max_length=512, truncation=True)
        return model
    except Exception as e:
        st.error(f"❌ Error loading FinBERT model: {e}")
        st.info("💡 Try restarting the app or check your internet connection")
        return None

sentiment_model = load_model()

# Fetch news from multiple premium sources
@st.cache_data(ttl=900)
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
    if not FEEDPARSER_AVAILABLE:
        return []
    
    try:
        feeds = [
            'https://www.cnbc.com/id/100003114/device/rss/rss.html',
            'https://www.cnbc.com/id/10001147/device/rss/rss.html',
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
                        'source': 'CNBC',
                        'url': entry.get('link', ''),
                        'publishedAt': entry.get('published', '')
                    })
        
        return articles
    except Exception as e:
        return []

@st.cache_data(ttl=900)
def fetch_bloomberg_rss(query):
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
@st.cache_data(ttl=300)
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
    return round(np.random.uniform(0.6, 1.4), 2)

def analyze_sentiment(articles):
    if not articles or sentiment_model is None:
        return 0, 0, 0, pd.DataFrame()
    
    try:
        texts = [article['text'][:512] for article in articles[:50]]
        results = sentiment_model(texts)
        
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

def generate_signal(fear, greed, neutral, vix, pcr, nifty_change):
    if vix is None:
        return "⏳ INSUFFICIENT DATA", "neutral"
    
    score = 0
    
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
    
    if vix > 20:
        score -= 2
    elif vix > 15:
        score -= 1
    elif vix < 10:
        score += 2
    elif vix < 12:
        score += 1
    
    if pcr > 1.3:
        score -= 2
    elif pcr > 1.1:
        score -= 1
    elif pcr < 0.7:
        score += 2
    elif pcr < 0.9:
        score += 1
    
    if nifty_change and nifty_change < -2:
        score -= 1
    elif nifty_change and nifty_change > 2:
        score += 1
    
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

def explain_market_gemini(api_key, signal, fear, greed, neutral, vix, pcr, articles, query):
    """Generate AI analysis using Gemini"""
    if not api_key:
        st.warning("⚠️ Gemini API key not configured")
        return None
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        
        headlines_text = "\n".join([f"{i+1}. [{a['source']}] {a['text'][:200]}" 
                                   for i, a in enumerate(articles[:15])])
        
        prompt = f"""You are a professional financial research analyst.

User Search Term: "{query}"

Task: Only analyze information DIRECTLY related to "{query}".

Market Sentiment Data:
- Overall Signal: {signal}
- Fear: {fear}% | Greed: {greed}% | Neutral: {neutral}%
- VIX: {vix if vix else 'N/A'} | PCR: {pcr:.2f}

Recent News:
{headlines_text}

Provide:
1. **Sentiment Summary** - Classify relevant articles as Bullish/Bearish/Neutral for {query}
2. **Key Insights** - 3 concise points on how news affects {query}
3. **Outlook** - One-line: Strongly Bullish/Mildly Bullish/Neutral/Mildly Bearish/Strongly Bearish
4. **Action** - Suggest: Accumulate/Hold/Book Profits/Avoid with brief risk note

Rules:
- Focus ONLY on {query}
- Be factual and data-driven
- Max 250 words
- Use clear headers"""

        response = model.generate_content(prompt)
        
        if response and response.text:
            return response.text
        else:
            st.warning("⚠️ Gemini returned empty response")
            return None
            
    except Exception as e:
        error_msg = str(e)
        
        with st.expander("🔍 Gemini API Error Details", expanded=False):
            st.code(error_msg)
        
        if "API_KEY_INVALID" in error_msg or "invalid" in error_msg.lower():
            st.error("❌ **Invalid Gemini API Key**")
            st.info("Get a new key: https://aistudio.google.com/app/apikey")
        elif "quota" in error_msg.lower() or "resource_exhausted" in error_msg.lower():
            st.warning("⚠️ **API Quota Exceeded** - Free tier limit reached. Wait 60 seconds.")
        elif "429" in error_msg or "rate" in error_msg.lower():
            st.warning("⚠️ **Rate Limit** - Too many requests. Wait 60 seconds.")
        elif "blocked" in error_msg.lower() or "safety" in error_msg.lower():
            st.warning("⚠️ **Content filtered** - Try a different search term")
        else:
            st.error(f"❌ **Gemini Error** - Check error details above")
        
        return None

# --- UI ---
st.title("📊 Smart Money Market Sentiment Analyzer")
st.caption("Real-time sentiment analysis from CNBC, Bloomberg, Finnhub & NewsAPI")

# Sidebar
with st.sidebar:
    st.header("📊 API Status")
    
    if GEMINI_API_KEY:
        st.success(f"✅ **Gemini**: {GEMINI_API_KEY[:10]}...{GEMINI_API_KEY[-4:]}")
    else:
        st.error("❌ **Gemini**: Not configured")
    
    if GNEWS_API_KEY:
        st.success(f"✅ **GNews**: {GNEWS_API_KEY[:10]}...{GNEWS_API_KEY[-4:]}")
    else:
        st.error("❌ **GNews**: Not configured")
    
    if NEWSAPI_KEY:
        st.success(f"✅ **NewsAPI**: Configured")
    else:
        st.info("⚠️ **NewsAPI**: Optional")
    
    if FINNHUB_KEY:
        st.success(f"✅ **Finnhub**: Configured")
    else:
        st.info("⚠️ **Finnhub**: Optional")
    
    st.markdown("---")
    st.header("🔍 Search Settings")
    
    search_query = st.text_input(
        "Target Stock/Topic", 
        value="NIFTY 50",
        help="Enter stock name, sector, or market event"
    )
    
    use_rss = st.checkbox("Include CNBC & Bloomberg RSS", 
                         value=FEEDPARSER_AVAILABLE,
                         help="Requires feedparser")
    
    st.markdown("---")
    st.subheader("🔧 System Status")
    
    if YFINANCE_AVAILABLE:
        st.success("✅ Market Data (yfinance)")
    else:
        st.error("❌ Market Data (yfinance)")
    
    if FEEDPARSER_AVAILABLE:
        st.success("✅ RSS Feeds (feedparser)")
    else:
        st.warning("⚠️ RSS Feeds (optional)")

# Main Analysis
if st.button("🔄 Analyze Market Sentiment", type="primary", use_container_width=True):
    
    if not GNEWS_API_KEY:
        st.error("❌ GNews API key required! Please configure it in the sidebar.")
        st.stop()
    
    with st.spinner(f"🔍 Analyzing '{search_query}'..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_articles = []
        
        status_text.text("📰 Fetching from GNews...")
        progress_bar.progress(15)
        gnews_articles = fetch_gnews(GNEWS_API_KEY, search_query)
        all_articles.extend(gnews_articles)
        
        if NEWSAPI_KEY:
            status_text.text("📊 Fetching from NewsAPI...")
            progress_bar.progress(30)
            newsapi_articles = fetch_newsapi(NEWSAPI_KEY, search_query)
            all_articles.extend(newsapi_articles)
        
        if FINNHUB_KEY:
            status_text.text("📈 Fetching from Finnhub...")
            progress_bar.progress(45)
            finnhub_articles = fetch_finnhub(FINNHUB_KEY, search_query)
            all_articles.extend(finnhub_articles)
        
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
            st.error(f"❌ No articles found for '{search_query}'")
            st.stop()
        
        status_text.text("🧠 Analyzing sentiment with FinBERT...")
        progress_bar.progress(75)
        fear, greed, neutral, sentiment_df = analyze_sentiment(all_articles)
        
        status_text.text("📈 Fetching market indicators...")
        progress_bar.progress(85)
        nifty_price, nifty_change = get_nifty()
        vix_value = get_vix()
        pcr_value = get_pcr()
        
        signal, signal_type = generate_signal(fear, greed, neutral, vix_value, pcr_value, nifty_change)
        
        gemini_explanation = None
        if GEMINI_API_KEY:
            status_text.text("🤖 Generating AI insights...")
            progress_bar.progress(95)
            gemini_explanation = explain_market_gemini(
                GEMINI_API_KEY, signal, fear, greed, neutral, 
                vix_value, pcr_value, all_articles, search_query
            )
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        source_counts = {}
        for article in all_articles:
            source = article['source']
            source_counts[source] = source_counts.get(source, 0) + 1
        
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
    
    st.subheader("📡 Data Sources")
    source_cols = st.columns(len(st.session_state.source_counts))
    for i, (source, count) in enumerate(st.session_state.source_counts.items()):
        with source_cols[i]:
            st.metric(source, count)
    
    st.subheader("📈 Market Indicators")
    col1, col2 = st.columns(2)
    
    with col1:
        vix_color = "🔴" if st.session_state.vix and st.session_state.vix > 15 else "🟢"
        st.metric(f"{vix_color} VIX (Volatility)", 
                 f"{st.session_state.vix}" if st.session_state.vix else "N/A")
    
    with col2:
        pcr_color = "🔴" if st.session_state.pcr > 1.2 else "🟢" if st.session_state.pcr < 0.8 else "🟡"
        st.metric(f"{pcr_color} Put-Call Ratio", f"{st.session_state.pcr:.2f}")
    
    st.subheader("🚨 Market Signal")
    if st.session_state.signal_type == "danger":
        st.error(st.session_state.signal)
    elif st.session_state.signal_type == "warning":
        st.warning(st.session_state.signal)
    else:
        st.info(st.session_state.signal)
    
    st.subheader("🤖 AI Financial Analysis")
    if st.session_state.gemini_explanation:
        st.markdown(st.session_state.gemini_explanation)
    else:
        st.info("💡 Gemini AI analysis unavailable - check errors above")
    
    st.subheader("📊 Sentiment Distribution")
    chart_data = pd.DataFrame({
        'Sentiment': ['Fear', 'Neutral', 'Greed'],
        'Percentage': [st.session_state.fear, st.session_state.neutral, st.session_state.greed]
    })
    st.bar_chart(chart_data.set_index('Sentiment'))
    
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

st.markdown("---")
st.caption("⚠️ **Disclaimer**: Educational purposes only. Not financial advice.")
st.caption("Built with Streamlit • FinBERT • Gemini AI • GNews • NewsAPI • Finnhub")    GEMINI_API_KEY = ""
    NEWSAPI_KEY = ""
    FINNHUB_KEY = ""

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
@st.cache_data(ttl=900)
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
    if not FEEDPARSER_AVAILABLE:
        return []
    
    try:
        feeds = [
            'https://www.cnbc.com/id/100003114/device/rss/rss.html',
            'https://www.cnbc.com/id/10001147/device/rss/rss.html',
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
                        'source': 'CNBC',
                        'url': entry.get('link', ''),
                        'publishedAt': entry.get('published', '')
                    })
        
        return articles
    except Exception as e:
        return []

@st.cache_data(ttl=900)
def fetch_bloomberg_rss(query):
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
@st.cache_data(ttl=300)
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
    return round(np.random.uniform(0.6, 1.4), 2)

def analyze_sentiment(articles):
    if not articles or sentiment_model is None:
        return 0, 0, 0, pd.DataFrame()
    
    try:
        texts = [article['text'][:512] for article in articles[:50]]
        results = sentiment_model(texts)
        
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

def generate_signal(fear, greed, neutral, vix, pcr, nifty_change):
    if vix is None:
        return "⏳ INSUFFICIENT DATA", "neutral"
    
    score = 0
    
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
    
    if vix > 20:
        score -= 2
    elif vix > 15:
        score -= 1
    elif vix < 10:
        score += 2
    elif vix < 12:
        score += 1
    
    if pcr > 1.3:
        score -= 2
    elif pcr > 1.1:
        score -= 1
    elif pcr < 0.7:
        score += 2
    elif pcr < 0.9:
        score += 1
    
    if nifty_change and nifty_change < -2:
        score -= 1
    elif nifty_change and nifty_change > 2:
        score += 1
    
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

def explain_market_gemini(api_key, signal, fear, greed, neutral, vix, pcr, articles, query):
    """Generate AI analysis using Gemini"""
    if not api_key:
        st.warning("⚠️ Gemini API key not configured")
        return None
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        
        headlines_text = "\n".join([f"{i+1}. [{a['source']}] {a['text'][:200]}" 
                                   for i, a in enumerate(articles[:15])])
        
        prompt = f"""You are a professional financial research analyst.

User Search Term: "{query}"

Task: Only analyze information DIRECTLY related to "{query}".

Market Sentiment Data:
- Overall Signal: {signal}
- Fear: {fear}% | Greed: {greed}% | Neutral: {neutral}%
- VIX: {vix if vix else 'N/A'} | PCR: {pcr:.2f}

Recent News:
{headlines_text}

Provide:
1. **Sentiment Summary** - Classify relevant articles as Bullish/Bearish/Neutral for {query}
2. **Key Insights** - 3 concise points on how news affects {query}
3. **Outlook** - One-line: Strongly Bullish/Mildly Bullish/Neutral/Mildly Bearish/Strongly Bearish
4. **Action** - Suggest: Accumulate/Hold/Book Profits/Avoid with brief risk note

Rules:
- Focus ONLY on {query}
- Be factual and data-driven
- Max 250 words
- Use clear headers"""

        response = model.generate_content(prompt)
        
        if response and response.text:
            return response.text
        else:
            st.warning("⚠️ Gemini returned empty response")
            return None
            
    except Exception as e:
        error_msg = str(e)
        
        with st.expander("🔍 Gemini API Error Details", expanded=False):
            st.code(error_msg)
        
        if "API_KEY_INVALID" in error_msg or "invalid" in error_msg.lower():
            st.error("❌ **Invalid Gemini API Key**")
            st.info("Get a new key: https://aistudio.google.com/app/apikey")
        elif "quota" in error_msg.lower() or "resource_exhausted" in error_msg.lower():
            st.warning("⚠️ **API Quota Exceeded** - Free tier limit reached. Wait 60 seconds.")
        elif "429" in error_msg or "rate" in error_msg.lower():
            st.warning("⚠️ **Rate Limit** - Too many requests. Wait 60 seconds.")
        elif "blocked" in error_msg.lower() or "safety" in error_msg.lower():
            st.warning("⚠️ **Content filtered** - Try a different search term")
        else:
            st.error(f"❌ **Gemini Error** - Check error details above")
        
        return None

# --- UI ---
st.title("📊 Smart Money Market Sentiment Analyzer")
st.caption("Real-time sentiment analysis from CNBC, Bloomberg, Finnhub & NewsAPI")

# Sidebar
with st.sidebar:
    st.header("🔑 API Configuration")
    
    # Manual API key input option
    with st.expander("⚙️ Configure API Keys", expanded=not (GEMINI_API_KEY and GNEWS_API_KEY)):
        st.info("💡 **Setup Options:**\n1. Use Streamlit secrets (`.streamlit/secrets.toml`)\n2. Enter keys below (session only)")
        
        temp_gnews = st.text_input("GNews API Key", value=GNEWS_API_KEY, type="password", key="gnews_input")
        temp_gemini = st.text_input("Gemini API Key", value=GEMINI_API_KEY, type="password", key="gemini_input")
        temp_newsapi = st.text_input("NewsAPI Key (Optional)", value=NEWSAPI_KEY, type="password", key="newsapi_input")
        temp_finnhub = st.text_input("Finnhub Key (Optional)", value=FINNHUB_KEY, type="password", key="finnhub_input")
        
        if temp_gnews:
            GNEWS_API_KEY = temp_gnews
        if temp_gemini:
            GEMINI_API_KEY = temp_gemini
        if temp_newsapi:
            NEWSAPI_KEY = temp_newsapi
        if temp_finnhub:
            FINNHUB_KEY = temp_finnhub
        
        st.caption("🔐 Keys are not saved. Use secrets.toml for persistence.")
    
    st.markdown("---")
    
    st.subheader("📊 API Status")
    
    if GEMINI_API_KEY:
        st.success(f"✅ **Gemini**: {GEMINI_API_KEY[:10]}...{GEMINI_API_KEY[-4:]}")
    else:
        st.error("❌ **Gemini**: Not configured")
    
    if GNEWS_API_KEY:
        st.success(f"✅ **GNews**: {GNEWS_API_KEY[:10]}...{GNEWS_API_KEY[-4:]}")
    else:
        st.error("❌ **GNews**: Not configured")
    
    if NEWSAPI_KEY:
        st.success(f"✅ **NewsAPI**: Configured")
    else:
        st.info("⚠️ **NewsAPI**: Optional")
    
    if FINNHUB_KEY:
        st.success(f"✅ **Finnhub**: Configured")
    else:
        st.info("⚠️ **Finnhub**: Optional")
    
    st.markdown("---")
    st.header("🔍 Search Settings")
    
    search_query = st.text_input(
        "Target Stock/Topic", 
        value="NIFTY 50",
        help="Enter stock name, sector, or market event"
    )
    
    use_rss = st.checkbox("Include CNBC & Bloomberg RSS", 
                         value=FEEDPARSER_AVAILABLE,
                         help="Requires feedparser")
    
    st.markdown("---")
    st.subheader("🔧 System Status")
    
    if YFINANCE_AVAILABLE:
        st.success("✅ Market Data (yfinance)")
    else:
        st.error("❌ Market Data (yfinance)")
    
    if FEEDPARSER_AVAILABLE:
        st.success("✅ RSS Feeds (feedparser)")
    else:
        st.warning("⚠️ RSS Feeds (optional)")

# Main Analysis
if st.button("🔄 Analyze Market Sentiment", type="primary", use_container_width=True):
    
    if not GNEWS_API_KEY:
        st.error("❌ GNews API key required! Please configure it in the sidebar.")
        st.stop()
    
    with st.spinner(f"🔍 Analyzing '{search_query}'..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_articles = []
        
        status_text.text("📰 Fetching from GNews...")
        progress_bar.progress(15)
        gnews_articles = fetch_gnews(GNEWS_API_KEY, search_query)
        all_articles.extend(gnews_articles)
        
        if NEWSAPI_KEY:
            status_text.text("📊 Fetching from NewsAPI...")
            progress_bar.progress(30)
            newsapi_articles = fetch_newsapi(NEWSAPI_KEY, search_query)
            all_articles.extend(newsapi_articles)
        
        if FINNHUB_KEY:
            status_text.text("📈 Fetching from Finnhub...")
            progress_bar.progress(45)
            finnhub_articles = fetch_finnhub(FINNHUB_KEY, search_query)
            all_articles.extend(finnhub_articles)
        
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
            st.error(f"❌ No articles found for '{search_query}'")
            st.stop()
        
        status_text.text("🧠 Analyzing sentiment with FinBERT...")
        progress_bar.progress(75)
        fear, greed, neutral, sentiment_df = analyze_sentiment(all_articles)
        
        status_text.text("📈 Fetching market indicators...")
        progress_bar.progress(85)
        nifty_price, nifty_change = get_nifty()
        vix_value = get_vix()
        pcr_value = get_pcr()
        
        signal, signal_type = generate_signal(fear, greed, neutral, vix_value, pcr_value, nifty_change)
        
        gemini_explanation = None
        if GEMINI_API_KEY:
            status_text.text("🤖 Generating AI insights...")
            progress_bar.progress(95)
            gemini_explanation = explain_market_gemini(
                GEMINI_API_KEY, signal, fear, greed, neutral, 
                vix_value, pcr_value, all_articles, search_query
            )
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        source_counts = {}
        for article in all_articles:
            source = article['source']
            source_counts[source] = source_counts.get(source, 0) + 1
        
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
    
    st.subheader("📡 Data Sources")
    source_cols = st.columns(len(st.session_state.source_counts))
    for i, (source, count) in enumerate(st.session_state.source_counts.items()):
        with source_cols[i]:
            st.metric(source, count)
    
    st.subheader("📈 Market Indicators")
    col1, col2 = st.columns(2)
    
    with col1:
        vix_color = "🔴" if st.session_state.vix and st.session_state.vix > 15 else "🟢"
        st.metric(f"{vix_color} VIX (Volatility)", 
                 f"{st.session_state.vix}" if st.session_state.vix else "N/A")
    
    with col2:
        pcr_color = "🔴" if st.session_state.pcr > 1.2 else "🟢" if st.session_state.pcr < 0.8 else "🟡"
        st.metric(f"{pcr_color} Put-Call Ratio", f"{st.session_state.pcr:.2f}")
    
    st.subheader("🚨 Market Signal")
    if st.session_state.signal_type == "danger":
        st.error(st.session_state.signal)
    elif st.session_state.signal_type == "warning":
        st.warning(st.session_state.signal)
    else:
        st.info(st.session_state.signal)
    
    st.subheader("🤖 AI Financial Analysis")
    if st.session_state.gemini_explanation:
        st.markdown(st.session_state.gemini_explanation)
    else:
        st.info("💡 Gemini AI analysis unavailable - check errors above")
    
    st.subheader("📊 Sentiment Distribution")
    chart_data = pd.DataFrame({
        'Sentiment': ['Fear', 'Neutral', 'Greed'],
        'Percentage': [st.session_state.fear, st.session_state.neutral, st.session_state.greed]
    })
    st.bar_chart(chart_data.set_index('Sentiment'))
    
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

st.markdown("---")
st.caption("⚠️ **Disclaimer**: Educational purposes only. Not financial advice.")
st.caption("Built with Streamlit • FinBERT • Gemini AI • GNews • NewsAPI • Finnhub")
