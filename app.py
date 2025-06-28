import streamlit as st
import requests
import pandas as pd
from transformers import pipeline
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

# --- PAGE CONFIG --- #
st.set_page_config(page_title="Market Signal AI", layout="wide")
st.title("📊 Daily AI Investment Signals")

# --- SELECT INPUTS --- #
index_options = ["S&P 500", "Nasdaq", "Dow Jones", "Nikkei", "Hang Seng", "Shanghai"]
stock_input = st.text_input("🔍 Enter a stock symbol (e.g., AAPL, TSLA, BABA):")
selected_index = st.selectbox("🌍 Select a Market Index:", index_options)

# --- SIMULATED PRICE CHANGE (Replace with real API) --- #
price_change = {
    "S&P 500": 0.8,
    "Nasdaq": -0.5,
    "Dow Jones": 1.1,
    "Nikkei": 0.3,
    "Hang Seng": -1.2,
    "Shanghai": 0.4,
    "default": 0.2,
}

# --- FETCH NEWS + SENTIMENT --- #
st.subheader("📰 News Sentiment")

@st.cache_data(ttl=3600)
def fetch_headlines():
    url = "https://www.reuters.com/markets/"
    r = requests.get(url)
    soup = BeautifulSoup(r.content, "html.parser")
    headlines = [h.text.strip() for h in soup.find_all("h3") if len(h.text.strip()) > 40]
    return headlines[:5]

sentiment_pipeline = pipeline("sentiment-analysis")
headlines = fetch_headlines()
sentiment_scores = []

for h in headlines:
    result = sentiment_pipeline(h)[0]
    sentiment_scores.append((h, result['label'], result['score']))

if sentiment_scores:
    for text, label, score in sentiment_scores:
        st.markdown(f"**{text}**")
        st.caption(f"Sentiment: {label} ({round(score * 100, 1)}%)")
else:
    st.warning("No news headlines found or failed to analyze sentiment.")

# --- DAILY SIGNAL LOGIC --- #
st.subheader("🚦 Daily Signal")

target = stock_input.upper() if stock_input else selected_index
index_move = price_change.get(target, price_change["default"])

if sentiment_scores:
    sentiment_value = sum([s[2] if s[1] == 'POSITIVE' else -s[2] for s in sentiment_scores]) / len(sentiment_scores)
else:
    sentiment_value = 0

signal = "Hold"
if index_move > 0.5 and sentiment_value > 0.2:
    signal = "Buy"
elif index_move < -0.5 and sentiment_value < -0.2:
    signal = "Sell"

st.metric(label=f"Recommendation for {target}", value=signal)

# --- EXPLANATION --- #
st.subheader("🧠 Explanation")
st.markdown(f"Price movement: **{index_move}%**")
st.markdown(f"Average news sentiment score: **{round(sentiment_value, 2)}**")

if signal == "Buy":
    st.success("Positive momentum and news sentiment suggest a buying opportunity.")
elif signal == "Sell":
    st.error("Negative sentiment and price drop indicate caution.")
else:
    st.info("Mixed signals – best to hold for now.")

# --- TOP GURU LINKS --- #
st.subheader("📚 Insights from Top Investors")
guru_sources = {
    "Warren Buffett (CNBC)": "https://www.cnbc.com/warren-buffett/",
    "Ray Dalio (LinkedIn)": "https://www.linkedin.com/in/raydalio/",
    "Cathie Wood (ARK Invest)": "https://ark-invest.com/news/",
    "Howard Marks (Oaktree Memos)": "https://www.oaktreecapital.com/insights/memo"
}
for name, url in guru_sources.items():
    st.markdown(f"🔗 [{name}]({url})")

# --- TRUMP TWEET IMPACT --- #
st.subheader("🦅 Trump Tweet Impact")

fake_tweet = st.text_area("📢 Enter or simulate a recent Trump post:",
                          value="BIDEN KILLED ENERGY INDEPENDENCE. OIL SHOULD BE $30!!!")

trump_result = sentiment_pipeline(fake_tweet)[0]
label = trump_result['label']
score = trump_result['score']

keywords = fake_tweet.lower()
action = "Hold"
suggested_asset = ""

if "oil" in keywords or "energy" in keywords:
    suggested_asset = "USO (Oil ETF)"
    action = "Sell" if label == "NEGATIVE" else "Buy"
elif "china" in keywords or "tariff" in keywords:
    suggested_asset = "SPY (S&P 500)"
    action = "Sell" if label == "NEGATIVE" else "Buy"
elif "inflation" in keywords or "gold" in keywords:
    suggested_asset = "GLD (Gold ETF)"
    action = "Buy" if label == "NEGATIVE" else "Sell"

if suggested_asset:
    st.metric(label=f"Trump Tweet Sentiment: {label}", value=f"{round(score * 100, 1)}%")
    st.metric(label=f"Suggested Action: {suggested_asset}", value=action)
    if action == "Buy":
        st.success(f"Trump's post suggests a BUY on {suggested_asset}.")
    elif action == "Sell":
        st.error(f"Trump's post suggests a SELL on {suggested_asset}.")
    else:
        st.info("Hold — tweet has no strong signal.")
else:
    st.warning("Tweet doesn't match known asset signals yet.")

# --- FOOTER --- #
st.markdown("---")
st.caption("Built with ❤️ using Reuters, HuggingFace, and curated guru insights.")