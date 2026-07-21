import requests
from textblob import TextBlob
import yfinance as yf
from datetime import datetime
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

print("🎯 Bullseye - Sentiment Analyzer")
print("=" * 40)


# --- CRYPTO SENTIMENT via Google News RSS ---
def get_crypto_sentiment(coin_name):
    try:
        url = f"https://news.google.com/rss/search?q={coin_name}+crypto&hl=en-US&gl=US&ceid=US:en"
        response = requests.get(url, timeout=10)

        # Parse headlines from RSS
        import xml.etree.ElementTree as ET
        root = ET.fromstring(response.content)
        items = root.findall('.//item')

        headlines = []
        sentiments = []

        for item in items[:10]:  # top 10 news
            title = item.find('title').text
            if title:
                headlines.append(title)
                score = TextBlob(title).sentiment.polarity
                sentiments.append(score)

        if not sentiments:
            return 0.0, []

        avg_sentiment = sum(sentiments) / len(sentiments)
        return round(avg_sentiment, 3), headlines

    except Exception as e:
        return 0.0, []


# --- STOCK SENTIMENT via Yahoo Finance News ---
def get_stock_sentiment(ticker):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news[:10]  # top 10 news

        headlines = []
        sentiments = []

        for article in news:
            # Handle both old and new yfinance news format
            title = None
            if isinstance(article, dict):
                title = article.get('title') or article.get('content', {}).get('title') if isinstance(
                    article.get('content'), dict) else None
            if title:
                headlines.append(title)
                score = TextBlob(title).sentiment.polarity
                sentiments.append(score)

        if not sentiments:
            return 0.0, []

        avg_sentiment = sum(sentiments) / len(sentiments)
        return round(avg_sentiment, 3), headlines

    except Exception as e:
        return 0.0, []


# --- SENTIMENT LABEL ---
def sentiment_label(score):
    if score > 0.15:
        return "🟢 BULLISH"
    elif score < -0.15:
        return "🔴 BEARISH"
    else:
        return "⚪ NEUTRAL"


# --- TEST ALL ASSETS ---
crypto_assets = {
    'Bitcoin': 'bitcoin',
    'Ethereum': 'ethereum',
    'BNB': 'bnb',
    'Litecoin': 'litecoin',
    'XRP': 'xrp',
    'Dogecoin': 'dogecoin',
}

stock_assets = {
    'Infosys': 'INFY.NS',
    'TCS': 'TCS.NS',
    'SBI': 'SBIN.NS',
    'Reliance': 'RELIANCE.NS',
}

print("\n📦 CRYPTO SENTIMENT")
print("=" * 40)
for name, keyword in crypto_assets.items():
    score, headlines = get_crypto_sentiment(keyword)
    label = sentiment_label(score)
    print(f"\n{name}")
    print(f"  Sentiment : {label} ({score:+.3f})")
    if headlines:
        print(f"  Top News  : {headlines[0][:60]}...")

print("\n📈 STOCK SENTIMENT")
print("=" * 40)
for name, ticker in stock_assets.items():
    score, headlines = get_stock_sentiment(ticker)
    label = sentiment_label(score)
    print(f"\n{name}")
    print(f"  Sentiment : {label} ({score:+.3f})")
    if headlines:
        print(f"  Top News  : {headlines[0][:60]}...")

print("\n" + "=" * 40)
print("✅ Sentiment analysis complete!!")