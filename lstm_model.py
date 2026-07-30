import ccxt
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import requests
import xml.etree.ElementTree as ET
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

print("🎯 Bullseye - LSTM Neural Network")
print("=" * 40)

# --- SENTIMENT ---
def get_sentiment_score(asset_name, atype, ticker=None):
    try:
        if atype == 'crypto':
            url      = f"https://news.google.com/rss/search?q={asset_name}+crypto&hl=en-US&gl=US&ceid=US:en"
            response = requests.get(url, timeout=10)
            root     = ET.fromstring(response.content)
            items    = root.findall('.//item')
            scores   = []
            for item in items[:10]:
                title = item.find('title').text
                if title:
                    scores.append(TextBlob(title).sentiment.polarity)
            return round(sum(scores) / len(scores), 3) if scores else 0.0
        else:
            stock  = yf.Ticker(ticker)
            news   = stock.news[:10]
            scores = []
            for article in news:
                title = None
                if isinstance(article, dict):
                    title = article.get('title') or (article.get('content', {}).get('title') if isinstance(article.get('content'), dict) else None)
                if title:
                    scores.append(TextBlob(title).sentiment.polarity)
            return round(sum(scores) / len(scores), 3) if scores else 0.0
    except:
        return 0.0

# --- FETCH DATA ---
def fetch_crypto(symbol, days=2000):
    binance = ccxt.binance()
    ohlcv   = binance.fetch_ohlcv(symbol, timeframe='1d', limit=days)
    df      = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def fetch_stock(ticker, period="5y"):
    df = yf.Ticker(ticker).history(period=period)
    df.columns = [c.lower() for c in df.columns]
    return df

# --- ADD FEATURES ---
def add_features(df, sentiment=0.0):
    close = df['close']
    high  = df['high']
    low   = df['low']
    vol   = df['volume']

    df['ma7']  = ta.sma(close, length=7)
    df['ma21'] = ta.sma(close, length=21)
    df['ma50'] = ta.sma(close, length=50)
    df['rsi']  = ta.rsi(close, length=14)

    macd = ta.macd(close)
    df['macd']        = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']
    df['macd_hist']   = macd['MACDh_12_26_9']

    bb      = ta.bbands(close, length=20)
    bb_cols = bb.columns.tolist()
    df['bb_upper'] = bb[bb_cols[2]]
    df['bb_lower'] = bb[bb_cols[0]]
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / bb[bb_cols[1]]

    df['price_vs_ma7']  = (close - df['ma7'])  / df['ma7']  * 100
    df['price_vs_ma21'] = (close - df['ma21']) / df['ma21'] * 100
    df['change_1d'] = close.pct_change(1) * 100
    df['change_3d'] = close.pct_change(3) * 100
    df['change_7d'] = close.pct_change(7) * 100
    df['vol_change'] = vol.pct_change(1) * 100
    df['vol_vs_avg'] = vol / vol.rolling(20).mean()
    df['atr']        = ta.atr(high, low, close, length=14)

    df['golden_cross']   = ((df['ma7'] > df['ma21']) & (df['ma7'].shift(1) <= df['ma21'].shift(1))).astype(int)
    df['death_cross']    = ((df['ma7'] < df['ma21']) & (df['ma7'].shift(1) >= df['ma21'].shift(1))).astype(int)
    df['ma7_above_ma21'] = (df['ma7'] > df['ma21']).astype(int)
    df['macd_cross_up']  = ((df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
    df['high_volume']    = (df['vol_vs_avg'] > 1.5).astype(int)
    df['sentiment']      = sentiment

    df['target'] = (close.shift(-7) > close).astype(int)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    return df

# --- BUILD SEQUENCES FOR LSTM ---
def build_sequences(df, features, sequence_length=30):
    X, y = [], []
    data = df[features].values
    tgt  = df['target'].values

    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
        y.append(tgt[i])

    return np.array(X), np.array(y)

# --- BUILD LSTM MODEL ---
def build_lstm(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        BatchNormalization(),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        BatchNormalization(),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# --- TRAIN AND EVALUATE ---
def train_lstm(df, asset_name, sequence_length=30):
    features = [
        'rsi', 'ma7', 'ma21', 'ma50',
        'macd', 'macd_signal', 'macd_hist',
        'bb_upper', 'bb_lower', 'bb_width',
        'price_vs_ma7', 'price_vs_ma21',
        'change_1d', 'change_3d', 'change_7d',
        'vol_change', 'vol_vs_avg', 'atr',
        'golden_cross', 'death_cross',
        'ma7_above_ma21', 'macd_cross_up',
        'high_volume', 'sentiment'
    ]

    # Normalize features — CRITICAL for LSTM!!
    scaler  = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[features] = scaler.fit_transform(df[features])

    X, y = build_sequences(df_scaled, features, sequence_length)

    if len(X) < 100:
        print(f"  ⚠️ Not enough data for LSTM")
        return None, 0

    # Train/Val/Test split — proper 3-way split!!
    train_size = int(len(X) * 0.70)
    val_size   = int(len(X) * 0.15)

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val   = X[train_size:train_size+val_size]
    y_val   = y[train_size:train_size+val_size]
    X_test  = X[train_size+val_size:]
    y_test  = y[train_size+val_size:]

    print(f"  📊 Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # Build and train
    model = build_lstm((sequence_length, len(features)))

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )

    # Evaluate on test set
    y_pred    = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    accuracy  = accuracy_score(y_test, y_pred)

    # Current prediction
    latest_seq = X[-1:]
    prob       = model.predict(latest_seq, verbose=0)[0][0]

    if prob >= 0.60:
        signal = "🟢 BUY"
    elif prob <= 0.40:
        signal = "🔴 SELL"
    else:
        signal = "⚪ HOLD"

    print(f"\n{'='*40}")
    print(f"🧠 {asset_name} LSTM Results")
    print(f"{'='*40}")
    print(f"  Accuracy  : {accuracy*100:.1f}%")
    print(f"  Signal    : {signal}")
    print(f"  Up Prob   : {prob*100:.1f}%")
    print(f"  Down Prob : {(1-prob)*100:.1f}%")

    return model, accuracy

# --- RUN ALL ASSETS ---
results = []

crypto_assets = {
    'Bitcoin'  : ('BTC/USDT', 'crypto', None),
    'Ethereum' : ('ETH/USDT', 'crypto', None),
    'BNB'      : ('BNB/USDT', 'crypto', None),
    'Litecoin' : ('LTC/USDT', 'crypto', None),
    'XRP'      : ('XRP/USDT', 'crypto', None),
    'Dogecoin' : ('DOGE/USDT','crypto', None),
}

stock_assets = {
    'Infosys'  : ('INFY.NS',     'stock', 'INFY.NS'),
    'TCS'      : ('TCS.NS',      'stock', 'TCS.NS'),
    'Reliance' : ('RELIANCE.NS', 'stock', 'RELIANCE.NS'),
    'HDFC Bank': ('HDFCBANK.NS', 'stock', 'HDFCBANK.NS'),
    'SBI'      : ('SBIN.NS',     'stock', 'SBIN.NS'),
}

print("\n📦 CRYPTO LSTM MODELS")
for name, (symbol, atype, ticker) in crypto_assets.items():
    print(f"\n⏳ Training {name} LSTM...")
    try:
        sentiment = get_sentiment_score(name, atype, ticker)
        print(f"  📰 Sentiment: {sentiment:+.3f}")
        df        = fetch_crypto(symbol, days=2000)
        df        = add_features(df, sentiment)
        model, acc = train_lstm(df, name)
        if model:
            results.append({'asset': name, 'accuracy': acc})
    except Exception as e:
        print(f"  ❌ Error: {e}")

print("\n📈 STOCK LSTM MODELS")
for name, (ticker, atype, tick) in stock_assets.items():
    print(f"\n⏳ Training {name} LSTM...")
    try:
        sentiment  = get_sentiment_score(name, atype, tick)
        print(f"  📰 Sentiment: {sentiment:+.3f}")
        df         = fetch_stock(ticker, period="5y")
        df         = add_features(df, sentiment)
        model, acc = train_lstm(df, name)
        if model:
            results.append({'asset': name, 'accuracy': acc})
    except Exception as e:
        print(f"  ❌ Error: {e}")

# --- SUMMARY ---
print(f"\n{'='*40}")
print(f"🎯 LSTM MODELS SUMMARY")
print(f"{'='*40}")
for r in results:
    bar = "🟢" if r['accuracy'] >= 0.55 else "⚪" if r['accuracy'] >= 0.50 else "🔴"
    print(f"  {bar} {r['asset']:15s} : {r['accuracy']*100:.1f}%")

avg = sum(r['accuracy'] for r in results) / len(results) if results else 0
print(f"\n  Average Accuracy : {avg*100:.1f}%")
print(f"\n✅ LSTM Training complete!!")