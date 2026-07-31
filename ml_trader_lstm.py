import ccxt
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import json
import csv
import os
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
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

print("🎯 Bullseye - LSTM Trading Engine")
print("=" * 40)

# --- SETTINGS ---
STARTING_BALANCE  = 10000
TRADE_PERCENT     = 0.20
STOP_LOSS_PCT     = 0.07
TARGET_PCT        = 0.15
LSTM_BUY_THRESH   = 0.60
LSTM_SELL_THRESH  = 0.40
MIN_ACCURACY      = 0.55
WALLET_FILE       = 'ml_wallet.json'
TRADE_LOG_FILE    = 'ml_trade_log.csv'
SEQUENCE_LENGTH   = 30

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

# --- WALLET ---
def load_wallet():
    if os.path.exists(WALLET_FILE):
        with open(WALLET_FILE, 'r') as f:
            return json.load(f)
    wallet = {
        'balance'      : STARTING_BALANCE,
        'starting'     : STARTING_BALANCE,
        'open_trades'  : {},
        'total_trades' : 0,
        'wins'         : 0,
        'losses'       : 0
    }
    save_wallet(wallet)
    return wallet

def save_wallet(wallet):
    with open(WALLET_FILE, 'w') as f:
        json.dump(wallet, f, indent=2)

def log_trade(asset, action, price, amount, profit_loss, reason):
    file_exists = os.path.exists(TRADE_LOG_FILE)
    with open(TRADE_LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'asset', 'action', 'price', 'amount', 'profit_loss', 'reason'])
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            asset, action, round(price, 2),
            round(amount, 2), round(profit_loss, 2), reason
        ])

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

# --- FEATURES ---
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

# --- LSTM SIGNAL ---
def get_lstm_signal(df):
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

    # Normalize
    scaler    = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[features] = scaler.fit_transform(df[features])

    # Build sequences
    X, y = [], []
    data = df_scaled[features].values
    tgt  = df_scaled['target'].values
    for i in range(SEQUENCE_LENGTH, len(data)):
        X.append(data[i-SEQUENCE_LENGTH:i])
        y.append(tgt[i])
    X, y = np.array(X), np.array(y)

    if len(X) < 100:
        return 'SKIP', 0, 0, df['close'].iloc[-1]

    # Split
    train_size = int(len(X) * 0.70)
    val_size   = int(len(X) * 0.15)
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val   = X[train_size:train_size+val_size]
    y_val   = y[train_size:train_size+val_size]
    X_test  = X[train_size+val_size:]
    y_test  = y[train_size+val_size:]

    # Build LSTM
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(SEQUENCE_LENGTH, len(features))),
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

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50, batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )

    # Accuracy
    y_pred   = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    accuracy = accuracy_score(y_test, y_pred)
    price    = df['close'].iloc[-1]

    if accuracy < MIN_ACCURACY:
        return 'SKIP', 0, accuracy, price

    # Current prediction
    prob = model.predict(X[-1:], verbose=0)[0][0]

    if prob >= LSTM_BUY_THRESH:
        return 'BUY', float(prob), accuracy, price
    elif prob <= LSTM_SELL_THRESH:
        return 'SELL', float(1-prob), accuracy, price
    else:
        return 'HOLD', float(max(prob, 1-prob)), accuracy, price

# --- MAIN TRADER ---
def run_lstm_trader():
    wallet = load_wallet()

    assets = {
        'BTC/USDT'   : ('crypto', None,            'Bitcoin'),
        'ETH/USDT'   : ('crypto', None,            'Ethereum'),
        'LTC/USDT'   : ('crypto', None,            'Litecoin'),
        'BNB/USDT'   : ('crypto', None,            'BNB'),
        'DOGE/USDT'  : ('crypto', None,            'Dogecoin'),
        'XRP/USDT'   : ('crypto', None,            'XRP'),
        'Infosys'    : ('stock',  'INFY.NS',       'Infosys'),
        'TCS'        : ('stock',  'TCS.NS',        'TCS'),
        'Reliance'   : ('stock',  'RELIANCE.NS',   'Reliance'),
        'HDFC Bank'  : ('stock',  'HDFCBANK.NS',   'HDFC Bank'),
        'SBI'        : ('stock',  'SBIN.NS',       'SBI'),
    }

    print(f"\n💰 Balance         : ₹{wallet['balance']:,.2f}")
    print(f"📊 Total Trades    : {wallet['total_trades']}")
    print(f"✅ Wins            : {wallet['wins']}")
    print(f"❌ Losses          : {wallet['losses']}")
    print("=" * 40)

    for asset, (atype, ticker, name) in assets.items():
        print(f"\n🔍 {asset} — training LSTM...")

        try:
            # Fetch data
            if atype == 'crypto':
                df       = fetch_crypto(asset, days=2000)
                currency = '$'
            else:
                df       = fetch_stock(ticker, period="5y")
                currency = '₹'

            # Get sentiment
            sentiment = get_sentiment_score(name, atype, ticker)
            print(f"  📰 Sentiment: {sentiment:+.3f}")

            # Add features with sentiment
            df = add_features(df, sentiment)

            if len(df) < 100:
                print(f"  ⚠️ Not enough data — skipping")
                continue

            # Get LSTM signal
            signal, confidence, accuracy, price = get_lstm_signal(df)

            # Latest indicators
            latest         = df.iloc[-1]
            golden_cross   = latest['golden_cross']   == 1
            death_cross    = latest['death_cross']    == 1
            ma7_above_ma21 = latest['ma7_above_ma21'] == 1
            macd_up        = latest['macd_cross_up']  == 1

            if signal == 'SKIP':
                # Try cross overrides even when accuracy low
                if golden_cross:
                    signal     = 'BUY'
                    confidence = 0.62
                    print(f"  ⚡ Golden Cross override!!")
                elif death_cross:
                    signal     = 'SELL'
                    confidence = 0.62
                    print(f"  ⚡ Death Cross override!!")
                elif ma7_above_ma21 and macd_up:
                    signal     = 'BUY'
                    confidence = 0.60
                    print(f"  ⚡ Uptrend + MACD override!!")
                else:
                    print(f"  ⏭️ Skipping — accuracy {accuracy*100:.1f}% below {MIN_ACCURACY*100:.0f}% threshold")
                    continue

            print(f"  🧠 LSTM Accuracy: {accuracy*100:.1f}% | Signal: {signal} | Confidence: {confidence*100:.1f}%")

            # --- CHECK EXITS ---
            if asset in wallet['open_trades']:
                trade      = wallet['open_trades'][asset]
                buy_price  = trade['buy_price']
                invested   = trade['invested']
                change_pct = (price - buy_price) / buy_price

                print(f"  📊 Open Trade | Bought: {currency}{buy_price:,.2f} | Now: {currency}{price:,.2f} | P/L: {change_pct*100:+.1f}%")

                exit_reason = None

                # Sentiment crash exit
                if sentiment < -0.3:
                    exit_reason = 'SENTIMENT CRASH'
                    print(f"  ⚠️ Sentiment very negative ({sentiment:+.3f}) — forcing exit!!")
                elif change_pct <= -STOP_LOSS_PCT:
                    exit_reason = 'STOP LOSS'
                elif change_pct >= TARGET_PCT:
                    exit_reason = 'TARGET HIT'
                elif signal == 'SELL':
                    exit_reason = 'LSTM SELL SIGNAL'

                if exit_reason:
                    current_value = invested * (1 + change_pct)
                    profit_loss   = current_value - invested
                    wallet['balance'] += current_value

                    if profit_loss >= 0:
                        wallet['wins'] += 1
                        print(f"  🎉 {exit_reason} → Profit: ₹{profit_loss:+,.2f} ({change_pct*100:+.1f}%)")
                    else:
                        wallet['losses'] += 1
                        print(f"  😬 {exit_reason} → Loss: ₹{profit_loss:+,.2f} ({change_pct*100:+.1f}%)")

                    log_trade(asset, 'SELL', price, current_value, profit_loss, exit_reason)
                    del wallet['open_trades'][asset]
                    wallet['total_trades'] += 1

            # --- CHECK BUY ---
            if signal == 'BUY' and asset not in wallet['open_trades']:

                # Don't buy if sentiment very negative
                if sentiment < -0.2:
                    print(f"  ⚠️ BUY blocked — sentiment negative ({sentiment:+.3f})")
                    continue

                # Position sizing based on confidence + sentiment
                if golden_cross and confidence > 0.70:
                    invest_pct = 0.30
                    print(f"  💪 Golden Cross + High Confidence — investing 30%!!")
                elif sentiment > 0.15 and confidence > 0.65:
                    invest_pct = 0.25
                    print(f"  💪 Positive sentiment + confidence — investing 25%!!")
                else:
                    invest_pct = TRADE_PERCENT

                invest_amount = wallet['balance'] * invest_pct

                if invest_amount > 100:
                    wallet['balance'] -= invest_amount
                    wallet['open_trades'][asset] = {
                        'buy_price' : price,
                        'invested'  : invest_amount,
                        'buy_time'  : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'sentiment' : sentiment,
                        'accuracy'  : accuracy
                    }
                    print(f"  🟢 LSTM BUY → {currency}{price:,.2f} | ₹{invest_amount:,.2f} | Conf: {confidence*100:.1f}% | Acc: {accuracy*100:.1f}%")
                    log_trade(asset, 'BUY', price, invest_amount, 0, f'LSTM BUY Conf:{confidence*100:.1f}% Acc:{accuracy*100:.1f}%')

        except Exception as e:
            print(f"  ❌ Error: {str(e)[:80]}")
            continue

    # --- SUMMARY ---
    total_invested = sum(t['invested'] for t in wallet['open_trades'].values())
    total_value    = wallet['balance'] + total_invested
    total_return   = ((total_value - wallet['starting']) / wallet['starting']) * 100

    print("\n" + "=" * 40)
    print(f"💰 Free Balance    : ₹{wallet['balance']:,.2f}")
    print(f"📈 Invested        : ₹{total_invested:,.2f}")
    print(f"💎 Total Value     : ₹{total_value:,.2f}")
    print(f"📊 Total Return    : {total_return:+.2f}%")

    if wallet['total_trades'] > 0:
        win_rate = (wallet['wins'] / wallet['total_trades']) * 100
        print(f"🎯 Win Rate        : {win_rate:.1f}%")
    else:
        print(f"🎯 Win Rate        : No closed trades yet")

    save_wallet(wallet)
    print("\n✅ LSTM Wallet saved!!")

run_lstm_trader()