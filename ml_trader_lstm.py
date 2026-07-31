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

print("🎯 Bullseye - LSTM Trading Engine v2")
print("=" * 40)

# --- SETTINGS ---
STARTING_BALANCE    = 10000
STOP_LOSS_PCT       = 0.07
TARGET_PCT          = 0.15
LSTM_BUY_THRESH     = 0.60
LSTM_SELL_THRESH    = 0.40
MIN_ACCURACY        = 0.55
SEQUENCE_LENGTH     = 30
WALLET_FILE         = 'ml_wallet.json'
TRADE_LOG_FILE      = 'ml_trade_log.csv'

# --- RISK MANAGEMENT SETTINGS ---
MAX_RISK_PER_TRADE  = 0.02   # never risk more than 2% of portfolio per trade
MAX_OPEN_TRADES     = 5      # max 5 open positions at once
MAX_DRAWDOWN        = 0.15   # stop trading if portfolio drops 15%
BASE_POSITION_SIZE  = 0.20   # default 20% per trade
MAX_POSITION_SIZE   = 0.30   # max 30% per trade (strong signals only)

# Assets that move together — don't hold both!!
CORRELATED_GROUPS = [
    ['BTC/USDT', 'ETH/USDT', 'LTC/USDT'],  # major crypto
    ['BNB/USDT', 'ADA/USDT', 'AVAX/USDT'], # altcoins
    ['Infosys', 'TCS', 'Wipro'],             # IT stocks
    ['HDFC Bank', 'SBI'],                    # banking stocks
]

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
        'balance'        : STARTING_BALANCE,
        'starting'       : STARTING_BALANCE,
        'peak_value'     : STARTING_BALANCE,  # for drawdown tracking
        'open_trades'    : {},
        'total_trades'   : 0,
        'wins'           : 0,
        'losses'         : 0,
        'trading_halted' : False
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

# --- RISK MANAGEMENT FUNCTIONS ---
def check_drawdown(wallet):
    total_invested = sum(t['invested'] for t in wallet['open_trades'].values())
    total_value    = wallet['balance'] + total_invested

    # Update peak value
    if total_value > wallet['peak_value']:
        wallet['peak_value'] = total_value

    # Calculate drawdown from peak
    drawdown = (wallet['peak_value'] - total_value) / wallet['peak_value']

    if drawdown >= MAX_DRAWDOWN:
        wallet['trading_halted'] = True
        print(f"\n🚨 DRAWDOWN ALERT!! Portfolio dropped {drawdown*100:.1f}% from peak!!")
        print(f"🚨 Trading HALTED to protect capital!!")
        print(f"🚨 Peak: ₹{wallet['peak_value']:,.2f} | Now: ₹{total_value:,.2f}")
    else:
        wallet['trading_halted'] = False

    return drawdown

def check_correlation(asset, open_trades):
    for group in CORRELATED_GROUPS:
        if asset in group:
            for held_asset in open_trades:
                if held_asset in group and held_asset != asset:
                    return True, held_asset
    return False, None

def calculate_position_size(wallet, confidence, accuracy, sentiment, golden_cross):
    total_value = wallet['balance'] + sum(t['invested'] for t in wallet['open_trades'].values())

    # Base position size
    position_pct = BASE_POSITION_SIZE

    # Boost for very strong signals
    if golden_cross and confidence > 0.75 and accuracy > 0.60:
        position_pct = MAX_POSITION_SIZE
        reason       = "Golden Cross + High Confidence + High Accuracy"
    elif sentiment > 0.15 and confidence > 0.70 and accuracy > 0.60:
        position_pct = 0.25
        reason       = "Positive Sentiment + High Confidence"
    elif confidence > 0.80 and accuracy > 0.65:
        position_pct = 0.25
        reason       = "Very High Confidence + Accuracy"
    else:
        reason = "Standard position"

    # Risk check — never risk more than MAX_RISK_PER_TRADE of portfolio
    max_loss_amount  = total_value * MAX_RISK_PER_TRADE
    invest_amount    = total_value * position_pct
    potential_loss   = invest_amount * STOP_LOSS_PCT

    if potential_loss > max_loss_amount:
        invest_amount = max_loss_amount / STOP_LOSS_PCT
        position_pct  = invest_amount / total_value
        reason       += " (Risk-adjusted)"

    return invest_amount, position_pct, reason

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

    scaler    = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[features] = scaler.fit_transform(df[features])

    X, y = [], []
    data = df_scaled[features].values
    tgt  = df_scaled['target'].values
    for i in range(SEQUENCE_LENGTH, len(data)):
        X.append(data[i-SEQUENCE_LENGTH:i])
        y.append(tgt[i])
    X, y = np.array(X), np.array(y)

    if len(X) < 100:
        return 'SKIP', 0, 0, df['close'].iloc[-1]

    train_size = int(len(X) * 0.70)
    val_size   = int(len(X) * 0.15)
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val   = X[train_size:train_size+val_size]
    y_val   = y[train_size:train_size+val_size]
    X_test  = X[train_size+val_size:]
    y_test  = y[train_size+val_size:]

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
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=50, batch_size=32, callbacks=[early_stop], verbose=0)

    y_pred   = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    accuracy = accuracy_score(y_test, y_pred)
    price    = df['close'].iloc[-1]

    if accuracy < MIN_ACCURACY:
        return 'SKIP', 0, accuracy, price

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

    # Check drawdown first!!
    drawdown = check_drawdown(wallet)

    total_invested = sum(t['invested'] for t in wallet['open_trades'].values())
    total_value    = wallet['balance'] + total_invested
    total_return   = ((total_value - wallet['starting']) / wallet['starting']) * 100

    print(f"\n💰 Balance         : ₹{wallet['balance']:,.2f}")
    print(f"📈 Invested        : ₹{total_invested:,.2f}")
    print(f"💎 Total Value     : ₹{total_value:,.2f}")
    print(f"📊 Total Return    : {total_return:+.2f}%")
    print(f"📉 Drawdown        : {drawdown*100:.1f}%")
    print(f"🔢 Open Positions  : {len(wallet['open_trades'])}/{MAX_OPEN_TRADES}")
    print(f"✅ Wins            : {wallet['wins']}")
    print(f"❌ Losses          : {wallet['losses']}")
    print("=" * 40)

    # HALT CHECK — stop all new trades if drawdown too high
    if wallet['trading_halted']:
        print("\n🚨 TRADING HALTED — Max drawdown reached!!")
        print("🚨 Only monitoring existing trades — no new buys!!")

    for asset, (atype, ticker, name) in assets.items():
        print(f"\n🔍 {asset} — training LSTM...")

        try:
            if atype == 'crypto':
                df       = fetch_crypto(asset, days=2000)
                currency = '$'
            else:
                df       = fetch_stock(ticker, period="5y")
                currency = '₹'

            sentiment = get_sentiment_score(name, atype, ticker)
            print(f"  📰 Sentiment: {sentiment:+.3f}")

            df = add_features(df, sentiment)

            if len(df) < 100:
                print(f"  ⚠️ Not enough data — skipping")
                continue

            signal, confidence, accuracy, price = get_lstm_signal(df)

            latest         = df.iloc[-1]
            golden_cross   = latest['golden_cross']   == 1
            death_cross    = latest['death_cross']    == 1
            ma7_above_ma21 = latest['ma7_above_ma21'] == 1
            macd_up        = latest['macd_cross_up']  == 1

            if signal == 'SKIP':
                if golden_cross:
                    signal = 'BUY'; confidence = 0.62
                    print(f"  ⚡ Golden Cross override!!")
                elif death_cross:
                    signal = 'SELL'; confidence = 0.62
                    print(f"  ⚡ Death Cross override!!")
                elif ma7_above_ma21 and macd_up:
                    signal = 'BUY'; confidence = 0.60
                    print(f"  ⚡ Uptrend + MACD override!!")
                else:
                    print(f"  ⏭️ Skipping — accuracy {accuracy*100:.1f}% below threshold")
                    continue

            print(f"  🧠 Accuracy: {accuracy*100:.1f}% | Signal: {signal} | Confidence: {confidence*100:.1f}%")

            # --- CHECK EXITS ---
            if asset in wallet['open_trades']:
                trade      = wallet['open_trades'][asset]
                buy_price  = trade['buy_price']
                invested   = trade['invested']
                change_pct = (price - buy_price) / buy_price

                # Show current P/L
                pl_color = "+" if change_pct >= 0 else ""
                print(f"  📊 Open | Bought: {currency}{buy_price:,.2f} | Now: {currency}{price:,.2f} | P/L: {pl_color}{change_pct*100:.1f}%")
                print(f"  🎯 Target: {currency}{buy_price*(1+TARGET_PCT):,.2f} | Stop: {currency}{buy_price*(1-STOP_LOSS_PCT):,.2f}")

                exit_reason = None
                if sentiment < -0.3:
                    exit_reason = 'SENTIMENT CRASH'
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

                # RISK CHECK 1 — trading halted?
                if wallet['trading_halted']:
                    print(f"  🚨 No new buys — trading halted (max drawdown reached)")
                    continue

                # RISK CHECK 2 — max positions reached?
                if len(wallet['open_trades']) >= MAX_OPEN_TRADES:
                    print(f"  ⚠️ Max positions ({MAX_OPEN_TRADES}) reached — skipping")
                    continue

                # RISK CHECK 3 — sentiment too negative?
                if sentiment < -0.2:
                    print(f"  ⚠️ BUY blocked — sentiment too negative ({sentiment:+.3f})")
                    continue

                # RISK CHECK 4 — correlated asset already held?
                is_correlated, held = check_correlation(asset, wallet['open_trades'])
                if is_correlated:
                    print(f"  ⚠️ BUY blocked — already holding correlated asset {held}")
                    continue

                # Calculate position size with risk management
                invest_amount, position_pct, size_reason = calculate_position_size(
                    wallet, confidence, accuracy, sentiment, golden_cross
                )

                print(f"  💼 Position Size: {position_pct*100:.1f}% — {size_reason}")

                if invest_amount > 100 and invest_amount <= wallet['balance']:
                    wallet['balance'] -= invest_amount
                    wallet['open_trades'][asset] = {
                        'buy_price'  : price,
                        'invested'   : invest_amount,
                        'buy_time'   : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'sentiment'  : sentiment,
                        'accuracy'   : accuracy,
                        'confidence' : confidence
                    }
                    print(f"  🟢 BUY → {currency}{price:,.2f} | ₹{invest_amount:,.2f} ({position_pct*100:.1f}%) | Acc:{accuracy*100:.1f}%")
                    log_trade(asset, 'BUY', price, invest_amount, 0, f'LSTM Acc:{accuracy*100:.1f}% Conf:{confidence*100:.1f}%')
                elif invest_amount > wallet['balance']:
                    print(f"  ⚠️ Insufficient balance (need ₹{invest_amount:,.2f}, have ₹{wallet['balance']:,.2f})")

        except Exception as e:
            print(f"  ❌ Error: {str(e)[:80]}")
            continue

    # Update drawdown after all trades
    check_drawdown(wallet)

    # --- FINAL SUMMARY ---
    total_invested = sum(t['invested'] for t in wallet['open_trades'].values())
    total_value    = wallet['balance'] + total_invested
    total_return   = ((total_value - wallet['starting']) / wallet['starting']) * 100
    win_rate       = (wallet['wins'] / wallet['total_trades'] * 100) if wallet['total_trades'] > 0 else 0

    print("\n" + "=" * 40)
    print(f"💰 Free Balance    : ₹{wallet['balance']:,.2f}")
    print(f"📈 Invested        : ₹{total_invested:,.2f}")
    print(f"💎 Total Value     : ₹{total_value:,.2f}")
    print(f"📊 Total Return    : {total_return:+.2f}%")
    print(f"🎯 Win Rate        : {f'{win_rate:.1f}%' if wallet['total_trades'] > 0 else 'No closed trades yet'}")
    print(f"🔢 Open Positions  : {len(wallet['open_trades'])}")

    if wallet['open_trades']:
        print(f"\n📋 Open Positions:")
        for asset, trade in wallet['open_trades'].items():
            print(f"   {asset:15s} | Invested: ₹{trade['invested']:,.0f} | Acc: {trade.get('accuracy', 0)*100:.1f}%")

    save_wallet(wallet)
    print("\n✅ Wallet saved!!")

run_lstm_trader()