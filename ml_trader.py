import ccxt
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import json
import csv
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("🎯 Bullseye - ML Trading Engine")
print("=" * 40)

# --- SETTINGS ---
STARTING_BALANCE  = 10000
TRADE_PERCENT     = 0.20
STOP_LOSS_PCT     = 0.07
TARGET_PCT        = 0.15
ML_BUY_THRESHOLD  = 0.60   # 60% confidence to buy
ML_SELL_THRESHOLD = 0.60   # 60% confidence to sell
WALLET_FILE       = 'ml_wallet.json'
TRADE_LOG_FILE    = 'ml_trade_log.csv'

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

# --- DATA FETCHING ---
def fetch_crypto(symbol, days=500):
    binance = ccxt.binance()
    ohlcv   = binance.fetch_ohlcv(symbol, timeframe='1d', limit=days)
    df      = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def fetch_stock(ticker, period="2y"):
    df = yf.Ticker(ticker).history(period=period)
    df.columns = [c.lower() for c in df.columns]
    return df

# --- FEATURES ---
def add_features(df):
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

    bb     = ta.bbands(close, length=20)
    bb_cols = bb.columns.tolist()
    df['bb_upper'] = bb[bb_cols[2]]
    df['bb_lower'] = bb[bb_cols[0]]
    df['bb_mid']   = bb[bb_cols[1]]
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']

    df['price_vs_ma7']  = (close - df['ma7'])  / df['ma7']  * 100
    df['price_vs_ma21'] = (close - df['ma21']) / df['ma21'] * 100
    df['change_1d'] = close.pct_change(1) * 100
    df['change_3d'] = close.pct_change(3) * 100
    df['change_7d'] = close.pct_change(7) * 100
    df['vol_change'] = vol.pct_change(1) * 100
    df['vol_vs_avg'] = vol / vol.rolling(20).mean()
    df['atr'] = ta.atr(high, low, close, length=14)

    df['target'] = (close.shift(-7) > close).astype(int)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    return df

# --- TRAIN + PREDICT ---
def get_ml_signal(df, asset_name, min_accuracy=0.52):
    features = [
        'rsi', 'ma7', 'ma21', 'ma50',
        'macd', 'macd_signal', 'macd_hist',
        'bb_upper', 'bb_lower', 'bb_width',
        'price_vs_ma7', 'price_vs_ma21',
        'change_1d', 'change_3d', 'change_7d',
        'vol_change', 'vol_vs_avg', 'atr'
    ]

    X = df[features]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, model.predict(X_test))

    # Skip low accuracy models
    if accuracy < min_accuracy:
        return 'SKIP', 0, accuracy, df['close'].iloc[-1]

    latest = df[features].iloc[-1:]
    prob   = model.predict_proba(latest)[0]
    up_prob   = prob[1]
    down_prob = prob[0]
    price     = df['close'].iloc[-1]

    if up_prob >= ML_BUY_THRESHOLD:
        return 'BUY', up_prob, accuracy, price
    elif down_prob >= ML_SELL_THRESHOLD:
        return 'SELL', down_prob, accuracy, price
    else:
        return 'HOLD', max(prob), accuracy, price

# --- MAIN TRADER ---
def run_ml_trader():
    wallet = load_wallet()

    assets = {
        'BTC/USDT'   : ('crypto', None),
        'ETH/USDT'   : ('crypto', None),
        'LTC/USDT'   : ('crypto', None),
        'BNB/USDT'   : ('crypto', None),
        'DOGE/USDT'  : ('crypto', None),
        'XRP/USDT'   : ('crypto', None),
        'ADA/USDT'   : ('crypto', None),
        'AVAX/USDT'  : ('crypto', None),
        'MATIC/USDT' : ('crypto', None),
        'Infosys'    : ('stock', 'INFY.NS'),
        'TCS'        : ('stock', 'TCS.NS'),
        'Wipro'      : ('stock', 'WIPRO.NS'),
        'Reliance'   : ('stock', 'RELIANCE.NS'),
        'HDFC Bank'  : ('stock', 'HDFCBANK.NS'),
        'Adani Ports': ('stock', 'ADANIPORTS.NS'),
        'SBI'        : ('stock', 'SBIN.NS'),
    }

    print(f"\n💰 Balance         : ₹{wallet['balance']:,.2f}")
    print(f"📊 Total Trades    : {wallet['total_trades']}")
    print(f"✅ Wins            : {wallet['wins']}")
    print(f"❌ Losses          : {wallet['losses']}")
    print("=" * 40)

    for asset, (atype, ticker) in assets.items():
        print(f"\n🔍 {asset} — training ML model...")

        try:
            if atype == 'crypto':
                df       = fetch_crypto(asset, days=500)
                currency = '$'
            else:
                df       = fetch_stock(ticker, period="2y")
                currency = '₹'

            df = add_features(df)

            if len(df) < 100:
                print(f"  ⚠️ Not enough data — skipping")
                continue

            signal, confidence, accuracy, price = get_ml_signal(df, asset)

            if signal == 'SKIP':
                print(f"  ⏭️ Skipping — model accuracy too low ({accuracy*100:.1f}%)")
                continue

            print(f"  🧠 Accuracy: {accuracy*100:.1f}% | Signal: {signal} | Confidence: {confidence*100:.1f}%")

            # --- CHECK EXITS ---
            if asset in wallet['open_trades']:
                trade      = wallet['open_trades'][asset]
                buy_price  = trade['buy_price']
                invested   = trade['invested']
                change_pct = (price - buy_price) / buy_price

                print(f"  📊 Open Trade | Bought: {currency}{buy_price:,.2f} | Now: {currency}{price:,.2f} | P/L: {change_pct*100:+.1f}%")

                exit_reason = None
                if change_pct <= -STOP_LOSS_PCT:
                    exit_reason = 'STOP LOSS'
                elif change_pct >= TARGET_PCT:
                    exit_reason = 'TARGET HIT'
                elif signal == 'SELL':
                    exit_reason = 'ML SELL SIGNAL'

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
                invest_amount = wallet['balance'] * TRADE_PERCENT
                if invest_amount > 100:
                    wallet['balance'] -= invest_amount
                    wallet['open_trades'][asset] = {
                        'buy_price': price,
                        'invested' : invest_amount,
                        'buy_time' : datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    print(f"  🟢 ML BUY → {currency}{price:,.2f} | Invested: ₹{invest_amount:,.2f} | Confidence: {confidence*100:.1f}%")
                    log_trade(asset, 'BUY', price, invest_amount, 0, f'ML BUY {confidence*100:.1f}%')

        except Exception as e:
            print(f"  ❌ Error: {str(e)[:60]}")
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
    print("\n✅ ML Wallet saved!!")

run_ml_trader()