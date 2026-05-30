import ccxt
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("🎯 Bullseye - ML Model")
print("=" * 40)

# --- FETCH & PREPARE DATA ---
def fetch_crypto_data(symbol, days=500):
    binance = ccxt.binance()
    ohlcv   = binance.fetch_ohlcv(symbol, timeframe='1d', limit=days)
    df      = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def fetch_stock_data(ticker, period="2y"):
    df = yf.Ticker(ticker).history(period=period)
    df.columns = [c.lower() for c in df.columns]
    return df

# --- FEATURE ENGINEERING ---
def add_features(df):
    close = df['close']
    high  = df['high']
    low   = df['low']
    vol   = df['volume']

    # Moving averages
    df['ma7']   = ta.sma(close, length=7)
    df['ma21']  = ta.sma(close, length=21)
    df['ma50']  = ta.sma(close, length=50)

    # RSI
    df['rsi']   = ta.rsi(close, length=14)

    # MACD
    macd        = ta.macd(close)
    df['macd']  = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']
    df['macd_hist']   = macd['MACDh_12_26_9']

    # Bollinger Bands
    bb = ta.bbands(close, length=20)
    bb_cols = bb.columns.tolist()
    df['bb_upper'] = bb[bb_cols[2]]  # upper band
    df['bb_lower'] = bb[bb_cols[0]]  # lower band
    df['bb_mid'] = bb[bb_cols[1]]  # middle band
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']

    # Price vs MAs (% difference)
    df['price_vs_ma7']  = (close - df['ma7'])  / df['ma7']  * 100
    df['price_vs_ma21'] = (close - df['ma21']) / df['ma21'] * 100

    # Price momentum
    df['change_1d'] = close.pct_change(1)  * 100
    df['change_3d'] = close.pct_change(3)  * 100
    df['change_7d'] = close.pct_change(7)  * 100

    # Volume (with infinity protection!!)
    df['vol_change'] = vol.pct_change(1) * 100
    df['vol_vs_avg'] = vol / vol.rolling(20).mean()

    # Replace infinity values with NaN then fill
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.dropna(inplace=True)

    # Volatility
    df['atr'] = ta.atr(high, low, close, length=14)

    # --- TARGET ---
    # 1 = price higher in 7 days (BUY)
    # 0 = price lower in 7 days (SELL/HOLD)
    df['target'] = (close.shift(-7) > close).astype(int)

    df.dropna(inplace=True)
    return df

# --- TRAIN MODEL ---
def train_model(df, asset_name):
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

    # Train on 80% test on 20%
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # shuffle=False keeps time order!!
    )

    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred    = model.predict(X_test)
    accuracy  = accuracy_score(y_test, y_pred)

    print(f"\n{'='*40}")
    print(f"🧠 {asset_name} ML Model Results")
    print(f"{'='*40}")
    print(f"  Training samples : {len(X_train)}")
    print(f"  Testing samples  : {len(X_test)}")
    print(f"  Accuracy         : {accuracy*100:.1f}%")

    # Feature importance
    importance = pd.Series(
        model.feature_importances_,
        index=features
    ).sort_values(ascending=False)

    print(f"\n  📊 Top 5 Most Important Features:")
    for feat, imp in importance.head(5).items():
        print(f"    {feat:20s} : {imp*100:.1f}%")

    # Current prediction
    latest  = df[features].iloc[-1:]
    pred    = model.predict(latest)[0]
    prob    = model.predict_proba(latest)[0]

    print(f"\n  🎯 Current Prediction:")
    print(f"    Signal      : {'🟢 BUY' if pred == 1 else '🔴 SELL/HOLD'}")
    print(f"    Confidence  : {max(prob)*100:.1f}%")
    print(f"    Up prob     : {prob[1]*100:.1f}%")
    print(f"    Down prob   : {prob[0]*100:.1f}%")

    return model, accuracy

# --- RUN FOR ALL ASSETS ---
results = []

crypto_assets = {
    'Bitcoin'  : 'BTC/USDT',
    'Ethereum' : 'ETH/USDT',
    'Litecoin' : 'LTC/USDT',
    'BNB'      : 'BNB/USDT',
    'Dogecoin' : 'DOGE/USDT',
    'XRP'      : 'XRP/USDT',
    'Cardano'  : 'ADA/USDT',
    'Avalanche': 'AVAX/USDT',
    'Matic'    : 'MATIC/USDT',
}

print("\n📦 CRYPTO ML MODELS")
for name, symbol in crypto_assets.items():
    print(f"\n⏳ Training {name} model...")
    try:
        df    = fetch_crypto_data(symbol, days=500)
        df    = add_features(df)
        model, acc = train_model(df, name)
        results.append({'asset': name, 'accuracy': acc})
    except Exception as e:
        print(f"  ❌ Error: {e}")

stock_assets = {
    'Infosys'    : 'INFY.NS',
    'TCS'        : 'TCS.NS',
    'Wipro'      : 'WIPRO.NS',
    'Reliance'   : 'RELIANCE.NS',
    'HDFC Bank'  : 'HDFCBANK.NS',
    'Adani Ports': 'ADANIPORTS.NS',
    'SBI'        : 'SBIN.NS',
}

print("\n📈 STOCK ML MODELS")
for name, ticker in stock_assets.items():
    print(f"\n⏳ Training {name} model...")
    try:
        df    = fetch_stock_data(ticker, period="2y")
        df    = add_features(df)
        model, acc = train_model(df, name)
        results.append({'asset': name, 'accuracy': acc})
    except Exception as e:
        print(f"  ❌ Error: {e}")

# --- SUMMARY ---
print(f"\n{'='*40}")
print(f"🎯 ML MODELS SUMMARY")
print(f"{'='*40}")
for r in results:
    bar = "🟢" if r['accuracy'] >= 0.55 else "🔴"
    print(f"  {bar} {r['asset']:15s} : {r['accuracy']*100:.1f}%")

print(f"\n✅ ML Training complete!!")