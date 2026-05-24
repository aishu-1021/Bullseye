import ccxt
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from datetime import datetime

print("🎯 Bullseye - Signal Generator")
print("=" * 40)
print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 40)

# --- SIGNAL FUNCTION ---
def generate_signal(df, price_col='close'):
    latest = df.iloc[-1]
    prev   = df.iloc[-2]

    rsi   = latest['RSI']
    ma7   = latest['MA7']
    ma21  = latest['MA21']
    close = latest[price_col]

    # BUY conditions
    buy = (
        rsi < 30 and
        close > ma7
    )

    # SELL conditions - smarter version
    sell = (
            rsi > 70 or
            (close < ma7 and rsi > 50)
    )

    if buy:
        return '🟢 BUY', rsi, close, ma7, ma21
    elif sell:
        return '🔴 SELL', rsi, close, ma7, ma21
    else:
        return '⚪ HOLD', rsi, close, ma7, ma21

# --- CRYPTO SIGNALS ---
print("\n📦 CRYPTO SIGNALS")
binance = ccxt.binance()

crypto_assets = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']

for asset in crypto_assets:
    ohlcv = binance.fetch_ohlcv(asset, timeframe='1d', limit=90)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df['MA7']  = ta.sma(df['close'], length=7)
    df['MA21'] = ta.sma(df['close'], length=21)
    df['RSI']  = ta.rsi(df['close'], length=14)
    df.dropna(inplace=True)

    signal, rsi, close, ma7, ma21 = generate_signal(df, price_col='close')

    print(f"\n{asset}")
    print(f"  Signal : {signal}")
    print(f"  Price  : ${close:,.2f}")
    print(f"  RSI    : {rsi:.2f}")
    print(f"  MA7    : ${ma7:,.2f}")
    print(f"  MA21   : ${ma21:,.2f}")

# --- STOCK SIGNALS ---
print("\n📈 INDIAN STOCK SIGNALS")

stock_assets = {
    'Infosys' : 'INFY.NS',
    'TCS'     : 'TCS.NS',
    'Wipro'   : 'WIPRO.NS',
    'Reliance': 'RELIANCE.NS'
}

for name, ticker in stock_assets.items():
    df = yf.Ticker(ticker).history(period="90d")
    df['MA7']  = ta.sma(df['Close'], length=7)
    df['MA21'] = ta.sma(df['Close'], length=21)
    df['RSI']  = ta.rsi(df['Close'], length=14)
    df.dropna(inplace=True)

    signal, rsi, close, ma7, ma21 = generate_signal(df, price_col='Close')

    print(f"\n{name}")
    print(f"  Signal : {signal}")
    print(f"  Price  : ₹{close:,.2f}")
    print(f"  RSI    : {rsi:.2f}")
    print(f"  MA7    : ₹{ma7:,.2f}")
    print(f"  MA21   : ₹{ma21:,.2f}")

print("\n" + "=" * 40)
print("✅ Signals generated successfully!")