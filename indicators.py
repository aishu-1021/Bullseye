import ccxt
import yfinance as yf
import pandas as pd
import pandas_ta as ta

print("🎯 Bullseye - Technical Indicators")
print("=" * 40)

# --- CRYPTO ASSETS ---
crypto_assets = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
binance = ccxt.binance()

print("\n📦 CRYPTO INDICATORS")
for asset in crypto_assets:
    ohlcv = binance.fetch_ohlcv(asset, timeframe='1d', limit=90)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    df['MA7']  = ta.sma(df['close'], length=7)
    df['MA21'] = ta.sma(df['close'], length=21)
    df['RSI']  = ta.rsi(df['close'], length=14)

    latest = df.iloc[-1]
    print(f"\n{asset}:")
    print(f"  Close : ${latest['close']:,.2f}")
    print(f"  MA7   : ${latest['MA7']:,.2f}")
    print(f"  MA21  : ${latest['MA21']:,.2f}")
    print(f"  RSI   : {latest['RSI']:.2f}")

# --- INDIAN STOCK ASSETS ---
stock_assets = {
    'Infosys' : 'INFY.NS',
    'TCS'     : 'TCS.NS',
    'Wipro'   : 'WIPRO.NS',
    'Reliance': 'RELIANCE.NS'
}

print("\n📈 INDIAN STOCK INDICATORS")
for name, ticker in stock_assets.items():
    df = yf.Ticker(ticker).history(period="90d")
    df['MA7']  = ta.sma(df['Close'], length=7)
    df['MA21'] = ta.sma(df['Close'], length=21)
    df['RSI']  = ta.rsi(df['Close'], length=14)

    latest = df.iloc[-1]
    print(f"\n{name}:")
    print(f"  Close : ₹{latest['Close']:,.2f}")
    print(f"  MA7   : ₹{latest['MA7']:,.2f}")
    print(f"  MA21  : ₹{latest['MA21']:,.2f}")
    print(f"  RSI   : {latest['RSI']:.2f}")

print("\n" + "=" * 40)
print("✅ All indicators calculated!")