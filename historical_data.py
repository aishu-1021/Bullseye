import yfinance as yf
import ccxt
import pandas as pd

print("🎯 Bullseye - Historical Data Fetcher")
print("=" * 40)

# --- CRYPTO Historical Data (Bitcoin - Last 90 days) ---
print("\n📦 Bitcoin Historical Data (Last 90 days)")

binance = ccxt.binance()

btc_ohlcv = binance.fetch_ohlcv('BTC/USDT', timeframe='1d', limit=90)

btc_df = pd.DataFrame(btc_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'], unit='ms')
btc_df.set_index('timestamp', inplace=True)

print(btc_df.tail(5))

# --- INDIAN STOCK Historical Data (Infosys - Last 90 days) ---
print("\n📈 Infosys Historical Data (Last 90 days)")

infy = yf.Ticker("INFY.NS")
infy_df = infy.history(period="90d")
infy_df = infy_df[['Open', 'High', 'Low', 'Close', 'Volume']]

print(infy_df.tail(5))

print("\n" + "=" * 40)
print("✅ Historical data fetched successfully!")