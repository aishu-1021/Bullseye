import yfinance as yf
import ccxt

print("🎯 Bullseye - Live Price Fetcher")
print("=" * 40)

# --- CRYPTO (Bitcoin) via Binance ---
print("\n📦 CRYPTO PRICES")
binance = ccxt.binance()

bitcoin = binance.fetch_ticker('BTC/USDT')
ethereum = binance.fetch_ticker('ETH/USDT')

print(f"Bitcoin  (BTC): ${bitcoin['last']:,.2f}")
print(f"Ethereum (ETH): ${ethereum['last']:,.2f}")

# --- INDIAN STOCKS via Yahoo Finance ---
print("\n📈 INDIAN STOCK PRICES")

reliance = yf.Ticker("RELIANCE.NS")
infosys = yf.Ticker("INFY.NS")
tcs = yf.Ticker("TCS.NS")

print(f"Reliance : ₹{reliance.fast_info['lastPrice']:,.2f}")
print(f"Infosys  : ₹{infosys.fast_info['lastPrice']:,.2f}")
print(f"TCS      : ₹{tcs.fast_info['lastPrice']:,.2f}")

print("\n" + "=" * 40)
print("✅ Prices fetched successfully!")