import ccxt
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from datetime import datetime

print("🎯 Bullseye - Backtester")
print("=" * 40)

# --- SETTINGS ---
STARTING_BALANCE = 10000
TRADE_PERCENT    = 0.20
STOP_LOSS_PCT    = 0.07
TARGET_PCT       = 0.15
RSI_BUY          = 35
RSI_SELL         = 70

# --- FETCH HISTORICAL DATA ---
def fetch_crypto_history(asset, days=365):
    binance = ccxt.binance()
    ohlcv   = binance.fetch_ohlcv(asset, timeframe='1d', limit=days)
    df      = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df['MA7']  = ta.sma(df['close'], length=7)
    df['MA21'] = ta.sma(df['close'], length=21)
    df['RSI']  = ta.rsi(df['close'], length=14)
    df.dropna(inplace=True)
    return df

def fetch_stock_history(ticker, period="1y"):
    df = yf.Ticker(ticker).history(period=period)
    df['MA7']  = ta.sma(df['Close'], length=7)
    df['MA21'] = ta.sma(df['Close'], length=21)
    df['RSI']  = ta.rsi(df['Close'], length=14)
    df.dropna(inplace=True)
    return df

# --- BACKTEST ENGINE ---
def backtest(df, asset_name, price_col='close', currency='$'):
    balance       = STARTING_BALANCE
    open_trade    = None
    trades        = []

    for i in range(1, len(df)):
        row   = df.iloc[i]
        price = row[price_col]
        rsi   = row['RSI']
        ma7   = row['MA7']

        # --- CHECK EXIT FIRST ---
        if open_trade:
            change_pct = (price - open_trade['buy_price']) / open_trade['buy_price']
            exit_reason = None

            if change_pct <= -STOP_LOSS_PCT:
                exit_reason = 'STOP LOSS'
            elif change_pct >= TARGET_PCT:
                exit_reason = 'TARGET HIT'
            elif rsi > RSI_SELL or (price < ma7 and rsi > 50):
                exit_reason = 'SELL SIGNAL'

            if exit_reason:
                current_value = open_trade['invested'] * (1 + change_pct)
                profit_loss   = current_value - open_trade['invested']
                balance      += current_value

                trades.append({
                    'asset'      : asset_name,
                    'buy_date'   : open_trade['buy_date'],
                    'sell_date'  : df.index[i].strftime('%Y-%m-%d'),
                    'buy_price'  : open_trade['buy_price'],
                    'sell_price' : price,
                    'invested'   : open_trade['invested'],
                    'profit_loss': profit_loss,
                    'change_pct' : change_pct * 100,
                    'result'     : 'WIN' if profit_loss > 0 else 'LOSS',
                    'reason'     : exit_reason
                })
                open_trade = None

        # --- CHECK BUY ---
        if not open_trade:
            buy = (rsi < RSI_BUY and price > ma7)
            if buy:
                invest_amount = balance * TRADE_PERCENT
                if invest_amount > 100:
                    balance -= invest_amount
                    open_trade = {
                        'buy_price' : price,
                        'invested'  : invest_amount,
                        'buy_date'  : df.index[i].strftime('%Y-%m-%d')
                    }

    # --- RESULTS ---
    if not trades:
        print(f"\n  ⚠️ No trades executed for {asset_name}")
        return

    trades_df  = pd.DataFrame(trades)
    wins       = len(trades_df[trades_df['result'] == 'WIN'])
    losses     = len(trades_df[trades_df['result'] == 'LOSS'])
    total      = len(trades_df)
    win_rate   = (wins / total) * 100
    total_pl   = trades_df['profit_loss'].sum()
    best_trade = trades_df.loc[trades_df['profit_loss'].idxmax()]
    worst_trade= trades_df.loc[trades_df['profit_loss'].idxmin()]
    final_val  = balance + (open_trade['invested'] if open_trade else 0)
    total_ret  = ((final_val - STARTING_BALANCE) / STARTING_BALANCE) * 100

    print(f"\n{'='*40}")
    print(f"📊 {asset_name} Backtest Results (1 Year)")
    print(f"{'='*40}")
    print(f"  Total Trades  : {total}")
    print(f"  Wins          : {wins} ✅")
    print(f"  Losses        : {losses} ❌")
    print(f"  Win Rate      : {win_rate:.1f}%")
    print(f"  Total P/L     : ₹{total_pl:+,.2f}")
    print(f"  Total Return  : {total_ret:+.2f}%")
    print(f"  Best Trade    : ₹{best_trade['profit_loss']:+,.2f} ({best_trade['change_pct']:+.1f}%) on {best_trade['asset']} [{best_trade['result']}]")
    print(f"  Worst Trade   : ₹{worst_trade['profit_loss']:+,.2f} ({worst_trade['change_pct']:+.1f}%) on {worst_trade['asset']} [{worst_trade['result']}]")

    return trades_df

# --- RUN BACKTESTS ---
all_trades = []

# Crypto
crypto_assets = {
    'Bitcoin'  : 'BTC/USDT',
    'Ethereum' : 'ETH/USDT',
    'Litecoin' : 'LTC/USDT',
    'Dogecoin' : 'DOGE/USDT',
    'XRP'      : 'XRP/USDT',
    'Cardano'  : 'ADA/USDT',
    'Avalanche': 'AVAX/USDT',
    'BNB'      : 'BNB/USDT',
    'Matic'    : 'MATIC/USDT',
}

print("\n📦 CRYPTO BACKTESTS")
for name, symbol in crypto_assets.items():
    print(f"\n⏳ Backtesting {name}...")
    try:
        df     = fetch_crypto_history(symbol, days=365)
        result = backtest(df, name, price_col='close', currency='$')
        if result is not None:
            all_trades.append(result)
    except Exception as e:
        print(f"  ❌ Error: {e}")

# Stocks
stock_assets = {
    'Infosys'    : 'INFY.NS',
    'TCS'        : 'TCS.NS',
    'Wipro'      : 'WIPRO.NS',
    'Reliance'   : 'RELIANCE.NS',
    'HDFC Bank'  : 'HDFCBANK.NS',
    'Adani Ports': 'ADANIPORTS.NS',
    'SBI'        : 'SBIN.NS',
}

print("\n📈 STOCK BACKTESTS")
for name, ticker in stock_assets.items():
    print(f"\n⏳ Backtesting {name}...")
    try:
        df     = fetch_stock_history(ticker, period="1y")
        result = backtest(df, name, price_col='Close', currency='₹')
        if result is not None:
            all_trades.append(result)
    except Exception as e:
        print(f"  ❌ Error: {e}")

# --- COMBINED SUMMARY ---
if all_trades:
    combined   = pd.concat(all_trades, ignore_index=True)
    total_wins = len(combined[combined['result'] == 'WIN'])
    total_loss = len(combined[combined['result'] == 'LOSS'])
    total      = len(combined)

    print(f"\n{'='*40}")
    print(f"🎯 OVERALL BACKTEST SUMMARY")
    print(f"{'='*40}")
    print(f"  Total Trades  : {total}")
    print(f"  Total Wins    : {total_wins} ✅")
    print(f"  Total Losses  : {total_loss} ❌")
    print(f"  Overall Win Rate : {(total_wins/total)*100:.1f}%")
    print(f"  Total P/L     : ₹{combined['profit_loss'].sum():+,.2f}")
    print(f"\n✅ Backtest complete!!")