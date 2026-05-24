import ccxt
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import json
import csv
import os
from datetime import datetime

print("🎯 Bullseye - Paper Trading Engine")
print("=" * 40)

# --- SETTINGS ---
STARTING_BALANCE  = 10000   # ₹10,000 fake money
TRADE_PERCENT     = 0.20    # 20% of wallet per trade
STOP_LOSS_PCT     = 0.05    # 5% stop loss
TARGET_PCT        = 0.08    # 8% target
WALLET_FILE       = 'wallet.json'
TRADE_LOG_FILE    = 'trade_log.csv'

# --- LOAD OR CREATE WALLET ---
def load_wallet():
    if os.path.exists(WALLET_FILE):
        with open(WALLET_FILE, 'r') as f:
            return json.load(f)
    else:
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

# --- LOG A TRADE ---
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

# --- SIGNAL GENERATOR ---
def generate_signal(df, price_col='close'):
    latest = df.iloc[-1]
    rsi    = latest['RSI']
    ma7    = latest['MA7']
    close  = latest[price_col]

    buy  = (rsi < 30 and close > ma7)
    sell = (rsi > 70 or (close < ma7 and rsi > 50))

    if buy:
        return 'BUY', close, rsi
    elif sell:
        return 'SELL', close, rsi
    else:
        return 'HOLD', close, rsi

# --- FETCH DATA ---
def fetch_crypto(asset):
    binance  = ccxt.binance()
    ohlcv    = binance.fetch_ohlcv(asset, timeframe='1d', limit=90)
    df       = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df['MA7']  = ta.sma(df['close'], length=7)
    df['MA21'] = ta.sma(df['close'], length=21)
    df['RSI']  = ta.rsi(df['close'], length=14)
    df.dropna(inplace=True)
    return df

def fetch_stock(ticker):
    df = yf.Ticker(ticker).history(period="90d")
    df['MA7']  = ta.sma(df['Close'], length=7)
    df['MA21'] = ta.sma(df['Close'], length=21)
    df['RSI']  = ta.rsi(df['Close'], length=14)
    df.dropna(inplace=True)
    return df

# --- EXECUTE TRADE ---
def run_trader():
    wallet = load_wallet()

    assets = {
        # --- CRYPTO ---
        'BTC/USDT': ('crypto', None),
        'ETH/USDT': ('crypto', None),
        'SOL/USDT': ('crypto', None),
        'DOGE/USDT': ('crypto', None),
        'XRP/USDT': ('crypto', None),
        'ADA/USDT': ('crypto', None),
        'AVAX/USDT': ('crypto', None),
        'BNB/USDT': ('crypto', None),
        'MATIC/USDT': ('crypto', None),
        'LTC/USDT': ('crypto', None),

        # --- INDIAN STOCKS ---
        'Infosys': ('stock', 'INFY.NS'),
        'TCS': ('stock', 'TCS.NS'),
        'Wipro': ('stock', 'WIPRO.NS'),
        'Reliance': ('stock', 'RELIANCE.NS'),
        'HDFC Bank': ('stock', 'HDFCBANK.NS'),
        'Zomato': ('stock', 'ZOMATO.NS'),
        'Bajaj Auto': ('stock', 'BAJAJ-AUTO.NS'),
        'Tata Motors': ('stock', 'TATAMOTORS.NS'),
        'Adani Ports': ('stock', 'ADANIPORTS.NS'),
        'SBI': ('stock', 'SBIN.NS'),
    }

    print(f"\n💰 Current Balance : ₹{wallet['balance']:,.2f}")
    print(f"📊 Total Trades    : {wallet['total_trades']}")
    print(f"✅ Wins            : {wallet['wins']}")
    print(f"❌ Losses          : {wallet['losses']}")
    print("=" * 40)

    for asset, (atype, ticker) in assets.items():
        print(f"\n🔍 Checking {asset}...")

        # Fetch data
        if atype == 'crypto':
            df         = fetch_crypto(asset)
            price_col  = 'close'
            currency   = '$'
        else:
            df         = fetch_stock(ticker)
            price_col  = 'Close'
            currency   = '₹'

        signal, price, rsi = generate_signal(df, price_col)

        # --- CHECK OPEN TRADE EXITS FIRST ---
        if asset in wallet['open_trades']:
            trade      = wallet['open_trades'][asset]
            buy_price  = trade['buy_price']
            invested   = trade['invested']
            change_pct = (price - buy_price) / buy_price

            exit_reason = None

            if change_pct <= -STOP_LOSS_PCT:
                exit_reason = 'STOP LOSS'
            elif change_pct >= TARGET_PCT:
                exit_reason = 'TARGET HIT'
            elif signal == 'SELL':
                exit_reason = 'SELL SIGNAL'

            if exit_reason:
                # Calculate profit/loss
                current_value = invested * (1 + change_pct)
                profit_loss   = current_value - invested
                wallet['balance'] += current_value

                if profit_loss >= 0:
                    wallet['wins'] += 1
                    print(f"  🎉 {exit_reason} → Sold {asset} | Profit: ₹{profit_loss:,.2f} (+{change_pct*100:.1f}%)")
                else:
                    wallet['losses'] += 1
                    print(f"  😬 {exit_reason} → Sold {asset} | Loss: ₹{profit_loss:,.2f} ({change_pct*100:.1f}%)")

                log_trade(asset, 'SELL', price, current_value, profit_loss, exit_reason)
                del wallet['open_trades'][asset]
                wallet['total_trades'] += 1

        # --- CHECK FOR NEW BUY ---
        if signal == 'BUY' and asset not in wallet['open_trades']:
            invest_amount = wallet['balance'] * TRADE_PERCENT

            if invest_amount > 100:  # minimum ₹100 per trade
                wallet['balance'] -= invest_amount
                wallet['open_trades'][asset] = {
                    'buy_price' : price,
                    'invested'  : invest_amount,
                    'buy_time'  : datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                print(f"  🟢 BUY signal → Bought {asset} at {currency}{price:,.2f} | Invested: ₹{invest_amount:,.2f}")
                log_trade(asset, 'BUY', price, invest_amount, 0, 'BUY SIGNAL')

        elif signal == 'HOLD':
            if asset in wallet['open_trades']:
                trade      = wallet['open_trades'][asset]
                change_pct = (price - trade['buy_price']) / trade['buy_price']
                print(f"  ⚪ HOLD | Current P/L: {change_pct*100:.1f}%")
            else:
                print(f"  ⚪ HOLD | RSI: {rsi:.1f}")

    # --- SUMMARY ---
    total_invested = sum(t['invested'] for t in wallet['open_trades'].values())
    total_value    = wallet['balance'] + total_invested
    total_return   = ((total_value - wallet['starting']) / wallet['starting']) * 100

    print("\n" + "=" * 40)
    print(f"💰 Free Balance    : ₹{wallet['balance']:,.2f}")
    print(f"📈 Invested        : ₹{total_invested:,.2f}")
    print(f"💎 Total Value     : ₹{total_value:,.2f}")
    print(f"📊 Total Return    : {total_return:+.2f}%")
    print(f"✅ Win Rate        : ", end="")

    if wallet['total_trades'] > 0:
        win_rate = (wallet['wins'] / wallet['total_trades']) * 100
        print(f"{win_rate:.1f}%")
    else:
        print("No closed trades yet")

    save_wallet(wallet)
    print("\n✅ Wallet saved successfully!")

run_trader()