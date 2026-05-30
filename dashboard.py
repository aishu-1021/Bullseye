import streamlit as st
import ccxt
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import json
import os
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="🎯 Bullseye Trading",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Bullseye — AI Trading Dashboard")
st.markdown("---")

# --- SETTINGS ---
WALLET_FILE      = 'ml_wallet.json'
TRADE_LOG_FILE   = 'ml_trade_log.csv'
ML_BUY_THRESHOLD = 0.60
STOP_LOSS_PCT    = 0.07
TARGET_PCT       = 0.15

# --- LOAD WALLET ---
def load_wallet():
    if os.path.exists(WALLET_FILE):
        with open(WALLET_FILE, 'r') as f:
            return json.load(f)
    return {
        'balance': 10000, 'starting': 10000,
        'open_trades': {}, 'total_trades': 0,
        'wins': 0, 'losses': 0
    }

# --- FETCH DATA ---
@st.cache_data(ttl=300)  # cache for 5 minutes
def fetch_crypto(symbol, days=500):
    binance = ccxt.binance()
    ohlcv   = binance.fetch_ohlcv(symbol, timeframe='1d', limit=days)
    df      = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

@st.cache_data(ttl=300)
def fetch_stock(ticker, period="2y"):
    df = yf.Ticker(ticker).history(period=period)
    df.columns = [c.lower() for c in df.columns]
    return df

# --- ADD FEATURES ---
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

    bb      = ta.bbands(close, length=20)
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

# --- ML SIGNAL ---
def get_ml_signal(df):
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

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    latest   = df[features].iloc[-1:]
    prob     = model.predict_proba(latest)[0]

    return prob[1], prob[0], accuracy

# --- CANDLESTICK CHART ---
def plot_chart(df, asset_name):
    fig = go.Figure()

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df.index[-60:],
        open=df['open'].tail(60),
        high=df['high'].tail(60),
        low=df['low'].tail(60),
        close=df['close'].tail(60),
        name='Price'
    ))

    # MA lines
    fig.add_trace(go.Scatter(
        x=df.index[-60:], y=df['ma7'].tail(60),
        line=dict(color='orange', width=1),
        name='MA7'
    ))
    fig.add_trace(go.Scatter(
        x=df.index[-60:], y=df['ma21'].tail(60),
        line=dict(color='blue', width=1),
        name='MA21'
    ))

    fig.update_layout(
        title=f"{asset_name} — Last 60 Days",
        xaxis_rangeslider_visible=False,
        height=400,
        template='plotly_dark'
    )
    return fig

# --- ASSETS ---
assets = {
    'BTC/USDT'   : ('crypto', None,       'Bitcoin'),
    'ETH/USDT'   : ('crypto', None,       'Ethereum'),
    'LTC/USDT'   : ('crypto', None,       'Litecoin'),
    'BNB/USDT'   : ('crypto', None,       'BNB'),
    'XRP/USDT'   : ('crypto', None,       'XRP'),
    'DOGE/USDT'  : ('crypto', None,       'Dogecoin'),
    'Infosys'    : ('stock',  'INFY.NS',  'Infosys'),
    'TCS'        : ('stock',  'TCS.NS',   'TCS'),
    'SBI'        : ('stock',  'SBIN.NS',  'SBI'),
    'Reliance'   : ('stock',  'RELIANCE.NS', 'Reliance'),
}

# =====================
# SECTION 1 — WALLET
# =====================
wallet = load_wallet()
total_invested = sum(t['invested'] for t in wallet['open_trades'].values())
total_value    = wallet['balance'] + total_invested
total_return   = ((total_value - wallet['starting']) / wallet['starting']) * 100
win_rate       = (wallet['wins'] / wallet['total_trades'] * 100) if wallet['total_trades'] > 0 else 0

st.subheader("💰 Wallet Overview")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Free Balance",   f"₹{wallet['balance']:,.0f}")
col2.metric("Invested",       f"₹{total_invested:,.0f}")
col3.metric("Total Value",    f"₹{total_value:,.0f}", f"{total_return:+.2f}%")
col4.metric("Total Trades",   wallet['total_trades'])
col5.metric("Win Rate",       f"{win_rate:.1f}%" if wallet['total_trades'] > 0 else "N/A")

st.markdown("---")

# =====================
# SECTION 2 — OPEN TRADES
# =====================
if wallet['open_trades']:
    st.subheader("📊 Open Trades")
    trade_data = []
    for asset, trade in wallet['open_trades'].items():
        trade_data.append({
            'Asset'     : asset,
            'Buy Price' : trade['buy_price'],
            'Invested'  : f"₹{trade['invested']:,.0f}",
            'Buy Time'  : trade['buy_time']
        })
    st.dataframe(pd.DataFrame(trade_data), use_container_width=True)
    st.markdown("---")

# =====================
# SECTION 3 — LIVE SIGNALS
# =====================
st.subheader("🧠 ML Signals — Live Market Analysis")
st.caption("⏳ Training ML models... this takes 2-3 minutes. Please wait!!")

signal_data = []
charts      = {}

for asset, (atype, ticker, name) in assets.items():
    with st.spinner(f"Analyzing {name}..."):
        try:
            if atype == 'crypto':
                df       = fetch_crypto(asset)
                currency = '$'
                price    = df['close'].iloc[-1]
            else:
                df       = fetch_stock(ticker)
                currency = '₹'
                price    = df['close'].iloc[-1]

            df = add_features(df)

            if len(df) < 100:
                continue

            up_prob, down_prob, accuracy = get_ml_signal(df)

            if accuracy < 0.52:
                signal     = "⏭️ SKIP"
                confidence = max(up_prob, down_prob)
            elif up_prob >= ML_BUY_THRESHOLD:
                signal     = "🟢 BUY"
                confidence = up_prob
            elif down_prob >= ML_BUY_THRESHOLD:
                signal     = "🔴 SELL"
                confidence = down_prob
            else:
                signal     = "⚪ HOLD"
                confidence = max(up_prob, down_prob)

            rsi = df['rsi'].iloc[-1]

            signal_data.append({
                'Asset'      : name,
                'Price'      : f"{currency}{price:,.2f}",
                'RSI'        : f"{rsi:.1f}",
                'Signal'     : signal,
                'Confidence' : f"{confidence*100:.1f}%",
                'Accuracy'   : f"{accuracy*100:.1f}%",
                'Up Prob'    : f"{up_prob*100:.1f}%",
                'Down Prob'  : f"{down_prob*100:.1f}%",
            })

            charts[name] = plot_chart(df, name)

        except Exception as e:
            signal_data.append({
                'Asset'      : name,
                'Price'      : 'Error',
                'RSI'        : '-',
                'Signal'     : '❌ Error',
                'Confidence' : '-',
                'Accuracy'   : '-',
                'Up Prob'    : '-',
                'Down Prob'  : '-',
            })

if signal_data:
    st.dataframe(pd.DataFrame(signal_data), use_container_width=True)

st.markdown("---")

# =====================
# SECTION 4 — CHARTS
# =====================
st.subheader("📈 Price Charts")
if charts:
    asset_names = list(charts.keys())
    cols        = st.columns(2)
    for i, name in enumerate(asset_names):
        with cols[i % 2]:
            st.plotly_chart(charts[name], use_container_width=True)

st.markdown("---")

# =====================
# SECTION 5 — TRADE LOG
# =====================
st.subheader("📝 Trade History")
if os.path.exists(TRADE_LOG_FILE):
    log_df = pd.read_csv(TRADE_LOG_FILE)
    if not log_df.empty:
        st.dataframe(log_df.sort_values('timestamp', ascending=False), use_container_width=True)
    else:
        st.info("No trades logged yet!!")
else:
    st.info("No trade log found yet!!")

st.markdown("---")
st.caption("🎯 Bullseye AI Trading Bot — Built by Aishu 🔥")