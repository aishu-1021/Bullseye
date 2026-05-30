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
    page_title="Bullseye Trading",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
:root {
    --bg-primary: #0a0a0a;
    --bg-card: #141414;
    --bg-hover: #1a1a1a;
    --border: #222222;
    --green: #00ff88;
    --green-dim: #00cc6a;
    --green-dark: #003d1f;
    --red: #ff3355;
    --red-dark: #3d0010;
    --yellow: #ffcc00;
    --text-primary: #e8e8e8;
    --text-secondary: #888888;
    --text-dim: #444444;
}
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}
.main .block-container { padding: 2rem 2.5rem !important; max-width: 1600px !important; }
#MainMenu, footer { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
.bull-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 0 2rem 0; border-bottom: 1px solid var(--border); margin-bottom: 2rem;
}
.bull-logo-text { font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem; font-weight: 700; color: var(--text-primary); }
.bull-logo-dot { color: var(--green); }
.bull-tagline { font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; color: var(--text-dim); letter-spacing: 0.15em; text-transform: uppercase; }
.bull-status { display: flex; align-items: center; gap: 0.5rem; font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: var(--green); }
.bull-status-dot { width: 6px; height: 6px; background: var(--green); border-radius: 50%; animation: pulse 2s infinite; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
.section-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.6rem; color: var(--text-dim);
    letter-spacing: 0.2em; text-transform: uppercase; margin-bottom: 1rem;
    padding-bottom: 0.5rem; border-bottom: 1px solid var(--border);
}
.metrics-row { display: grid; grid-template-columns: repeat(5, 1fr); gap: 1px; background: var(--border); border: 1px solid var(--border); margin-bottom: 2rem; }
.metric-card { background: var(--bg-card); padding: 1.25rem 1.5rem; }
.metric-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.6rem; color: var(--text-dim); letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 0.5rem; }
.metric-value { font-family: 'IBM Plex Mono', monospace; font-size: 1.4rem; font-weight: 600; color: var(--text-primary); line-height: 1; }
.metric-sub { font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; margin-top: 0.25rem; }
.pos { color: var(--green); } .neg { color: var(--red); } .neu { color: var(--text-secondary); }
.trade-card {
    background: var(--bg-card); border: 1px solid var(--border); border-left: 3px solid var(--green);
    padding: 1rem 1.25rem; margin-bottom: 0.5rem;
    display: grid; grid-template-columns: 2fr 1fr 1fr 1fr 1fr; align-items: center; gap: 1rem;
    font-family: 'IBM Plex Mono', monospace;
}
.trade-asset { font-size: 0.85rem; font-weight: 600; color: var(--text-primary); }
.trade-label { font-size: 0.55rem; color: var(--text-dim); letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 2px; }
.trade-val { font-size: 0.8rem; color: var(--text-secondary); }
</style>
""", unsafe_allow_html=True)

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
    return {'balance': 10000, 'starting': 10000, 'open_trades': {}, 'total_trades': 0, 'wins': 0, 'losses': 0}

# --- FETCH DATA ---
@st.cache_data(ttl=300)
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    prob     = model.predict_proba(df[features].iloc[-1:])[0]
    return prob[1], prob[0], accuracy

# --- CHART ---
def plot_chart(df, asset_name, currency):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index[-60:],
        open=df['open'].tail(60), high=df['high'].tail(60),
        low=df['low'].tail(60),   close=df['close'].tail(60),
        name='Price',
        increasing_line_color='#00ff88', decreasing_line_color='#ff3355',
        increasing_fillcolor='#003d1f',  decreasing_fillcolor='#3d0010',
    ))
    fig.add_trace(go.Scatter(x=df.index[-60:], y=df['ma7'].tail(60),
        line=dict(color='#ffcc00', width=1, dash='dot'), name='MA7', opacity=0.8))
    fig.add_trace(go.Scatter(x=df.index[-60:], y=df['ma21'].tail(60),
        line=dict(color='#0088ff', width=1), name='MA21', opacity=0.8))
    fig.update_layout(
        title=dict(text=asset_name, font=dict(family='IBM Plex Mono', size=12, color='#888888')),
        xaxis_rangeslider_visible=False, height=280,
        paper_bgcolor='#0a0a0a', plot_bgcolor='#0a0a0a',
        font=dict(family='IBM Plex Mono', color='#888888'),
        xaxis=dict(gridcolor='#1a1a1a', showgrid=True, zeroline=False),
        yaxis=dict(gridcolor='#1a1a1a', showgrid=True, zeroline=False, tickprefix=currency),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=9)),
        margin=dict(l=10, r=10, t=35, b=10),
    )
    return fig

# --- ASSETS ---
assets = {
    'BTC/USDT'  : ('crypto', None,         'Bitcoin',  '$'),
    'ETH/USDT'  : ('crypto', None,         'Ethereum', '$'),
    'LTC/USDT'  : ('crypto', None,         'Litecoin', '$'),
    'BNB/USDT'  : ('crypto', None,         'BNB',      '$'),
    'XRP/USDT'  : ('crypto', None,         'XRP',      '$'),
    'DOGE/USDT' : ('crypto', None,         'Dogecoin', '$'),
    'Infosys'   : ('stock',  'INFY.NS',    'Infosys',  '₹'),
    'TCS'       : ('stock',  'TCS.NS',     'TCS',      '₹'),
    'SBI'       : ('stock',  'SBIN.NS',    'SBI',      '₹'),
    'Reliance'  : ('stock',  'RELIANCE.NS','Reliance', '₹'),
}

# =====================
# HEADER
# =====================
st.markdown("""
<div class="bull-header">
    <div>
        <div class="bull-logo-text">BULL<span class="bull-logo-dot">.</span>SEYE</div>
        <div class="bull-tagline">AI-Powered Trading Intelligence</div>
    </div>
    <div class="bull-status">
        <div class="bull-status-dot"></div> LIVE
    </div>
</div>
""", unsafe_allow_html=True)

# =====================
# WALLET
# =====================
wallet         = load_wallet()
total_invested = sum(t['invested'] for t in wallet['open_trades'].values())
total_value    = wallet['balance'] + total_invested
total_return   = ((total_value - wallet['starting']) / wallet['starting']) * 100
win_rate       = (wallet['wins'] / wallet['total_trades'] * 100) if wallet['total_trades'] > 0 else 0
ret_class      = "pos" if total_return >= 0 else "neg"
ret_sign       = "+" if total_return >= 0 else ""

st.markdown('<div class="section-label">Portfolio Overview</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="metrics-row">
    <div class="metric-card">
        <div class="metric-label">Free Balance</div>
        <div class="metric-value">₹{wallet['balance']:,.0f}</div>
        <div class="metric-sub neu">Available</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Invested</div>
        <div class="metric-value">₹{total_invested:,.0f}</div>
        <div class="metric-sub neu">{len(wallet['open_trades'])} positions</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Total Value</div>
        <div class="metric-value">₹{total_value:,.0f}</div>
        <div class="metric-sub {ret_class}">{ret_sign}{total_return:.2f}%</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Total Trades</div>
        <div class="metric-value">{wallet['total_trades']}</div>
        <div class="metric-sub neu">{wallet['wins']}W / {wallet['losses']}L</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Win Rate</div>
        <div class="metric-value">{"N/A" if wallet['total_trades'] == 0 else f"{win_rate:.1f}%"}</div>
        <div class="metric-sub neu">{"No trades yet" if wallet['total_trades'] == 0 else "Recorded"}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# =====================
# OPEN TRADES
# =====================
if wallet['open_trades']:
    st.markdown('<div class="section-label">Open Positions</div>', unsafe_allow_html=True)
    for asset, trade in wallet['open_trades'].items():
        buy_price = trade['buy_price']
        invested  = trade['invested']
        stop_loss = buy_price * (1 - STOP_LOSS_PCT)
        target    = buy_price * (1 + TARGET_PCT)
        st.markdown(f"""
        <div class="trade-card">
            <div><div class="trade-asset">{asset}</div><div class="trade-label">Active Position</div></div>
            <div><div class="trade-label">Buy Price</div><div class="trade-val">{buy_price:,.2f}</div></div>
            <div><div class="trade-label">Invested</div><div class="trade-val">₹{invested:,.0f}</div></div>
            <div><div class="trade-label">Stop Loss</div><div class="trade-val" style="color:#ff3355">{stop_loss:,.2f}</div></div>
            <div><div class="trade-label">Target</div><div class="trade-val" style="color:#00ff88">{target:,.2f}</div></div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# =====================
# ML SIGNALS
# =====================
st.markdown('<div class="section-label">ML Signal Engine — Live Analysis</div>', unsafe_allow_html=True)

signal_rows = []
chart_data  = {}
progress    = st.progress(0, text="Initializing models...")
total_count = len(assets)

for i, (asset, (atype, ticker, name, currency)) in enumerate(assets.items()):
    progress.progress((i + 1) / total_count, text=f"Analyzing {name}...")
    try:
        df    = fetch_crypto(asset) if atype == 'crypto' else fetch_stock(ticker)
        df    = add_features(df)
        price = df['close'].iloc[-1]
        rsi   = df['rsi'].iloc[-1]
        if len(df) < 100:
            continue
        up_prob, down_prob, accuracy = get_ml_signal(df)
        if accuracy < 0.52:
            signal = "SKIP"
            confidence = max(up_prob, down_prob)
        elif up_prob >= ML_BUY_THRESHOLD:
            signal = "BUY"
            confidence = up_prob
        elif down_prob >= ML_BUY_THRESHOLD:
            signal = "SELL"
            confidence = down_prob
        else:
            signal = "HOLD"
            confidence = max(up_prob, down_prob)

        signal_icon = "🟢 BUY" if signal == "BUY" else "🔴 SELL" if signal == "SELL" else "🟡 HOLD" if signal == "HOLD" else "⏭ SKIP"
        signal_rows.append({
            'Asset'      : name,
            'Price'      : f"{currency}{price:,.2f}",
            'RSI'        : f"{rsi:.1f}",
            'Signal'     : signal_icon,
            'Confidence' : f"{confidence*100:.1f}%",
            'Model Acc.' : f"{accuracy*100:.1f}%",
            'Up Prob'    : f"{up_prob*100:.1f}%",
            'Down Prob'  : f"{down_prob*100:.1f}%",
        })
        chart_data[name] = plot_chart(df, name, currency)

    except Exception as e:
        signal_rows.append({
            'Asset': name, 'Price': 'ERR', 'RSI': '-',
            'Signal': '❌ ERROR', 'Confidence': '-',
            'Model Acc.': '-', 'Up Prob': '-', 'Down Prob': '-',
        })

progress.empty()

if signal_rows:
    st.dataframe(pd.DataFrame(signal_rows), use_container_width=True, hide_index=True)

# =====================
# CHARTS
# =====================
st.markdown('<div class="section-label">Price Charts — 60 Day View</div>', unsafe_allow_html=True)
chart_items = list(chart_data.items())
for i in range(0, len(chart_items), 2):
    cols = st.columns(2)
    for j, col in enumerate(cols):
        if i + j < len(chart_items):
            name, fig = chart_items[i + j]
            with col:
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# =====================
# TRADE LOG
# =====================
st.markdown('<div class="section-label">Trade History</div>', unsafe_allow_html=True)
if os.path.exists(TRADE_LOG_FILE):
    log_df = pd.read_csv(TRADE_LOG_FILE)
    if not log_df.empty:
        st.dataframe(log_df.sort_values('timestamp', ascending=False), use_container_width=True, hide_index=True)
    else:
        st.markdown('<p style="color:#444;font-family:IBM Plex Mono;font-size:0.8rem">No trades recorded yet.</p>', unsafe_allow_html=True)
else:
    st.markdown('<p style="color:#444;font-family:IBM Plex Mono;font-size:0.8rem">No trade log found.</p>', unsafe_allow_html=True)

st.markdown("""
<div style="margin-top:3rem;padding-top:1rem;border-top:1px solid #1a1a1a;
     font-family:IBM Plex Mono;font-size:0.6rem;color:#333;letter-spacing:0.1em;
     display:flex;justify-content:space-between;">
    <span>BULLSEYE TRADING INTELLIGENCE</span>
    <span>FOR PAPER TRADING PURPOSES ONLY</span>
    <span>BUILT BY AISHU</span>
</div>
""", unsafe_allow_html=True)