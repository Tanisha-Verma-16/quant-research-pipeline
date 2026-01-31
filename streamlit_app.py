import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# Configuration
API_URL = "https://quant-research-api.onrender.com"  # Your Render URL
st.set_page_config(page_title="Quant Research Platform", layout="wide")

# Header
st.title("🚀 Quantitative Research Platform")
st.markdown("**Hybrid XGBoost-LLM System for Volatility Regime Prediction**")

# Sidebar
st.sidebar.header("Settings")

# Fetch tickers
@st.cache_data(ttl=3600)
def get_tickers():
    response = requests.get(f"{API_URL}/api/tickers")
    return response.json()

tickers_data = get_tickers()
all_tickers = [ticker for tickers in tickers_data.values() for ticker in tickers]

selected_ticker = st.sidebar.selectbox("Select Ticker", all_tickers)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎯 Prediction", "📈 Strategy", "🤖 AI Summary"])

# Tab 1: Market Data
with tab1:
    st.header(f"Market Data: {selected_ticker}")
    
    # Fetch market data
    response = requests.get(f"{API_URL}/api/market-data/{selected_ticker}?limit=100")
    market_data = pd.DataFrame(response.json())
    
    # Price chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=market_data['Date'], 
        y=market_data['Close'],
        mode='lines',
        name='Close Price'
    ))
    fig.update_layout(title=f"{selected_ticker} Price History", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Price", f"${market_data['Close'].iloc[-1]:.2f}")
    with col2:
        returns = market_data['Close'].pct_change().iloc[-1] * 100
        st.metric("Daily Return", f"{returns:.2f}%")
    with col3:
        vol = market_data['Close'].pct_change().std() * 100 * (252**0.5)
        st.metric("Annualized Vol", f"{vol:.2f}%")

# Tab 2: Regime Prediction
with tab2:
    st.header("Volatility Regime Prediction")
    
    response = requests.get(f"{API_URL}/api/predict-regime/{selected_ticker}")
    prediction = response.json()
    
    col1, col2 = st.columns(2)
    
    with col1:
        regime = prediction['predicted_regime']
        confidence = prediction['confidence'] * 100
        
        # Color-coded regime
        color = {"LOW_VOL": "green", "NORMAL_VOL": "orange", "HIGH_VOL": "red"}[regime]
        st.markdown(f"### Predicted Regime (21 days ahead)")
        st.markdown(f"<h2 style='color: {color};'>{regime}</h2>", unsafe_allow_html=True)
        st.metric("Confidence", f"{confidence:.1f}%")
    
    with col2:
        st.markdown("### Top Drivers")
        drivers = prediction.get('top_drivers', [])
        for i, driver in enumerate(drivers[:5], 1):
            st.write(f"{i}. `{driver}`")

# Tab 3: Portfolio Strategy
with tab3:
    st.header("Recommended Portfolio Strategy")
    
    response = requests.get(f"{API_URL}/api/strategy-recommendation/{selected_ticker}")
    strategy = response.json()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Risk Posture", strategy['risk_posture'])
        st.metric("Equity Exposure", f"{strategy['equity_exposure']}%")
    
    with col2:
        st.metric("Primary Strategy", strategy['primary_strategy'])
        st.metric("Rebalance Frequency", strategy['rebalance_frequency'])
    
    # Strategy details
    st.subheader("Strategy Details")
    st.write(strategy.get('rationale', 'N/A'))

# Tab 4: AI Summary
with tab4:
    st.header("Executive Summary (Powered by Gemini)")
    
    with st.spinner("Generating AI summary..."):
        response = requests.get(f"{API_URL}/api/executive-summary/{selected_ticker}")
        summary = response.json()
    
    st.markdown(summary['summary'])
    st.caption(f"Generated at: {summary['timestamp']}")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Built for Quantifying the Markets Hackathon")