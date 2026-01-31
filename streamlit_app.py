"""
Streamlit Dashboard for Quantitative Research Platform
Production-ready version with proper error handling
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Quantitative Research Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #0A192F; }
    h1, h2, h3 { color: #E6F1FF !important; }
    .stMetric { 
        background-color: #112240;
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #FF6B35;
    }
    .stButton>button {
        background-color: #FF6B35;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# API FUNCTIONS
# =============================================================================

@st.cache_data(ttl=300)
def fetch_api(endpoint, params=None):
    """Fetch data from API with caching"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Cannot connect to API at {API_BASE_URL}")
        st.info("💡 The backend may be waking up (cold start ~50s). Please wait and refresh.")
        return None
    except requests.exceptions.Timeout:
        st.warning("⏱️ Request timed out. The backend may be starting up. Please try again.")
        return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

def check_api_health():
    """Check if API is online"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        return response.status_code == 200
    except:
        try:
            response = requests.get(f"{API_BASE_URL}/", timeout=10)
            return response.status_code == 200
        except:
            return False

@st.cache_data(ttl=600)
def get_tickers():
    """Get available tickers"""
    data = fetch_api("/api/tickers")
    if data and "tickers" in data:
        return sorted(data["tickers"])
    return ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "SPY", "QQQ"]

def get_market_data(ticker, limit=100):
    """Get market data"""
    data = fetch_api(f"/api/market-data/{ticker}", params={"limit": limit})
    if data and "data" in data:
        return pd.DataFrame(data["data"])
    return None

def get_prediction(ticker):
    """Get regime prediction"""
    return fetch_api(f"/api/predict-regime/{ticker}")

def get_strategy(ticker):
    """Get strategy recommendation"""
    return fetch_api(f"/api/strategy-recommendation/{ticker}")

def get_risk_metrics():
    """Get risk metrics"""
    return fetch_api("/api/risk-metrics")

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_price_chart(df):
    """Create price chart"""
    if df is None or df.empty:
        return None
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Close'],
        mode='lines',
        name='Price',
        line=dict(color='#E6F1FF', width=2)
    ))
    
    fig.update_layout(
        title=f"{df['Ticker'].iloc[0]} Price Chart",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        template="plotly_dark",
        height=400
    )
    
    return fig

def create_regime_chart(prediction):
    """Create regime probability chart"""
    if not prediction or 'probabilities' not in prediction:
        return None
    
    probs = prediction['probabilities']
    labels = ['Low Vol', 'Normal Vol', 'High Vol']
    values = [
        probs.get('LOW_VOL', 0) * 100,
        probs.get('NORMAL_VOL', 0) * 100,
        probs.get('HIGH_VOL', 0) * 100
    ]
    colors = ['#64FFDA', '#FF8E53', '#FF6B35']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.6,
        marker=dict(colors=colors),
        textinfo='percent+label'
    )])
    
    fig.update_layout(
        title="Regime Probabilities",
        template="plotly_dark",
        height=300,
        showlegend=False
    )
    
    return fig

def get_regime_color(regime):
    """Get color for regime"""
    colors = {
        "HIGH_VOL": "#FF6B35",
        "NORMAL_VOL": "#FF8E53",
        "LOW_VOL": "#64FFDA"
    }
    return colors.get(regime, "#8892B0")

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Initialize session state
    if 'ticker' not in st.session_state:
        st.session_state.ticker = "AAPL"
    
    # Sidebar
    with st.sidebar:
        st.markdown("# 📈 Quantitative Research")
        st.markdown("---")
        
        # API Status
        api_online = check_api_health()
        status_icon = "🟢" if api_online else "🔴"
        status_text = "Connected" if api_online else "Disconnected"
        
        st.markdown(f"""
        <div style="padding: 15px; background-color: #112240; border-radius: 10px; 
                    border-left: 4px solid {'#64FFDA' if api_online else '#FF6B35'};">
            <div style="font-size: 24px; text-align: center;">{status_icon}</div>
            <div style="color: #E6F1FF; font-weight: bold; text-align: center;">API Status</div>
            <div style="color: #8892B0; text-align: center;">{status_text}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Ticker Selection
        tickers = get_tickers()
        selected_ticker = st.selectbox(
            "Select Ticker",
            options=tickers,
            index=tickers.index(st.session_state.ticker) if st.session_state.ticker in tickers else 0
        )
        
        if st.button("🔍 Analyze", use_container_width=True):
            st.session_state.ticker = selected_ticker
            st.rerun()
        
        st.markdown("---")
        
        # Quick Actions
        st.markdown("### ⚡ Quick Actions")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        # System Info
        st.markdown("---")
        st.markdown("### 📊 System Info")
        st.metric("Total Tickers", len(tickers))
        st.metric("Model Accuracy", "81.7%")
    
    # Main Content
    st.title("Quantitative Research Platform")
    st.markdown(f"### Analysis for {st.session_state.ticker}")
    st.markdown("---")
    
    # Check API
    if not api_online:
        st.warning("⚠️ Backend API is disconnected")
        st.info(f"Attempting to connect to: {API_BASE_URL}")
        st.info("💡 If using Render free tier, the backend may be waking up (~50 seconds)")
        
        if st.button("🔄 Retry Connection"):
            st.rerun()
        
        st.stop()
    
    # Fetch data
    ticker = st.session_state.ticker
    
    with st.spinner(f"Loading data for {ticker}..."):
        market_data = get_market_data(ticker, limit=100)
        prediction = get_prediction(ticker)
        strategy = get_strategy(ticker)
        risk_metrics = get_risk_metrics()
    
    # Top Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if prediction:
            regime = prediction.get('predicted_regime_21d', 'UNKNOWN')
            regime_color = get_regime_color(regime)
            st.markdown(f"""
            <div style="padding: 20px; background-color: #112240; border-radius: 10px; 
                        border-left: 4px solid {regime_color};">
                <h4 style="color: #8892B0; margin: 0;">Predicted Regime</h4>
                <h2 style="color: {regime_color}; margin: 10px 0;">{regime.replace('_', ' ')}</h2>
                <p style="color: #8892B0; margin: 0;">21-day forecast</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if prediction:
            confidence = prediction.get('confidence', 0) * 100
            st.metric("Model Confidence", f"{confidence:.1f}%")
    
    with col3:
        if strategy and 'recommendation' in strategy:
            exposure = strategy['recommendation'].get('equity_exposure', 0.6) * 100
            st.metric("Equity Exposure", f"{exposure:.0f}%")
    
    with col4:
        if risk_metrics:
            sharpe = risk_metrics.get('sharpe_ratio', 0)
            st.metric("Sharpe Ratio", f"{sharpe:.2f}")
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Market Data", 
        "🎯 Prediction", 
        "💼 Strategy",
        "📊 Risk Metrics"
    ])
    
    with tab1:
        st.markdown("### Price Chart")
        if market_data is not None and not market_data.empty:
            fig = create_price_chart(market_data)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Show data table
            st.markdown("### Recent Data")
            st.dataframe(
                market_data[['Date', 'Close', 'Volume']].head(10),
                use_container_width=True
            )
        else:
            st.info("No market data available")
    
    with tab2:
        st.markdown("### Regime Prediction")
        
        if prediction:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Show prediction details
                current_regime = prediction.get('current_regime', 'UNKNOWN')
                predicted_regime = prediction.get('predicted_regime_21d', 'UNKNOWN')
                confidence = prediction.get('confidence', 0)
                
                st.markdown(f"""
                **Current Regime:** {current_regime}  
                **Predicted Regime (21d):** {predicted_regime}  
                **Confidence:** {confidence*100:.1f}%
                """)
                
                # Show probabilities
                if 'probabilities' in prediction:
                    st.markdown("#### Probabilities")
                    probs = prediction['probabilities']
                    for regime, prob in probs.items():
                        st.progress(prob, text=f"{regime}: {prob*100:.1f}%")
            
            with col2:
                fig = create_regime_chart(prediction)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No prediction data available")
    
    with tab3:
        st.markdown("### Portfolio Strategy")
        
        if strategy and 'recommendation' in strategy:
            rec = strategy['recommendation']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Risk Posture", rec.get('risk_posture', 'N/A'))
                st.metric("Primary Strategy", rec.get('primary_strategy', 'N/A'))
            
            with col2:
                st.metric("Equity Exposure", f"{rec.get('equity_exposure', 0)*100:.0f}%")
                st.metric("Rebalancing", rec.get('rebalancing_frequency', 'N/A'))
            
            # Rationale
            st.markdown("### Strategy Rationale")
            st.info(f"""
            Based on the {rec.get('regime', 'NORMAL_VOL')} regime prediction, 
            we recommend a {rec.get('risk_posture', 'NEUTRAL').lower()} stance with 
            {rec.get('primary_strategy', 'BALANCED').replace('_', ' ').lower()} approach.
            """)
        else:
            st.info("No strategy recommendation available")
    
    with tab4:
        st.markdown("### Risk Metrics")
        
        if risk_metrics:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Sharpe Ratio", f"{risk_metrics.get('sharpe_ratio', 0):.2f}")
                st.metric("Calmar Ratio", f"{risk_metrics.get('calmar_ratio', 0):.2f}")
            
            with col2:
                st.metric("Max Drawdown", f"{risk_metrics.get('max_drawdown', 0):.1f}%")
                st.metric("VaR (95%)", f"{risk_metrics.get('var_95', 0):.1f}%")
            
            with col3:
                st.metric("CVaR (95%)", f"{risk_metrics.get('cvar_95', 0):.1f}%")
                st.metric("Prob Loss", f"{risk_metrics.get('prob_loss', 0):.1f}%")
        else:
            st.info("No risk metrics available")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #8892B0; padding: 20px;">
        <p>Powered by XGBoost, SHAP, and Gemini AI</p>
        <p>Data updated every 5 minutes | Model accuracy: 81.7%</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()