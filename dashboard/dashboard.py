import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import yfinance as yf
from tab1 import tab_1
from tab2 import tab_2
from tab3 import tab_3
from tab4 import tab_4
from tab5 import tab_5
from dashboard_helpers import load_master_news, source_indicators
import logging


DATA_DIR = "../data"
PLOTS_DIR = "../plots"
LOGS_DIR = "../logs"
CACHE_DIR = "../cache"

if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

if os.path.exists(os.path.join(LOGS_DIR, 'dashboard.log')):
    os.remove(os.path.join(LOGS_DIR, 'dashboard.log'))

logging.basicConfig(filename=os.path.join(LOGS_DIR, 'dashboard.log'),
                    level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def clear_market_data():
    all_files = os.listdir(CACHE_DIR)
    files_to_remove = [file for file in all_files if file.endswith('.parquet') and file != 'master_news.parquet']
    for file in files_to_remove:
        os.remove(os.path.join(CACHE_DIR, file))
    logging.info("Market data cleared.")



def main(period="5y"):
    timeframe = "1mo"

    timeframes = {
        "1d": "Daily",
        "1wk": "Weekly",
        "1mo": "Monthly",
        "3mo": "Quarterly",
    }

    periods = {
        "ytd": "Year to Date",
        "1y": "1 Year",
        "2y": "2 Years",
        "5y": "5 Years",
        "10y": "10 Years",
        "max": "All time",
    }

    st.set_page_config(
        page_title="NVDA Stock Analysis with Sentiment",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("NVIDIA (NVDA) Stock Analysis with News Sentiment")

    st.markdown("""
    This dashboard analyzes the relationship between NVIDIA stock price movements, technical indicators, 
    and news sentiment. It explores whether sentiment from news articles correlates with stock price changes.
    """)

    st.button("Refresh Market Data", key="refresh_data", on_click=clear_market_data)
    
    # Initialize session state for tracking loaded data
    if 'data_loaded' not in st.session_state:
        st.session_state['data_loaded'] = False
    
    # Load data
    with st.spinner("Loading sentiment data..."):
        data_sentiment = load_master_news()
    
    with st.spinner("Loading indicators data..."):
        data_indicators = source_indicators(period, timeframe)
    
    with st.spinner("Loading company financials..."):
        data_financials = yf.Ticker("NVDA").financials

    critical_data_loaded = (
        data_sentiment is not None and
        data_indicators is not None and
        data_financials is not None
    )
    
    if data_sentiment is None:
        st.error("Failed to load sentiment data.")
    
    if data_indicators is None:
        st.error("Failed to load indicators data.")

    if data_financials is None:
        st.error("Failed to load financial data.")
    
    if critical_data_loaded:
        st.session_state['data_loaded'] = True
        st.success("Data loaded successfully!", icon="✅")
    else:
        st.warning("Some data could not be loaded. Please check the logs for more details.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Stock Overview", 
        "News Sentiment", 
        "Technical Indicators", 
        "Correlation Analysis",
        "Insights & Findings"
    ])

    with tab1:
        tab_1(timeframes=timeframes, CACHE_DIR=CACHE_DIR, periods=periods)

    with tab3:
        tab_3()

    with tab4:
        tab_4()

    with tab5:
        tab_5()
        
    with tab2:
        tab_2(CACHE_DIR=CACHE_DIR)

if __name__ == "__main__":

    main()