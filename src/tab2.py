import streamlit as st
import polars as pl
import os
import json
from datetime import datetime


def tab_2(sentiment_data, indicators_data, CACHE_DIR):
    st.header("News Sentiment Analysis")

    with open(os.path.join(CACHE_DIR, "master_news.json"), "r") as f:
        data = json.loads(f)
    
    col1, col2, col3 = st.columns(3)

    dates = [datetime.strptime(date, "%Y-%m-%dT%H:%M:%S") for date in data["articles"].keys()]

    with col1:
        st.selectbox(
            label="Start Date",
            value=min(dates),
            min_value=min(dates),
            max_value=max(dates)
        )

    with col2:
        st.selectbox(
            label="End Date",
            value=max(dates),
            min_value=min(dates),
            max_values=max(dates)
        )

    with col3:
        st.selectbox(
            "Select Sentiment type",
            ("Headlines", "Articles"),
            value="Articles"
        )

    

    if sentiment_data.get('daily_sentiment') is not None and sentiment_data.get('sentiment_df') is not None:
        daily_sentiment = data['daily_sentiment']
        sentiment_df = data['sentiment_df']
        
        # Display sentiment over time
        st.subheader("News Sentiment Over Time")
        
        # Create a date filter for sentiment data
        sentiment_start_date = st.date_input(
            "Sentiment Start Date",
            value=daily_sentiment.index.min().date(),
            min_value=daily_sentiment.index.min().date(),
            max_value=daily_sentiment.index.max().date(),
            key="sentiment_start_date"
        )
        
        sentiment_end_date = st.date_input(
            "Sentiment End Date",
            value=daily_sentiment.index.max().date(),
            min_value=daily_sentiment.index.min().date(),
            max_value=daily_sentiment.index.max().date(),
            key="sentiment_end_date"
        )
