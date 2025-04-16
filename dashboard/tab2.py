import streamlit as st
import polars as pl
import plotly.graph_objects as go
import os
import json
import time
from datetime import datetime, date
from nltk.tokenize import word_tokenize
from dashboard_helpers import get_filtered_articles_df, get_filtered_headlines_df, get_sentiments, load_master_news,source_indicators
from events import filter_events, get_corpus, create_wordcloud

def tab_2(CACHE_DIR):
    st.header("News Sentiment Analysis")

    period = "10y"

    data, ai_analysis = load_master_news()
    
    col1, col2, col3, col4, col5 = st.columns(5)

    with col3:
        sentiment_type = st.selectbox(
            "Select Sentiment type",
            ("Headlines", "Articles"),
            index=1
        )

    dates = data["date"].to_list()

    with col1:
        start_date = st.date_input(
            label="Start Date",
            value=min(dates),
            min_value=min(dates),
            max_value=max(dates)
        )

    with col2:
        end_date = st.date_input(
            label="End Date",
            value=max(dates),
            min_value=min(dates),
            max_value=max(dates)
        )

    with col4:
        time_frame2 = st.selectbox(
            "Select Time Frame",
            ("1d", "1wk","1mo", "3mo", "1y"),
            index=2
        )

        timeframes = {
            "1d": "1d",
            "1wk": "1w",
            "1mo": "1mo",
            "3mo": "1q",
            "1y": "1y",
        }

        time_frame = timeframes[time_frame2]

    with col5:
        overlay = st.selectbox(
            "Overlay",
            (None, "close","volume", "open", "high", "low"),
            index=0
        )


    filtered_df = data.filter(
        (pl.col("date") > start_date) & (pl.col("date") < end_date)
    )

    sentiments = get_sentiments(filtered_df, time_frame, sentiment_type)


    st.subheader("News sentiment with market")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name = "Positive Sentiment",
        x=sentiments["date"],
        y=sentiments["positive"],
        marker_color='green',
        opacity=0.7
    ))

    fig.add_trace(go.Bar(
        name = "Negative Sentiment",
        x=sentiments["date"],
        y=sentiments["negative"],
        marker_color='red',
        opacity=0.7
    ))

    if overlay is not None:
        indicators_df = source_indicators(period, time_frame2)

        filtered_indicators = indicators_df.filter(
            (pl.col("date") > start_date) & (pl.col("date") < end_date)
        )

        fig.add_trace(go.Scatter(
            x=filtered_indicators["date"],
            y=filtered_indicators[overlay],
            mode='lines',
            name=overlay,
            line=dict(color='white', width=2),
            yaxis='y2'
        ))

    fig.update_layout(
        barmode="group",
        title=f"{time_frame} News Sentiment Analysis",
        xaxis_title="Date",
        yaxis_title="Sentiment Count",
        hovermode="x unified",
        yaxis2 = dict(
            title=overlay,
            overlaying='y',
            side='right',
            showgrid=False,
            zeroline=False,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sentiment Wordcloud")

    col1, col2, col3, col4 = st.columns(4)

    with col3:
        sentiment = st.selectbox(
            "Select Sentiment",
            ("positive", "negative", "neutral", "all"),
            index=0
        )

        if sentiment != "all":
            wc_df = filtered_df.filter(
                pl.col("finbert_sentiment") == sentiment
            )
        else:
            wc_df = filtered_df

    with col4:
        impact = st.selectbox(
            "Select Impact",
            ("high", "medium", "low", "all"),
            index=0
        )

        impact_map = {
            "high": "high_event_tokens",
            "medium": "medium_event_tokens",
            "low": "low_event_tokens",
            "all": "tokens"
        }

    with col1:
        desired_start = datetime(2025, 1, 1)
        min_date = min(filtered_df["date"])
        max_date = max(filtered_df["date"])
        # Use desired date if within range, otherwise fallback to min date
        default_date = desired_start if min_date <= desired_start <= max_date else min_date
        
        start_date1 = st.date_input(
            label="Start Date",
            value=default_date,
            min_value=min_date,
            max_value=max_date,
        )

    with col2:
        end_date1 = st.date_input(
            label="End Date",
            value=max(filtered_df["date"]),
            min_value=min(filtered_df["date"]),
            max_value=max(filtered_df["date"]),
        )

    events_key = f"events_{start_date1}_{end_date1}_{sentiment}_{impact}"
    
    wc_df = wc_df.filter(
        (pl.col("date") > start_date1) & (pl.col("date") < end_date1)
    )


    time.sleep(3)

    # Check if we've already computed these events
    if events_key not in st.session_state and "RUNNING" not in st.session_state.values():
        st.session_state[events_key] = "RUNNING"
        with st.spinner("Finding event impacts..."):
            events = filter_events(start_date1, end_date1, wc_df, sentiment)
            st.session_state[events_key] = events

    if not events_key in st.session_state.keys():
        st.rerun()

    if st.session_state[events_key] == "RUNNING":
        if os.path.exists(os.path.join(CACHE_DIR, f"events_{start_date1}_{end_date1}_{sentiment}.json")):
            with open(os.path.join(CACHE_DIR, f"events_{start_date1}_{end_date1}_{sentiment}.json"), "r") as file:
                events = json.load(file)
            st.session_state[events_key] = events

        else:
            st.rerun()

    tokens_key = f"tokens_{start_date1}_{end_date1}_{sentiment}_{impact}"
    

    if tokens_key not in st.session_state:
        st.session_state[tokens_key] = "RUNNING"
        with st.spinner("Creating tokens..."):
            corpus = get_corpus(start_date1, end_date1, events, wc_df, sentiment)
            st.session_state[tokens_key] = corpus

            with st.spinner("Creating wordcloud..."):
                create_wordcloud(start_date1, end_date1, corpus, sentiment)
    
    elif st.session_state[tokens_key] == "RUNNING":
        if os.path.exists(os.path.join(CACHE_DIR, f"tokens_{start_date1}_{end_date1}_{sentiment}.json")):
            with open(os.path.join(CACHE_DIR, f"tokens_{start_date1}_{end_date1}_{sentiment}.json"), "r") as file:
                corpus = json.load(file)
            st.session_state[tokens_key] = corpus

            with st.spinner("Creating wordcloud..."):
                create_wordcloud(start_date1, end_date1, corpus, sentiment)

        else:
            st.rerun()

    if os.path.exists(os.path.join(CACHE_DIR, f"{impact_map[impact]}_{start_date1}_{end_date1}_{sentiment}.png")):
        st.image(image=os.path.join(CACHE_DIR, f"{impact_map[impact]}_{start_date1}_{end_date1}_{sentiment}.png"), caption="Wordcloud", use_container_width=True)  
    else:
        st.warning("Wordcloud not found. Please check the input parameters.") 
        st.write(f"{os.path.join(CACHE_DIR, f"{impact_map[impact]}_{start_date1}_{end_date1}_{sentiment}.png")}")


    st.subheader("Key Insights")

    col1, col2, col3, col4 = st.columns(4)

    # with col1:
    #     st.metric(
    #         "Average Positive Sentiment",
    #         f"{sentiments['avg_finbert_positive_sentiment'].mean() * 100:.2f}%",
    #         f"{(sentiments['avg_finbert_positive_sentiment'][-1] - sentiments["avg_finbert_positive_sentiment"].mean()) * 100:.2f}%",
    #     )

    # with col2:
    #     st.metric(
    #         "Average Negative Sentiment",
    #         f"{sentiments['avg_finbert_negative_sentiment'].mean() * 100:.2f}%",
    #         f"{sentiments['avg_finbert_negative_sentiment'][-1] * 100:.2f}%",
    #     )

    # with col3:
    #     st.metric(
    #         "Highest Positive Sentiment",
    #         f"{sentiments['max_finbert_positive_sentiment'].max() * 100:.2f}%",
    #         f"{sentiments['max_finbert_positive_sentiment'][-1] * 100:.2f}%",
    #     )

    # with col4:
    #     st.metric(
    #         "Highest Negative Sentiment",
    #         f"{sentiments['max_finbert_negative_sentiment'].max() * 100:.2f}%",
    #         f"{sentiments['max_finbert_negative_sentiment'][-1] * 100:.2f}%",
    #     )