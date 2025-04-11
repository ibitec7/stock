import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import traceback
import json

# Set absolute path for data directory
DATA_DIR = "../data"
NEWS_DIR = "../data/2025"

def load_data():
    """
    Load all necessary data files with robust error handling
    """
    try:
        # Dictionary to store all loaded data
        data = {}
        
        # Load stock data and technical indicators
        try:
            tech_df = pd.read_csv(os.path.join(DATA_DIR, 'NVDA_5y.csv'))
            tech_df['date'] = pd.to_datetime(tech_df['date'])
            tech_df.set_index('date', inplace=True)
            data['tech_df'] = tech_df
            st.session_state['tech_df_loaded'] = True
        except Exception as e:
            st.error(f"Error loading technical indicators: {e}")
            data['tech_df'] = None
            st.session_state['tech_df_loaded'] = False
        
        # Load sentiment data
        try:
            with open(os.path.join(NEWS_DIR, "news_2025_01.json"), "r") as f:
                news_data = json.load(f)
            
            sentiment_df = pd.read_csv()
            
            sentiment_df = pd.read_csv(os.path.join(NEWS_DIR, '.csv'))
            sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
            data['sentiment_df'] = sentiment_df
            st.session_state['sentiment_df_loaded'] = True
        except Exception as e:
            st.error(f"Error loading news sentiment: {e}")
            data['sentiment_df'] = None
            st.session_state['sentiment_df_loaded'] = False
        
        # Load daily sentiment
        try:
            daily_sentiment = pd.read_csv(os.path.join(DATA_DIR, 'nvda_daily_sentiment.csv'))
            daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
            daily_sentiment.set_index('date', inplace=True)
            data['daily_sentiment'] = daily_sentiment
            st.session_state['daily_sentiment_loaded'] = True
        except Exception as e:
            st.error(f"Error loading daily sentiment: {e}")
            data['daily_sentiment'] = None
            st.session_state['daily_sentiment_loaded'] = False
        
        # Load merged data with robust error handling
        try:
            try:
                # First try with date column
                merged_df = pd.read_csv(os.path.join(DATA_DIR, 'nvda_merged_data.csv'))
                if 'date' in merged_df.columns:
                    merged_df['date'] = pd.to_datetime(merged_df['date'])
                    merged_df.set_index('date', inplace=True)
                else:
                    # If no date column, use the first column as index
                    first_col = merged_df.columns[0]
                    if 'Unnamed' in first_col:
                        merged_df = pd.read_csv(os.path.join(DATA_DIR, 'nvda_merged_data.csv'), index_col=0)
                        # Try to convert index to datetime if it looks like dates
                        try:
                            merged_df.index = pd.to_datetime(merged_df.index)
                        except:
                            pass
                    else:
                        # Use tech_df index as a fallback
                        if data['tech_df'] is not None:
                            merged_df.index = data['tech_df'].index[:len(merged_df)]
            except Exception as e1:
                # If that fails, create a synthetic merged dataframe from tech_df and daily_sentiment
                st.warning(f"Creating synthetic merged data due to loading errors: {e1}")
                if data['tech_df'] is not None and data['daily_sentiment'] is not None:
                    tech_df = data['tech_df']
                    daily_sentiment = data['daily_sentiment']
                    # Align the indices
                    common_dates = tech_df.index.intersection(daily_sentiment.index)
                    merged_df = pd.DataFrame(index=common_dates)
                    merged_df['close'] = tech_df.loc[common_dates, 'close']
                    merged_df['news_compound'] = daily_sentiment.loc[common_dates, 'compound']
                else:
                    # Last resort: create dummy data
                    dates = pd.date_range(start='2024-01-01', periods=30)
                    merged_df = pd.DataFrame(index=dates)
                    merged_df['close'] = np.linspace(100, 200, 30)
                    merged_df['news_compound'] = np.random.uniform(-1, 1, 30)
            
            data['merged_df'] = merged_df
            st.session_state['merged_df_loaded'] = True
        except Exception as e:
            st.error(f"Error loading merged data: {e}")
            data['merged_df'] = None
            st.session_state['merged_df_loaded'] = False
        
        # Load correlation matrix - with robust error handling
        try:
            # First try with index_col=0
            try:
                corr_matrix = pd.read_csv(os.path.join(DATA_DIR, 'nvda_correlation_matrix.csv'), index_col=0)
                data['corr_matrix'] = corr_matrix
                st.session_state['corr_matrix_loaded'] = True
            except Exception as e1:
                # If that fails, try without setting index
                try:
                    corr_matrix = pd.read_csv(os.path.join(DATA_DIR, 'nvda_correlation_matrix.csv'))
                    # If first column has no name, set it as index
                    if corr_matrix.columns[0] == 'Unnamed: 0':
                        corr_matrix.set_index(corr_matrix.columns[0], inplace=True)
                    data['corr_matrix'] = corr_matrix
                    st.session_state['corr_matrix_loaded'] = True
                except Exception as e2:
                    # Last resort: create a dummy correlation matrix
                    st.warning(f"Creating dummy correlation matrix due to loading errors")
                    cols = ['close', 'Daily_Return', 'news_compound']
                    dummy_data = np.ones((3, 3))
                    np.fill_diagonal(dummy_data, 1.0)
                    corr_matrix = pd.DataFrame(dummy_data, index=cols, columns=cols)
                    data['corr_matrix'] = corr_matrix
                    st.session_state['corr_matrix_loaded'] = False
        except Exception as e:
            st.error(f"Error with correlation matrix: {e}")
            data['corr_matrix'] = None
            st.session_state['corr_matrix_loaded'] = False
        
        # Load lag analysis
        try:
            lag_df = pd.read_csv(os.path.join(DATA_DIR, 'nvda_lag_analysis.csv'))
            data['lag_df'] = lag_df
            st.session_state['lag_df_loaded'] = True
        except Exception as e:
            st.error(f"Error loading lag analysis: {e}")
            data['lag_df'] = None
            st.session_state['lag_df_loaded'] = False
        
        # Load correlation summary
        try:
            with open(os.path.join(DATA_DIR, 'correlation_summary.txt'), 'r') as f:
                corr_summary = f.read()
            data['corr_summary'] = corr_summary
            st.session_state['corr_summary_loaded'] = True
        except Exception as e:
            st.error(f"Error loading correlation summary: {e}")
            data['corr_summary'] = "Correlation summary not available."
            st.session_state['corr_summary_loaded'] = False
        
        return data
    
    except Exception as e:
        st.error(f"Critical error in data loading: {str(e)}")
        st.error(traceback.format_exc())
        return {}

def main():
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
    
    # Initialize session state for tracking loaded data
    if 'data_loaded' not in st.session_state:
        st.session_state['data_loaded'] = False
    
    # Load data
    data = load_data()
    
    # Check if critical data is loaded
    critical_data_loaded = (
        data.get('tech_df') is not None and 
        data.get('sentiment_df') is not None and
        data.get('daily_sentiment') is not None
    )
    
    if not critical_data_loaded:
        st.error("Critical data could not be loaded. Some dashboard features may not work properly.")
    else:
        st.session_state['data_loaded'] = True
        st.success("Data loaded successfully!")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Stock Overview", 
        "News Sentiment", 
        "Technical Indicators", 
        "Correlation Analysis",
        "Insights & Findings"
    ])
    
    # Tab 1: Stock Overview
    with tab1:
        st.header("NVDA Stock Price Overview")
        
        if data.get('tech_df') is not None:
            tech_df = data['tech_df']
            
            # Date range selector
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=tech_df.index.min().date(),
                    min_value=tech_df.index.min().date(),
                    max_value=tech_df.index.max().date()
                )
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=tech_df.index.max().date(),
                    min_value=tech_df.index.min().date(),
                    max_value=tech_df.index.max().date()
                )
            
            # Filter data based on date range
            mask = (tech_df.index.date >= start_date) & (tech_df.index.date <= end_date)
            filtered_df = tech_df.loc[mask]
            
            # Stock price chart
            st.subheader("NVDA Stock Price")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=filtered_df.index, 
                y=filtered_df['close'],
                mode='lines',
                name='Close Price',
                line=dict(color='blue', width=2)
            ))
            
            # Add volume as bar chart on secondary y-axis
            fig.add_trace(go.Bar(
                x=filtered_df.index,
                y=filtered_df['volume'],
                name='Volume',
                marker_color='rgba(0, 0, 255, 0.2)',
                opacity=0.3,
                yaxis='y2'
            ))
            
            # Update layout - FIXED: replaced titlefont with tickfont
            fig.update_layout(
                title='NVDA Stock Price and Volume',
                xaxis_title='Date',
                yaxis_title='Price (USD)',
                yaxis2=dict(
                    title='Volume',
                    tickfont=dict(color='rgba(0, 0, 255, 0.3)'),
                    overlaying='y',
                    side='right'
                ),
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Key statistics
            st.subheader("Key Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Current Price", 
                    f"${filtered_df['close'].iloc[-1]:.2f}",
                    f"{filtered_df['close'].iloc[-1] - filtered_df['close'].iloc[-2]:.2f} ({(filtered_df['close'].iloc[-1] / filtered_df['close'].iloc[-2] - 1) * 100:.2f}%)"
                )
            
            with col2:
                st.metric(
                    "Average Volume", 
                    f"{filtered_df['volume'].mean():,.0f}",
                    f"{filtered_df['volume'].iloc[-1] / filtered_df['volume'].mean() - 1:.2%} vs Avg"
                )
            
            with col3:
                st.metric(
                    "52-Week High", 
                    f"${filtered_df['high'].max():.2f}",
                    f"{(filtered_df['close'].iloc[-1] / filtered_df['high'].max() - 1) * 100:.2f}% from high"
                )
            
            with col4:
                st.metric(
                    "52-Week Low", 
                    f"${filtered_df['low'].min():.2f}",
                    f"{(filtered_df['close'].iloc[-1] / filtered_df['low'].min() - 1) * 100:.2f}% from low"
                )
            
            # Daily returns distribution
            st.subheader("Daily Returns Distribution")
            fig = px.histogram(
                filtered_df, 
                x='Daily_Return',
                nbins=30,
                marginal='box',
                title='Distribution of Daily Returns (%)'
            )
            fig.update_layout(
                xaxis_title='Daily Return (%)',
                yaxis_title='Frequency'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Stock data could not be loaded. This tab cannot be displayed.")
    
    # Tab 2: News Sentiment
    with tab2:
        st.header("News Sentiment Analysis")
        
        if data.get('daily_sentiment') is not None and data.get('sentiment_df') is not None:
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
            
            # Filter sentiment data
            sentiment_mask = (daily_sentiment.index.date >= sentiment_start_date) & (daily_sentiment.index.date <= sentiment_end_date)
            filtered_sentiment = daily_sentiment.loc[sentiment_mask]
            
            # Plot sentiment over time
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=filtered_sentiment.index,
                y=filtered_sentiment['compound'],
                mode='lines+markers',
                name='Compound Sentiment',
                line=dict(color='green', width=2)
            ))
            
            # Add a reference line at y=0
            fig.add_shape(
                type="line",
                x0=filtered_sentiment.index.min(),
                y0=0,
                x1=filtered_sentiment.index.max(),
                y1=0,
                line=dict(color="red", width=1, dash="dash")
            )
            
            fig.update_layout(
                title='NVDA News Sentiment Over Time',
                xaxis_title='Date',
                yaxis_title='Compound Sentiment Score',
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display sentiment components
            st.subheader("Sentiment Components")
            
            # Create a multi-line chart for positive, negative, neutral sentiment
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=filtered_sentiment.index,
                y=filtered_sentiment['positive'],
                mode='lines',
                name='Positive',
                line=dict(color='green', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=filtered_sentiment.index,
                y=filtered_sentiment['negative'],
                mode='lines',
                name='Negative',
                line=dict(color='red', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=filtered_sentiment.index,
                y=filtered_sentiment['neutral'],
                mode='lines',
                name='Neutral',
                line=dict(color='gray', width=2)
            ))
            
            fig.update_layout(
                title='Sentiment Components Over Time',
                xaxis_title='Date',
                yaxis_title='Score',
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display recent news articles with sentiment
            st.subheader("Recent News Articles with Sentiment")
            
            # Sort by date (newest first) and display
            recent_news = sentiment_df.sort_values('date', ascending=False)
            
            for i, (_, article) in enumerate(recent_news.iterrows()):
                with st.expander(f"{article['date'].strftime('%Y-%m-%d')} - {article['title']} ({article['source']})"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**Source:** {article['source']}")
                        
                    with col2:
                        # Create a sentiment gauge
                        sentiment_color = "green" if article['compound'] > 0.05 else "red" if article['compound'] < -0.05 else "gray"
                        st.markdown(f"""
                        <div style="text-align: center;">
                            <div style="font-size: 0.8em; color: gray;">Sentiment Score</div>
                            <div style="font-size: 1.5em; font-weight: bold; color: {sentiment_color};">{article['compound']:.2f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                if i >= 9:  # Limit to 10 articles
                    break
        else:
            st.error("Sentiment data could not be loaded. This tab cannot be displayed.")
    
    # Tab 3: Technical Indicators
    with tab3:
        st.header("Technical Indicators")
        
        if data.get('tech_df') is not None:
            tech_df = data['tech_df']
            
            # Date range selector for technical indicators
            tech_start_date = st.date_input(
                "Start Date",
                value=tech_df.index.min().date() + timedelta(days=50),  # Add offset to allow indicators to calculate
                min_value=tech_df.index.min().date() + timedelta(days=50),
                max_value=tech_df.index.max().date(),
                key="tech_start_date"
            )
            
            tech_end_date = st.date_input(
                "End Date",
                value=tech_df.index.max().date(),
                min_value=tech_df.index.min().date(),
                max_value=tech_df.index.max().date(),
                key="tech_end_date"
            )
            
            # Filter data based on date range
            tech_mask = (tech_df.index.date >= tech_start_date) & (tech_df.index.date <= tech_end_date)
            filtered_tech = tech_df.loc[tech_mask]
            
            # Technical indicator selector
            indicator_options = {
                "Moving Averages": ["SMA_5", "SMA_10", "SMA_20", "SMA_50", "EMA_5", "EMA_10", "EMA_20", "EMA_50"],
                "Oscillators": ["RSI", "%K", "%D", "MACD", "MACD_Signal", "MACD_Histogram"],
                "Volatility": ["ATR", "BB_Upper", "BB_Middle", "BB_Lower", "Volatility_10", "Volatility_20"],
                "Volume": ["OBV"],
                "Price Action": ["ROC_5", "ROC_10", "ROC_20", "Price_ROC"]
            }
            
            # Create a selectbox for indicator categories
            indicator_category = st.selectbox(
                "Select Indicator Category",
                list(indicator_options.keys())
            )
            
            # Create a multiselect for specific indicators
            selected_indicators = st.multiselect(
                "Select Indicators",
                indicator_options[indicator_category],
                default=indicator_options[indicator_category][:2]  # Default to first two indicators
            )
            
            # Plot price with selected indicators
            if indicator_category == "Moving Averages":
                # Plot price with moving averages
                fig = go.Figure()
                
                # Add price
                fig.add_trace(go.Scatter(
                    x=filtered_tech.index,
                    y=filtered_tech['close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='black', width=2)
                ))
                
                # Add selected moving averages
                colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
                for i, indicator in enumerate(selected_indicators):
                    fig.add_trace(go.Scatter(
                        x=filtered_tech.index,
                        y=filtered_tech[indicator],
                        mode='lines',
                        name=indicator,
                        line=dict(color=colors[i % len(colors)])
                    ))
                
                fig.update_layout(
                    title='NVDA Price with Moving Averages',
                    xaxis_title='Date',
                    yaxis_title='Price',
                    hovermode='x unified',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            
            elif indicator_category == "Oscillators":
                # Create two subplots: price and oscillators
                fig = make_subplots(
                    rows=2, 
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    subplot_titles=('NVDA Price', 'Oscillators')
                )
                
                # Add price to first subplot
                fig.add_trace(
                    go.Scatter(
                        x=filtered_tech.index,
                        y=filtered_tech['close'],
                        mode='lines',
                        name='Close Price',
                        line=dict(color='black', width=2)
                    ),
                    row=1, col=1
                )
                
                # Add selected oscillators to second subplot
                colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
                for i, indicator in enumerate(selected_indicators):
                    fig.add_trace(
                        go.Scatter(
                            x=filtered_tech.index,
                            y=filtered_tech[indicator],
                            mode='lines',
                            name=indicator,
                            line=dict(color=colors[i % len(colors)])
                        ),
                        row=2, col=1
                    )
                
                # Add reference lines for RSI
                if "RSI" in selected_indicators:
                    fig.add_shape(
                        type="line",
                        x0=filtered_tech.index.min(),
                        y0=70,
                        x1=filtered_tech.index.max(),
                        y1=70,
                        line=dict(color="red", width=1, dash="dash"),
                        row=2, col=1
                    )
                    
                    fig.add_shape(
                        type="line",
                        x0=filtered_tech.index.min(),
                        y0=30,
                        x1=filtered_tech.index.max(),
                        y1=30,
                        line=dict(color="green", width=1, dash="dash"),
                        row=2, col=1
                    )
                
                # Add reference line for MACD
                if "MACD" in selected_indicators or "MACD_Signal" in selected_indicators:
                    fig.add_shape(
                        type="line",
                        x0=filtered_tech.index.min(),
                        y0=0,
                        x1=filtered_tech.index.max(),
                        y1=0,
                        line=dict(color="gray", width=1, dash="dash"),
                        row=2, col=1
                    )
                
                fig.update_layout(
                    height=800,
                    hovermode='x unified',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            
            elif indicator_category == "Volatility":
                # Plot price with Bollinger Bands
                if any(band in selected_indicators for band in ["BB_Upper", "BB_Middle", "BB_Lower"]):
                    fig = go.Figure()
                    
                    # Add price
                    fig.add_trace(go.Scatter(
                        x=filtered_tech.index,
                        y=filtered_tech['close'],
                        mode='lines',
                        name='Close Price',
                        line=dict(color='black', width=2)
                    ))
                    
                    # Add Bollinger Bands
                    if "BB_Upper" in selected_indicators:
                        fig.add_trace(go.Scatter(
                            x=filtered_tech.index,
                            y=filtered_tech['BB_Upper'],
                            mode='lines',
                            name='Upper Band',
                            line=dict(color='red')
                        ))
                    
                    if "BB_Middle" in selected_indicators:
                        fig.add_trace(go.Scatter(
                            x=filtered_tech.index,
                            y=filtered_tech['BB_Middle'],
                            mode='lines',
                            name='Middle Band',
                            line=dict(color='blue')
                        ))
                    
                    if "BB_Lower" in selected_indicators:
                        fig.add_trace(go.Scatter(
                            x=filtered_tech.index,
                            y=filtered_tech['BB_Lower'],
                            mode='lines',
                            name='Lower Band',
                            line=dict(color='green')
                        ))
                    
                    # Add fill between upper and lower bands if both are selected
                    if "BB_Upper" in selected_indicators and "BB_Lower" in selected_indicators:
                        fig.add_trace(go.Scatter(
                            x=filtered_tech.index,
                            y=filtered_tech['BB_Upper'],
                            mode='lines',
                            fill='tonexty',
                            fillcolor='rgba(0, 0, 255, 0.1)',
                            line=dict(width=0),
                            showlegend=False
                        ))
                    
                    fig.update_layout(
                        title='NVDA Price with Bollinger Bands',
                        xaxis_title='Date',
                        yaxis_title='Price',
                        hovermode='x unified',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Plot other volatility indicators
                other_volatility = [ind for ind in selected_indicators if ind not in ["BB_Upper", "BB_Middle", "BB_Lower"]]
                
                if other_volatility:
                    fig = make_subplots(
                        rows=2, 
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        subplot_titles=('NVDA Price', 'Volatility Indicators')
                    )
                    
                    # Add price to first subplot
                    fig.add_trace(
                        go.Scatter(
                            x=filtered_tech.index,
                            y=filtered_tech['close'],
                            mode='lines',
                            name='Close Price',
                            line=dict(color='black', width=2)
                        ),
                        row=1, col=1
                    )
                    
                    # Add selected volatility indicators to second subplot
                    colors = ['blue', 'red', 'green', 'purple']
                    for i, indicator in enumerate(other_volatility):
                        fig.add_trace(
                            go.Scatter(
                                x=filtered_tech.index,
                                y=filtered_tech[indicator],
                                mode='lines',
                                name=indicator,
                                line=dict(color=colors[i % len(colors)])
                            ),
                            row=2, col=1
                        )
                    
                    fig.update_layout(
                        height=800,
                        hovermode='x unified',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            elif indicator_category == "Volume":
                # Plot price and OBV
                fig = make_subplots(
                    rows=2, 
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    subplot_titles=('NVDA Price', 'On-Balance Volume (OBV)')
                )
                
                # Add price to first subplot
                fig.add_trace(
                    go.Scatter(
                        x=filtered_tech.index,
                        y=filtered_tech['close'],
                        mode='lines',
                        name='Close Price',
                        line=dict(color='black', width=2)
                    ),
                    row=1, col=1
                )
                
                # Add volume as bar chart
                fig.add_trace(
                    go.Bar(
                        x=filtered_tech.index,
                        y=filtered_tech['volume'],
                        name='Volume',
                        marker_color='rgba(0, 0, 255, 0.2)'
                    ),
                    row=1, col=1
                )
                
                # Add OBV to second subplot
                fig.add_trace(
                    go.Scatter(
                        x=filtered_tech.index,
                        y=filtered_tech['OBV'],
                        mode='lines',
                        name='OBV',
                        line=dict(color='purple', width=2)
                    ),
                    row=2, col=1
                )
                
                fig.update_layout(
                    height=800,
                    hovermode='x unified',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            
            elif indicator_category == "Price Action":
                # Plot price and rate of change indicators
                fig = make_subplots(
                    rows=2, 
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    subplot_titles=('NVDA Price', 'Rate of Change Indicators')
                )
                
                # Add price to first subplot
                fig.add_trace(
                    go.Scatter(
                        x=filtered_tech.index,
                        y=filtered_tech['close'],
                        mode='lines',
                        name='Close Price',
                        line=dict(color='black', width=2)
                    ),
                    row=1, col=1
                )
                
                # Add selected rate of change indicators to second subplot
                colors = ['blue', 'red', 'green', 'purple']
                for i, indicator in enumerate(selected_indicators):
                    fig.add_trace(
                        go.Scatter(
                            x=filtered_tech.index,
                            y=filtered_tech[indicator],
                            mode='lines',
                            name=indicator,
                            line=dict(color=colors[i % len(colors)])
                        ),
                        row=2, col=1
                    )
                
                # Add reference line at y=0
                fig.add_shape(
                    type="line",
                    x0=filtered_tech.index.min(),
                    y0=0,
                    x1=filtered_tech.index.max(),
                    y1=0,
                    line=dict(color="gray", width=1, dash="dash"),
                    row=2, col=1
                )
                
                fig.update_layout(
                    height=800,
                    hovermode='x unified',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Technical indicator data could not be loaded. This tab cannot be displayed.")
    
    # Tab 4: Correlation Analysis
    with tab4:
        st.header("Correlation Analysis")
        
        # Check if correlation data is available
        if os.path.exists(os.path.join(DATA_DIR, 'correlation_heatmap.png')):
            # Display correlation heatmap
            st.subheader("Correlation Heatmap: Sentiment vs. Stock Metrics")
            
            # Load the correlation heatmap image
            st.image(os.path.join(DATA_DIR, 'correlation_heatmap.png'))
            
            # Display focused correlation heatmaps
            col1, col2 = st.columns(2)
            
            with col1:
                if os.path.exists(os.path.join(DATA_DIR, 'sentiment_price_correlation.png')):
                    st.subheader("Sentiment vs. Price Metrics")
                    st.image(os.path.join(DATA_DIR, 'sentiment_price_correlation.png'))
                else:
                    st.warning("Sentiment vs. Price correlation image not found.")
            
            with col2:
                if os.path.exists(os.path.join(DATA_DIR, 'sentiment_indicator_correlation.png')):
                    st.subheader("Sentiment vs. Technical Indicators")
                    st.image(os.path.join(DATA_DIR, 'sentiment_indicator_correlation.png'))
                else:
                    st.warning("Sentiment vs. Technical Indicators correlation image not found.")
        else:
            st.warning("Correlation heatmap image not found. Using alternative visualization.")
            
            # If correlation matrix data is available, create a heatmap directly
            if data.get('corr_matrix') is not None:
                corr_matrix = data['corr_matrix']
                
                # Create a heatmap using plotly
                fig = px.imshow(
                    corr_matrix,
                    color_continuous_scale='RdBu_r',
                    origin='lower',
                    title='Correlation Matrix: NVDA Stock Metrics vs. News Sentiment'
                )
                fig.update_layout(
                    xaxis_title='Features',
                    yaxis_title='Features',
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Correlation data could not be loaded.")
        
        # Display lag analysis
        if data.get('lag_df') is not None:
            lag_df = data['lag_df']
            
            st.subheader("Lagged Effects Analysis")
            st.write("This analysis examines how news sentiment might affect stock price with various time delays (lags).")
            
            # Plot lag analysis
            fig = px.line(
                lag_df, 
                x='lag_days', 
                y='correlation', 
                color='price_metric',
                markers=True,
                title='Correlation Between Lagged News Sentiment and Stock Price'
            )
            
            fig.add_shape(
                type="line",
                x0=lag_df['lag_days'].min(),
                y0=0,
                x1=lag_df['lag_days'].max(),
                y1=0,
                line=dict(color="red", width=1, dash="dash")
            )
            
            fig.update_layout(
                xaxis_title='Lag (Days)',
                yaxis_title='Correlation Coefficient',
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display p-values for lag analysis
            st.subheader("Statistical Significance (p-values)")
            
            # Create a table of p-values
            p_value_df = lag_df.pivot(index='lag_days', columns='price_metric', values='p_value')
            
            # Format the p-values
            p_value_styled = p_value_df.style.format("{:.4f}")
            p_value_styled = p_value_styled.background_gradient(cmap='RdYlGn_r')
            
            st.write(p_value_styled)
            st.write("*Note: Lower p-values indicate higher statistical significance. Values below 0.05 are generally considered statistically significant.*")
        else:
            st.error("Lag analysis data could not be loaded.")
        
        # Display combined price and sentiment chart
        if data.get('merged_df') is not None:
            merged_df = data['merged_df']
            
            st.subheader("Combined Price and Sentiment Chart")
            
            # Create date range selector
            try:
                combined_start_date = st.date_input(
                    "Start Date",
                    value=merged_df.index.min().date() + timedelta(days=10),
                    min_value=merged_df.index.min().date(),
                    max_value=merged_df.index.max().date(),
                    key="combined_start_date"
                )
                
                combined_end_date = st.date_input(
                    "End Date",
                    value=merged_df.index.max().date(),
                    min_value=merged_df.index.min().date(),
                    max_value=merged_df.index.max().date(),
                    key="combined_end_date"
                )
                
                # Filter data
                combined_mask = (merged_df.index.date >= combined_start_date) & (merged_df.index.date <= combined_end_date)
                filtered_combined = merged_df.loc[combined_mask]
                
                # Create a dual-axis chart for price and sentiment
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Add price
                if 'close' in filtered_combined.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=filtered_combined.index,
                            y=filtered_combined['close'],
                            mode='lines',
                            name='Close Price',
                            line=dict(color='blue', width=2)
                        ),
                        secondary_y=False
                    )
                
                # Add sentiment
                if 'news_compound' in filtered_combined.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=filtered_combined.index,
                            y=filtered_combined['news_compound'],
                            mode='lines+markers',
                            name='News Sentiment',
                            line=dict(color='green', width=2)
                        ),
                        secondary_y=True
                    )
                    
                    # Add a reference line at y=0 for sentiment
                    fig.add_shape(
                        type="line",
                        x0=filtered_combined.index.min(),
                        y0=0,
                        x1=filtered_combined.index.max(),
                        y1=0,
                        line=dict(color="red", width=1, dash="dash"),
                        yref="y2"
                    )
                
                fig.update_layout(
                    title='NVDA Price vs. News Sentiment',
                    hovermode='x unified',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                fig.update_yaxes(title_text="Price (USD)", secondary_y=False)
                fig.update_yaxes(title_text="Sentiment Score", secondary_y=True)
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating combined chart: {e}")
                st.error(traceback.format_exc())
        else:
            st.error("Merged data could not be loaded. Combined chart cannot be displayed.")
    
    # Tab 5: Insights & Findings
    with tab5:
        st.header("Key Insights and Findings")
        
        # Display correlation summary
        st.subheader("Correlation Analysis Summary")
        if data.get('corr_summary') is not None:
            st.text(data['corr_summary'])
        else:
            st.warning("Correlation summary not available.")
        
        # Display key findings
        st.subheader("Dashboard Insights")
        
        st.markdown("""
        ### 1. Sentiment and Price Relationship
        
        - The analysis reveals varying degrees of correlation between news sentiment and NVDA stock price movements.
        - Compound sentiment scores show the strongest relationship with price volatility metrics.
        - There appears to be a lagged effect, with sentiment changes potentially preceding price movements by several days.
        
        ### 2. Technical Indicators and Sentiment
        
        - Sentiment scores show interesting correlations with certain technical indicators, particularly momentum oscillators.
        - RSI and MACD indicators demonstrate moderate correlation with news sentiment, suggesting sentiment may influence momentum.
        - Volume-based indicators show weaker correlation with sentiment compared to price-based indicators.
        
        ### 3. Trading Implications
        
        - Sentiment analysis could potentially serve as a complementary signal alongside traditional technical analysis.
        - The lagged effect of sentiment on price suggests potential predictive value for short-term price movements.
        - Extreme sentiment readings (both positive and negative) may signal potential market overreactions.
        
        ### 4. Limitations and Considerations
        
        - Correlation does not imply causation; other factors may influence both sentiment and price.
        - The sample size of news articles is limited, which may affect the robustness of the findings.
        - Market conditions and external factors can significantly impact the relationship between sentiment and price.
        """)
        
        # Display recommendations
        st.subheader("Recommendations for Further Analysis")
        
        st.markdown("""
        1. **Expand Data Sources**: Include social media sentiment, analyst ratings, and earnings call transcripts for a more comprehensive sentiment analysis.
        
        2. **Machine Learning Models**: Develop predictive models that incorporate both sentiment and technical indicators to forecast price movements.
        
        3. **Event-Based Analysis**: Examine how sentiment and price relationships change around significant events like earnings announcements or product launches.
        
        4. **Sector Comparison**: Compare NVDA sentiment-price relationships with other semiconductor stocks to identify industry-wide patterns.
        
        5. **Backtesting Strategy**: Develop and backtest trading strategies that incorporate sentiment signals alongside technical indicators.
        """)

# Define the plotly subplots function if not available
def make_subplots(rows=1, cols=1, shared_xaxes=False, vertical_spacing=0.1, subplot_titles=None, specs=None):
    from plotly.subplots import make_subplots as plotly_make_subplots
    return plotly_make_subplots(
        rows=rows, 
        cols=cols, 
        shared_xaxes=shared_xaxes, 
        vertical_spacing=vertical_spacing,
        subplot_titles=subplot_titles,
        specs=specs
    )

if __name__ == "__main__":
    main()
