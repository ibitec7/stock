import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import json # Added for JSON handling
import glob # Added for finding JSON files
import traceback

# Use relative paths assuming the script is run from 'Code Folder for gemini'
DATA_DIR = "../data"
NEWS_DIR = "../data/2025" # Assuming news json files are in the root of 'Code Folder for gemini'

def load_data():
    """
    Load all necessary data files using relative paths and handle JSON news files.
    """
    data = {}

    # --- Load Stock Data and Technical Indicators ---
    tech_file_path = os.path.join(DATA_DIR, 'NVDA_10y.csv')
    try:
        tech_df = pd.read_csv(tech_file_path)
        # Ensure 'date' column exists and convert to datetime
        if 'date' in tech_df.columns:
             # Handle potential timezone info if present
            if tech_df['date'].str.contains('T', na=False).any():
                 tech_df['date'] = pd.to_datetime(tech_df['date'].str.split('T').str[0])
            else:
                 tech_df['date'] = pd.to_datetime(tech_df['date'])
            tech_df.set_index('date', inplace=True)
            data['tech_df'] = tech_df
            st.session_state['tech_df_loaded'] = True
            st.success(f"Successfully loaded technical indicators from {tech_file_path}")
        else:
            st.error(f"'date' column not found in {tech_file_path}")
            data['tech_df'] = None
            st.session_state['tech_df_loaded'] = False

    except FileNotFoundError:
        st.error(f"Error: Technical indicators file not found at {tech_file_path}. Make sure 'NVDA_10y.csv' is inside the '{DATA_DIR}' folder.")
        data['tech_df'] = None
        st.session_state['tech_df_loaded'] = False
    except Exception as e:
        st.error(f"Error loading technical indicators from {tech_file_path}: {e}")
        # print(traceback.format_exc()) # Uncomment for detailed debugging if needed
        data['tech_df'] = None
        st.session_state['tech_df_loaded'] = False


    # --- Load and Process News Sentiment Data from JSON ---
    all_headlines = []
    news_files = glob.glob(os.path.join(NEWS_DIR, 'news_*.json'))

    if not news_files:
        st.warning(f"No news JSON files found matching 'news_*.json' in '{NEWS_DIR}'. Sentiment analysis will be limited.")
        data['sentiment_df'] = None
        st.session_state['sentiment_df_loaded'] = False
        data['daily_sentiment'] = None
        st.session_state['daily_sentiment_loaded'] = False

    else:
        st.info(f"Found news files: {', '.join([os.path.basename(f) for f in news_files])}")
        for news_file in news_files:
            try:
                with open(news_file, 'r') as f:
                    news_data = json.load(f)
                    if 'articles' in news_data:
                        for article in news_data['articles']:
                            # Extract date - handle potential errors
                            published_date = None
                            if 'published_parsed' in article and article['published_parsed']:
                                try:
                                    # Assuming published_parsed is [year, month, day, ...]
                                    dt_tuple = tuple(article['published_parsed'][:6])
                                    published_date = datetime(*dt_tuple)
                                except (TypeError, ValueError):
                                     st.warning(f"Could not parse date from 'published_parsed' in {os.path.basename(news_file)} for article: {article.get('title')}")
                                     # Try 'published' field as fallback
                                     if 'published' in article:
                                         try:
                                             published_date = pd.to_datetime(article['published']).tz_localize(None).normalize()
                                         except:
                                             st.warning(f"Could not parse date from 'published' field either.")


                            if published_date and 'headline' in article and 'finbert_sentiment' in article:
                                sentiment = article['finbert_sentiment']
                                # Calculate a simple compound score: positive - negative
                                compound_score = sentiment.get('positive', 0) - sentiment.get('negative', 0)

                                all_headlines.append({
                                    'date': published_date.date(), # Store only date part
                                    'headline': article['headline'],
                                    'positive': sentiment.get('positive', 0),
                                    'neutral': sentiment.get('neutral', 0),
                                    'negative': sentiment.get('negative', 0),
                                    'compound_score': compound_score # Add compound score
                                })
                    else:
                         st.warning(f"'articles' key not found in {os.path.basename(news_file)}")

            except FileNotFoundError:
                 st.error(f"Error: News file not found at {news_file}")
            except json.JSONDecodeError:
                 st.error(f"Error decoding JSON from {news_file}. Please check the file format.")
            except Exception as e:
                 st.error(f"Error loading news data from {news_file}: {e}")
                 # print(traceback.format_exc()) # Uncomment for detailed debugging

        if all_headlines:
            sentiment_df = pd.DataFrame(all_headlines)
            sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
            data['sentiment_df'] = sentiment_df
            st.session_state['sentiment_df_loaded'] = True
            st.success(f"Successfully loaded and processed sentiment data from {len(news_files)} JSON files.")

            # --- Aggregate Sentiment Daily ---
            try:
                 # Group by date and calculate mean sentiment scores
                 daily_sentiment = sentiment_df.groupby(sentiment_df['date'].dt.date)[['positive', 'neutral', 'negative', 'compound_score']].mean()
                 # Also count headlines per day
                 daily_sentiment['headline_count'] = sentiment_df.groupby(sentiment_df['date'].dt.date)['headline'].count()
                 daily_sentiment.index = pd.to_datetime(daily_sentiment.index) # Convert index back to datetime
                 daily_sentiment.index.name = 'date' # Rename index to 'date'
                 data['daily_sentiment'] = daily_sentiment
                 st.session_state['daily_sentiment_loaded'] = True
                 st.success("Successfully aggregated daily sentiment.")
            except Exception as e:
                 st.error(f"Error aggregating daily sentiment: {e}")
                 # print(traceback.format_exc())
                 data['daily_sentiment'] = None
                 st.session_state['daily_sentiment_loaded'] = False

        else:
             st.warning("No headlines found in JSON files. Cannot calculate daily sentiment.")
             data['sentiment_df'] = None
             st.session_state['sentiment_df_loaded'] = False
             data['daily_sentiment'] = None
             st.session_state['daily_sentiment_loaded'] = False


    # --- Create Merged Data ---
    # Attempt to merge tech data with daily sentiment
    if data.get('tech_df') is not None and data.get('daily_sentiment') is not None:
        try:
            # Ensure both indices are DatetimeIndex without timezone for clean merge
            tech_index = data['tech_df'].index.normalize()
            sentiment_index = data['daily_sentiment'].index.normalize()

            # Reindex daily_sentiment to match tech_df's dates, filling missing sentiment dates with NaN
            aligned_sentiment = data['daily_sentiment'].reindex(tech_index)

            # Merge based on index
            merged_df = data['tech_df'].copy() # Start with tech_df
            # Add sentiment columns, matching by date index
            sentiment_cols_to_add = ['positive', 'neutral', 'negative', 'compound_score', 'headline_count']
            for col in sentiment_cols_to_add:
                 if col in aligned_sentiment.columns:
                       merged_df[col] = aligned_sentiment[col]
                 else:
                      merged_df[col] = np.nan # Add NaN if column missing in sentiment data


            # Fill NaN values in sentiment columns if needed (e.g., with 0 or forward fill)
            # Example: Fill with 0 for scores, maybe ffill for count or keep NaN
            for col in ['positive', 'neutral', 'negative', 'compound_score']:
                 if col in merged_df.columns:
                     merged_df[col].fillna(0, inplace=True)
            if 'headline_count' in merged_df.columns:
                 merged_df['headline_count'].fillna(0, inplace=True) # Or use ffill()

            data['merged_df'] = merged_df
            st.session_state['merged_df_loaded'] = True
            st.success("Successfully merged technical and daily sentiment data.")

        except Exception as e:
            st.error(f"Error merging technical and daily sentiment data: {e}")
            # print(traceback.format_exc())
            data['merged_df'] = data.get('tech_df') # Fallback to just tech_df
            st.session_state['merged_df_loaded'] = st.session_state.get('tech_df_loaded', False)
            if data['merged_df'] is not None:
                st.warning("Proceeding with only technical data due to merge error.")
            else:
                st.error("Cannot proceed without technical data.")


    elif data.get('tech_df') is not None:
         st.warning("Daily sentiment data not available or failed to load/process. Merged data will only contain technical indicators.")
         data['merged_df'] = data['tech_df'] # Use only tech data if sentiment is missing
         st.session_state['merged_df_loaded'] = True
    else:
         st.error("Neither technical nor sentiment data could be loaded. Cannot create merged data.")
         data['merged_df'] = None
         st.session_state['merged_df_loaded'] = False


    return data

# --- Plotting Functions ---

def plot_stock_price(df):
    """ Plot stock closing price over time. """
    if df is None or 'close' not in df.columns:
        st.warning("Stock price data ('close' column) not available to plot.")
        return
    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Close Price'))
        fig.update_layout(title='NVDA Stock Price Over Time', xaxis_title='Date', yaxis_title='Price (USD)')
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error plotting stock price: {e}")

def plot_technical_indicators(df):
    """ Plot selected technical indicators. """
    if df is None:
        st.warning("Technical indicator data not available to plot.")
        return

    st.subheader("Technical Indicators")

    sma = [col for col in df.columns if 'sma' in col.lower()]

    indicators_to_plot = {
        'SMA': f"{sma[0]}", # Using SMA 10 based on NVDA_10y.csv
        'MACD': ['macd', 'signal_line'],
        'RSI': 'rsi',
        'Bollinger Bands': ['close', 'bb_upper', 'bb_lower'],
        'Rate of Change (ROC)': 'roc'
    }

    available_indicators = {name: cols for name, cols in indicators_to_plot.items()
                            if isinstance(cols, list) and all(col in df.columns for col in cols) or
                            isinstance(cols, str) and cols in df.columns}

    if not available_indicators:
        st.warning("None of the selected technical indicators (SMA, MACD, RSI, Bollinger Bands, ROC) are available in the loaded data.")
        return

    selected_indicator = st.selectbox("Select Indicator to Plot", list(available_indicators.keys()))

    try:
        fig = go.Figure()
        indicator_cols = available_indicators[selected_indicator]

        if selected_indicator == 'SMA':
            fig.add_trace(go.Scatter(x=df.index, y=df[indicator_cols], mode='lines', name=selected_indicator))
            fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Close Price', opacity=0.5))
            fig.update_layout(title=f'NVDA {selected_indicator} and Close Price')
        elif selected_indicator == 'MACD':
            fig.add_trace(go.Scatter(x=df.index, y=df[indicator_cols[0]], mode='lines', name='MACD'))
            fig.add_trace(go.Scatter(x=df.index, y=df[indicator_cols[1]], mode='lines', name='Signal Line'))
            # Optionally plot MACD Histogram if available
            if 'macd_hist' in df.columns:
                 fig.add_trace(go.Bar(x=df.index, y=df['macd_hist'], name='MACD Histogram'))
            fig.update_layout(title=f'NVDA {selected_indicator}')
        elif selected_indicator == 'RSI':
            fig.add_trace(go.Scatter(x=df.index, y=df[indicator_cols], mode='lines', name=selected_indicator))
            fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
            fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
            fig.update_layout(title=f'NVDA {selected_indicator}', yaxis_range=[0,100])
        elif selected_indicator == 'Bollinger Bands':
            fig.add_trace(go.Scatter(x=df.index, y=df[indicator_cols[0]], mode='lines', name='Close Price', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=df.index, y=df[indicator_cols[1]], mode='lines', name='Upper Band', line=dict(color='rgba(255,0,0,0.5)')))
            fig.add_trace(go.Scatter(x=df.index, y=df[indicator_cols[2]], mode='lines', name='Lower Band', line=dict(color='rgba(0,255,0,0.5)'), fill='tonexty', fillcolor='rgba(0,100,80,0.2)'))
            fig.update_layout(title=f'NVDA {selected_indicator}')
        elif selected_indicator == 'Rate of Change (ROC)':
             fig.add_trace(go.Scatter(x=df.index, y=df[indicator_cols], mode='lines', name=selected_indicator))
             fig.add_hline(y=0, line_dash="dash", line_color="grey")
             fig.update_layout(title=f'NVDA {selected_indicator}')

        fig.update_layout(xaxis_title='Date', yaxis_title='Value')
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error plotting {selected_indicator}: {e}")


def plot_sentiment(df):
    """ Plot average daily compound sentiment score over time. """
    if df is None or 'compound_score' not in df.columns:
        st.warning("Daily sentiment score data ('compound_score') not available to plot.")
        return
    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['compound_score'], mode='lines', name='Avg. Daily Compound Sentiment'))
        fig.add_hline(y=0, line_dash="dash", line_color="grey")
        fig.update_layout(title='Average Daily News Sentiment Over Time', xaxis_title='Date', yaxis_title='Avg. Compound Score')
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error plotting sentiment: {e}")


def plot_daily_sentiment_distribution(df):
     """ Plot the distribution of daily sentiment scores. """
     if df is None or 'compound_score' not in df.columns:
         st.warning("Daily sentiment score data ('compound_score') not available for distribution plot.")
         return
     try:
        fig = px.histogram(df, x='compound_score', nbins=50, title='Distribution of Daily Compound Sentiment Scores')
        fig.update_layout(xaxis_title='Compound Sentiment Score', yaxis_title='Frequency')
        st.plotly_chart(fig, use_container_width=True)

        # Also show positive/neutral/negative distribution if available
        if all(col in df.columns for col in ['positive', 'neutral', 'negative']):
            avg_scores = df[['positive', 'neutral', 'negative']].mean()
            fig_pie = px.pie(values=avg_scores.values, names=avg_scores.index, title='Average Sentiment Category Proportion (Daily Avg.)',
                             color_discrete_map={'positive':'green', 'neutral':'grey', 'negative':'red'})
            st.plotly_chart(fig_pie, use_container_width=True)

     except Exception as e:
        st.error(f"Error plotting sentiment distribution: {e}")


def plot_correlation_heatmap(df):
    """ Plot correlation heatmap of numerical features. """
    if df is None or df.empty:
        st.warning("Merged data not available for correlation heatmap.")
        return
    try:
        # Select only numerical columns for correlation
        numerical_df = df.select_dtypes(include=np.number)
        # Drop columns with too many NaNs or constant values if they cause issues
        numerical_df = numerical_df.dropna(axis=1, how='all')
        numerical_df = numerical_df.loc[:, (numerical_df != numerical_df.iloc[0]).any()]

        if numerical_df.shape[1] < 2:
             st.warning("Not enough numerical data with variance to calculate correlations.")
             return

        corr = numerical_df.corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title('Feature Correlation Heatmap')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting correlation heatmap: {e}")
        # print(traceback.format_exc()) # Uncomment for debugging

def display_recent_news(df):
    """ Display recent news headlines and their sentiment. """
    if df is None or df.empty:
        st.warning("News sentiment data not available.")
        return
    try:
        st.subheader("Recent News Headlines")
        # Sort by date descending and take top N
        recent_news = df.sort_values(by='date', ascending=False).head(20)

        for _, row in recent_news.iterrows():
             # Determine sentiment label based on compound score
             if row['compound_score'] > 0.1:
                 sentiment_label = "Positive 😊"
                 color = "green"
             elif row['compound_score'] < -0.1:
                 sentiment_label = "Negative 😞"
                 color = "red"
             else:
                 sentiment_label = "Neutral 😐"
                 color = "grey"

             # Display headline with sentiment color coding (using markdown)
             st.markdown(f"**[{row['date'].strftime('%Y-%m-%d')}]** <span style='color:{color};'>({sentiment_label})</span>: {row['headline']}", unsafe_allow_html=True)
             # Optional: Show detailed scores
             # st.caption(f"  Pos: {row['positive']:.2f}, Neu: {row['neutral']:.2f}, Neg: {row['negative']:.2f}, Comp: {row['compound_score']:.2f}")

    except Exception as e:
        st.error(f"Error displaying recent news: {e}")

# --- Streamlit App Layout ---

st.set_page_config(layout="wide")
st.title("NVDA Stock Analysis Dashboard")

# Load data once and store in session state
if 'data' not in st.session_state:
    with st.spinner('Loading data... Please wait.'):
        st.session_state.data = load_data()

data = st.session_state.data
tech_df = data.get('tech_df')
sentiment_df = data.get('sentiment_df')
daily_sentiment = data.get('daily_sentiment')
merged_df = data.get('merged_df')

# --- Sidebar for Date Range Selection ---
st.sidebar.header("Date Range")
if tech_df is not None and not tech_df.empty:
    min_date = tech_df.index.min().to_pydatetime()
    max_date = tech_df.index.max().to_pydatetime()

    # Default to last 1 year if data is available
    default_start_date = max_date - timedelta(days=365) if (max_date - min_date).days > 365 else min_date

    start_date = st.sidebar.date_input("Start Date", default_start_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End Date", max_date, min_value=start_date, max_value=max_date)

    # Convert sidebar dates to datetime objects for filtering
    start_datetime = pd.to_datetime(start_date)
    end_datetime = pd.to_datetime(end_date)

    # Filter DataFrames based on selected date range
    filtered_tech = tech_df[start_datetime:end_datetime] if tech_df is not None else None
    filtered_sentiment = sentiment_df[(sentiment_df['date'] >= start_datetime) & (sentiment_df['date'] <= end_datetime)] if sentiment_df is not None else None
    filtered_daily_sentiment = daily_sentiment[start_datetime:end_datetime] if daily_sentiment is not None else None
    filtered_merged = merged_df[start_datetime:end_datetime] if merged_df is not None else None

else:
    st.sidebar.warning("Stock data not loaded. Date range selection disabled.")
    filtered_tech = None
    filtered_sentiment = sentiment_df # Show all if no tech data for range
    filtered_daily_sentiment = daily_sentiment
    filtered_merged = None
    start_datetime, end_datetime = None, None


# --- Main Dashboard Area ---

# Tabbed Interface
tab1, tab2, tab3, tab4 = st.tabs(["Stock & Technicals", "News Sentiment", "Correlations", "Raw Data"])

with tab1:
    st.header("Stock Price and Technical Analysis")
    plot_stock_price(filtered_tech)
    if filtered_tech is not None:
         plot_technical_indicators(filtered_tech)
    else:
         st.warning("Technical indicator data not loaded or available for the selected date range.")


with tab2:
    st.header("News Sentiment Analysis")
    plot_sentiment(filtered_daily_sentiment)
    plot_daily_sentiment_distribution(filtered_daily_sentiment)
    display_recent_news(filtered_sentiment) # Use the non-aggregated df for headlines

with tab3:
    st.header("Correlation Analysis")
    plot_correlation_heatmap(filtered_merged)

with tab4:
    st.header("Raw Data Viewer")
    st.write("Filtered Technical Data:")
    st.dataframe(filtered_tech)
    st.write("Filtered Daily Sentiment Data:")
    st.dataframe(filtered_daily_sentiment)
    st.write("Filtered Merged Data:")
    st.dataframe(filtered_merged)
    # Optionally display the raw headline sentiment data
    # st.write("Headline Sentiment Data (filtered):")
    # st.dataframe(filtered_sentiment)


# --- Footer / Info ---
st.markdown("---")
st.markdown("Dashboard created to visualize stock data, technical indicators, and news sentiment for NVDA.")
st.markdown(f"Data loaded for range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}" if 'min_date' in locals() else "Data loading issues encountered.")
if start_datetime and end_datetime:
     st.markdown(f"Displaying data for selected range: **{start_datetime.strftime('%Y-%m-%d')}** to **{end_datetime.strftime('%Y-%m-%d')}**")