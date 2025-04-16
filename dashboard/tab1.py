import streamlit as st
import polars as pl
import plotly.graph_objects as go
from dashboard_helpers import source_indicators

cache = "../cache"

def tab_1(timeframes, periods, CACHE_DIR):
    st.header("NVDIA Stock Price Overview")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col4:
        period = st.selectbox(
            "Select Period",
            ("ytd", "1y", "2y", "5y", "10y", "max"),
            index=3,
        )
    
    with col5:
        timeframe = st.selectbox(
            "Select Timeframe",
            ("1d", "1wk", "1mo", "3mo"),
            index=2,
        )

    data_indicators = source_indicators(period, timeframe= timeframe)

    with col1:
        start_date = st.date_input(
            "Start Date",
            value=data_indicators["date"].min(),
            min_value=data_indicators["date"].min(),
            max_value=data_indicators["date"].max()
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=data_indicators["date"].max(),
            min_value=data_indicators["date"].min(),
            max_value=data_indicators["date"].max()
        )

    with col3:
        type = st.selectbox(
            "Select Chart Type",
            ("Candlestick", "Line", "OHLC")
        )

    filtered_df = data_indicators.filter(
        (pl.col("date") >= start_date) & (pl.col("date") <= end_date)
    )
    
    # Stock price chart
    st.subheader("NVDA Stock Price")
    fig = go.Figure()

    if type == "Candlestick":
        fig.add_trace(go.Candlestick(
            x = filtered_df["date"],
            open = filtered_df["open"],
            high = filtered_df['high'],
            low = filtered_df['low'],
            close = filtered_df['close'],
            name = 'Candlestick',
            opacity=1
        ))

    elif type == "Line":
        fig.add_trace(go.Scatter(
            x=filtered_df["date"], 
            y=filtered_df['close'],
            mode='lines',
            name='Close Price',
            line=dict(color='orange', width=2)
        ))
    
    elif type == "OHLC":
        fig.add_trace(go.Ohlc(
            x=filtered_df["date"],
            open=filtered_df["open"],
            high=filtered_df['high'],
            low=filtered_df['low'],
            close=filtered_df['close'],
            name='OHLC',
            opacity=1
        ))
    
    # Add volume as bar chart on secondary y-axis
    fig.add_trace(go.Bar(
        x=filtered_df["date"],
        y=filtered_df['volume'],
        name='Volume',
        marker_color='rgba(255, 255, 255, 0.5)',
        opacity=0.5,
        yaxis='y2'
    ))
    
    # Update layout - FIXED: replaced titlefont with tickfont
    fig.update_layout(
        title='NVDA Stock Price and Volume',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        yaxis2=dict(
            title='Volume',
            tickfont=dict(color='rgba(255, 255, 255, 1)'),
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

    timeframe_name = timeframes[timeframe]
    period_name = periods[period]

    st.subheader(f"{timeframe_name} Returns")

    fig2 = go.Figure()

    colors = ["rgba(20, 232, 20, 0.7)" if ret > 0 else "rgba(232, 20, 20, 0.7)" for ret in filtered_df["returns"].to_list()]

    fig2.add_trace(go.Bar(
        x = filtered_df["date"],
        y = filtered_df["returns"],
        name = "Returns",
        marker_color = colors,
        opacity=0.8,
    ))

    fig2.update_layout(
        title=f"{timeframe_name} Returns with conditional coloring",
        xaxis_title = "Date",
        yaxis_title = "Returns (%)",
        hovermode = "x unified",
        legend = dict(
            orientation = "h",
            yanchor = "bottom",
            y = 1.02,
            xanchor = "right",
            x = 1
        ),
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Key Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
            st.metric(
                f"Current {timeframe_name} Close Price", 
                f"${filtered_df['close'][-1]:.2f}",
                f"{filtered_df['close'][-1] - filtered_df['close'][-2]:.2f} ({(filtered_df['close'][-1] / filtered_df['close'][-2] - 1) * 100:.2f}%)"
            )
        
    with col2:
        st.metric(
            f"{period_name} Average {timeframe_name} Volume", 
            f"{filtered_df['volume'].mean():,.0f}",
            f"{filtered_df['volume'][-1] / filtered_df['volume'].mean() - 1:.2%} vs Avg"
        )
    
    with col3:
        st.metric(
            f"{period_name} High", 
            f"${filtered_df['high'].max():.2f}",
            f"{(filtered_df['close'][-1] / filtered_df['high'].max() - 1) * 100:.2f}% from high"
        )
    
    with col4:
        st.metric(
            f"{period_name} Low", 
            f"${filtered_df['low'].min():.2f}",
            f"{(filtered_df['close'][-1] / filtered_df['low'].min() - 1) * 100:.2f}% from low"
        )