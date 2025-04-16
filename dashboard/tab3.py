import streamlit as st
import polars as pl
import plotly.graph_objects as go
import os
import json
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from dashboard_helpers import source_indicators

CACHE_DIR = "../cache"

CONFIG_KEYS = {
    "SMA": "sma_window",
    "EMA": "ema_window",
    "Bollinger Bands": "bb_window",
    "MACD_LONG": "macd_long",
    "MACD_SHORT": "macd_short",
    "MACD_SIGNAL": "macd_signal",
    "RSI": "rsi_window",
    "ATR": "atr_window",
    "Stochastic": "stochastic_window",
    "ROC": "roc_window",
}

def set_config(indicators, config_type, indicators_config, period, timeframe):
    if config_type == "Custom" and indicators: 

        indicators = [indicator for indicator in indicators if indicator not in ["True Range", "OBV"]]

        if "MACD" in indicators:
            indicators.remove("MACD")
            indicators.extend(["MACD_LONG", "MACD_SHORT", "MACD_SIGNAL"])

        if indicators_config is None:
            indicators_config = dict()

        param_cols = st.columns(min(len(indicators), 6))
        
        for i, indicator in enumerate(indicators):
            with param_cols[i % len(param_cols)]:
                value = st.number_input(
                    f"{indicator} Parameters",
                    min_value=1,
                    max_value=200,
                    value=10,
                    key=f"{indicator}_period_input"
                )
                # Initialize the nested dictionary structure if needed
                if CONFIG_KEYS[indicator] not in indicators_config:
                    indicators_config[CONFIG_KEYS[indicator]] = {}
                if period not in indicators_config[CONFIG_KEYS[indicator]]:
                    indicators_config[CONFIG_KEYS[indicator]][period] = {}
                    
                # Now safely assign the value
                indicators_config[CONFIG_KEYS[indicator]][period][timeframe] = value

    else:
        indicators_config = None

    return indicators_config

def tab_3():
    st.header("NVDIA Stock Technical Analysis")

    col1, col2, col3, col4, col5 = st.columns(5)

    col21, col22, col23, col24, col25 = st.columns(5)

    with col4:
        period = st.selectbox(
            "Select Period",
            ("ytd", "1y", "2y", "5y", "10y", "max"),
            index=3,
            key="period_selectbox"
        )
    
    with col5:
        timeframe = st.selectbox(
            "Select Timeframe",
            ("1d", "1wk", "1mo", "3mo"),
            index=2,
            key="timeframe_selectbox"
        )

    data_indicators = source_indicators(period, timeframe)

    with col1:
        start_date = st.date_input(
            "Start Date",
            value=data_indicators["date"].min(),
            min_value=data_indicators["date"].min(),
            max_value=data_indicators["date"].max(),
            key="start_date_input"
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=data_indicators["date"].max(),
            min_value=data_indicators["date"].min(),
            max_value=data_indicators["date"].max(),
            key="end_date_input"
        )

    with col3:
        type = st.selectbox(
            "Select Chart Type",
            ("Candlestick", "Line", "OHLC"),
            key="chart_type_selectbox"
        )

    with col21:
        indicators = st.multiselect(
            "Select Technical Indicators",
            ["SMA", "EMA", "Bollinger Bands"],
            default=["SMA", "EMA"],
            key="indicators_selectbox"
        )

    with col24:
        overlay = st.selectbox(
            "Select Overlay",
            ("None", "Volume"),
            index=1,
            key="overlay_selectbox"
        )

    with col25:
        config_type = st.selectbox(
            "Configuration",
            ("Default", "Custom"),
            key="config_type_selectbox"
        )

    with col22:
        momentum_indicators_selected = st.multiselect(
            "Select Momentum Indicators",
            ["RSI", "MACD", "ROC", "Stochastic"],
            default=["RSI"],
            key="momentum_indicators_selectbox"
        )

    with col23:
        vv_indicators_selected = st.multiselect(
            "Select Volume and Volatility Indicators",
            ["OBV", "True Range", "ATR"],
            default=["ATR"],
            key="vv_indicators_selectbox"
        )

    indicators_config = None

    momentum_indicators = ["RSI", "MACD", "ROC", "Stochastic"]
    vv_indicators = ["OBV", "True Range", "ATR"]

    indicators_config = set_config(indicators, config_type, indicators_config, period, timeframe)
    indicators_config = set_config(momentum_indicators, config_type, indicators_config, period, timeframe)
    indicators_config = set_config(vv_indicators, config_type, indicators_config, period, timeframe)

    if indicators_config is not None:
        with open(os.path.join(CACHE_DIR, "indicators_config.json"), "w") as f:
            json.dump(indicators_config, f)

        data_indicators = source_indicators(period=period, timeframe=timeframe, indicators_config="indicators_config.json")

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
            line=dict(color="rgba(255,255,255,1)", width=2)
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

    for indicator in indicators:
        if indicator == "SMA":
            fig.add_trace(go.Scatter(
                x=filtered_df["date"],
                y=filtered_df['sma'],
                mode='lines',
                name='SMA',
                line=dict(color='blue', width=2)
            ))
        elif indicator == "EMA":
            fig.add_trace(go.Scatter(
                x=filtered_df["date"],
                y=filtered_df['ema'],
                mode='lines',
                name='EMA',
                line=dict(color='orange', width=2)
            ))


        elif indicator == "Bollinger Bands":
            fig.add_trace(go.Scatter(
                x=filtered_df["date"],
                y=filtered_df['bb_upper'],
                mode='lines',
                name='Bollinger Upper Band',
                line=dict(color='rgba(47, 251, 243, 0.8)', width=1),
                showlegend=True
            ))
            fig.add_trace(go.Scatter(
                x=filtered_df["date"],
                y=filtered_df['bb_lower'],
                mode='lines',
                name='Bollinger Lower Band',
                line=dict(color='rgba(47, 251, 243, 0.8)', width=1),
                fill='tonexty',
                fillcolor='rgba(47, 251, 243, 0.1)',
                showlegend=True
            ))
    
    # Add volume as bar chart on secondary y-axis
    if overlay == "Volume":
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
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        )
    )

    st.plotly_chart(fig, use_container_width=True, key="stock_price_chart")
    indicators = ["SMA", "EMA", "Bollinger Bands"]
    # Add Trend Indicators Analysis section
    if indicators:        
        # Create columns for metrics
        col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
        
        # Get current values (most recent data point)
        current_data = filtered_df.tail(1).row(0, named=True)
        prev_data = filtered_df.tail(2).head(1).row(0, named=True) if len(filtered_df) > 1 else None
        
        if "SMA" in indicators and "sma" in filtered_df.columns:
            # Calculate SMA metrics
            sma_current = current_data['sma']
            price_current = current_data['close']
            
            # Determine price relation to SMA
            price_vs_sma = price_current / sma_current - 1
            relation = "Above" if price_current > sma_current else "Below"
            direction = "Increasing" if prev_data and sma_current > prev_data['sma'] else "- Decreasing"
            
            # Display SMA metrics
            col_metrics1.metric(
                "SMA (Simple Moving Average)", 
                f"${sma_current:.2f}", 
                delta=direction
            )
            col_metrics1.markdown(f"**Price is {relation} SMA by {abs(price_vs_sma)*100:.2f}%**")
            
            if relation == "Above":
                col_metrics1.markdown("*Price above SMA suggests bullish trend*")
            else:
                col_metrics1.markdown("*Price below SMA suggests bearish trend*")
        
        if "EMA" in indicators and "ema" in filtered_df.columns:
            # Calculate EMA metrics
            ema_current = current_data['ema']
            price_current = current_data['close']
            
            # Determine price relation to EMA
            price_vs_ema = price_current / ema_current - 1
            relation = "Above" if price_current > ema_current else "Below"
            direction = "Increasing" if prev_data and ema_current > prev_data['ema'] else "- Decreasing"
            
            # Display EMA metrics
            col_metrics2.metric(
                "EMA (Exponential Moving Average)", 
                f"${ema_current:.2f}", 
                delta=direction
            )
            col_metrics2.markdown(f"**Price is {relation} EMA by {abs(price_vs_ema)*100:.2f}%**")
            
            if "sma" in filtered_df.columns and "SMA" in indicators:
                sma_ema_relation = "EMA > SMA" if ema_current > sma_current else "EMA < SMA"
                col_metrics2.markdown(f"*{sma_ema_relation}: {'Strengthening' if ema_current > sma_current else 'Weakening'} trend*")
            else:
                if relation == "Above":
                    col_metrics2.markdown("*Price above EMA indicates upward momentum*")
                else:
                    col_metrics2.markdown("*Price below EMA indicates downward momentum*")
        
        if "Bollinger Bands" in indicators and all(col in filtered_df.columns for col in ['bb_upper', 'bb_lower']):
            # Calculate Bollinger Bands metrics
            bb_upper = current_data['bb_upper']
            bb_lower = current_data['bb_lower']
            price_current = current_data['close']
            
            # Calculate width and position
            band_width = (bb_upper - bb_lower) / sma_current if "sma" in current_data else (bb_upper - bb_lower) / ema_current
            
            # Determine position within bands (0-100%)
            position = (price_current - bb_lower) / (bb_upper - bb_lower) * 100 if bb_upper != bb_lower else 50
            
            # Position interpretation
            if position > 80:
                band_position = "Upper Band"
                interpretation = "Near upper band (possible overbought)"
            elif position < 20:
                band_position = "Lower Band"
                interpretation = "Near lower band (possible oversold)"
            else:
                band_position = "Middle"
                interpretation = "In middle of bands (neutral)"
            
            # Display Bollinger Bands metrics
            col_metrics3.metric(
                "Bollinger Bands", 
                f"Width: ${band_width:.2f}", 
                delta=f"{position:.1f}% position"
            )
            col_metrics3.markdown(f"**{interpretation}**")
            col_metrics3.markdown(f"*Range: \${bb_lower:.2f} - \${bb_upper:.2f}*")
        
        # Additional combined insights
        st.markdown("### Trend Indicator Signals")
        
        col_trend1, col_trend2, col_trend3, col_trend4, col_trend5 = st.columns(5)
        
        # Overall trend score
        trend_score = 0
        signals = 0
        
        # SMA signal
        if "SMA" in indicators and "sma" in filtered_df.columns:
            sma_score = 1 if price_current > sma_current else -1
            
            # Check for recent crossover
            sma_cross = ""
            if len(filtered_df) >= 5:
                prev_5days = filtered_df.tail(5)
                # Create a comparison column and count changes
                crosses = prev_5days.with_columns(
                    (pl.col('close') > pl.col('sma')).cast(pl.Int8).alias('cross_check')
                ).select(
                    pl.col('cross_check').diff().alias('cross_diff')
                ).filter(
                    pl.col('cross_diff') != 0
                ).height
                
                if crosses > 0:
                    sma_cross = "Recent crossover detected"
            
            sma_signal = "Bullish" if price_current > sma_current else "Bearish"
            col_trend1.metric("SMA Signal", sma_signal)
            if sma_cross:
                col_trend1.markdown(f"*{sma_cross}*")
            
            trend_score += sma_score
            signals += 1
        
        # EMA signal
        if "EMA" in indicators and "ema" in filtered_df.columns:
            ema_score = 1 if price_current > ema_current else -1
            
            # Check for recent crossover
            ema_cross = ""
            if len(filtered_df) >= 5:
                prev_5days = filtered_df.tail(5)
                # Create a comparison column and count changes
                crosses = prev_5days.with_columns(
                    (pl.col('close') > pl.col('ema')).cast(pl.Int8).alias('cross_check')
                ).select(
                    pl.col('cross_check').diff().alias('cross_diff')
                ).filter(
                    pl.col('cross_diff') != 0
                ).height
                
                if crosses > 0:
                    ema_cross = "Recent crossover detected"
            
            ema_signal = "Bullish" if price_current > ema_current else "Bearish"
            col_trend2.metric("EMA Signal", ema_signal)
            if ema_cross:
                col_trend2.markdown(f"*{ema_cross}*")
            
            trend_score += ema_score
            signals += 1
        
        # SMA/EMA Cross
        if "SMA" in indicators and "EMA" in indicators and "sma" in filtered_df.columns and "ema" in filtered_df.columns:
            cross_signal = "Bullish" if ema_current > sma_current else "Bearish"
            
            # Check for recent SMA/EMA crossover
            cross_detected = ""
            if len(filtered_df) >= 5:
                prev_5days = filtered_df.tail(5)
                # Create a comparison column and count changes
                crosses = prev_5days.with_columns(
                    (pl.col('ema') > pl.col('sma')).cast(pl.Int8).alias('cross_check')
                ).select(
                    pl.col('cross_check').diff().alias('cross_diff')
                ).filter(
                    pl.col('cross_diff') != 0
                ).height
                
                if crosses > 0:
                    cross_detected = "Recent EMA/SMA crossover"
            
            cross_score = 1 if ema_current > sma_current else -1
            col_trend3.metric("EMA/SMA Cross", cross_signal)
            if cross_detected:
                col_trend3.markdown(f"*{cross_detected}*")
            
            trend_score += cross_score
            signals += 1
        
        # Bollinger Bands signal
        if "Bollinger Bands" in indicators and all(col in filtered_df.columns for col in ['bb_upper', 'bb_lower']):
            bb_score = 0
            
            if position < 20:
                bb_score = 1  # Near lower band (possible buy)
                bb_signal = "Potential Buy"
            elif position > 80:
                bb_score = -1  # Near upper band (possible sell)
                bb_signal = "Potential Sell"
            else:
                bb_signal = "Neutral"
            
            # Check for band squeeze (volatility contraction)
            squeeze = ""
            if len(filtered_df) >= 10:
                current_width = band_width
                # Calculate previous width using polars expressions
                prev_width_df = filtered_df.tail(10).select(
                    ((pl.col('bb_upper') - pl.col('bb_lower')) / pl.col('close')).mean().alias('prev_width')
                )
                prev_width = prev_width_df.row(0)[0]
                
                if current_width < prev_width * 0.8:
                    squeeze = "Band squeeze detected (volatility contraction)"
            
            col_trend4.metric("Bollinger Bands", bb_signal)
            if squeeze:
                col_trend4.markdown(f"*{squeeze}*")
            
            trend_score += bb_score
            signals += 1
        
        # Overall trend signal
        if signals > 0:
            trend_status = trend_score / signals
            if trend_status > 0.5:
                signal = "Strong Uptrend"
            elif trend_status > 0:
                signal = "Moderate Uptrend"
            elif trend_status > -0.5:
                signal = "Moderate Downtrend"
            else:
                signal = "Strong Downtrend"
                
            trend_strength = abs(trend_status) * 100
            col_trend5.metric("Overall Trend", signal, f"{trend_strength:.0f}% strength")

    st.subheader("Oscillators and Momentum Indicators")

    # Create momentum indicators chart
    if momentum_indicators_selected:
        fig_momentum = go.Figure()
        
        # Add momentum indicators
        if "RSI" in momentum_indicators_selected:
            fig_momentum.add_trace(go.Scatter(
                x=filtered_df["date"],
                y=filtered_df['rsi'],
                mode='lines',
                name='RSI',
                line=dict(color='purple', width=2)
            ))
            # Add overbought/oversold lines
            fig_momentum.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
            fig_momentum.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        
        if "MACD" in momentum_indicators_selected:
            fig_momentum.add_trace(go.Scatter(
                x=filtered_df["date"],
                y=filtered_df['macd'],
                mode='lines',
                name='MACD Line',
                line=dict(color='blue', width=2)
            ))
            fig_momentum.add_trace(go.Scatter(
                x=filtered_df["date"],
                y=filtered_df['signal_line'],
                mode='lines',
                name='Signal Line',
                line=dict(color='red', width=1)
            ))

            colors = ['rgba(0, 255, 0, 0.5)' if val >= 0 else 'rgba(255, 0, 0, 0.5)' for val in filtered_df['macd_hist']]

            fig_momentum.add_trace(go.Bar(
                x=filtered_df["date"],
                y=filtered_df['macd_hist'],
                name='MACD Histogram',
                marker_color=colors
            ))
        
        if "ROC" in momentum_indicators_selected:
            fig_momentum.add_trace(go.Scatter(
                x=filtered_df["date"],
                y=filtered_df['roc'],
                mode='lines',
                name='ROC',
                line=dict(color='orange', width=2)
            ))
            fig_momentum.add_hline(y=0, line_dash="dash", line_color="white")
        
        if "Stochastic" in momentum_indicators_selected:
            fig_momentum.add_trace(go.Scatter(
                x=filtered_df["date"],
                y=filtered_df['k'],
                mode='lines',
                name='%K',
                line=dict(color='cyan', width=2)
            ))
            fig_momentum.add_trace(go.Scatter(
                x=filtered_df["date"],
                y=filtered_df['d'],
                mode='lines',
                name='%D',
                line=dict(color='magenta', width=2)
            ))
            fig_momentum.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Overbought")
            fig_momentum.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="Oversold")
        
        # Update layout
        fig_momentum.update_layout(
            title='Momentum Indicators',
            xaxis_title='Date',
            yaxis_title='Value',
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig_momentum, use_container_width=True, key="momentum_indicators_chart")
    else:
        st.info("Select momentum indicators to display")

    if momentum_indicators:

        col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
        
        current_data = filtered_df.tail(1).row(0, named=True)
        
        short_trend = filtered_df.tail(5)
        long_trend = filtered_df.tail(20)
        
        if "RSI" in momentum_indicators and "rsi" in filtered_df.columns:
            # Calculate RSI metrics
            rsi_current = current_data['rsi']
            rsi_prev = filtered_df['rsi'].tail(5).to_list()
            rsi_direction = "Increasing" if rsi_prev[-1] > rsi_prev[0] else "- Decreasing"
            
            # Determine overbought/oversold condition
            rsi_condition = "Overbought" if rsi_current > 70 else "Oversold" if rsi_current < 30 else "Neutral"
            
            # Display RSI metrics
            col_metrics1.metric(
                "RSI (Relative Strength Index)", 
                f"{rsi_current:.2f}", 
                delta=rsi_direction
            )
            col_metrics1.markdown(f"**Current condition: {rsi_condition}**")
            col_metrics1.markdown("*Values above 70 indicate overbought conditions, below 30 oversold*")
        
        if "MACD" in momentum_indicators and all(col in filtered_df.columns for col in ['macd', 'signal_line']):
            # Calculate MACD metrics
            macd_current = current_data['macd']
            signal_current = current_data['signal_line']
            histogram = macd_current - signal_current
            
            # Determine trend and crossover
            macd_signal = "Bullish" if macd_current > signal_current else "- Bearish"
            crossover = ""
            if len(filtered_df) >= 2:
                prev_macd = filtered_df['macd'].tail(2).to_list()[0]
                prev_signal = filtered_df['signal_line'].tail(2).to_list()[0]
                if (macd_current > signal_current and prev_macd < prev_signal):
                    crossover = "Bullish crossover detected"
                elif (macd_current < signal_current and prev_macd > prev_signal):
                    crossover = "Bearish crossover detected"
            
            # Display MACD metrics
            col_metrics2.metric(
                "MACD", 
                f"{macd_current:.2f}", 
                delta=macd_signal
            )
            col_metrics2.markdown(f"**Signal line: {signal_current:.2f} (Diff: {histogram:.2f})**")
            if crossover:
                col_metrics2.markdown(f"*{crossover}*")
            else:
                col_metrics2.markdown("*MACD crossovers indicate potential trend changes*")
        
        if "Stochastic" in momentum_indicators and all(col in filtered_df.columns for col in ['k', 'd']):
            # Calculate Stochastic metrics
            k_current = current_data['k']
            d_current = current_data['d']
            
            # Determine overbought/oversold condition
            stoch_condition = "Overbought" if k_current > 80 else "Oversold" if k_current < 20 else "Neutral"
            stoch_signal = "Bullish" if k_current > d_current else "- Bearish"
            
            # Display Stochastic metrics
            col_metrics3.metric(
                "Stochastic Oscillator", 
                f"%K: {k_current:.2f}", 
                delta=stoch_signal
            )
            col_metrics3.markdown(f"**%D: {d_current:.2f} | Condition: {stoch_condition}**")
            col_metrics3.markdown("*K crossing above D is bullish; below D is bearish*")
        
        # Additional combined insights
        st.markdown("### Momentum Indicator Signals")
        
        col_metric1, col_metric2, col_metric3, col_metric4, col_metric5 = st.columns(5)
        
        # Overall momentum score
        momentum_score = 0
        signals = 0
        
        if "RSI" in momentum_indicators and "rsi" in filtered_df.columns:
            # RSI signal
            rsi_score = 0
            if rsi_current < 30:
                rsi_score = 1  # Bullish (oversold)
            elif rsi_current > 70:
                rsi_score = -1  # Bearish (overbought)
            
            rsi_trend = "Oversold" if rsi_current < 30 else "Overbought" if rsi_current > 70 else "Neutral"
            col_metric1.metric("RSI Signal", rsi_trend)
            momentum_score += rsi_score
            signals += 1
        
        # MACD signal
        if "MACD" in momentum_indicators and all(col in filtered_df.columns for col in ['macd', 'signal_line']):
            macd_score = 1 if macd_current > signal_current else -1
            macd_trend = "Bullish" if macd_current > signal_current else "Bearish"
            
            # Check for recent crossover
            if crossover:
                macd_trend = crossover
            
            col_metric2.metric("MACD Signal", macd_trend)
            momentum_score += macd_score
            signals += 1
        
        # Stochastic signal
        if "Stochastic" in momentum_indicators and all(col in filtered_df.columns for col in ['k', 'd']):
            stoch_score = 0
            if k_current < 20 and k_current > d_current:
                stoch_score = 1  # Bullish reversal from oversold
            elif k_current > 80 and k_current < d_current:
                stoch_score = -1  # Bearish reversal from overbought
                
            stoch_condition_detail = stoch_condition
            if k_current < 20 and k_current > d_current:
                stoch_condition_detail = "Bullish reversal"
            elif k_current > 80 and k_current < d_current:
                stoch_condition_detail = "Bearish reversal"
                
            col_metric3.metric("Stochastic Signal", stoch_condition_detail)
            momentum_score += stoch_score
            signals += 1
        
        # ROC signal
        if "ROC" in momentum_indicators and "roc" in filtered_df.columns:
            roc_current = current_data['roc']
            roc_score = 1 if roc_current > 0 else -1
            roc_trend = "Positive" if roc_current > 0 else "- Negative"
            
            # Trend strength
            roc_avg = filtered_df['roc'].mean()
            strength = "Strong" if abs(roc_current) > abs(roc_avg) * 1.5 else "Moderate" if abs(roc_current) > abs(roc_avg) else "Weak"
            
            col_metric4.metric("ROC Signal", f"{strength}", delta=roc_trend)
            momentum_score += roc_score
            signals += 1
        
        # Overall momentum signal
        if signals > 0:
            momentum_status = momentum_score / signals
            if momentum_status > 0.5:
                signal = "Strong Buy"
            elif momentum_status > 0:
                signal = "Buy"
            elif momentum_status > -0.5:
                signal = "Sell"
            else:
                signal = "Strong Sell"
                
            momentum_strength = abs(momentum_status) * 100
            col_metric5.metric("Overall Momentum", signal, f"{momentum_strength:.0f}% strength")



    st.subheader("Volume and Volatility Indicators")

    # Create volume and volatility indicators chart
    if vv_indicators_selected:
        fig_vv = go.Figure()
        
        # Add volume and volatility indicators
        if "OBV" in vv_indicators_selected:
            fig_vv.add_trace(go.Scatter(
                x=filtered_df["date"],
                y=filtered_df['obv'],
                mode='lines',
                name='On-Balance Volume',
                line=dict(color='yellow', width=2)
            ))
        
        if "True Range" in vv_indicators_selected:
            fig_vv.add_trace(go.Scatter(
                x=filtered_df["date"],
                y=filtered_df['true_range'],
                mode='lines',
                name='True Range',
                line=dict(color='cyan', width=2)
            ))
        
        if "ATR" in vv_indicators_selected:
            fig_vv.add_trace(go.Scatter(
                x=filtered_df["date"],
                y=filtered_df['atr'],
                mode='lines',
                name='ATR',
                line=dict(color='magenta', width=2)
            ))
        
        # Update layout
        fig_vv.update_layout(
            title='Volume and Volatility Indicators',
            xaxis_title='Date',
            yaxis_title='Value',
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig_vv, use_container_width=True, key="vv_indicators_chart")
    else:
        st.info("Select volume and volatility indicators to display")

    if vv_indicators:
        # Create columns for metrics
        col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
        
        # Get current values (most recent data point)
        current_data = filtered_df.tail(1).row(0, named=True)
        
        # Compute trend indicators
        short_trend = filtered_df.tail(5)
        long_trend = filtered_df.tail(20)
        
        if "OBV" in vv_indicators and "obv" in filtered_df.columns:
            # Calculate OBV metrics
            obv_current = current_data['obv']
            obv_change = filtered_df['obv'].tail(2).to_list()
            obv_direction = "Increasing" if obv_change[1] > obv_change[0] else "Decreasing"
            
            # Check if OBV confirms price movement
            price_change = filtered_df['close'].tail(5).to_list()
            price_trend = "Up" if price_change[-1] > price_change[0] else "Down"
            obv_trend = "Up" if filtered_df['obv'].tail(5).to_list()[-1] > filtered_df['obv'].tail(5).to_list()[0] else "Down"
            confirmation = "Confirming" if price_trend == obv_trend else "Diverging from"
            
            # Display OBV metrics
            col_metrics1.metric(
                "On-Balance Volume", 
                f"{obv_current:,.0f}", 
                delta=f"{obv_direction}"
            )
            col_metrics1.markdown(f"**OBV is {confirmation} price trend**")
            col_metrics1.markdown("*Higher OBV during price increases suggests strong buying pressure*")
        
        if "True Range" in vv_indicators and "true_range" in filtered_df.columns:
            tr_current = current_data['true_range']
            tr_avg = filtered_df['true_range'].mean()
            tr_status = "Above Average" if tr_current > tr_avg else "- Below Average"
            
            col_metrics2.metric(
                "True Range", 
                f"${tr_current:.2f}", 
                delta=tr_status,
            )
            
            volatility_level = "High" if tr_current > tr_avg * 1.5 else "Normal" if tr_current > tr_avg * 0.5 else "Low"
            col_metrics2.markdown(f"**Current volatility is {volatility_level}**")
            col_metrics2.markdown("*True Range measures daily volatility based on price movement*")
        
        if "ATR" in vv_indicators and "atr" in filtered_df.columns:
            # Calculate ATR metrics
            atr_current = current_data['atr']
            atr_percent = (atr_current / current_data['close']) * 100
            
            # Calculate ATR trend
            atr_5day = short_trend['atr'].mean()
            atr_20day = long_trend['atr'].mean()
            atr_trend = "Increasing" if atr_5day > atr_20day else "- Decreasing"
            
            # Display ATR metrics
            col_metrics3.metric(
                "ATR (Average True Range)", 
                f"${atr_current:.2f}", 
                delta=f"{atr_trend}",
            )
            col_metrics3.markdown(f"**{atr_percent:.2f}% of current price**")
            
            # Interpretation guide
            if atr_trend == "Increasing":
                col_metrics3.markdown("*Increasing volatility suggests potential for larger price swings*")
            else:
                col_metrics3.markdown("*Decreasing volatility often precedes significant price movements*")
        
        # Additional combined insights
        st.markdown("### Volatility and Volume Analysis")
        
        col_metric1, col_metric2, col_metric3, col_metric4, col_metric5 = st.columns(5)

        # Expected daily move based on ATR
        if "ATR" in vv_indicators and "atr" in filtered_df.columns:
            expected_move = atr_current
            col_metric1.metric("Expected Daily Move", f"${expected_move:.2f}")

        # Position sizing suggestion based on ATR
        suggested_stop = expected_move * 1.5
        col_metric2.metric("Suggested Stop Loss", f"${suggested_stop:.2f}")

        # OBV trend signal
        if "OBV" in vv_indicators and "obv" in filtered_df.columns:
            signal_value = "Bullish" if obv_trend == "Up" else "- Bearish"
            signal_delta = "Confirming price" if confirmation == "Confirming" else "Diverging"
            col_metric3.metric("Volume Trend Signal", signal_value, delta=signal_delta)

        # Volatility metric
        if "ATR" in vv_indicators and "atr" in filtered_df.columns:
            historical_atr_percent = (filtered_df['atr'] / filtered_df['close']).mean() * 100
            current_atr_percent = (atr_current / current_data['close']) * 100
            relative_vol = current_atr_percent / historical_atr_percent
            delta = "- " if relative_vol < 1 else ""
            delta_str = f"{delta}{relative_vol:.2f}x historical"
            col_metric4.metric("Volatility", f"{current_atr_percent:.2f}%", delta=delta_str)

        # Volume trend analysis
        if "OBV" in vv_indicators and "obv" in filtered_df.columns and "volume" in filtered_df.columns:
            recent_vol = filtered_df['volume'].tail(10).mean()
            historical_vol = filtered_df['volume'].mean()
            vol_ratio = recent_vol / historical_vol
            delta = "- " if vol_ratio < 1 else ""
            vol_delta = f"{delta}{vol_ratio:.2f}x average"
            col_metric5.metric("Trading Volume", f"{recent_vol:,.0f}", delta=vol_delta)


    st.subheader("Investment Strategy Analysis")

    col1_invest, col2, col3, col4, col5 = st.columns(5)

    with col1_invest:
        initial_investment = st.number_input(
            "Dollar Cost Investment ($)",
            min_value=1,
            max_value=10000,
            value=1,
            step=1
        )

    with col2:
        lumpsum_investment = st.number_input(
            "Lump Sum Investment ($)",
            min_value=1,
            max_value=10000,
            value=1,
            step=1
        )

    with col3:
        start_date_invest = st.date_input(
            "Investment Start Date",
            value=filtered_df["date"].min(),
            min_value=filtered_df["date"].min(),
            max_value=filtered_df["date"].max(),
            key="investment_start_date_input"
        )

    with col4:
        end_date_invest = st.date_input(
            "Investment End Date",
            value=filtered_df["date"].max(),
            min_value=filtered_df["date"].min(),
            max_value=filtered_df["date"].max(),
            key="investment_end_date_input"
        )
    
    with col5:
        investment_frequency = st.selectbox(
            "Investment Frequency",
            ("Daily", "Weekly", "Monthly", "Quarterly"),
            index=0,
            key="investment_frequency_selectbox"
        )


    df_invest = filtered_df.select(["date", "close"]).with_columns(
        pl.col("date").cast(pl.Date)
    )

    full_dates = pl.date_range(
        start=start_date_invest,
        end=end_date_invest,
        interval="1d",
        eager=True
    ).to_frame(name="date")

    df_invest_full = full_dates.join(df_invest, on="date", how="left") \
        .sort("date") \
        .with_columns(pl.col("close").fill_null(strategy="forward"))


    first_valid_row = df_invest_full.filter(pl.col("close").is_not_null()).head(1)
    if first_valid_row.height > 0:
        lump_price = first_valid_row.select(pl.col("close")).row(0)[0]
        lump_shares = lumpsum_investment / lump_price
        lump_portfolio = df_invest_full.with_columns(
            (pl.lit(lump_shares) * pl.col("close")).alias("lump_sum")
        )
    else:
        st.error("No valid price data available for the selected date range")
        lump_shares = 0
        lump_portfolio = df_invest_full.with_columns(
            (pl.lit(0) * pl.col("close")).alias("lump_sum")
        )

    dca_dates = []
    current = start_date_invest
    while current <= end_date_invest:
        dca_dates.append(current)
        if investment_frequency == "Daily":
            current += timedelta(days=1)
        elif investment_frequency == "Weekly":
            current += timedelta(weeks=1)
        elif investment_frequency == "Monthly":
            current += relativedelta(months=1)
        elif investment_frequency == "Quarterly":
            current += relativedelta(months=3)

    dca_transactions = (
        pl.DataFrame({"date": dca_dates})
        .join(df_invest_full.select(["date", "close"]), on="date", how="left")
        .with_columns((pl.lit(initial_investment) / pl.col("close")).alias("shares"))
    )


    dca_daily = df_invest_full.join(
        dca_transactions.select(["date", "shares"]), on="date", how="left"
    ).with_columns(
        pl.col("shares").fill_null(0)
    ).sort("date").with_columns(
        pl.col("shares").cum_sum().alias("cumulative_shares")
    )

    dca_daily = dca_daily.with_columns(
        (pl.col("cumulative_shares") * pl.col("close")).alias("dca_value")
    )


    comparison_df = df_invest_full.join(
        lump_portfolio.select(["date", "lump_sum"]), on="date", how="left"
    ).join(
        dca_daily.select(["date", "dca_value"]), on="date", how="left"
    ).select(["date", "lump_sum", "dca_value"])

    # Plotting with Plotly
    fig_invest = go.Figure()
    fig_invest.add_trace(go.Scatter(
        x=comparison_df["date"].to_list(),
        y=comparison_df["lump_sum"].to_list(),
        mode="lines",
        name="Lump Sum"
    ))
    fig_invest.add_trace(go.Scatter(
        x=comparison_df["date"].to_list(),
        y=comparison_df["dca_value"].to_list(),
        mode="lines",
        name="Dollar Cost Averaging"
    ))
    fig_invest.update_layout(
        title="Investment Strategy Comparison",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode="x unified"
    )
    st.plotly_chart(fig_invest, use_container_width=True)


    lump_final = comparison_df.select(pl.col("lump_sum")).tail(1).row(0)[0]
    dca_final = comparison_df.select(pl.col("dca_value")).tail(1).row(0)[0]

    lump_growth = (lump_final / lumpsum_investment - 1) * 100

    num_transactions = dca_transactions.height
    total_dca_invested = initial_investment * num_transactions
    dca_growth = (dca_final / total_dca_invested - 1) * 100

    col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
    col_metric1.metric("DCA Total Invested", f"${total_dca_invested:,.2f}", f"{dca_growth:.2f}%")
    col_metric2.metric("DCA Final Value", f"${dca_final:,.2f}", f"{dca_growth:.2f}%")

    col_metric3.metric("Lump Sum Total Invested", f"${lumpsum_investment:,.2f}", f"{lump_growth:.2f}%")
    col_metric4.metric("Lump Sum Final Value", f"${lump_final:,.2f}", f"{lump_growth:.2f}%")

