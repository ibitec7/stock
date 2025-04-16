import streamlit as st
import polars as pl
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
from dashboard_helpers import source_indicators
from sklearn.preprocessing import MinMaxScaler

CACHE_DIR = "../cache"

def tab_4():
    st.header("Comparative Analysis")

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    indicators = source_indicators(period="5y", timeframe="1wk")

    with col1:
        start_date = st.date_input(
            label="Start Date",
            value=indicators["date"].min(),
            min_value=indicators["date"].min(),
            max_value=indicators["date"].max(),
            key="correlation_start_date"
        )

    with col2:
        end_date = st.date_input(
            label="End Date",
            value=indicators["date"].max(),
            min_value=indicators["date"].min(),
            max_value=indicators["date"].max(),
            key="correlation_end_date"
        )

    with col3:
        period = st.selectbox(
            "Select Period",
            ("ytd", "1y", "2y", "5y", "10y", "max"),
            index=3,
            key="correlation_period",
        )

    with col4:
        timeframe = st.selectbox(
            "Select Timeframe",
            ("1d", "1wk", "1mo", "3mo"),
            index=2,
            key="correlation_timeframe",
        )

    with col5:
        compare = st.selectbox(
            "Select Comparison",
            ("Markets", "Competitors", "Dependents"),
            index=0,
            key="correlation_compare",
        )

    with col6:
        if compare == "Markets":
            entities = st.multiselect(
                "Select Markets",
                ["^GSPC", "^IXIC", "^DJI", "^TWII", "^FTSE"],
                default=["^GSPC", "^IXIC"],
                key="correlation_markets",
            )

        elif compare == "Competitors":
            entities = st.multiselect(
                "Select Competitors",
                ["AMD", "INTC", "TSM", "AAPL", "MSFT"],
                default=["AMD", "INTC"],
                key="correlation_competitors",
            )
        
        else:
            entities = st.multiselect(
                "Select Dependents",
                ["TSLA", "GOOGL", "AMZN", "MSFT", "AAPL"],
                default=["TSLA", "GOOGL"],
                key="correlation_dependents",
            )

    market_data = pl.DataFrame({})

    st.subheader("Correlation Analysis")


    for entity in entities:
        # Check if it's the first entity (empty DataFrame)
        if market_data.is_empty():
            # For first entity, include date column
            market_data = source_indicators(period, timeframe, ticker=entity).filter(
            (pl.col("date") > start_date) & (pl.col("date") < end_date)
            ).select(
            [
                pl.col("date"),
                pl.col("close").alias(f"{entity}_close"),
                pl.col("volume").alias(f"{entity}_volume"),
                pl.col("open").alias(f"{entity}_open"),
                pl.col("high").alias(f"{entity}_high"),
                pl.col("low").alias(f"{entity}_low"),
            ]
            )
        else:
            # For subsequent entities, join on date instead of concatenating
            new_data = source_indicators(period, timeframe, ticker=entity).filter(
            (pl.col("date") > start_date) & (pl.col("date") < end_date)
            ).select(
            [
                pl.col("date"),
                pl.col("close").alias(f"{entity}_close"),
                pl.col("volume").alias(f"{entity}_volume"),
                pl.col("open").alias(f"{entity}_open"),
                pl.col("high").alias(f"{entity}_high"),
                pl.col("low").alias(f"{entity}_low"),
            ]
            )
            market_data = market_data.join(new_data, on="date", how="outer")

    combined_data = indicators.join(
        market_data,
        on="date",
        how="inner"
    ).filter(
        (pl.col("date") > start_date) & (pl.col("date") < end_date)
    )

    entities.append("NVDA")
    
    correlation_matrix = combined_data.select(
        [pl.col(f"{entity}_close" if entity != "NVDA" else "close") for entity in entities]
    ).corr()

    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.to_numpy(),
        x=entities,
        y=entities,
        colorbar=dict(title='Correlation Coefficient'),
    ))

    fig.update_layout(
        title='Correlation Matrix',
        xaxis_title='Entities',
        yaxis_title='Entities',
        width=800,
        height=800,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Financial Analysis")

    metrics = ["Interest Expense", "Total Revenue", "Gross Profit", "Operating Income", "Operating Expense", "Diluted EPS", "EBITDA", "EBIT"]

    financials_entities = dict()

    col1, col2 = st.columns(2)

    with col1:
        dependents = st.multiselect(
            "Select Dependents",
            ["TSLA", "GOOGL", "AMZN", "META", "AAPL"],
            default=["TSLA", "GOOGL"],
            key="financial_dependents",
        )

    with col2:
        competitors = st.multiselect(
            "Select Competitors",
            ["AMD", "INTC", "TSM", "MSFT"],
            default=["AMD", "INTC"],
            key="financial_competitors",
        )

    entities = ["NVDA"]
    entities.extend(dependents)
    entities.extend(competitors)

    for entity in entities:
        financials = yf.Ticker(entity).financials.fillna(0.0).iloc[:,0]

        for metric in metrics:

            if not entity in financials_entities:
                financials_entities[entity] = {}

                # if not metric in financials_entities[entity]:
                #     financials_entities[entity][metric] = float(0)
            if metric in ["Operating Expense", "Interest Expense"]:
                financials_entities[entity][metric] = 1 / (financials[metric] / financials["Total Revenue"])
            elif metric in ["EBITDA", "EBIT"]:
                financials_entities[entity][metric] = financials[metric] / financials["Total Revenue"]
            else:
                financials_entities[entity][metric]=financials[metric]

    scaler = MinMaxScaler()

    scaled = dict()

    metrics_only = {}

    for entity in financials_entities.keys():
        for metric in metrics:
            if not metric in metrics_only:
                metrics_only[metric] = []
            metrics_only[metric].append(financials_entities[entity][metric])

    for metric in metrics_only.keys():
        scaled[metric] = scaler.fit_transform(np.array(metrics_only[metric]).reshape(-1, 1)).flatten()

    final = {}

    for i,entity in enumerate(entities):
        if not entity in final:
            final[entity] = []
        for metric in metrics:
            final[entity].append(scaled[metric][i])

    fig2 = go.Figure()

    for entity in final.keys():
        fig2.add_trace(go.Scatterpolar(
            r = final[entity],
            theta = metrics,
            fill = "toself",
            name = entity,
        ))

    # fig2.update_traces(
    #     hoverinfo="name+text",
    #     text=list(final.keys()),
    # )
    fig2.update_layout(
        title="Financial Metrics",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
            )
        ),
        showlegend=True,
    )

    st.plotly_chart(fig2, use_container_width=True)