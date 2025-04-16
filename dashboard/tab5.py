import streamlit as st
import json
import os
import polars as pl
import ollama
import random
from datetime import datetime
import numpy as np

CACHE_DIR = "../cache"

summary_prompt = """
**System Prompt:**
You are an expert financial analyst with advanced reasoning capabilities. Your task is to analyze and summarize the "news_summary" section of multiple AI-generated stock analyses. Focus on identifying key themes, sentiment trends, and market reactions across the articles.
Make a single unified summary of the articles, including the title, date, key points, sentiment, market reactions, and analyst concerns. Some articles may not be related to NVIDIA so ignore those.

For all articles, extract:
- The title as the core summary of all the titles as to what is going on.
- Key points from the articles.
- Overall sentiment of all articles (Bullish/Bearish/Neutral).
- Important Market reactions and analyst concerns.

Finally, provide a concise summary of the overall trends and insights derived from all articles.

**User Prompt:**

Analyze the following "news_summary" sections from multiple AI analyses:
{news_summaries}

Generate a summary using the following markdown format:
* Overall Trends
Summary of overarching trends across all articles

* Articles Summary
** Title
Title of the articles

** Key Points
- Key point 1
- Key point 2

** Sentiment
**Overall**: Bullish/Bearish/Neutral  
**Market Reaction**: Summary of market reaction  

** Analyst Concerns
- Concern 1
- Concern 2
"""

impact_prompt = """
**System Prompt:**
You are an expert financial analyst specializing in stock performance evaluation. Your task is to analyze and summarize the "stock_impact" section of multiple AI-generated stock analyses. Focus on identifying short-term and long-term impacts, including price movements, catalysts, risks, and strategic implications. Some articles may not be related to NVIDIA so ignore those.

For all articles, extract:
- Short-term price movement trends.
- Catalysts driving growth or risks to watch out for.
- Long-term strategic implications, growth opportunities, and competitive positioning.

Finally, provide a concise summary of the overall trends and insights derived from all articles.

**User Prompt:**

Analyze the following "stock_impact" sections from multiple AI analyses:
{stock_impacts}

Generate a summary using the following markdown format:
* Short-term Summary
Summary of short-term impacts across all articles

* Long-term Summary
Summary of long-term impacts across all articles

* Articles Analysis
** Short-term Impacts
*** Price Movement
Summary of price movement most recent

*** Catalysts
- Catalyst 1
- Catalyst 2

*** Risks
- Risk 1
- Risk 2

** Long-term Impacts
*** Strategic Implications
- Implication 1
- Implication 2

*** Growth Opportunities
- Opportunity 1
- Opportunity 2

*** Competitive Position
Summary of competitive position
"""

technical_prompt = """
**System Prompt:**
You are an expert in technology and innovation trends. Your task is to analyze and summarize the "technical_factors" section of multiple AI-generated stock analyses. Focus on identifying key product launches, AI innovations, and data center growth trends.
Some articles may not be related to NVIDIA so ignore those.

For all articles, extract:
- Product launches and their significance.
- AI innovations and their potential impact.
- Data center growth trends.

Finally, provide a concise summary of the overall trends and insights derived from all articles.

**User Prompt:**

Analyze the following "technical_factors" sections from multiple AI analyses:
{technical_factors}

Generate a summary using the following markdown format:
* Overall Technical Trends
Summary of overarching technical trends across all articles

* Articles Analysis
** Product Launches
- Product Launch 1
- Product Launch 2

** AI Innovation
- Innovation 1
- Innovation 2

** Data Center Growth
Summary of data center growth
"""

financial_prompt = """
**System Prompt:**
You are an expert in financial analysis. Your task is to analyze and summarize the "financial_metrics" section of multiple AI-generated stock analyses. Focus on identifying trends in revenue impact, margin pressure, and guidance changes.
Some articles may not be related to NVIDIA so ignore those.

For all articles, extract:
- Revenue impact and its significance.
- Margin pressure trends.
- Guidance changes and their implications.

Finally, provide a concise summary of the overall trends and insights derived from all articles.

**User Prompt:**

Analyze the following "financial_metrics" sections from multiple AI analyses:
{financial_metrics}

Generate a summary using the following markdown format:
* Overall Financial Trends
Summary of overarching financial trends across all articles

* Articles Analysis
** Revenue Impact
Summary of revenue impact

** Margin Pressure
Summary of margin pressure

** Guidance Changes
Summary of guidance changes
"""

recommendations_prompt = """
**System Prompt:**
You are an expert in investment strategy and decision-making. Your task is to analyze and summarize the "recommendations" section of multiple AI-generated stock analyses. Focus on identifying trading strategies, buy/hold/sell recommendations, and factors to monitor.
Some articles may not be related to NVIDIA so ignore those.

For all articles, extract:
- Trading strategy recommendations.
- Buy/Hold/Sell recommendation.
- Factors to monitor for future updates.

Finally, provide a concise summary of the overall trends and insights derived from all articles.

**User Prompt:**

Analyze the following "recommendations" sections from multiple AI analyses:
{recommendations}

Generate a summary using the following markdown format:
* Overall Recommendation Trends
Summary of overarching recommendation trends across all articles

* Articles Analysis
** Trading Strategy
Summary of trading strategy

** Buy/Hold/Sell Recommendation
Buy/Hold/Sell recommendation

** Factors to Monitor
- Factor 1
- Factor 2
"""

impact_prompt = """
**System Prompt:**
You are an expert financial analyst specializing in stock performance evaluation. Your task is to analyze and summarize the "stock_impact" section of multiple AI-generated stock analyses. Focus on identifying short-term and long-term impacts, including price movements, catalysts, risks, and strategic implications.
Some articles may not be related to NVIDIA so ignore those.

For all articles, extract:
- Short-term price movement trends.
- Catalysts driving growth or risks to watch out for.
- Long-term strategic implications, growth opportunities, and competitive positioning.

Finally, provide a concise summary of the overall trends and insights derived from all articles.

**User Prompt:**

Analyze the following "stock_impact" sections from multiple AI analyses:
{stock_impacts}

Generate a summary using the following markdown format:
* Short-term Summary
Summary of short-term impacts across all articles

* Long-term Summary
Summary of long-term impacts across all articles

* Articles Analysis
** Short-term Impacts
*** Price Movement
Summary of price movement most recent

*** Catalysts
- Catalyst 1
- Catalyst 2

*** Risks
- Risk 1
- Risk 2

** Long-term Impacts
*** Strategic Implications
- Implication 1
- Implication 2

*** Growth Opportunities
- Opportunity 1
- Opportunity 2

*** Competitive Position
Summary of competitive position
"""

CACHE_DIR = "../cache"

summary_prompt = """
**System Prompt:**
You are an expert financial analyst with advanced reasoning capabilities. Your task is to analyze and summarize the "news_summary" section of multiple AI-generated stock analyses. Focus on identifying key themes, sentiment trends, and market reactions across the articles.
Make a single unified summary of the articles, including the title, date, key points, sentiment, market reactions, and analyst concerns.
Some articles may not be related to NVIDIA so ignore those.

For all articles, extract:
- The title as the core summary of all the titles as to what is going on.
- Key points from the articles.
- Overall sentiment of all articles (Bullish/Bearish/Neutral).
- Important Market reactions and analyst concerns.

Finally, provide a concise summary of the overall trends and insights derived from all articles.

**User Prompt:**

Analyze the following "news_summary" sections from multiple AI analyses:
{news_summaries}

Generate a summary using the following markdown format:
* Overall Trends
Summary of overarching trends across all articles

* Articles Summary
** Title
Title of the articles

** Key Points
- Key point 1
- Key point 2

** Sentiment
**Overall**: Bullish/Bearish/Neutral
**Market Reaction**: Summary of market reaction

** Analyst Concerns
- Concern 1
- Concern 2
"""

impact_prompt = """
**System Prompt:**
You are an expert financial analyst specializing in stock performance evaluation. Your task is to analyze and summarize the "stock_impact" section of multiple AI-generated stock analyses. Focus on identifying short-term and long-term impacts, including price movements, catalysts, risks, and strategic implications.
Some articles may not be related to NVIDIA so ignore those.

For all articles, extract:
- Short-term price movement trends.
- Catalysts driving growth or risks to watch out for.
- Long-term strategic implications, growth opportunities, and competitive positioning.

Finally, provide a concise summary of the overall trends and insights derived from all articles.

**User Prompt:**

Analyze the following "stock_impact" sections from multiple AI analyses:
{stock_impacts}

Generate a summary using the following markdown format:
* Short-term Summary
Summary of short-term impacts across all articles

* Long-term Summary
Summary of long-term impacts across all articles

* Articles Analysis
** Short-term Impacts
*** Price Movement
Summary of price movement most recent

*** Catalysts
- Catalyst 1
- Catalyst 2

*** Risks
- Risk 1
- Risk 2

** Long-term Impacts
*** Strategic Implications
- Implication 1
- Implication 2

*** Growth Opportunities
- Opportunity 1
- Opportunity 2

*** Competitive Position
Summary of competitive position
"""

technical_prompt = """
**System Prompt:**
You are an expert in technology and innovation trends. Your task is to analyze and summarize the "technical_factors" section of multiple AI-generated stock analyses. Focus on identifying key product launches, AI innovations, and data center growth trends.
Some articles may not be related to NVIDIA so ignore those.

For all articles, extract:
- Product launches and their significance.
- AI innovations and their potential impact.
- Data center growth trends.

Finally, provide a concise summary of the overall trends and insights derived from all articles.

**User Prompt:**

Analyze the following "technical_factors" sections from multiple AI analyses:
{technical_factors}

Generate a summary using the following markdown format:
* Overall Technical Trends
Summary of overarching technical trends across all articles

* Articles Analysis
** Product Launches
- Product Launch 1
- Product Launch 2

** AI Innovation
- Innovation 1
- Innovation 2

** Data Center Growth
Summary of data center growth
"""

financial_prompt = """
**System Prompt:**
You are an expert in financial analysis. Your task is to analyze and summarize the "financial_metrics" section of multiple AI-generated stock analyses. Focus on identifying trends in revenue impact, margin pressure, and guidance changes.
Some articles may not be related to NVIDIA so ignore those.

For all articles, extract:
- Revenue impact and its significance.
- Margin pressure trends.
- Guidance changes and their implications.

Finally, provide a concise summary of the overall trends and insights derived from all articles.

**User Prompt:**

Analyze the following "financial_metrics" sections from multiple AI analyses:
{financial_metrics}

Generate a summary using the following markdown format:
* Overall Financial Trends
Summary of overarching financial trends across all articles

* Articles Analysis
** Revenue Impact
Summary of revenue impact

** Margin Pressure
Summary of margin pressure

** Guidance Changes
Summary of guidance changes
"""

recommendations_prompt = """
**System Prompt:**
You are an expert in investment strategy and decision-making. Your task is to analyze and summarize the "recommendations" section of multiple AI-generated stock analyses. Focus on identifying trading strategies, buy/hold/sell recommendations, and factors to monitor.
Some articles may not be related to NVIDIA so ignore those.

For all articles, extract:
- Trading strategy recommendations.
- Buy/Hold/Sell recommendation.
- Factors to monitor for future updates.

Finally, provide a concise summary of the overall trends and insights derived from all articles.

**User Prompt:**

Analyze the following "recommendations" sections from multiple AI analyses:
{recommendations}

Generate a summary using the following markdown format:
* Overall Recommendation Trends
Summary of overarching recommendation trends across all articles

* Articles Analysis
** Trading Strategy
Summary of trading strategy

** Buy/Hold/Sell Recommendation
Buy/Hold/Sell recommendation

** Factors to Monitor
- Factor 1
- Factor 2
"""


def summarize_text(type, object):
    if type == "news_summary":
        prompt = summary_prompt.format(news_summaries=object)
    elif type == "stock_impact":
        prompt = impact_prompt.format(stock_impacts=object)
    elif type == "technical_factors":
        prompt = technical_prompt.format(technical_factors=object)
    elif type == "financial_metrics":
        prompt = financial_prompt.format(financial_metrics=object)
    elif type == "recommendations":
        prompt = recommendations_prompt.format(recommendations=object)
    else:
        return "Invalid type specified"

    response = ollama.chat(
            model="qwen2.5:1.5b",
            messages=[
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.7},
        )


    return response.message.content

def tab_5():
    st.header("AI Insights and Analysis")

    ai_analysis_path = os.path.join(CACHE_DIR, "ai_analysis.json")
    if not os.path.exists(ai_analysis_path):
        st.error(f"AI analysis file not found at {ai_analysis_path}")
        return

    try:
        with open(ai_analysis_path, "r") as f:
            ai_analysis_data = json.load(f)
    except json.JSONDecodeError:
        st.error(f"Error decoding JSON from {ai_analysis_path}")
        return
    except Exception as e:
        st.error(f"Error reading file {ai_analysis_path}: {e}")
        return

    if "ai_analysis" not in ai_analysis_data or not isinstance(ai_analysis_data["ai_analysis"], list):
        st.error("Invalid format in ai_analysis.json: 'ai_analysis' key missing or not a list.")
        return
    if "date" not in ai_analysis_data or not isinstance(ai_analysis_data["date"], list):
         st.error("Invalid format in ai_analysis.json: 'date' key missing or not a list.")
         return
    if len(ai_analysis_data["ai_analysis"]) != len(ai_analysis_data["date"]):
        st.error("Mismatch in lengths between 'ai_analysis' and 'date' lists in ai_analysis.json.")
        return

    processed_analysis = []
    for i, analysis in enumerate(ai_analysis_data["ai_analysis"]):
        if isinstance(analysis, dict):
             processed_analysis.append(json.dumps(analysis))
        elif isinstance(analysis, str):
             try:
                 json.loads(analysis)
                 processed_analysis.append(analysis)
             except json.JSONDecodeError:
                 st.warning(f"Skipping invalid JSON string at index {i} in 'ai_analysis'.")
                 continue
        else:
            st.warning(f"Skipping unexpected data type at index {i} in 'ai_analysis'.")
            continue

    valid_dates = [ai_analysis_data["date"][i] for i, analysis in enumerate(ai_analysis_data["ai_analysis"]) if isinstance(json.loads(processed_analysis[i]) if isinstance(processed_analysis[i], str) else None, dict)]
    
    if len(processed_analysis) != len(ai_analysis_data["date"]):
         st.error("Data processing led to mismatch between analysis entries and dates. Aborting.")
         return

    data_dict = {"date": ai_analysis_data["date"], "ai_analysis": processed_analysis}


    try:
        data = pl.DataFrame(data_dict)
    except Exception as e:
        st.error(f"Error creating Polars DataFrame: {e}")
        return

    try:
        data = data.with_columns(
            pl.col("date").str.to_datetime().alias("date_dt") 
        ).sort("date_dt")
    except Exception as e:
        st.error(f"Error processing dates in DataFrame: {e}")
        return

    if data.height == 0:
        st.warning("No valid AI analysis data found.")
        return

    min_date = data["date_dt"].min()
    max_date = data["date_dt"].max()

    col1, col2, col3 = st.columns(3)

    with col1:
        start_date_ai = st.date_input(
            label="Start Date",
            value=min_date.date(), 
            min_value=min_date.date(),
            max_value=max_date.date(),
            key="ai_start_date"
        )

    with col2:
        end_date_ai = st.date_input(
            label="End Date",
            value=max_date.date(),
            min_value=min_date.date(),
            max_value=max_date.date(),
            key="ai_end_date"
        )

    start_datetime = datetime.combine(start_date_ai, datetime.min.time())
    end_datetime = datetime.combine(end_date_ai, datetime.max.time())


    filtered_df = data.filter(
        (pl.col("date_dt") >= start_datetime) & (pl.col("date_dt") <= end_datetime)
    )

    st.dataframe(filtered_df)

    if filtered_df.height == 0:
        st.warning("No AI analysis data found for the selected date range.")
        return

    analysis_list_str = filtered_df["ai_analysis"].to_list()

    analysis_list = []
    for i, analysis_str in enumerate(analysis_list_str):
        try:
            analysis_list.append(json.loads(analysis_str))
        except json.JSONDecodeError:
            st.warning(f"Skipping invalid JSON data during filtering at index {i}.")
            continue

    if not analysis_list:
        st.warning("No valid analysis data could be parsed for the selected range.")
        return

    summaries = [analysis.get("news_summary", "") for analysis in analysis_list if analysis.get("news_summary")]
    stock_impacts = [analysis.get("stock_impact", "") for analysis in analysis_list if analysis.get("stock_impact")]
    technical_factors = [analysis.get("technical_factors", "") for analysis in analysis_list if analysis.get("technical_factors")]
    financial_metrics = [analysis.get("financial_metrics", "") for analysis in analysis_list if analysis.get("financial_metrics")]
    recommendations = [analysis.get("recommendations", "") for analysis in analysis_list if analysis.get("recommendations")]

    st.subheader("AI Analysis Summaries")

    if summaries:
        st.subheader("News Summary")
        with st.spinner("Generating News Summary..."):
            news_summary_md = summarize_text("news_summary", "\n\n".join(summaries))
            st.markdown(news_summary_md)
    else:
        st.info("No 'news_summary' data available for the selected period.")

    if stock_impacts:
        st.subheader("Stock Impact")
        with st.spinner("Generating Stock Impact Summary..."):
            stock_impact_md = summarize_text("stock_impact", "\n\n".join(stock_impacts))
            st.markdown(stock_impact_md)
    else:
        st.info("No 'stock_impact' data available for the selected period.")

    if technical_factors:
        st.subheader("Technical Factors")
        with st.spinner("Generating Technical Factors Summary..."):
            technical_factors_md = summarize_text("technical_factors", "\n\n".join(technical_factors))
            st.markdown(technical_factors_md)
    else:
        st.info("No 'technical_factors' data available for the selected period.")

    if financial_metrics:
        st.subheader("Financial Metrics")
        with st.spinner("Generating Financial Metrics Summary..."):
            financial_metrics_md = summarize_text("financial_metrics", "\n\n".join(financial_metrics))
            st.markdown(financial_metrics_md)
    else:
        st.info("No 'financial_metrics' data available for the selected period.")

    if recommendations:
        st.subheader("Recommendations")
        with st.spinner("Generating Recommendations Summary..."):
            recommendations_md = summarize_text("recommendations", "\n\n".join(recommendations))
            st.markdown(recommendations_md)
    else:
        st.info("No 'recommendations' data available for the selected period.")

financial_prompt = """
**System Prompt:**
You are an expert in financial analysis. Your task is to analyze and summarize the "financial_metrics" section of multiple AI-generated stock analyses. Focus on identifying trends in revenue impact, margin pressure, and guidance changes.
Some articles may not be related to NVIDIA so ignore those.

For all articles, extract:
- Revenue impact and its significance.
- Margin pressure trends.
- Guidance changes and their implications.

Finally, provide a concise summary of the overall trends and insights derived from all articles.

**User Prompt:**

Analyze the following "financial_metrics" sections from multiple AI analyses:
{financial_metrics}

Generate a summary using the following markdown format:
* Overall Financial Trends
Summary of overarching financial trends across all articles

* Articles Analysis
** Revenue Impact
Summary of revenue impact

** Margin Pressure
Summary of margin pressure

** Guidance Changes
Summary of guidance changes
"""

recommendations_prompt = """
**System Prompt:**
You are an expert in investment strategy and decision-making. Your task is to analyze and summarize the "recommendations" section of multiple AI-generated stock analyses. Focus on identifying trading strategies, buy/hold/sell recommendations, and factors to monitor.
Some articles may not be related to NVIDIA so ignore those.

For all articles, extract:
- Trading strategy recommendations.
- Buy/Hold/Sell recommendation.
- Factors to monitor for future updates.

Finally, provide a concise summary of the overall trends and insights derived from all articles.

**User Prompt:**

Analyze the following "recommendations" sections from multiple AI analyses:
{recommendations}

Generate a summary using the following markdown format:
* Overall Recommendation Trends
Summary of overarching recommendation trends across all articles

* Articles Analysis
** Trading Strategy
Summary of trading strategy

** Buy/Hold/Sell Recommendation
Buy/Hold/Sell recommendation

** Factors to Monitor
- Factor 1
- Factor 2
"""

def summarize_text(type, object):
    if type == "news_summary":
        prompt = summary_prompt.format(news_summaries=object)
    elif type == "stock_impact":
        prompt = impact_prompt.format(stock_impacts=object)
    elif type == "technical_factors":
        prompt = technical_prompt.format(technical_factors=object)
    elif type == "financial_metrics":
        prompt = financial_prompt.format(financial_metrics=object)
    elif type == "recommendations":
        prompt = recommendations_prompt.format(recommendations=object)

    response = ollama.chat(
            model="qwen2.5:1.5b",
            messages=[
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.7},
        )
    
    return response.message.content

def tab_5():
    st.header("AI Insights and Analysis")

    with open(os.path.join(CACHE_DIR, "ai_analysis.json"), "r") as f:
        ai_analysis = json.load(f)

    for i,analysis in enumerate(ai_analysis["ai_analysis"]):
        ai_analysis["ai_analysis"][i] = json.dumps(analysis)

    data = pl.DataFrame(ai_analysis)

    data = data.with_columns(
        pl.col("date").str.to_datetime().sort()
    )
    
    col1, col2 = st.columns(2)

    with col1:
        start_date_ai = st.date_input(
            label="Start Date",
            value=data["date"].min(),
            min_value=data["date"].min(),
            max_value=data["date"].max(),
            key="ai_start_date"
        )

    with col2:
        end_date_ai = st.date_input(
            label="End Date",
            value=data["date"].max(),
            min_value=data["date"].min(),
            max_value=data["date"].max(),
            key="ai_end_date"
        )
    
    filtered_df = data.filter(
        (pl.col("date") >= start_date_ai) & (pl.col("date") <= end_date_ai),
        pl.col("ai_analysis").is_not_null()
    )

    analysis_list = filtered_df["ai_analysis"].to_list()

    print(analysis_list)

    analysis_list = analysis_list[:7]

    print(analysis_list)

    print("second time\n\n")
    analysis_list = [json.loads(analysis) for analysis in analysis_list]

    analysis_list = [analysis for analysis in analysis_list if analysis is not None]

    if len(analysis_list) > 0:
        analysis_list = analysis_list[:7]

    print(analysis_list)

    print("third time\n\n")

    for i, analysis in enumerate(analysis_list):
        try:
            assert analysis is not None
        except:
            st.error(f"Error parsing JSON at index {i}")

    summaries = [analysis["news_summary"] for analysis in analysis_list]
    stock_impacts = [analysis["stock_impact"] for analysis in analysis_list]
    technical_factors = [analysis["technical_factors"] for analysis in analysis_list]
    financial_metrics = [analysis["financial_metrics"] for analysis in analysis_list]
    recommendations = [analysis["recommendations"] for analysis in analysis_list]

    st.subheader("AI Analysis Summaries")

    if not os.path.exists(os.path.join(CACHE_DIR, f"summary_{start_date_ai}_{end_date_ai}.md")):
        with open(os.path.join(CACHE_DIR, f"summary_{start_date_ai}_{end_date_ai}.md"), "w") as f:
            st.subheader("News Summary")
            news_summary = summarize_text("news_summary", summaries)
            f.write(news_summary)
            st.markdown(news_summary)

    else:
        with open(os.path.join(CACHE_DIR, f"summary_{start_date_ai}_{end_date_ai}.md"), "r") as f:
            news_summary = f.read()
            st.subheader("News Summary")
            st.markdown(news_summary)

    if not os.path.exists(os.path.join(CACHE_DIR, f"stock_impact_{start_date_ai}_{end_date_ai}.md")):
        with open(os.path.join(CACHE_DIR, f"stock_impact_{start_date_ai}_{end_date_ai}.md"), "w") as f:
            stock_impact = summarize_text("stock_impact", stock_impacts)
            st.subheader("Stock Impact")
            st.markdown(stock_impact)
            f.write(stock_impact)
    
    else:
        with open(os.path.join(CACHE_DIR, f"stock_impact_{start_date_ai}_{end_date_ai}.md"), "r") as f:
            stock_impact = f.read()
            st.subheader("Stock Impact")
            st.markdown(stock_impact)

    if not os.path.exists(os.path.join(CACHE_DIR, f"technical_factors_{start_date_ai}_{end_date_ai}.md")):
        with open(os.path.join(CACHE_DIR, f"technical_factors_{start_date_ai}_{end_date_ai}.md"), "w") as f:
            technical_factors = summarize_text("technical_factors", technical_factors)
            st.subheader("Technical Factors")
            st.markdown(technical_factors)
            f.write(technical_factors)


    else:
        with open(os.path.join(CACHE_DIR, f"technical_factors_{start_date_ai}_{end_date_ai}.md"), "r") as f:
            technical_factors = f.read()
            st.subheader("Technical Factors")
            st.markdown(technical_factors)

    if not os.path.exists(os.path.join(CACHE_DIR, f"financial_metrics_{start_date_ai}_{end_date_ai}.md")):
        with open(os.path.join(CACHE_DIR, f"financial_metrics_{start_date_ai}_{end_date_ai}.md"), "w") as f:
            financial_metrics = summarize_text("financial_metrics", financial_metrics)
            st.subheader("Financial Metrics")
            st.markdown(financial_metrics)
            f.write(financial_metrics)
        
    else:
        with open(os.path.join(CACHE_DIR, f"financial_metrics_{start_date_ai}_{end_date_ai}.md"), "r") as f:
            financial_metrics = f.read()
            st.subheader("Financial Metrics")
            st.markdown(financial_metrics)
        
    if not os.path.exists(os.path.join(CACHE_DIR, f"recommendations_{start_date_ai}_{end_date_ai}.md")):
        with open(os.path.join(CACHE_DIR, f"recommendations_{start_date_ai}_{end_date_ai}.md"), "w") as f:
            recommendations = summarize_text("recommendations", recommendations)
            st.subheader("Recommendations")
            st.markdown(recommendations)
            f.write(recommendations)

    else:
        with open(os.path.join(CACHE_DIR, f"recommendations_{start_date_ai}_{end_date_ai}.md"), "r") as f:
            recommendations = f.read()
            st.subheader("Recommendations")
            st.markdown(recommendations)