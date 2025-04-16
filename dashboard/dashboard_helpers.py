
import os
import logging
import pandas as pd
import json
import traceback
import subprocess
import polars as pl
from datetime import datetime
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import streamlit as st
import cupy as cp

LOGS_DIR = "../logs"
CACHE_DIR = "../cache"
DATA_DIR = "../data"
YEARS = ["2020", "2021", "2022", "2023", "2024", "2025"]


if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

if os.path.exists(os.path.join(LOGS_DIR, 'dashboard.log')):
    os.remove(os.path.join(LOGS_DIR, 'dashboard.log'))

logging.basicConfig(filename=os.path.join(LOGS_DIR, 'dashboard.log'),
                    level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')


def get_embeddings() -> cp.ndarray:
    embeddings = cp.array([])

    for year in YEARS:
        PATHS = os.listdir(os.path.join(DATA_DIR, year))

        for path in PATHS:

            with open(os.path.join(DATA_DIR, year, path), "r") as file:
                data = json.load(file)

                for article in data["articles"]:

                    if embeddings.shape[0] == 0:
                        embeddings = cp.array(article["embedding"])
                    else:
                        embeddings = cp.vstack((embeddings, cp.array(article["embedding"])))

    return embeddings

def load_master_news():
    if os.path.exists(os.path.join(CACHE_DIR, "master_news.parquet")):
        df = pl.read_parquet(os.path.join(CACHE_DIR, "master_news.parquet"))

        with open(os.path.join(CACHE_DIR, "ai_analysis.json"), 'r') as f:
            ai_analysis = json.load(f)

        return df, ai_analysis

    if not os.path.exists(os.path.join(CACHE_DIR, "master_news.parquet")):
        logging.info("Master news data not found, loading from data directory")

        embeddings = get_embeddings()

        date_list = []

        article_list = []

        headlines_list = []

        title_list = []

        sentiment = cp.array([])
        sia_scores = cp.array([])

        ai_analyses = []

        headline_sentiment = cp.array([])

        for year in YEARS:
            PATHS = os.listdir(os.path.join(DATA_DIR, year))

            for path in PATHS:

                with open(os.path.join(DATA_DIR, year, path), "r") as file:
                    data = json.load(file)

                    for headline, article in zip(data["headlines"], data["articles"]):
                        if "published_parsed" in article.keys():
                            date = datetime(year = article["published_parsed"][0], month = article["published_parsed"][1], day = article["published_parsed"][2], hour = article["published_parsed"][3], minute = article["published_parsed"][4], second = article["published_parsed"][5])
                        else:
                            date = None

                        date_list.append(date)
                        article_list.append(article["response"])
                        title_list.append(article["title"])
                        headlines_list.append(headline["headline"])

                        if "ai_analysis" in article.keys():
                            ai_analyses.append(article["ai_analysis"])
                        else:
                            ai_analyses.append(None)

                        if sia_scores.shape[0] == 0:
                            sia_scores = cp.array([article["sentiment_scores"]["pos"], article["sentiment_scores"]["neu"], article["sentiment_scores"]["neg"], article["sentiment_scores"]["compound"]])
                        else:
                            sia_scores = cp.vstack((sia_scores, cp.array([article["sentiment_scores"]["pos"], article["sentiment_scores"]["neu"], article["sentiment_scores"]["neg"], article["sentiment_scores"]["compound"]])))

                        if sentiment.shape[0] == 0:
                            sentiment = cp.array([article["finbert_sentiment"]["positive"], article["finbert_sentiment"]["neutral"], article["finbert_sentiment"]["negative"]])
                        else:
                            sentiment = cp.vstack((sentiment, cp.array([article["finbert_sentiment"]["positive"], article["finbert_sentiment"]["neutral"], article["finbert_sentiment"]["negative"]])))

                        if headline_sentiment.shape[0] == 0:
                            headline_sentiment = cp.array([headline["finbert_sentiment"]["positive"], headline["finbert_sentiment"]["neutral"], headline["finbert_sentiment"]["negative"]])
                        else:
                            headline_sentiment = cp.vstack((headline_sentiment, cp.array([headline["finbert_sentiment"]["positive"], headline["finbert_sentiment"]["neutral"], headline["finbert_sentiment"]["negative"]])))

        date_series = pl.Series("date", date_list)
        headlines_list = pl.Series("headlines", headlines_list)
        article_list = pl.Series("article", article_list)
        title_list = pl.Series("title", title_list)

        df = pl.DataFrame({
            "date": date_series,
            "headlines": headlines_list,
            "article": article_list,
            "title": title_list,
            "embeddings": embeddings.get(),
            "headline_positive_sentiment": headline_sentiment[:, 0].get(),
            "headline_neutral_sentiment": headline_sentiment[:, 1].get(),
            "headline_negative_sentiment": headline_sentiment[:, 2].get(),
            "finbert_positive_sentiment": sentiment[:, 0].get(),
            "finbert_neutral_sentiment": sentiment[:, 1].get(),
            "finbert_negative_sentiment": sentiment[:, 2].get(),
            "sia_positive_sentiment": sia_scores[:, 0].get(),
            "sia_neutral_sentiment": sia_scores[:, 1].get(),
            "sia_negative_sentiment": sia_scores[:, 2].get(),
            "sia_compound_sentiment": sia_scores[:, 3].get(),
        })

        df = df.with_columns(
            pl.col("date").sort()
        )

        df = df.with_columns(
            pl.struct(["finbert_positive_sentiment", "finbert_neutral_sentiment", "finbert_negative_sentiment"])
            .map_elements(lambda x: sentiment_classifier(x["finbert_positive_sentiment"], x["finbert_neutral_sentiment"], x["finbert_negative_sentiment"]))
                .alias("finbert_sentiment"),

            pl.struct(["sia_positive_sentiment", "sia_neutral_sentiment", "sia_negative_sentiment"])
            .map_elements(lambda x: sentiment_classifier(x["sia_positive_sentiment"], x["sia_neutral_sentiment"], x["sia_negative_sentiment"]))
                .alias("sia_sentiment"),

            pl.struct(["headline_positive_sentiment", "headline_neutral_sentiment", "headline_negative_sentiment"])
            .map_elements(lambda x: sentiment_classifier(x["headline_positive_sentiment"], x["headline_neutral_sentiment"], x["headline_negative_sentiment"]))
                .alias("headline_sentiment"),
        )

        df.write_parquet(os.path.join(CACHE_DIR, "master_news.parquet"))
        logging.info("Master news data loaded successfully")

        ai_analysis_final = {"date": [], "ai_analysis": []}

        for date in date_list:
            ai_analysis_final["date"].append(date.isoformat())
            ai_analysis_final["ai_analysis"].append(ai_analyses[date_list.index(date)])

        with open(os.path.join(CACHE_DIR, "ai_analysis.json"), 'w') as f:
            json.dump(ai_analysis_final, f)

        return (df, ai_analysis_final)

def load_sentiment(DATA_DIR, CACHE_DIR):
    if os.path.exists(os.path.join(CACHE_DIR, "master_news.json")):
        logging.info("Loading sentiment data from cache")
        with open(os.path.join(CACHE_DIR, "master_news.json"), 'r') as f:
            master_news = json.load(f)
            return master_news

    all_news_dir = [directory for directory in os.listdir(DATA_DIR)\
                     if os.path.isdir(os.path.join(DATA_DIR, directory))\
                          and not (directory == "events" or directory == "processed")]

    master_news = {
        "title": "Master NVIDIA news data",
        "totalResults": 0,
    }

    headlines_list = dict()

    articles_list = dict()

    for dir in all_news_dir:
        paths = os.listdir(os.path.join(DATA_DIR, dir))

        logging.info("Starting to read all the files")
        
        for path in paths:
            
            with open(os.path.join(DATA_DIR, dir, path), 'r') as f:
                try:
                    data = json.load(f)

                    logging.info(f"Loaded file {path} in directory {dir}")

                    master_news["totalResults"] += data["totalResults"]

                    for article, headline in zip(data["articles"], data["headlines"]):

                        date = article["published_parsed"]

                        date_time = datetime(date[0], date[1], date[2], date[3], date[4], date[5]).isoformat()

                        articles_list[date_time] = article

                        headlines_list[date_time] = headline

                    logging.info(f"File {path} in directory {dir} loaded successfully")

                except Exception as e:
                    print(f"Error loading file {path} in directory {dir}: {e}")
                    traceback.print_exc()

    assert len(articles_list.keys()) != 0
    assert len(headlines_list.keys()) != 0

    article_dates = [datetime.strptime(date, "%Y-%m-%dT%H:%M:%S") for date in articles_list.keys()]
    headline_dates = [datetime.strptime(date, "%Y-%m-%dT%H:%M:%S") for date in headlines_list.keys()]

    article_dates = sorted(article_dates)
    headline_dates = sorted(headline_dates)

    article_dates = [date.isoformat() for date in article_dates]
    headline_dates = [date.isoformat() for date in headline_dates]

    articles_sorted = [articles_list[date] for date in article_dates]
    headlines_sorted = [headlines_list[date] for date in headline_dates]

    articles_list = {}
    headlines_sorted = {}

    for date, article in zip(article_dates, articles_sorted):
        articles_list[date] = article

    for date, headline in zip(headline_dates, headlines_sorted):
        headlines_sorted[date] = headline

    master_news["articles"] = articles_list
    master_news["headlines"] = headlines_list

    assert len(master_news["headlines"].keys()) != 0 

    logging.info("Caching news data")

    with open(os.path.join(CACHE_DIR, 'master_news.json'), 'w') as f:
        json.dump(master_news, f)

    logging.info("All files logged successfully")

    return master_news

def source_indicators(period, timeframe=None, indicators_config=None, ticker="NVDA") -> pl.DataFrame:
    

    if timeframe is None:
        default_tf = {
            "ytd": "1d",
            "1y": "1d",
            "2y": "1wk",
            "5y": "1mo",
            "10y": "1mo",
            "max": "3mo"
        }

        timeframe = default_tf[period]

    logging.info("Sourcing indicators data")

    try:

        if os.path.exists(os.path.join(CACHE_DIR, f"{ticker}_{period}_{timeframe}.parquet")) and indicators_config is None:
            logging.info("Loading indicators data from cache")
            indicators = pl.read_parquet(os.path.join(CACHE_DIR, f"{ticker}_{period}_{timeframe}.parquet"))
            
            indicators.columns = [col.lower() for col in indicators.columns]

            indicators = indicators.fill_null(0)

            return indicators

        else:

            logging.info("Indicators data not found, generating new data")

            if indicators_config is not None:
               subprocess.run(["indicators", ticker, "-p", period, "-t", timeframe, "-o", os.path.join(CACHE_DIR, f"{ticker}_{period}_{timeframe}_custom.parquet"), "-c", os.path.join(CACHE_DIR, indicators_config)])

            else:
                subprocess.run(["indicators", ticker, "-p", period, "-t", timeframe, "-o", os.path.join(CACHE_DIR, f"{ticker}_{period}_{timeframe}.parquet")])

    except Exception as e:
        logging.error(f"Error sourcing indicators data: {e}")
        traceback.print_exc()

    try:
            
        if indicators_config is None:
            indicators = pl.read_parquet(os.path.join(CACHE_DIR, f"{ticker}_{period}_{timeframe}.parquet"))

            indicators.columns = [col.lower() for col in indicators.columns]

            indicators = indicators.with_columns(
                pl.col("returns").fill_null(0).alias("returns")
            )

            return indicators

        else:
            indicators = pl.read_parquet(os.path.join(CACHE_DIR, f"{ticker}_{period}_{timeframe}_custom.parquet"))

            indicators.columns = [col.lower() for col in indicators.columns]

            indicators = indicators.with_columns(
                pl.col("returns").fill_null(0).alias("returns")
            )

            return indicators

    except Exception as e:
        logging.error(f"Error loading indicators data: {e}")
        traceback.print_exc()

    logging.info("Indicators data loaded successfully")

def get_filtered_headlines_df(filtered_dates, data) -> pl.DataFrame:

    finbert_pos_sentiment = [article["finbert_sentiment"]["positive"] for date in filtered_dates for article in data["headlines"][date]]
    finbert_neg_sentiment = [article["finbert_sentiment"]["negative"] for date in filtered_dates for article in data["headlines"][date]]
    finbert_neutral_sentiment = [article["finbert_sentiment"]["neutral"] for date in filtered_dates for article in data["headlines"][date]]

    headlines = [article["headline"] for date in filtered_dates for article in data["headlines"][date]]


    filtered_df = pl.DataFrame({
        "date": filtered_dates,
        "headlines": headlines,
        "positive_sentiment": finbert_pos_sentiment,
        "negative_sentiment": finbert_neg_sentiment,
        "neutral_sentiment": finbert_neutral_sentiment,
    })

    return filtered_df

def sentiment_classifier(pos, neu, neg) -> str:
    if pos > neu and pos > neg:
        return str("positive")
    elif neu > pos and neu > neg:
        return str("neutral")
    else:
        return str("negative")

def get_filtered_articles_df(filtered_dates, data) -> tuple[pl.DataFrame, list, list]:
        
    filtered_dates = [date.isoformat() for date in filtered_dates]

    finbert_pos_sentiment = [data["articles"][date]["finbert_sentiment"]["positive"] for date in filtered_dates]
    finbert_neg_sentiment = [data["articles"][date]["finbert_sentiment"]["negative"] for date in filtered_dates]
    finbert_neutral_sentiment = [data["articles"][date]["finbert_sentiment"]["neutral"] for date in filtered_dates]

    sia_positive_sentiment = [data["articles"][date]["sentiment_scores"]["pos"] for date in filtered_dates]
    sia_negative_sentiment = [data["articles"][date]["sentiment_scores"]["neg"] for date in filtered_dates]
    sia_neutral_sentiment = [data["articles"][date]["sentiment_scores"]["neu"] for date in filtered_dates]
    sia_compound_sentiment = [data["articles"][date]["sentiment_scores"]["compound"] for date in filtered_dates]

    embeddings = [data["articles"][date]["embedding"] for date in filtered_dates]

    titles = [data["articles"][date]["title"] for date in filtered_dates]

    filtered_df = pl.DataFrame({
        "date": filtered_dates,
        "finbert_positive_sentiment": finbert_pos_sentiment,
        "finbert_negative_sentiment": finbert_neg_sentiment,
        "finbert_neutral_sentiment": finbert_neutral_sentiment,
        "sia_positive_sentiment": sia_positive_sentiment,
        "sia_negative_sentiment": sia_negative_sentiment,
        "sia_neutral_sentiment": sia_neutral_sentiment,
        "sia_compound_sentiment": sia_compound_sentiment,
    })

    filtered_df = filtered_df.with_columns(
        pl.col("date").str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S").alias("date"),
    )

    filtered_df = filtered_df.with_columns(
        pl.struct(["finbert_positive_sentiment", "finbert_neutral_sentiment", "finbert_negative_sentiment"])
        .map_elements(lambda x: sentiment_classifier(x["finbert_positive_sentiment"], x["finbert_neutral_sentiment"], x["finbert_negative_sentiment"]))
            .alias("finbert_sentiment"),
    )

    filtered_df = filtered_df.with_columns(
        pl.struct(["sia_positive_sentiment", "sia_neutral_sentiment", "sia_negative_sentiment"])
        .map_elements(lambda x: sentiment_classifier(x["sia_positive_sentiment"], x["sia_neutral_sentiment"], x["sia_negative_sentiment"]))
            .alias("sia_sentiment"),
    )

    return (filtered_df, embeddings, titles)

def get_sentiments(filtered_df, time_frame, sentiment_type="Articles") -> pl.DataFrame:

    if sentiment_type == "Articles":
        key_name = "finbert_sentiment"
    elif sentiment_type == "Headlines":
        key_name = "headline_sentiment"

    sentiments = filtered_df.group_by_dynamic("date", every=time_frame).agg(
            pl.col("finbert_sentiment").value_counts(),
            pl.col("headline_sentiment").value_counts(),
            pl.col("sia_sentiment").value_counts(),
        )
    
    struct_list = sentiments[key_name].to_list()

    all_values = {
        "positive": [],
        "neutral": [],
        "negative": []
    }

    for struct in struct_list:
        new_vals = {
            "positive": 0,
            "neutral": 0,
            "negative": 0
        }
        for val in struct:
            if val[key_name] == "positive":
                new_vals["positive"] = val["count"]
            elif val[key_name] == "neutral":
                new_vals["neutral"] = val["count"]
            elif val[key_name] == "negative":
                new_vals["negative"] = val["count"]
        
        all_values["positive"].append(new_vals["positive"])
        all_values["neutral"].append(new_vals["neutral"])
        all_values["negative"].append(new_vals["negative"])

    sentiments = sentiments.with_columns(
        pl.Series("positive", all_values["positive"]),
        pl.Series("neutral", all_values["neutral"]),
        pl.Series("negative", all_values["negative"])
    )

    return sentiments


# def get_corpus(filtered_dates, data):

#     titles = [article["title"] for date in filtered_dates for article in data["articles"][date]]

#     responses = [article["response"] for date in filtered_dates for article in data["articles"][date]]

#     corpus = str("")

#     for response in responses:
#         corpus += response["response"] + " "

#     with st.progress()