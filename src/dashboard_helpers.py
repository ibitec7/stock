
import os
import logging
import pandas as pd
import json
import traceback
import subprocess
import polars as pl
from datetime import datetime
import logging

LOGS_DIR = "../logs"

if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

if os.path.exists(os.path.join(LOGS_DIR, 'dashboard.log')):
    os.remove(os.path.join(LOGS_DIR, 'dashboard.log'))

logging.basicConfig(filename=os.path.join(LOGS_DIR, 'dashboard.log'),
                    level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

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

    headlines_list = {}

    articles_list = {}

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

    article_dates = [datetime.strptime(date, "%Y-%m-%DT%H:%M:%S") for date in articles_list.keys()]
    headline_dates = [datetime.strptime(date, "%Y-%m-%DT%H:%M:%S") for date in headlines_list.keys()]

    article_dates = sorted(article_dates)
    headline_dates = sorted(headline_dates)

    article_dates = [date.isoformat() for date in article_dates]
    headline_dates = [date.isoformat() for date in headline_dates]

    articles_sorted = [articles_list[date] for date in article_dates]
    headlines_sorted = [headlines_list[date] for date in headline_dates]

    articles_list = {}
    headlines_sorted = {}

    for dates, article in zip(article_dates, articles_sorted):
        articles_list[date] = article

    for dates, headline in zip(headline_dates, headlines_sorted):
        headlines_sorted[date] = headline

    master_news["articles"] = articles_list
    master_news["headlines"] = headlines_list

    assert len(master_news["headlines"].keys()) != 0 

    logging.info("Caching news data")

    with open(os.path.join(CACHE_DIR, 'master_news.json'), 'w') as f:
        json.dump(master_news, f)

    logging.info("All files logged successfully")

    return master_news

def source_indicators(period, CACHE_DIR, timeframe=None) -> pl.DataFrame:
    

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

        if os.path.exists(os.path.join(CACHE_DIR, f"NVDA_{period}_{timeframe}.csv")):
            logging.info("Loading indicators data from cache")
            indicators = pl.read_csv(os.path.join(CACHE_DIR, f"NVDA_{period}_{timeframe}.csv"))
            
            indicators.columns = [col.lower() for col in indicators.columns]

            indicators = indicators.with_columns(
                pl.col("date").str.to_datetime().alias("date"),
            )

            indicators = indicators.fill_null(0)

            return indicators

        else:

            logging.info("Indicators data not found, generating new data")

            subprocess.run(["indicators", "NVDA", "-p", period, "-t", timeframe, "-o", f"NVDA_{period}_{timeframe}.csv", "-d", CACHE_DIR])

            if os.path.exists(os.path.join(CACHE_DIR, "N")):
                subprocess.run(["mv", os.path.join(CACHE_DIR, "N"), os.path.join(CACHE_DIR, f"NVDA_{period}_{timeframe}.csv")])

    except Exception as e:
        logging.error(f"Error sourcing indicators data: {e}")
        traceback.print_exc()

    try:

        indicators = pl.read_csv(os.path.join(CACHE_DIR, f"NVDA_{period}_{timeframe}.csv"))

        indicators.columns = [col.lower() for col in indicators.columns]

        indicators = indicators.with_columns(
            pl.col("date").str.to_datetime().alias("date"),
            pl.col("returns").fill_null(0).alias("returns")
        )

        return indicators

    except Exception as e:
        logging.error(f"Error loading indicators data: {e}")
        traceback.print_exc()

    logging.info("Indicators data loaded successfully")