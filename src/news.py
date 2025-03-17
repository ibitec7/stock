import yfinance as yf
from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
import httpx
import asyncio
from bs4 import BeautifulSoup
import os
import logging
import json

OUTPUT_PATH = "/home/ibrahim/stock/data/historical.json"
KEY_PATH = "/home/ibrahim/stock/keys.json"

async def get_response(client: httpx.AsyncClient, url: tuple):
    response = await client.get(url[1])
    response_tuple = (url[0], response)

    if response.status_code == 200:
        logging.debug(f"Fetched {url[1]} successfully!")
    else:
        logging.warning(f"Bad response from {url[1]}")

    return response_tuple

def get_response_sync(client: httpx.Client, url: tuple):
    response = client.get(url[1])
    response_tuple = (url[0], response)

    if response.status_code == 200:
        logging.debug(f"Fetched {url[1]} successfully!")
    else:
        logging.warning(f"Bad response from {url[1]}")

    return response_tuple

def scrape_selenium(urls: list):
    options = Options()
    options.add_argument("--headless")

    logging.info("Created Selenium web driver")

    driver = Chrome(options=options)

    soups = []

    logging.info("Fetching urls using selenium")

    for i, url in urls:
        driver.get(url)
        html = driver.page_source
        soups.append((i, BeautifulSoup(html, "html.parser")))

    return soups

async def fetch_urls(urls: list) -> list:

    logging.info("Configuring async client")

    async with httpx.AsyncClient() as client:
        tasks = [asyncio.create_task(get_response(client, url)) for url in urls]

        logging.info("Created tasks now gathering them")
        responses = await asyncio.gather(*tasks)

    return responses

def run_async(urls: list) -> list:
    logging.info("Fetching URL responses through async client")
    return asyncio.run(main(urls))

def get_news(api_key: str, ticker: str, date_from: str, date_to, limit: int = 50) -> httpx.Response:
    logging.info("Fetching news from Alpha Vantage API")
    if date_from == None:
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&limit={limit}&sort=RELEVANCE&apikey={api_key}"
    else:
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&time_from={date_from}&time_to={date_to}&limit={limit}&apikey={api_key}"

    response = httpx.get(url)

    if response.status_code == 200:
        logging.info(f"Fetched {url[1]} successfully!")
    else:
        logging.warning(f"Bad response from {url[1]}")

    return response


### MAIN FUNCTION ###

def main(ticker: str, from_date: str, to_date: str, limit: int, output_path: str, key_path: str, log_path: str):
    
    with open(key_path, "r") as file:
        keys = json.load(file)

    if os.path.exists(output_path) == False:
        response = get_news(api_key=keys["ALPHA_VANTAGE_API_KEY"], ticker=ticker, date_from=from_date, limit=limit)
        data = response.json()
    
    else:
        logging.info("Reading from existing news file")
        with open(output_path, "r") as file:
            data = json.load(file)

    urls = [(i,feed["url"]) for i, feed in enumerate(data["feed"])]

    logging.info("fetching the news from the URLs")
    responses = run_async(urls)

    bad_urls = []

    logging.info("Fetching detailed news from URLs")

    for i, response in enumerate(responses):
        if response[1].status_code == 200:
            soup = BeautifulSoup(responses[i][1].content, "html.parser")
            data["feed"][i]["response"] = soup.text
        elif response[1].status_code == 429:
            bad_urls.append((i, str(response.url)))

    if len(bad_urls) == 0:
        logging.info("No bad URLs found")
    else:
        logging.info("Some bad urls found scraping from selenium")

        soups = scrape_selenium(bad_urls)

        for i, soup in soups:
            data["feed"][i]["response"] = soup.text

    logging.info("Saving the json file now")

    with open(output_path, "w") as file:
        json.dump(data, file, indent=4)

    logging.info("Done saving the json file")
