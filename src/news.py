from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
import httpx
import asyncio
from bs4 import BeautifulSoup
from pygooglenews import GoogleNews
import os
import logging
import time
from urllib.parse import quote, urlparse
import json

OUTPUT_PATH = "/home/ibrahim/stock/data/historical.json"
KEY_PATH = "/home/ibrahim/stock/keys.json"

async def get_decoding_params(gn_art_id, client):
    response = await client.get(f"https://news.google.com/rss/articles/{gn_art_id}", follow_redirects=True)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "lxml")
    div = soup.select_one("c-wiz > div")
    return {
        "signature": div.get("data-n-a-sg"),
        "timestamp": div.get("data-n-a-ts"),
        "gn_art_id": gn_art_id,
    }

async def decode_urls(articles, client):
    articles_reqs = [
        [
            "Fbv4je",
            f'["garturlreq",[["X","X",["X","X"],null,null,1,1,"US:en",null,1,null,null,null,null,null,0,1],"X","X",1,[1,1,1],1,1,null,0,0,null,0],"{art["gn_art_id"]}",{art["timestamp"]},"{art["signature"]}"]',
        ]
        for art in articles
    ]
    payload = f"f.req={quote(json.dumps([articles_reqs]))}"
    headers = {"content-type": "application/x-www-form-urlencoded;charset=UTF-8", "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0"}
    response = await client.post("https://news.google.com/_/DotsSplashUi/data/batchexecute", headers=headers, data=payload)
    time.sleep(5)
    response.raise_for_status()
    return [json.loads(res[2])[1] for res in json.loads(response.text.split("\n\n")[1])[:-2]]

async def decode(encoded_urls):
    async with httpx.AsyncClient() as client:
        tasks = [get_decoding_params(urlparse(url).path.split("/")[-1], client) for url in encoded_urls]
        articles_params = await asyncio.gather(*tasks)
        decoded_urls = await decode_urls(articles_params, client)
        print(decoded_urls)
        return decoded_urls

def decode_async(urls):
    return asyncio.run(decode(urls))

async def get_response(client: httpx.AsyncClient, url: tuple):
    try:
        response = await client.get(url[1])

        if response.status_code == 200:
            logging.debug(f"Fetched {url[1]} successfully!")
        elif response.status_code == 302:
            try:
                location = response.headers.get("location", "")
                
                if location.startswith('/'):
                    parsed_url = urlparse(str(url[1]))
                    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
                    location = f"{base_url}{location}"
                    
                logging.debug(f"Following redirect from {url[1]} to {location}")
                
                redirect_response = await client.get(location)
                if redirect_response.status_code == 200:
                    logging.debug(f"Successfully followed redirect to {location}")
                    response = redirect_response
                else:
                    logging.warning(f"Redirect to {location} failed with status {redirect_response.status_code}")
            except (ValueError, httpx.RequestError) as e:
                logging.warning(f"Error following redirect from {url[1]}: {str(e)}")
        else:
            logging.warning(f"Bad response from {url[1]} with status {response.status_code}")
    except httpx.RequestError as e:
        logging.error(f"Request error for {url[1]}: {str(e)}")

        response = httpx.Response(status_code=0, request=httpx.Request("GET", url[1]))
        response.error_type = type(e).__name__
        response.error_msg = str(e)
    except Exception as e:
        logging.error(f"Unexpected error processing {url[1]}: {str(e)}")

        response = httpx.Response(status_code=0, request=httpx.Request("GET", url[1]))
        response.error_type = type(e).__name__
        response.error_msg = str(e)
        
    response_tuple = (url[0], response)
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
    return asyncio.run(fetch_urls(urls))

def get_news(api_key: str, ticker: str, date_from=None, date_to=None, limit: int = 50) -> httpx.Response:
    logging.info("Fetching news from Google News API")
    # if date_from == None:
    #     av_url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&limit={limit}&sort=RELEVANCE&apikey={api_key}"
    #     news_url = f"https://newsapi.org/v2/everything?q={ticker}&sortBy=relevancy&apiKey={api_key}"
    # else:
    #     av_url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&time_from={date_from}&time_to={date_to}&limit={limit}&apikey={api_key}"
    #     news_url = f"https://newsapi.org/v2/everything?q={ticker}&from={date_from}&to={date_to}&sortBy=relevancy&apiKey={api_key}"

    # response = httpx.get(news_url)

    gn = GoogleNews()
    nvda_news = gn.search(ticker, from_=date_from, to_=date_to)
    nvidia_news = gn.search("Nvidia", from_=date_from, to_=date_to)

    all_news = {
        "title": f"News for {ticker} from {date_from} to {date_to}",
        "totalResults": 0,
        "headlines": [],
        "articles": []
    }

    for news in nvda_news["entries"]:
        all_news["articles"].append(news)
        all_news["headlines"].append(news["title"])

    for news in nvidia_news["entries"]:
        if news["title"] not in all_news["headlines"]:
            all_news["articles"].append(news)

    encoded_urls = [article["link"] for article in all_news["articles"]]
    decoded_urls = decode_async(encoded_urls)
    
    for i, article in enumerate(all_news["articles"]):
        article["link"] = decoded_urls[i]
        

    all_news["totalResults"] = len(all_news["articles"])

    return all_news


### MAIN FUNCTION ###

def main(ticker: str, from_date: str, to_date: str, limit: int, output_path: str, key_path: str, log_path: str) -> int:
    logging.basicConfig(
        filename=log_path,
        filemode='a',
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    with open(key_path, "r") as file:
        keys = json.load(file)

    if os.path.exists(output_path) == False:
        data = get_news(api_key=keys["NEWS_API_KEY"], ticker=ticker, date_from=from_date, date_to=to_date, limit=limit)
    else:
        logging.info("News file already exists. Moving to the next step")
        return 1

    if "Information" in data.keys():
        logging.warning("No news found for the given date range")
        print(data)
        return -1

    urls = [(i,feed["link"]) for i, feed in enumerate(data["articles"])]

    logging.info("fetching the news from the URLs")
    responses = run_async(urls)

    bad_urls = []

    logging.info("Fetching detailed news from URLs")

    for i, response in enumerate(responses):
        try:
            if hasattr(response[1], 'error_type'):
                logging.warning(f"Skipping article at index {i} due to error: {response[1].error_type}: {response[1].error_msg}")
                data["articles"][i]["response"] = f"Error fetching content: {response[1].error_msg}"
                continue
                
            if response[1].status_code == 200:
                soup = BeautifulSoup(response[1].content, "html.parser")
                data["articles"][i]["response"] = soup.text
            elif response[1].status_code == 429:
                bad_urls.append((i, str(response[1].url)))
            else:
                logging.warning(f"Skipping article at index {i} due to status code {response[1].status_code}")
                data["articles"][i]["response"] = f"Error fetching content: HTTP {response[1].status_code}"
        except Exception as e:
            logging.error(f"Error processing response for index {i}: {str(e)}")
            data["articles"][i]["response"] = f"Error processing content: {str(e)}"

    if len(bad_urls) == 0:
        logging.info("No bad URLs found")
    else:
        logging.info("Some bad urls found scraping from selenium")

        soups = scrape_selenium(bad_urls)

        for i, soup in soups:
            data["articles"][i]["response"] = soup.text

    logging.info("Saving the json file now")

    with open(output_path, "w") as file:
        json.dump(data, file, indent=4)

    logging.info("Done saving the json file")

    return 1

if __name__ == "__main__":
    years = ["2023"]
    months = ["01"]

    root_path = "/home/ibrahim/stock/new_data"
    if os.path.exists(root_path) == False:
        os.makedirs(root_path)

    for year in years:
        for i, month in enumerate(months):
            start_date = f"{year}-{month}-01"

            if month in ["04", "06", "09", "11"]:
                end_date = f"{year}-{month}-30"
            elif month == "02":
                if int(year) % 4 == 0:
                    end_date = f"{year}-{month}-29"
                else:
                    end_date = f"{year}-{month}-28"
            else:
                end_date = f"{year}-{month}-31"

            dir_path = os.path.join(root_path, f"{year}")
            if os.path.exists(dir_path) == False:
                os.makedirs(dir_path)

            file_path = os.path.join(dir_path, f"news_{year}_{month}.json")

            status_code = main(
                ticker="NVDA",
                from_date=start_date,
                to_date=end_date,
                limit=50,
                output_path=file_path,
                key_path="/home/ibrahim/stock/keys.json",
                log_path="/home/ibrahim/stock/logs/news.log"
            )

            if status_code == -1:
                print(f"No news found for {year}-{month}")