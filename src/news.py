# %%
import yfinance as yf
import polars as pl
import httpx
import asyncio
from bs4 import BeautifulSoup
import time

# %%
news = yf.Ticker("AAPL").news
# %%
news[0]["content"].keys()
# %%
async def get_response(client: httpx.AsyncClient, url: str) -> httpx.Response:
    response = await client.get(url)
    return response
# So basically I am interested in the ['title', 'description', 'summary', 'pubDate', 'canonicalUrl']
# %%
relevant_news = []
urls = []
for story in news:

    # if any(sub in story["content"]["title"] for sub in ["nvidia", "NVDA", "Nvidia", "NVIDIA"]):
        urls.append(story["content"]["canonicalUrl"]["url"])
        relevant_news.append(story)

# %%
async def main(urls: list):
    async with httpx.AsyncClient() as client:
        tasks = [asyncio.create_task(get_response(client, url)) for url in urls]
        responses = await asyncio.gather(*tasks)
    
    return responses

if __name__ == "__main__":
    responses = asyncio.run(main(urls))
    for i,response in enumerate(responses):
        print(response.url)

    print(responses)


