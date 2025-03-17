import click
from src import news
import logging
import os

@click.command()
@click.argument("ticker")
@click.argument("api_key")
@click.option("--from", default=None, help="Start date of news")
@click.option("--to", default=None, help="End date of news")
@click.option("--limit", default=50, help="Limit the number of news maximum is 50")
@click.option("--output", "-o", default="news.json", help="Output path in JSON format")
@click.option("--logs", default="news.log", help="Logging file path")

def main(ticker, api_key, from_date, to_date, limit, output_path, log_path):
    logger = logging.getLogger()

    if os.path.exists(log_path) == False:
        open(log_path, "a").close()

    logging.basicConfig(filename=log_path, encoding="utf-8", level=logging.DEBUG)

    news.main(ticker, from_date, to_date, limit, output_path, api_key, log_path)

if __name__ == "__main__":
    main()