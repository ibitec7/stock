import os
import json
from dateutil import parser
from datetime import datetime
import re
import logging


def contains_keywords(text, keywords):
    lower_text = text.lower()
    for keyword in keywords:
        if keyword.lower() in lower_text:
            return True
    return False

def contains_error(text, phrase):
    return phrase.lower() in text.lower()


def generate_url_suffix(title):

    title = ' '.join(title.split()[:5])
    title = re.sub(r'[^a-zA-Z0-9\s]', '', title)

    formatted_title = title.strip().lower().replace(" ", "-").replace("?", "")
    return f"{formatted_title}"

def find_best_match(article_index, articles, used_indices):
    """Find the best matching article based on search query, excluding used indices."""
    search_query = generate_url_suffix(articles[article_index]["title"][:50])
    best_match_index = None
    
    for i, other_article in enumerate(articles):
        if i != article_index and i not in used_indices and search_query in other_article["link"]:
            best_match_index = i
            break  # Stop after finding the first match

    return best_match_index

def swap_response(articles, swaps):
    """Swap responses between articles based on provided swaps."""
    for a, b in swaps:
        # Check if indices are valid
        if 0 <= a < len(articles) and 0 <= b < len(articles):
            # Swap link and response
            articles[a]["link"], articles[b]["link"] = articles[b]["link"], articles[a]["link"]
            articles[a]["response"], articles[b]["response"] = articles[b]["response"], articles[a]["response"]

            # Swap ai_analysis if it exists
            if "ai_analysis" in articles[a] and "ai_analysis" in articles[b]:
                articles[a]["ai_analysis"], articles[b]["ai_analysis"] = articles[b]["ai_analysis"], articles[a]["ai_analysis"]
            elif "ai_analysis" in articles[a]:
                # Handle case where only article a has ai_analysis
                articles[b]["ai_analysis"] = articles[a].pop("ai_analysis")
            elif "ai_analysis" in articles[b]:
                # Handle case where only article b has ai_analysis
                articles[a]["ai_analysis"] = articles[b].pop("ai_analysis")
        else:
            logging.error(f"Invalid swap indices: {a}, {b}")
    return articles


if __name__ == "__main__":
    keywords = ["NVIDIA", "data", "AI", "GPU", "NVDA", "Graphics", "Semiconductor"]
    error_message = "Error fetching content: HTTP"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    logging.info("Starting cleaning process...")

    root_dir = "/home/ibrahim/stock/data"
    years = ["2025", "2024", "2023", "2022", "2021", "2020"]

    total = 0
    total_swaps = 0  # Initialize swap counter

    for year in years:
        for path in os.listdir(os.path.join(root_dir, year)):
            with open(os.path.join(root_dir, year, path), "r") as f:
                logging.info(f"Cleaning file: {path}")

                data = json.load(f)

            cleaned_headlines = [headline for headline in data["headlines"] if contains_keywords(headline["headline"], keywords)]
            cleaned_article = [article for article in data["articles"] if contains_keywords(article["response"], keywords) and contains_keywords(article["title"], keywords) and not contains_error(article["response"], error_message)]

            data["headlines"] = cleaned_headlines

            data["articles"] = cleaned_article
            data["totalResults"] = len(data["articles"])
            total += len(data["articles"])

            # Cleaning up any jumbled articles
            
            swaps = []
            used_indices = set()  # Keep track of used indices
            for i, article in enumerate(data["articles"]):
                if i not in used_indices:  # Skip if already used
                    search_query = generate_url_suffix(article["title"][:50])
                    if search_query not in article["link"]:
                        best_match_index = find_best_match(i, data["articles"], used_indices)
                        if best_match_index is not None:
                            swaps.append((i, best_match_index))
                            logging.info(f"Found response for article {i} in {best_match_index}")
                            used_indices.add(i)
                            used_indices.add(best_match_index)

            data["articles"] = swap_response(data["articles"], swaps)
            total_swaps += len(swaps)  # Increment swap counter

            with open(os.path.join(root_dir, year, path), "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)

    logging.info(f"Total clean articles: {total}")
    logging.info(f"Total swaps performed: {total_swaps}")  # Log total swaps
