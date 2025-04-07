from nltk.sentiment.vader import SentimentIntensityAnalyzer
import json
import nltk
from transformers import BertTokenizer, BertForSequenceClassification
import logging
import os
from tqdm import tqdm
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()

device = "cuda" if torch.cuda.is_available() else "cpu"

finbert = BertForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")

ROOT_DIR = "/home/ibrahim/stock/data"

dirs = [os.path.join(ROOT_DIR, dir) for dir in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, dir))]
file_paths = [os.path.join(dir, file) for dir in dirs for file in os.listdir(dir) if file.endswith(".json")]

for file in file_paths:
    logging.info(f"Processing file: {file}")
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    headline_obj = {}

    for i, headline in enumerate(data["headlines"]):
        if isinstance(headline, str):
            headline_obj = {
                "headline": headline,
                "finbert_sentiment": None
            }
            data["headlines"][i] = headline_obj


    logging.info(f"Loaded {len(data['articles'])} articles from {file}")

    for headline in tqdm(data["headlines"], desc="Processing headlines"):

        if headline["finbert_sentiment"] is None:
            inputs = tokenizer(headline["headline"], return_tensors="pt", truncation=True, padding=True)
            outputs = finbert(**inputs)
            logits = outputs.logits
            probabilities = logits.softmax(dim=-1).tolist()[0]

            headline["finbert_sentiment"] = {
                "positive": probabilities[0],
                "neutral": probabilities[2],
                "negative": probabilities[1]
            }

    for i, article in enumerate(tqdm(data["articles"], desc="Processing articles")):

        if not article.get("sentiment_scores"):
            sentiment = sia.polarity_scores(text)
            article["sentiment_scores"] = sentiment

        if article.get("finbert_sentiment") and article.get("embedding"):
            continue

        else:
            text = article["response"]

            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
            outputs = finbert(**inputs, output_hidden_states=True)

            if not article.get("finbert_sentiment"):
                logits = outputs.logits
                probabilities = logits.softmax(dim=-1).tolist()[0]

                article["finbert_sentiment"] = {
                    "positive": probabilities[0],
                    "neutral": probabilities[2],
                    "negative": probabilities[1]
                }

            if not article.get("embedding"):
                article["embedding"] = outputs.hidden_states[-1].detach().mean(dim=1).tolist()

    logging.info(f"Calculated sentiment scores and embeddings for {len(data['articles'])} articles")

    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    logging.info(f"Saved sentiment scores to {file}")

