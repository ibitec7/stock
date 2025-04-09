import json
import os
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from tqdm import tqdm
import logging

def calculate_relevance(headline, title, vectorizer):
    matrix = vectorizer.fit_transform([headline, title])
    return cosine_similarity(matrix[0:1], matrix[1:2])[0][0]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logging.info("Starting corpus generation...")

ROOT_DIR = "/home/ibrahim/stock/data"

DIRS = os.listdir(ROOT_DIR)
DIRS.remove("events")
DIRS.remove("processed")

news_corpus = str("")

# for dir in DIRS:
#     # if os.path.exists(f"processed/processed_tokens_{dir}.json"):
#     #     logging.info(f"Processed file for {dir} already exists, skipping...")
#     #     continue

#     paths = []

#     for file in os.listdir(os.path.join(ROOT_DIR, dir)):
#         paths.append(os.path.join(ROOT_DIR, dir, file))

#     corpus = str("")

logging.info("Reading files...")

for dir in DIRS:

    PATHS = [os.path.join(ROOT_DIR, dir, file) for file in os.listdir(os.path.join(ROOT_DIR, dir)) if file.endswith(".json")]

    corpus = str("")
    high_event_corpus = str("")
    medium_event_corpus = str("")
    low_event_corpus = str("")

    for i, path in tqdm(enumerate(PATHS), total=len(PATHS), desc="Creating corpus", unit="file"):

        file_name = path.split("/")[-1]
        year = path.split("/")[-2]


        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

        except Exception as e:
            logging.error(f"Error reading file {path}: {e}")
            continue

        try:
            with open(os.path.join(ROOT_DIR, "events", year, file_name), "r", encoding="utf-8") as f:
                event_data = json.load(f)

        except Exception as e:
            logging.error(f"Error reading events file {path}: {e}")
            continue

        high_impact_headlines = [event["headline"] for event in event_data["high_impact_events"]]
        medium_impact_headlines = [event["headline"] for event in event_data["medium_impact_events"]]
        low_impact_headlines = [event["headline"] for event in event_data["low_impact_events"]]

        # Find the headlines in articles and add the response to the corpus

        vectorizer = TfidfVectorizer()

        for headline in high_impact_headlines:
            best_match_article = None
            best_relevance_score = -1

            for article in data["articles"]:
                relevance_score = calculate_relevance(headline, article["title"], vectorizer)
                if relevance_score > best_relevance_score:
                    best_match_article = article
                    best_relevance_score = relevance_score

            if best_match_article:
                high_event_corpus += best_match_article["response"] + "\n"

        for headline in medium_impact_headlines:
            best_match_article = None
            best_relevance_score = -1

            for article in data["articles"]:
                relevance_score = calculate_relevance(headline, article["title"], vectorizer)
                if relevance_score > best_relevance_score:
                    best_match_article = article
                    best_relevance_score = relevance_score

            if best_match_article:
                medium_event_corpus += best_match_article["response"] + "\n"

        for headline in low_impact_headlines:
            best_match_article = None
            best_relevance_score = -1

            for article in data["articles"]:
                relevance_score = calculate_relevance(headline, article["title"], vectorizer)
                if relevance_score > best_relevance_score:
                    best_match_article = article
                    best_relevance_score = relevance_score

            if best_match_article:
                low_event_corpus += best_match_article["response"] + "\n"

        for article in data["articles"]:
            corpus += article["response"] + "\n"

        corpus = corpus.lower()
        high_event_corpus = high_event_corpus.lower()
        medium_event_corpus = medium_event_corpus.lower()
        low_event_corpus = low_event_corpus.lower()

        corpus = corpus.translate(str.maketrans("", "", string.punctuation))
        high_event_corpus = high_event_corpus.translate(str.maketrans("", "", string.punctuation))
        medium_event_corpus = medium_event_corpus.translate(str.maketrans("", "", string.punctuation))
        low_event_corpus = low_event_corpus.translate(str.maketrans("", "", string.punctuation))

    logging.info("Tokenizing corpus")

    tokens = word_tokenize(corpus)
    high_event_tokens = word_tokenize(high_event_corpus)
    medium_event_tokens = word_tokenize(medium_event_corpus)
    low_event_tokens = word_tokenize(low_event_corpus)

    tokens_arr = []
    high_event_tokens_arr = []
    medium_event_tokens_arr = []
    low_event_tokens_arr = []

    for i in range(0, int(len(corpus)/50000)):
        tokens_arr.append(tokens[i*50000:(i+1)*50000])

    tokens_arr.append(tokens[(i+1)*50000:])

    for i in range(0, int(len(high_event_corpus)/50000)):
        high_event_tokens_arr.append(high_event_tokens[i*50000:(i+1)*50000])

    high_event_tokens_arr.append(high_event_tokens[(i+1)*50000:])

    for i in range(0, int(len(medium_event_corpus)/50000)):
        medium_event_tokens_arr.append(medium_event_tokens[i*50000:(i+1)*50000])

    medium_event_tokens_arr.append(medium_event_tokens[(i+1)*50000:])
    
    for i in range(0, int(len(low_event_corpus)/50000)):
        low_event_tokens_arr.append(low_event_tokens[i*50000:(i+1)*50000])

    low_event_tokens_arr.append(low_event_tokens[(i+1)*50000:])

    logging.info("Tokenizing finished")

    lemmatizer = WordNetLemmatizer()

    stop_words = set(stopwords.words("english")).difference(set(["up", "down"]))

    for i in tqdm(range(len(tokens_arr)), total=len(tokens_arr), desc="Processing chunks", unit="chunk"):
        tokens_arr[i] = [token.strip(string.punctuation) for token in tokens_arr[i]]

        tokens_arr[i] = [token for token in tokens_arr[i] if not token.isdigit()]
        
        tokens_arr[i] = [lemmatizer.lemmatize(token) for token in tokens_arr[i] if token.isalpha() and token not in stop_words]

    for i in tqdm(range(len(high_event_tokens_arr)), total=len(high_event_tokens_arr), desc="Processing high event chunks", unit="chunk"):
        high_event_tokens_arr[i] = [token.strip(string.punctuation) for token in high_event_tokens_arr[i]]

        high_event_tokens_arr[i] = [token for token in high_event_tokens_arr[i] if not token.isdigit()]

        high_event_tokens_arr[i] = [lemmatizer.lemmatize(token) for token in high_event_tokens_arr[i] if token.isalpha() and token not in stop_words]

    for i in tqdm(range(len(medium_event_tokens_arr)), total=len(medium_event_tokens_arr), desc="Processing medium event chunks", unit="chunk"):
        medium_event_tokens_arr[i] = [word for word in medium_event_tokens_arr[i] if word not in stop_words]

        medium_event_tokens_arr[i] = [lemmatizer.lemmatize(token) for token in medium_event_tokens_arr[i]]

    for i in tqdm(range(len(low_event_tokens_arr)), total=len(low_event_tokens_arr), desc="Processing low event chunks", unit="chunk"):
        low_event_tokens_arr[i] = [word for word in low_event_tokens_arr[i] if word not in stop_words]

        low_event_tokens_arr[i] = [lemmatizer.lemmatize(token) for token in low_event_tokens_arr[i]]


    # Flatten the tokens_arr back into a single list
    tokens_flat = [token for sublist in tokens_arr for token in sublist]
    high_event_tokens_flat = [token for sublist in high_event_tokens_arr for token in sublist]
    medium_event_tokens_flat = [token for sublist in medium_event_tokens_arr for token in sublist]
    low_event_tokens_flat = [token for sublist in low_event_tokens_arr for token in sublist]

    # Define output path
    output_dir = os.path.join(ROOT_DIR, "processed", year)
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"processed_tokens_{year}.json")
    high_event_output_path = os.path.join(output_dir, f"high_event_tokens_{year}.json")
    medium_event_output_path = os.path.join(output_dir, f"medium_event_tokens_{year}.json")
    low_event_output_path = os.path.join(output_dir, f"low_event_tokens_{year}.json")


    # Write tokens to file
    total_tokens = len(tokens_flat) + len(high_event_tokens_flat) + len(medium_event_tokens_flat) + len(low_event_tokens_flat)
    logging.info(f"Writing {total_tokens} tokens to {os.path.join(ROOT_DIR, "processed", year)}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tokens_flat, f)

    with open(high_event_output_path, "w", encoding="utf-8") as f:
        json.dump(high_event_tokens_flat, f)

    with open(medium_event_output_path, "w", encoding="utf-8") as f:
        json.dump(medium_event_tokens_flat, f)

    with open(low_event_output_path, "w", encoding="utf-8") as f:
        json.dump(low_event_tokens_flat, f)

    logging.info("Tokens successfully written to file")