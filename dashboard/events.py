import ollama
import json
import os
import time
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
from wordcloud import WordCloud
import matplotlib.pyplot as plt



# ============================================ GLOBAL VARIABLES ============================================ #
ROOT_DIR = "/home/ibrahim/stock/cache"

KEYWORDS = [
    "earnings",
    "product launch",
    "acquisition",
    "merger",
    "regulation",
    "market share",
    "innovation",
    "investment",
    "partnership",
    "new chip",
    "technology breakthrough",
    "legal",
]

PROMPT = """
**System Prompt:**
<|im_start|>system You are an expert financial analyst with advanced reasoning capabilities. Your task is to evaluate a list of news headlines
 related to NVIDIA and classify each headline into one of three impact categories: High , Medium , or Low . [
    [
      8
    ]
] Focus on identifying events that could significantly affect NVIDIA's stock price, reputation, market position, or long-term strategy.
 Consider factors such as major product launches, earnings reports, regulatory actions, mergers & acquisitions, innovations, competitive dynamics,
   or other transformative events.For each headline, provide an analysis including the type of event,
     a brief summary, the assigned impact level, and a rationale explaining your classification. <|im_end|>

**User Prompt:**

<|im_start|>user
Analyze the following news headlines and select the most impactful ones for NVIDIA.
{headlines}

Generate an analysis of the selected headline events using the following JSON format:
```json
{{
  "selected_events": [
  {{
    "headline": "Original headline text",
    "event_type": "Type of event",
    "summary": "Brief summary of the event",
    "impact": "High/Medium/Low",
    "rationale": "Explanation for the selection"
  }}
  ]
}}
"""

# ============================================ CODE ============================================ #

def calculate_relevance(headline, title, vectorizer):
    assert isinstance(headline, str)

    assert isinstance(title, str)

    matrix = vectorizer.fit_transform([headline, title])
    return cosine_similarity(matrix[0:1], matrix[1:2])[0][0]

def filter_events(start_date, end_date, filtered_df, sentiment) -> dict:

    start = time.time()

    if os.path.exists(os.path.join(ROOT_DIR, f"events_{start_date}_{end_date}_{sentiment}.json")):
        with open(os.path.join(ROOT_DIR, f"events_{start_date}_{end_date}_{sentiment}.json"), "r") as f:
            events = json.load(f)
            return events
        
    headlines = filtered_df["headlines"].to_list()

    chunk_size = 10
    iterations = (len(headlines) + chunk_size - 1) // chunk_size

    headline_chunks = []
    for i in range(iterations):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(headlines))
        headline_chunks.append(headlines[start_idx:end_idx])

    high_impact_events = []
    medium_impact_events = []
    low_impact_events = []


    for headlines in tqdm(headline_chunks, desc="Processing headlines", unit="chunk"):

        prompt = PROMPT.format(headlines=headlines)

        response = ollama.chat(
            model="qwen2.5:1.5b",
            messages=[
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.7},
            format="json",
        )

        json_response = json.loads(response.message.content)

        assert isinstance(json_response, dict)
        assert isinstance(json_response["selected_events"], list)            


        high_impact_events.extend([event for event in json_response["selected_events"] if event["impact"].lower() == "high"])
        medium_impact_events.extend([event for event in json_response["selected_events"] if event["impact"].lower() == "medium"])
        low_impact_events.extend([event for event in json_response["selected_events"] if event["impact"].lower() == "low"])

    events = {
        "high_impact_events": high_impact_events,
        "medium_impact_events": medium_impact_events,
        "low_impact_events": low_impact_events
    }

    with open(os.path.join(ROOT_DIR, f"events_{start_date}_{end_date}_{sentiment}.json"), "w") as f:
        json.dump(events, f, indent=4)

    elapsed_time = time.time() - start

    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    return events

def get_corpus(start_date, end_date, events, filtered_df, sentiment):

    if os.path.exists(os.path.join(ROOT_DIR, f"tokens_{start_date}_{end_date}_{sentiment}.json")):
        with open(os.path.join(ROOT_DIR, f"tokens_{start_date}_{end_date}_{sentiment}.json"), "r") as f:
            tokens = json.load(f)
            return tokens
        
    def extract_headlines(events):
        headlines = [item['headline'] for item in events]
        return headlines if headlines else None
    
    high_impact_headlines = extract_headlines(events.get("high_impact_events", []))
    medium_impact_headlines = extract_headlines(events.get("medium_impact_events", []))
    low_impact_headlines = extract_headlines(events.get("low_impact_events", []))

    news = filtered_df["article"].to_list()

    assert len(high_impact_headlines) > 0, "No high impact headlines found"

    high_event_corpus = ""
    medium_event_corpus = ""
    low_event_corpus = ""
    news_corpus = "\n".join(news)

    vectorizer = TfidfVectorizer()

    titles = filtered_df["title"].to_list()
    articles = filtered_df["article"].to_list()


    if high_impact_headlines is not None:
        print("High impact headlines found")
        for headline in high_impact_headlines:
                best_match_article = None
                best_relevance_score = -1

                for (title, article) in zip(titles, articles):

                    relevance_score = calculate_relevance(headline, title, vectorizer)

                    if relevance_score > best_relevance_score:
                        best_match_article = article
                        best_relevance_score = relevance_score

                if best_match_article:
                    high_event_corpus += best_match_article + "\n"
    else:
        high_event_corpus = "This is a test message"

    if medium_impact_headlines is not None:
        print("medium headlines")
        for headline in medium_impact_headlines:
                best_match_article = None
                best_relevance_score = -1

                for (title, article) in zip(titles, articles):

                    relevance_score = calculate_relevance(headline, title, vectorizer)

                    if relevance_score > best_relevance_score:
                        best_highmatch_article = article
                        best_relevance_score = relevance_score

                if best_match_article:
                    medium_event_corpus += best_match_article + "\n"
    else:
        medium_event_corpus = "This is a test message"
        
    if low_impact_headlines is not None:
        print("low headlines")
        for headline in low_impact_headlines:
                best_match_article = None
                best_relevance_score = -1

                for (title, article) in zip(titles, articles):

                    relevance_score = calculate_relevance(headline, title, vectorizer)

                    if relevance_score > best_relevance_score:
                        best_match_article = article
                        best_relevance_score = relevance_score

                if best_match_article:
                    low_event_corpus += best_match_article + "\n"

    else:
        low_event_corpus = "This is a test message"

    lemmatizer = WordNetLemmatizer()

    high_event_tokens = word_tokenize(high_event_corpus.lower().translate(str.maketrans("", "", string.punctuation)))
    medium_event_tokens = word_tokenize(medium_event_corpus.lower().translate(str.maketrans("", "", string.punctuation)))
    low_event_tokens = word_tokenize(low_event_corpus.lower().translate(str.maketrans("", "", string.punctuation)))
    tokens = word_tokenize(news_corpus.lower().translate(str.maketrans("", "", string.punctuation)))

    if len(high_event_tokens) == 0:
        print("no high event tokens found")

    if len(medium_event_tokens) == 0:
        print("no medium event tokens found")
    if len(low_event_tokens) == 0:
        print("no low event tokens found")

    high_event_tokens_arr = []
    medium_event_tokens_arr = []
    low_event_tokens_arr = []

    def chunk_tokens(tokens, chunk_size=50000):
        # If tokens is empty, return a list with an empty list
        if not tokens:
            return [[]]
        
        chunks = []
        for i in range(0, len(tokens), chunk_size):
            chunks.append(tokens[i:i + chunk_size])
        
        return chunks

    tokens_arr = chunk_tokens(tokens)
    high_event_tokens_arr = chunk_tokens(high_event_tokens)
    medium_event_tokens_arr = chunk_tokens(medium_event_tokens)
    low_event_tokens_arr = chunk_tokens(low_event_tokens)

    if len(tokens_arr) == 0:
        print("no tokens found")
    
    if len(high_event_tokens_arr) == 0:
        print("no high event tokens found1")
    if len(medium_event_tokens_arr) == 0:
        print("no medium event tokens found1")
    if len(low_event_tokens_arr) == 0:
        print("no low event tokens found1")

    def preprocess_tokens(token_array, lemmatizer, stop_words):
        processed_array = []
        for chunk in tqdm(token_array, desc="Processing token chunks", unit="chunk"):
            processed_chunk = [
                lemmatizer.lemmatize(token) 
                for token in chunk 
                if token.isalpha() and token not in stop_words and not token.isdigit()
            ]
            processed_array.append(processed_chunk)
        return [token for sub_list in processed_array for token in sub_list]

    if len(tokens_arr) == 0:
        print("no tokens found2")
    if len(high_event_tokens_arr) == 0:
        print("no high event tokens found2")
    if len(medium_event_tokens_arr) == 0:
        print("no medium event tokens found2")
    if len(low_event_tokens_arr) == 0:
        print("no low event tokens found2")

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    tokens_arr = preprocess_tokens(tokens_arr, lemmatizer, stop_words)
    high_event_tokens_arr = preprocess_tokens(high_event_tokens_arr, lemmatizer, stop_words)
    medium_event_tokens_arr = preprocess_tokens(medium_event_tokens_arr, lemmatizer, stop_words)
    low_event_tokens_arr = preprocess_tokens(low_event_tokens_arr, lemmatizer, stop_words)

    with open(os.path.join(ROOT_DIR, f"tokens_{start_date}_{end_date}_{sentiment}.json"), "w") as f:
        json.dump({
            "tokens": tokens_arr,
            "high_event_tokens": high_event_tokens_arr,
            "medium_event_tokens": medium_event_tokens_arr,
            "low_event_tokens": low_event_tokens_arr
        }, f, indent=4)

    return {
        "tokens": tokens_arr,
        "high_event_tokens": high_event_tokens_arr,
        "medium_event_tokens": medium_event_tokens_arr,
        "low_event_tokens": low_event_tokens_arr
    }

def create_wordcloud(start_date, end_date, corpus, sentiment):

    keys = [key for key in corpus.keys()]

    for key in keys:
        if not os.path.exists(os.path.join(ROOT_DIR, f"{key}_{start_date}_{end_date}_{sentiment}.png")):
            text = " ".join(corpus[key])
            text = text.replace("nvidia", " ")

            if text == " " or text == "":
                text = "Here is some sample text to generate a word cloud  as there are no tokens to plot"

            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color="white",
                colormap="viridis",
                max_words=200,
                min_font_size=10,
            ).generate(text)
            
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.title(f"{key} Word Cloud")
            plt.savefig(os.path.join(ROOT_DIR, f"{key}_{start_date}_{end_date}_{sentiment}.png"), bbox_inches="tight")