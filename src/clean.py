import os
import json

def contains_keywords(text, keywords):
    lower_text = text.lower()
    for keyword in keywords:
        if keyword.lower() in lower_text:
            return True
    return False

def contains_error(text, phrase):
    return phrase.lower() in text.lower()

if __name__ == "__main__":
    keywords = ["NVIDIA", "data", "AI", "GPU", "NVDA", "Graphics", "Semiconductor"]
    error_message = "Error fetching content: HTTP"

    root_dir = "/home/ibrahim/stock/data"
    years = ["2020", "2021", "2022", "2023", "2024"]

    total = 0

    for year in years:
        for path in os.listdir(os.path.join(root_dir, year)):
            with open(os.path.join(root_dir, year, path), "r") as f:
                print(f"Cleaning {path}")
                data = json.load(f)

            cleaned_article = [article for article in data["articles"] if contains_keywords(article["response"], keywords) and not contains_error(article["response"], error_message)]

            data["articles"] = cleaned_article
            data["totalResults"] = len(data["articles"])
            total += len(data["articles"])

            with open(os.path.join(root_dir, year, path), "w") as f:
                json.dump(data, f, indent=4)

    print(f"Total clean articles: {total}")
