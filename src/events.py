import ollama
import json
import os
import time
from tqdm import tqdm

# ============================================ GLOBAL VARIABLES ============================================ #
ROOT_DIR = "/home/ibrahim/stock/data"
DIRS = [os.path.join(ROOT_DIR, dir) for dir in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, dir))]
FILE_PATHS = [os.path.join(dir, file) for dir in DIRS for file in os.listdir(dir) if file.endswith(".json")]
YEARS = ["2020", "2021", "2022", "2023", "2024", "2025"]

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
<|im_start|>system You are an expert financial analyst evaluating a month's worth of news headlines related to NVIDIA. 
Use your reasoning capabilities [
    [
      8
    ]
] to select the single most impactful event headline from the list, focusing on events that could significantly affect NVIDIA's stock price, reputation, or market position. 
Consider major product launches, earnings reports, regulatory actions, mergers & acquisitions, innovations, or other transformative events. <|im_end|>

**User Prompt:**

<|im_start|>user
Analyze the following news headlines and select the most impactful ones for NVIDIA.
{headlines}

Generate an analysis of the selected headline events using the following JSON format:
```json
{{
  "selected_events": {{
    "headline": "Original headline text",
    "event_type": "Type of event",
    "summary": "Brief summary of the event",
    "impact": "High/Medium/Low",
    "rationale": "Explanation for the selection"
  }}
}}
"""

# ============================================ CODE ============================================ #

def main():

    start = time.time()

    for year in YEARS:
        if not os.path.exists(os.path.join(ROOT_DIR, "events", year)):
            os.makedirs(os.path.join(ROOT_DIR, "events", year))

    for path in tqdm(FILE_PATHS, desc="Processing files", unit="file"):
        
        year = str("")

        for year_it in YEARS:
            if year_it in path:
                year = year_it
                break

        if os.path.exists(os.path.join(ROOT_DIR, "events", year, os.path.basename("events" + path))):
            print(f"Events file for {path} already exists, skipping...")
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        headlines = [headline["headline"] for headline in data["headlines"]]

        prompt = PROMPT.format(headlines=headlines)

        response = ollama.chat(
            model="qwen2.5:3b",
            messages=[
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.7},
            format="json",
        )

        json_response = json.loads(response.message.content)

        high_impact_events = [headline for headline in json_response["selected_events"] if headline["impact"].lower() == "high"]
        medium_impact_events = [headline for headline in json_response["selected_events"] if headline["impact"].lower() == "medium"]
        low_impact_events = [headline for headline in json_response["selected_events"] if headline["impact"].lower() == "low"]

        events = {
            "high_impact_events": high_impact_events,
            "medium_impact_events": medium_impact_events,
            "low_impact_events": low_impact_events
        }

        if not os.path.exists(os.path.join(ROOT_DIR, "events")):
            os.makedirs(os.path.join(ROOT_DIR, "events"))

        with open(os.path.join(ROOT_DIR, "events", year, os.path.basename("events" + path)), "w") as f:
            json.dump(events, f, indent=4)

    elapsed_time = time.time() - start

    print(f"Elapsed time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
