import ollama
import os
import json
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm

if __name__ == "__main__":

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["OLLAMA_TRT"] = '1'

    prompt = """**System Prompt:**
    "<|im_start|>system
    You are a financial analysis expert specializing in semiconductor companies. Use the reasoning capabilities [
        [
            5
        ]
    ] to analyze NVIDIA news content and generate structured investment insights. Maintain adherence to the specified JSON format but make some additions or changes to it depending on if it is fit for the certain news while incorporating technical and market factors.<|im_end|>
    "

    **User Prompt Template:**
    "<|im_start|>user
    Analyze the following NVIDIA news content:  {news}

    Generate a comprehensive analysis using this JSON structure:
    ```json
    {{
        "analysis": {{
            "news_summary": {{
                "title": "string",
                "date": "string",
                "key_points": [
                    "string"
                ],
                "sentiment": {{
                    "overall": "string (Bullish/Bearish/Neutral)",
                    "market_reaction": "string",
                    "analyst_concerns": [
                        "string"
                    ]
                }}
            }},
            "stock_impact": {{
                "short_term": {{
                    "price_movement": "string",
                    "catalysts": [
                        "string"
                    ],
                    "risks": [
                        "string"
                    ]
                }},
                "long_term": {{
                    "strategic_implications": [
                        "string"
                    ],
                    "growth_opportunities": [
                        "string"
                    ],
                    "competitive_position": "string"
                }}
            }},
            "technical_factors": {{
                "product_launches": [
                    "string"
                ],
                "AI_innovation": [
                    "string"
                ],
                "data_center_growth": "string"
            }},
            "financial_metrics": {{
                "revenue_impact": "string",
                "margin_pressure": "string",
                "guidance_changes": "string"
            }}, 
            "recommendations": {{
                "trading_strategy": "string",
                "hold/buy/sell": "string",
                "monitor_factors": [
                    "string"
                ]
            }}
        }}
    }}
    """

    model_options = {
        "trt": True,
        "gpu_layers": -1
    }


    root_dir = "/home/ibrahim/stock/data"
    years = ["2025"]
    for year in years:
        for path in os.listdir(os.path.join(root_dir, year)):
            print(f"Working on {path}")
            with open(os.path.join(root_dir, year, path), "r") as f:
                data = json.load(f)
            
            for i, article in enumerate(tqdm(data["articles"], desc="Processing articles")):
                if "ai_analysis" in data["articles"][i].keys():
                    print(f"Skipping {i+1}/{len(data['articles'])}")
                    continue
                input = prompt.format(news=article["response"])
                start_time = time.time()
                response = ollama.chat(messages=[{
                    "role": "user",
                    "content": input
                }],
                model="qwen2.5:1.5b",
                format="json",
                ).message.content

                try:
                    analysis = json.loads(response)
                    
                except:

                    response_text = ollama.chat(messages=[{
                        "role": "user",
                        "content": input
                    }],
                    model="qwen2.5:1.5b",
                    format="json",
                    ).message.content

                    analysis = json.loads(response_text)

                elapsed_time = time.time() - start_time

                
                data["articles"][i]["ai_analysis"] = analysis

                with open(os.path.join(root_dir, year, path), "w") as f:
                    json.dump(data, f, indent=4)