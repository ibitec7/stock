import ollama
import os
import json
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

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

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct-AWQ")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct-AWQ", device_map = "cuda", low_cpu_mem_usage = True)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    root_dir = "/home/ibrahim/stock/data"
    years = ["2024", "2023", "2022", "2021","2020"]
    for year in years:
        for path in os.listdir(os.path.join(root_dir, year)):
            print(f"Working on {path}")
            with open(os.path.join(root_dir, year, path), "r") as f:
                data = json.load(f)
            
            for i, article in enumerate(data["articles"]):
                if "ai_analysis" in data["articles"][i].keys():
                    print(f"Skipping {i+1}/{len(data['articles'])}")
                    continue
                input = prompt.format(news=article["response"])
                start_time = time.time()
                response = generator(input, max_length=32000)[0]["generated_text"]
                try:
                    analysis = json.loads(response)["analysis"]
                    print(analysis)
                except:
                    # Load your model and tokenizer (consider moving these outside the loop for efficiency)
                    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct-AWQ")
                    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct-AWQ")
                    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

                    response_text = generator(input, max_length=32000)[0]["generated_text"]

                    analysis = json.loads(response_text)["analysis"]
                    print(analysis)
                    analysis = json.loads(response.message["content"])["analysis"]
                    print(analysis)


                elapsed_time = time.time() - start_time

                
                data["articles"][i]["ai_analysis"] = analysis
                print(f"Completed {i+1}/{len(data['articles'])}   |   Response time: {elapsed_time:.2f} seconds")

                print("Saving data")

                with open(os.path.join(root_dir, year, path), "w") as f:
                    json.dump(data, f, indent=4)