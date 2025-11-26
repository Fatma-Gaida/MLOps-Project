import requests

API_URL = "https://router.huggingface.co/hf-inference/models/gpt2"
headers = {"Authorization": "Bearer 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"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        print("Error:", response.status_code, response.text)
        return None
    try:
        return response.json()
    except Exception as e:
        print("Failed to decode JSON:", e, response.text)
        return None

prompt = (
    "You are an intelligent routing system.\n"
    "Text: Hello world\n"
    "If the text is simple or short, respond only with TFIDF.\n"
    "If the text is complex or nuanced, respond only with TRANSFORMER."
)

output = query({"inputs": prompt, "parameters": {"max_new_tokens": 30, "temperature":0.7}})
print(output)
