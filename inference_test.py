import requests
from sglang.utils import print_highlight

url = f"http://localhost:{30044}/v1/chat/completions"

data = {
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
}

response = requests.post(url, json=data)
print_highlight(response.json())
