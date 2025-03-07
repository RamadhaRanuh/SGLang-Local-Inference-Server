import requests
from sglang.utils import print_highlight

url = f"http://localhost:{35526}/v1/chat/completions"

data = {
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
}

response = requests.post(url, json=data)
print_highlight(response.json())
