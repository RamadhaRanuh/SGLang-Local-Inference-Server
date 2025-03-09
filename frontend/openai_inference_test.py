port = 30045
def chat_completions():
    import openai
    from sglang.utils import print_highlight

    client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

    response = client.chat.completions.create(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    messages=[
        {
            "role": "system",
            "content": "You are a knowledgeable historian who provides concise responses.",
        },
        {"role": "user", "content": "Tell me about ancient Rome"},
        {
            "role": "assistant",
            "content": "Ancient Rome was a civilization centered in Italy.",
        },
        {"role": "user", "content": "What were their major achievements?"},
    ],
    temperature=0.3,  # Lower temperature for more focused responses
    max_tokens=128,  # Reasonable length for a concise response
    top_p=0.95,  # Slightly higher for better fluency
    presence_penalty=0.2,  # Mild penalty to avoid repetition
    frequency_penalty=0.2,  # Mild penalty for more natural language
    n=1,  # Single response is usually more stable
    seed=42,  # Keep for reproducibility
    )

    print_highlight(f"Response: {response}")
    print_highlight(f"Text: {response.choices[0].message.content}")


def completions():
    import openai
    from sglang.utils import print_highlight

    client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

    response = client.completions.create(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    prompt="<|user|>\nWrite a short story about a space explorer.\n<|assistant|>\n",
    temperature=0.7,  # Moderate temperature for creative writing
    max_tokens=150,  # Longer response for a story
    top_p=0.9,  # Balanced diversity in word choice
    stop=["\n\n", "THE END"],  # Multiple stop sequences
    presence_penalty=0.3,  # Encourage novel elements
    frequency_penalty=0.3,  # Reduce repetitive phrases
    n=1,  # Generate one completion
    seed=123,  # For reproducible results
    )

    print_highlight(f"Response: {response}")
    print_highlight(f"Text: {response.choices[0].text}")



def batches():
import json # Encoding and decoding JSON data
import time # Pausing the executin to check batch job status
from openai import OpenAI # Provides OpenAI API client

# Creates an instance of the OpenAI client
client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="None") 

# Create Multiple Requests
requests = [
    {
        "custom_id": "request-1",
        "method": "POST",
        "url": "/chat/completions",
        "body": {
            "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "messages": [
                {"role": "user", "content": "Tell me a joke about programming"}
            ],
            "max_tokens": 50,
        },
    },
    {
        "custom_id": "request-2",
        "method": "POST",
        "url": "/chat/completions",
        "body": {
            "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "messages": [{"role": "user", "content": "What is Python?"}],
            "max_tokens": 50,
        },
    },
]

# Creating the JSONL File
input_file_path = "batch_requests.jsonl"

with open(input_file_path, "w") as f:
    for req in requests:
        f.write(json.dumps(req) + "\n")

# Uploading the File
with open(input_file_path, "rb") as f:
    file_response = client.files.create(file=f, purpose="batch")

# Creating the Batch Job
batch_response = client.batches.create(
    input_file_id=file_response.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
)

# Printing the Batch Job ID
print_highlight(f"Batch job created with ID: {batch_response.id}")


if __name__ == "__main__":
    # chat_completions()
    completions()
    #batches()
