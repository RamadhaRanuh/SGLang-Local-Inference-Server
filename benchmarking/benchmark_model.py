import requests
import json
import time
import statistics


port = 30045
url = f"http://localhost:{port}/v1/chat/completions"
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
num_runs = 5
prompt = "Explain quantum computing in simple terms."
max_tokens = 100


def run_benchmark():
    results = []

    for i in range(num_runs):
        print(f"Run {i + 1}/{num_runs}")
        start_time = time.time()

        data = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }

        response = requests.post(url, json=data)
        result = response.json()
        end_time = time.time()

        generated_text = result["choices"][0]["message"]["content"]

        # output_tokens = len(generated_text.split())
        output_tokens = result["usage"]["completion_tokens"]

        elapsed_time = end_time - start_time
        tokens_per_second = output_tokens / elapsed_time

        print(f" Time: {elapsed_time:.2f}s, Tokens: {output_tokens}, Tokens/s: {tokens_per_second:.2f}")


        results.append({
            "elapsed_time": elapsed_time,
            "output_tokens": output_tokens,
            "tokens_per_second": tokens_per_second
        })

        avg_time = statistics.mean([r["elapsed_time"] for r in results])
        avg_tokens = statistics.mean([r["output_tokens"] for r in results])
        avg_tps = statistics.mean([r["tokens_per_second"] for r in results])

        print("\nResults:")
        print(f"Average time: {avg_time:.2f} seconds")
        print(f"Average tokens: {avg_tokens:.2f}")
        print(f"Average tokens per second: {avg_tps:.2f}")

if __name__ == "__main__":
    run_benchmark()

