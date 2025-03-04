import requests # Http Request to the SGLang Server
import time # Measuring execution time (TPS)
import statistics # Calculating Mean, Median, Min, Max, Std Dev
from concurrent.futures import ThreadPoolExecutor # Running task concurrently
import matplotlib.pyplot as plt # Plotting the results
import numpy as np # Numeric Operation
from transformers import AutoTokenizer # Tokenizing text

# Configuration
port = 30045  # Replace with your actual port
url = f"http://localhost:{port}/v1/chat/completions" # URL of the SGLang Server's endpoint
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # Name of the model
num_runs = 5 # How many times to run each test
concurrent_requests = [1, 2, 4]  # Test with different levels of concurrency. 1 is sequential.
tokenizer = AutoTokenizer.from_pretrained(model_name) # Load the tokenizer for the model

# Sample prompts of different lengths
prompts = [
    "What is the capital of France?",
    "Write a paragraph about machine learning.",
    "Explain the process of photosynthesis in detail, including the light-dependent and light-independent reactions.",
    "Compare and contrast deep learning and traditional machine learning algorithms. Discuss their applications, advantages, and limitations in various domains."
]

def count_tokens(text):
    """
    This function takes text as input and returns the number of tokens in that token according to the loaded tokenizer. 
    This is used to measure the zize of prompts and generated responses.
    """
    tokens = tokenizer.encode(text)
    return len(tokens)
    
# Takes a prompt and an optional max_tokens parameter, 
# and returns a dictionary with the results of a single inference request.
def single_inference(prompt, max_tokens=100): 
    """
    This is the core function that performs a single inference request to the SGLang server.
    """
    # Record the start time
    start_time = time.time() 
    
    # Create JSON data for the request
    data = { 
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }
    
    # sent POST request to the SGLang server with the payload
    response = requests.post(url, json=data) 

    # Check if the response is successful
    if response.status_code != 200: 
        print(f"Error: {response.status_code}")
        print(response.text)
        return None
    
    # Parse the response JSON
    result = response.json()

    # Record the end time
    end_time = time.time()
    
    # Extract generated text
    generated_text = result["choices"][0]["message"]["content"]
    
    # Count tokens in the generated text
    output_tokens = count_tokens(generated_text)
    
    # Calculate time and tokens per second
    elapsed_time = end_time - start_time
    tokens_per_second = output_tokens / elapsed_time
    
    # Return a dictionary
    return {
        "prompt": prompt,
        "prompt_tokens": count_tokens(prompt),
        "output_tokens": output_tokens,
        "elapsed_time": elapsed_time,
        "tokens_per_second": tokens_per_second
    }

def concurrent_benchmark(prompt, concurrency, max_tokens=100):
    results = []
    
    def worker():
        return single_inference(prompt, max_tokens)
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker) for _ in range(num_runs)]
        for future in futures:
            result = future.result()
            if result:
                results.append(result)
    
    return results

# Run benchmarks for different concurrency levels
all_results = {}
for concurrency in concurrent_requests:
    print(f"\nRunning benchmark with concurrency: {concurrency}")
    concurrency_results = {}
    
    for prompt in prompts:
        print(f"  Testing prompt: {prompt[:30]}...")
        results = concurrent_benchmark(prompt, concurrency)
        concurrency_results[prompt] = results
        
        # Calculate average tokens per second
        avg_tps = statistics.mean([r["tokens_per_second"] for r in results])
        print(f"  Average tokens per second: {avg_tps:.2f}")
    
    all_results[concurrency] = concurrency_results

# Analyze and visualize results
def plot_results(all_results):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bar_width = 0.2
    index = np.arange(len(prompts))
    
    for i, concurrency in enumerate(concurrent_requests):
        avg_tps_values = []
        
        for j, prompt in enumerate(prompts):
            results = all_results[concurrency][prompt]
            avg_tps = statistics.mean([r["tokens_per_second"] for r in results])
            avg_tps_values.append(avg_tps)
        
        offset = (i - len(concurrent_requests)/2 + 0.5) * bar_width
        ax.bar(index + offset, avg_tps_values, bar_width, label=f'Concurrency: {concurrency}')
    
    ax.set_xlabel('Prompt')
    ax.set_ylabel('Tokens per Second')
    ax.set_title('Inference Throughput by Prompt Length and Concurrency')
    ax.set_xticks(index)
    ax.set_xticklabels([f"Prompt {i+1}\n({count_tokens(prompt)} tokens)" for i, prompt in enumerate(prompts)])
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('throughput_benchmark.png')
    plt.show()

# Generate summary statistics
print("\nSummary Statistics:")
for concurrency in concurrent_requests:
    print(f"\nConcurrency: {concurrency}")
    for prompt in prompts:
        results = all_results[concurrency][prompt]
        tps_values = [r["tokens_per_second"] for r in results]
        
        print(f"  Prompt: {prompt[:30]}...")
        print(f"    Mean TPS: {statistics.mean(tps_values):.2f}")
        print(f"    Median TPS: {statistics.median(tps_values):.2f}")
        print(f"    Min TPS: {min(tps_values):.2f}")
        print(f"    Max TPS: {max(tps_values):.2f}")
        print(f"    Std Dev: {statistics.stdev(tps_values) if len(tps_values) > 1 else 0:.2f}")

# Try to generate the plot if matplotlib is available
try:
    plot_results(all_results)
except Exception as e:
    print(f"Could not generate plot: {e}")
    print("Raw data is still available in the all_results variable")