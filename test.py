from transformers import AutoTokenizer
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
prompts = [
    "What is the capital of France?",
    "Write a paragraph about machine learning.",
    "Explain the process of photosynthesis in detail, including the light-dependent and light-independent reactions.",
    "Compare and contrast deep learning and traditional machine learning algorithms. Discuss their applications, advantages, and limitations in various domains."
]

def count_tokens(text):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokens = tokenizer.encode(text)
    return len(tokens)


for prompt in prompts:
    token_count= count_tokens(prompt)
    print(f"Prompt: '{prompt[:30]}...' has {token_count} tokens")

token_counts = [count_tokens(prompt) for prompt in prompts]
print("\nAll token counts:", token_counts)