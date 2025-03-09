import os
import torch
import datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

from trl import SFTTrainer


# Configuration
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "./lora-adapter-output"
LORA_R = 8 # LoRA rank
LORA_ALPHA = 16 # LoRA alpha
LORA_DROPOUT = 0.05 # LoRA dropout
BATCH_SIZE = 4
MICRO_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 2e-4
EPOCHS = 3

# 1. Prepare the quantization config for loading the base model
quantization_config = BitsAndBytesConfig(
    load_in_4_bit = True,
    bnb_4bit_compute_dtype = torch.float16,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_use_double_quant = True
)

# 2. Load the base model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config = quantization_config,
    device_map = "auto",
    trust_remote_code = True
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code = True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 3. Prepare the model for LoRA fine-tuning
model = prepare_model_for_kbit_training(model)

# 4. Define LoRA configuration
lora_config = LoraConfig(
    r = LORA_R,
    lora_alpha = LORA_ALPHA,
    lora_dropout = LORA_DROPOUT,
    bias = "none",
    task_type = "CAUSAL_LM",
    target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"] # Target attention layers
)


# 5. Get the PEFT model
model = get_peft_model(model, lora_config)
print(f"Trainable parameters: {model.print_trainable_parameters()}")


# 6. Create simple dataset

def create_sample_dataset():
    data = {
        "instruction": [
            "Summarize the following paragraph:",
            "Explain the concept of machine learning:",
            "Translate this sentence to French:",
            "Write a short poem about nature:",
        ],
        "input": [
            "The economy has been experiencing significant fluctuations due to global events. Inflation rates have risen while unemployment numbers remain stable.",
            "",
            "The quick brown fox jumps over the lazy dog.",
            "",
        ],
        "output": [
            "The economy is fluctuating due to global events, with rising inflation but stable unemployment.",
            "Machine learning is a subset of AI where algorithms learn patterns from data to make predictions or decisions without explicit programming.",
            "Le renard brun rapide saute par-dessus le chien paresseux.",
            "Gentle breeze through swaying trees,\nSunlight dancing on morning dew.\nBirds sing melodies with ease,\nNature's beauty ever new.",
        ]
    }
    return datasets.Dataset.from_dict(data)


dataset = create_sample_dataset()


def format_instruction(example):
    """Format the instruction, input, and output into a prompt template"""
    if example["input"]:
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    else:
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    return {"text": prompt}

formatted_dataset = dataset.map(format_instruction)


# 7. Configure training arguments
training_args = TrainingArguments(
    output_dir = OUTPUT_DIR,
    per_device_train_batch_size = MICRO_BATCH_SIZE,
    gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS,
    learning_rate = LEARNING_RATE,
    num_train_epochs = EPOCHS,
    logging_steps = 10,
    save_strategy = "epoch",
    evaluation_strategy = "cosine",
    warmup_ratio = 0.1,
    report_to = "none",
    bf16 = True, 
    remove_unused_columns = False
)

# 8. Create the SFT trainer
trainer = SFTTrainer(
    model = model,
    args = training_args,
    train_dataset = formatted_dataset,
    tokenizer = tokenizer,
    packing = True,
    max_seq_length = 2048
)

# 9. Train the model
trainer.train()

# 10. Save the trained LoRA adapter
model.save_pretrained(OUTPUT_DIR)

# 11. Function to use the LoRA adapter with SGLang
def setup_sglang_with_lora(lora_adapter_path, base_model_path):
    """
    Set up SGLang server with a LoRA adapter

    Args:
        lora_adapter_path (str): Path to the LoRA adapter
        base_model_path (str): Path to the base model
    """
    from sglang.utils import launch_server_cmd, wait_for_server

    # Launch SGLang server with LoRA adapter(s)
    server_process, port = launch_server_cmd(
        f"""
        python -m sglang.launch_server --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --host 0.0.0.0 --lora-paths {lora_adapter_path} --max-loras-per-batch 1
        """
    )

    wait_for_server(f"http://localhost:{port}")

    return server_process, port

# 12. Test a query with the LoRA adapter
def test_lora_adapter(port, prompt):
    """
    Test a prompt with the LoRA adapter
    
    Args:
        port: Port of the SGLang server
        prompt: Prompt to test
    """
    import requests
    import json
    
    url = f"http://localhost:{port}/v1/chat/completions"
    data = {
        "model": BASE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "lora_id": os.path.basename(OUTPUT_DIR),  # Use the directory name as lora_id
    }
    
    response = requests.post(url, json=data)
    print(f"Status code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))