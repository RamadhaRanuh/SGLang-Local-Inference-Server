from transformers import AutoTokenizer
from llmcompressor.transformers import SparseAutoModelForCausalLM
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from datasets import load_dataset

# Step 1: Load the original model.
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

model = SparseAutoModelForCausalLM.from_pretrained(
  MODEL_ID, device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

from datasets import load_dataset

NUM_CALIBRATION_SAMPLES=512
MAX_SEQUENCE_LENGTH=2048

# Load dataset.
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

# Preprocess the data into the format the model is trained with.
def preprocess(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False,)}
ds = ds.map(preprocess)

# Tokenize the data (be careful with bos tokens - we need add_special_tokens=False since the chat_template already added it).
def tokenize(sample):
    return tokenizer(sample["text"], padding=False, max_length=MAX_SEQUENCE_LENGTH, truncation=True, add_special_tokens=False)
ds = ds.map(tokenize, remove_columns=ds.column_names)

# Step 2: Perform offline quantization.
# Step 2.1: Configure mixed quantization with GPTQ
# Use a dictionary for scheme configuration
recipe = GPTQModifier(
  targets="Linear", 
  scheme="W8A8",
  ignore=["lm_head"]  # Don't quantize the language model head
)

# Step 2.2: Apply the quantization algorithm.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Step 3: Save the model.
SAVE_DIR = MODEL_ID.split("/")[1] + "-GPTQ-INT8"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)