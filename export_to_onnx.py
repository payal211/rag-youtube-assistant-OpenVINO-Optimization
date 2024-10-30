# export_to_onnx.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("./Phi-3-mini-128k-instruct")
model = AutoModelForCausalLM.from_pretrained("./Phi-3-mini-128k-instruct")
model.eval()

# Create a dummy input tensor using the vocabulary size
vocab_size = tokenizer.vocab_size
dummy_input = torch.randint(0, vocab_size, (1, 32))  # Adjust the shape as needed

# Export the model to ONNX format
torch.onnx.export(
    model,
    dummy_input,
    "./Phi-3-mini-128k-instruct_onnx/Phi-3-mini-128k-instruct.onnx",
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"}}
)

print("Model exported to ONNX format.")
