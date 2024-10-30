import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("./Phi-3-mini-128k-instruct")

# Load the ONNX model
model_path = "./Phi-3-mini-128k-instruct_onnx/Phi-3-mini-128k-instruct.onnx"

# Load the ONNX Runtime session
ort_session = ort.InferenceSession(model_path)

# Check input names
input_names = [input.name for input in ort_session.get_inputs()]
print("Input names:", input_names)

# Prepare the input message
message = "What is OpenVINO?"

# Tokenize the input and convert to numpy array with int64 type
inputs = tokenizer(message, return_tensors="np")
onnx_inputs = {input_names[0]: inputs["input_ids"].astype(np.int64)}  # Ensure int64

# Run inference
outputs = ort_session.run(None, onnx_inputs)

# Extract the generated text from the output
logits = outputs[0]

# Convert logits to token IDs and decode to text
predicted_token_ids = np.argmax(logits, axis=-1)  # Get the index of the highest logit
predicted_text = tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True)

# Print the generated text
print(predicted_text)
