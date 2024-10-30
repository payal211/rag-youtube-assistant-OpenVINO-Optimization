import openvino_genai as ov_genai

model_path = "./model/Phi-3-mini-128k-instruct-fp32-openvino/fp32"

device = "CPU"
pipe = ov_genai.LLMPipeline(model_path, device)
print(pipe.generate("What is OpenVINO?", max_length=2000))
