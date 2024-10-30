from optimum import __version__
from optimum.openvino import OVModel
print(__version__)
exit()

# Path to your ONNX model
onnx_model_path = "./Phi-3-mini-128k-instruct_onnx/Phi-3-mini-128k-instruct.onnx"

# Output path for the OpenVINO model
openvino_model_path = "./Phi-3-mini-128k-instruct_openvino"

# Load the ONNX model
onnx_model = OVModel.from_pretrained(onnx_model_path)

# Export the model to OpenVINO format
onnx_model.save_pretrained(openvino_model_path)

print("Model converted to OpenVINO format.")
