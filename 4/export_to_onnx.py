from transformers import ViTForImageClassification, ViTImageProcessor
import torch
from PIL import Image

# Load the model and processor
model = ViTForImageClassification.from_pretrained("agent593/Thyroid-Ultrasound-Image-Classification-ViTModel")
processor = ViTImageProcessor.from_pretrained("agent593/Thyroid-Ultrasound-Image-Classification-ViTModel")

# Set the model to evaluation mode
model.eval()

# Path to the image you want to use for preprocessing
image_path = r"C:\Users\USER\Documents\MyPythonProjects\ThyroidClassification\images\images.jpg"

# Load and preprocess the image
image = Image.open(image_path).convert("RGB")
inputs = processor(images=image, return_tensors="pt")  # Process the image into a tensor
pixel_values = inputs["pixel_values"]  # This will be the actual input for the model

# Export the model to ONNX format using real input data
torch.onnx.export(
    model,  # The PyTorch model
    pixel_values,  # Use the preprocessed image tensor
    "thyroid_ultrasound_model.onnx",  # Output ONNX file path
    input_names=["pixel_values"],  # Input tensor name
    output_names=["logits"],  # Output tensor name
    dynamic_axes={"pixel_values": {0: "batch_size"}, "logits": {0: "batch_size"}},  # Dynamic axes
    opset_version=14,  # Compatible ONNX opset version
)

print("ONNX model has been exported successfully.")
