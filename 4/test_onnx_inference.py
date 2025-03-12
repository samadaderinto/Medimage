import onnxruntime as ort
from transformers import ViTImageProcessor
from PIL import Image
import numpy as np

# Path to ONNX model
model_path = r"C:\Users\USER\Documents\MyPythonProjects\ThyroidClassification\thyroid_ultrasound_model.onnx"

# Load the ONNX model
session = ort.InferenceSession(model_path)

# Path to the image you want to classify
image_path = r"C:\Users\USER\Documents\MyPythonProjects\ThyroidClassification\images\images.jpg"

# Preprocess the image
processor = ViTImageProcessor.from_pretrained("agent593/Thyroid-Ultrasound-Image-Classification-ViTModel")
image = Image.open(image_path).convert("RGB")
inputs = processor(images=image, return_tensors="np")  # Return numpy arrays for ONNX
pixel_values = inputs["pixel_values"]

# Perform inference
outputs = session.run(None, {"pixel_values": pixel_values})  # Run ONNX inference
logits = outputs[0]  # Extract logits

# Interpret the output
id2label = {0: "normal thyroid", 1: "malignant", 2: "benign"}
predicted_class = np.argmax(logits, axis=1)[0]  # Get the predicted class
print("Predicted Diagnosis:", id2label[predicted_class])
