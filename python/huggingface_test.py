from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

# Load model and processor
MODEL_NAME = "Anwarkh1/Skin_Cancer-Image_Classification"
processor = AutoImageProcessor.from_pretrained("Anwarkh1/Skin_Cancer-Image_Classification")
model = AutoModelForImageClassification.from_pretrained("Anwarkh1/Skin_Cancer-Image_Classification")

# Function to preprocess image and run inference
def classify_image(image):
    # Open the image file
    processed_image = image.convert("RGB")

    # Preprocess the image using the processor
    inputs = processor(images=processed_image, return_tensors="pt")

    # Run inference (no gradients required)
    with torch.no_grad():
        outputs = model(**inputs)

    # Get predicted class and confidence score
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()  # Get class index
    label = model.config.id2label.get(predicted_class_idx, f"Class {predicted_class_idx}")
    predicted_class_score = torch.nn.functional.softmax(logits, dim=-1)[0][predicted_class_idx].item()  # Get class score

    return label, predicted_class_idx, predicted_class_score

# # Path to your image file
# image_path = "content/mole.jpg"
# label, predicted_class_idx, predicted_class_score = classify_image(image_path)

# Output result
# print(f"Predicted Class: {label}")
# print(f"Predicted Class Index: {predicted_class_idx}")
# print(f"Predicted Confidence Score: {predicted_class_score:.4f}")
