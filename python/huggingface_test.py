from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

# Load model and processor
MODEL_NAME = "Anwarkh1/Skin_Cancer-Image_Classification"
processor = AutoImageProcessor.from_pretrained("Anwarkh1/Skin_Cancer-Image_Classification")
model = AutoModelForImageClassification.from_pretrained("Anwarkh1/Skin_Cancer-Image_Classification")

# Function to preprocess image and run inference
def classify_image(image_path):
    # Open the image file
    image = Image.open(image_path).convert("RGB")

    # Preprocess the image using the processor
    inputs = processor(images=image, return_tensors="pt")

    # Run inference (no gradients required)
    with torch.no_grad():
        outputs = model(**inputs)

    # Get predicted class and confidence score
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()  # Get class index
    label = model.config.id2label.get(predicted_class_idx, f"Class {predicted_class_idx}")
    predicted_class_score = torch.nn.functional.softmax(logits, dim=-1)[0][predicted_class_idx].item()  # Get class score

    # Classify the image based on the index
    if predicted_class_idx in [0,4,6]:
        result = "Not Cancer"
    elif predicted_class_idx in [1, 2, 3, 5]:
        result = "Cancer or Problematic"
    else:
        if predicted_class_score < 0.5:
            result = "Unknown"

    return label, predicted_class_idx, predicted_class_score, result

# Path to your image file
image_path = "./benign.jpg"
try:
    label, predicted_class_idx, predicted_class_score, result = classify_image(image_path)
    string = "".join(label).replace("_", " ").title()

    # Output result
    print(f"Predicted Class: {string}")
    print(f"Predicted Class Index: {predicted_class_idx}")
    print(f"Predicted Confidence Score: {predicted_class_score:.4f}")
    print(f"Classification Result: {result}")

except Exception as e:
    print(f"Error: {e}")
