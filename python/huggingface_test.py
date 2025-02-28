

import requests
import base64
from PIL import Image, ImageDraw, ImageFont
import io

API_URL = "https://api-inference.huggingface.co/models/facebook/detr-resnet-50"
headers = {"Authorization": "Bearer "}  # Replace with your actual API key


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


def detect_objects(image_path):
    # Open the image file in binary read mode
    with open(image_path, "rb") as f:
        image_data = f.read()

    # Encode the image data to base64
    image_base64 = base64.b64encode(image_data).decode('utf-8')

    # Prepare the request payload with the base64 encoded image
    response = query({"inputs": image_base64})

    # Handle errors in the response
    if 'error' in response:
        print(f"Error: {response['error']}")
        return None

    return response


def draw_boxes(image_path, detections, output_path="output_with_boxes.jpg"):
    # Open the image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Define font (adjust font size as necessary)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    # Draw each bounding box on the image
    for detection in detections:
        box = detection['box']
        label = detection['label']
        score = detection['score']

        # Bounding box coordinates
        xmin, ymin, xmax, ymax = box['xmin'], box['ymin'], box['xmax'], box['ymax']

        # Draw the bounding box
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=3)

        # Add label and score above the bounding box
        text = f"{label} ({score:.2f})"

        # Get text size using textbbox (works in Pillow 8.0+)
        text_bbox = draw.textbbox((xmin, ymin), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Draw background rectangle for text
        draw.rectangle([(xmin, ymin - text_height), (xmin + text_width, ymin)], fill="red")

        # Draw the label text
        draw.text((xmin, ymin - text_height), text, fill="white", font=font)

    # Save the image with bounding boxes
    image.save(output_path)
    print(f"Image saved with bounding boxes as {output_path}")


# Path to your image file
image_path = "./animal images.jpg"
result = detect_objects(image_path)

# Draw boxes and save the result if detection is successful
if result is not None:
    draw_boxes(image_path, result)
