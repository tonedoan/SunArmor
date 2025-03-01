import requests

url = "http://127.0.0.1:5000/predict"
image_path = "path_to_your_image.jpg"

with open(image_path, "rb") as img_file:
    files = {"file": img_file}
    response = requests.post(url, files=files)

print(response.json())  # Prints the prediction result