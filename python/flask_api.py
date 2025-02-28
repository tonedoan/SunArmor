from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("model.h5")


def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to the model's expected size
    image = np.array(image) / 255.0   # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]  # Get uploaded image
    image = Image.open(file.stream)
    processed_image = preprocess_image(image)  # Process for model
    prediction = model.predict(processed_image)  # Get prediction
    return jsonify({"prediction": prediction.tolist()})




if __name__ == "__main__":
    app.run(debug=True)


