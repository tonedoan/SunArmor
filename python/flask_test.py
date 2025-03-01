# Import libraries
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok
from PIL import Image
import numpy as np
import io
from huggingface_test import classify_image
test_string = 'you did it'
# Create upload folder
UPLOAD_FOLDER = './content'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Create a Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

@app.route('/upload', methods=['POST'])
def upload_image():
    # Check if request has the file part
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400

    file = request.files['image']

    # If user doesn't select file, browser might submit an empty file
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    if file:
        # Save the file
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Optional: Process the image
        try:
            img = Image.open(filename)
            # Example processing: get image dimensions
            width, height = img.size
            label, predicted_class_idx, predicted_class_score, result = classify_image(img)
            accuracy = str(predicted_class_score * 100)[0:2] + "%"


            return jsonify({

                'message': 'Image uploaded successfully',
                'label': label,
                'predicted_class_score': accuracy,
                'result': result
           }), 200

        except Exception as e:
            return jsonify({'error': str(e)}), 500

# Health check endpoint
@app.route('/', methods=['GET'])
def hello():
    return jsonify({'status': 'API is running'}), 200

# Start ngrok tunnel
public_url = ngrok.connect(5000)  # Just specify the port
print(f"Public URL: {public_url}")

# Run the app
app.run(port=5000)
print(public_url)