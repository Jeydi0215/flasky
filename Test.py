import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import logging
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

# Configure CORS
CORS(app, resources={
    r"/translate": {
        "origins": [
            "https://salinterpret.vercel.app",  # React frontend
            "https://middleman-psi-five.vercel.app"  # Middleman server
        ],
        "methods": ["POST"],
        "allow_headers": ["Content-Type"]
    }
})

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths to model and labels
MODEL_PATH = os.path.join(os.getcwd(), "Model", "keras_model.h5")
LABELS_PATH = os.path.join(os.getcwd(), "Model", "labels.txt")

# Global variables for model and labels
model = None
labels = []

# Load model and labels at startup
try:
    logger.info(f"Loading model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")

try:
    logger.info(f"Loading labels from: {LABELS_PATH}")
    with open(LABELS_PATH, "r") as file:
        labels = file.read().splitlines()
    logger.info("Labels loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load labels: {e}")


@app.route('/')
def home():
    """Home route to check server status."""
    return "Flask server is running!", 200


@app.route('/translate', methods=['POST'])
def translate():
    """
    Endpoint to handle image translation requests.
    Expects an image file in the request and returns the translation.
    """
    try:
        logger.info("Received /translate request.")

        # Check if an image is present in the request
        if 'image' not in request.files:
            logger.warning("No 'image' key in request.files.")
            return jsonify({"error": "No image provided"}), 400

        file = request.files['image']
        logger.info(f"Received file: {file.filename}")

        # Validate file type
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            logger.warning("Unsupported file type.")
            return jsonify({"error": "Unsupported file type"}), 400

        # Process the image
        image = Image.open(file).resize((224, 224))  # Resize based on model input
        image_array = np.array(image) / 255.0  # Normalize pixel values
        image_tensor = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Make predictions
        predictions = model.predict(image_tensor)
        label_index = np.argmax(predictions)
        translation = labels[label_index]

        logger.info(f"Predictions: {predictions}")
        logger.info(f"Translated label: {translation}")

        # Convert the processed image to Base64 for frontend display (optional)
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Return the response
        return jsonify({"img": image_base64, "translation": translation}), 200

    except Exception as e:
        logger.error(f"Error in /translate: {e}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
