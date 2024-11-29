import os
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
import psutil  # For monitoring system resources
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the model and labels during server startup
MODEL_PATH = os.path.join(os.getcwd(), "Model", "keras_model.h5")
LABELS_PATH = os.path.join(os.getcwd(), "Model", "labels.txt")

try:
    app.logger.info(f"Loading model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    app.logger.info("Model loaded successfully.")
except Exception as e:
    app.logger.error(f"Failed to load model: {e}")
    model = None

try:
    app.logger.info(f"Loading labels from: {LABELS_PATH}")
    with open(LABELS_PATH, "r") as file:
        labels = file.read().splitlines()
    app.logger.info("Labels loaded successfully.")
except Exception as e:
    app.logger.error(f"Failed to load labels: {e}")
    labels = []

@app.route('/')
def home():
    return "Flask server is running!"

@app.route('/translate', methods=['POST'])
def translate():
    # Debug: Log incoming request details
    app.logger.info("Received /translate request.")
    app.logger.info(f"Headers: {request.headers}")
    app.logger.info(f"Content-Type: {request.content_type}")

    # Monitor resource usage
    memory_usage = psutil.virtual_memory()
    cpu_usage = psutil.cpu_percent(interval=1)
    app.logger.info(f"Memory Usage: {memory_usage}")
    app.logger.info(f"CPU Usage: {cpu_usage}%")

    try:
        # Debug: Log request data
        request_data = request.get_json()
        app.logger.info(f"Request data: {request_data}")

        # Mock input validation
        if 'image' not in request_data:
            raise ValueError("No image data provided in request.")

        # Process input (example placeholder logic)
        image_data = request_data['image']
        app.logger.info(f"Processing image data: {len(image_data)} bytes.")

        # Predict using preloaded model
        result = model.predict(image_data)  # Replace with real preprocessed data
        label_index = result.argmax()
        translation = labels[label_index]

        # Debug: Log prediction result
        app.logger.info(f"Prediction: {result}")
        app.logger.info(f"Translated label: {translation}")

        return jsonify({"translation": translation})

    except Exception as e:
        app.logger.error(f"Error in /translate: {e}")
        return jsonify({"error": "Translation failed.", "details": str(e)}), 500

# Run app (development only)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
