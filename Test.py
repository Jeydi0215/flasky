import os
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
import psutil  # For monitoring system resources
import logging
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://your-frontend.vercel.app"}})  # Update with your frontend URL

# Configure logging
logging.basicConfig(level=logging.INFO)
app.logger.info("Starting Flask server...")

# Load model and labels
MODEL_PATH = os.path.join(os.getcwd(), "Model", "keras_model.h5")
LABELS_PATH = os.path.join(os.getcwd(), "Model", "labels.txt")

try:
    app.logger.info(f"Loading model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    app.logger.info(f"Model loaded successfully. Input shape: {model.input_shape}")
except Exception as e:
    app.logger.error(f"Failed to load model: {e}")
    model = None

try:
    app.logger.info(f"Loading labels from: {LABELS_PATH}")
    with open(LABELS_PATH, "r") as file:
        labels = file.read().splitlines()
    app.logger.info(f"Labels loaded successfully: {labels}")
except Exception as e:
    app.logger.error(f"Failed to load labels: {e}")
    labels = []

# Preprocessing function
def preprocess_image(image_data):
    try:
        # Decode base64 image (assuming the data is sent in base64 format)
        decoded_image = tf.io.decode_base64(image_data)
        
        # Convert to an image tensor
        image = tf.image.decode_image(decoded_image, channels=3)
        image = tf.image.resize(image, (224, 224))  # Replace with your model's input size
        image_array = img_to_array(image) / 255.0  # Normalize pixel values
        
        # Expand dimensions for model input
        return np.expand_dims(image_array, axis=0)
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {e}")

# Routes
@app.route('/')
def home():
    return "Flask server is running!"

@app.route('/translate', methods=['POST'])
def translate():
    app.logger.info("Received /translate request.")
    
    # Monitor resource usage
    memory_usage = psutil.virtual_memory()
    cpu_usage = psutil.cpu_percent(interval=1)
    app.logger.info(f"Memory Usage: {memory_usage}")
    app.logger.info(f"CPU Usage: {cpu_usage}%")

    try:
        # Validate and log incoming request data
        request_data = request.get_json()
        app.logger.info(f"Request data: {request_data}")

        # Check for 'image' key in request
        if 'image' not in request_data:
            raise ValueError("Missing 'image' key in request payload.")

        # Preprocess image
        image_data = request_data['image']
        if not image_data or len(image_data) == 0:
            raise ValueError("Image data is empty or invalid.")
        processed_image = preprocess_image(image_data)

        # Predict using the preloaded model
        app.logger.info("Making prediction...")
        result = model.predict(processed_image)
        label_index = result.argmax()
        translation = labels[label_index]

        # Log prediction result
        app.logger.info(f"Prediction: {result}")
        app.logger.info(f"Translated label: {translation}")
        return jsonify({"translation": translation})

    except ValueError as ve:
        app.logger.error(f"ValueError: {ve}")
        return jsonify({"error": str(ve)}), 400

    except Exception as e:
        app.logger.error(f"Unhandled error: {e}")
        return jsonify({"error": "Internal server error.", "details": str(e)}), 500

# Run the app in development mode
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
