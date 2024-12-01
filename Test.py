import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and labels at startup
MODEL_PATH = os.path.join(os.getcwd(), "Model", "keras_model.h5")
LABELS_PATH = os.path.join(os.getcwd(), "Model", "labels.txt")

# Global variables for the model and labels
model = None
labels = []

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
    return "Flask server is running!", 200

@app.route('/translate', methods=['POST'])
def translate():
    try:
        logger.info("Received /translate request.")
        
        # Ensure 'image' is present in the request
        if 'image' not in request.files:
            logger.warning("No 'image' key in request.files.")
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files['image']
        logger.info(f"Received file: {file.filename}")
        
        # Validate file type (you can customize this as needed)
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            logger.warning("Unsupported file type.")
            return jsonify({"error": "Unsupported file type"}), 400
        
        # Process the file (placeholder for actual preprocessing)
        import numpy as np
        from PIL import Image
        image = Image.open(file).resize((224, 224))  # Adjust size based on your model
        image_array = np.array(image) / 255.0  # Normalize pixel values
        image_tensor = np.expand_dims(image_array, axis=0)
        
        # Make predictions
        predictions = model.predict(image_tensor)
        label_index = np.argmax(predictions)
        translation = labels[label_index]
        
        logger.info(f"Prediction: {predictions}")
        logger.info(f"Translated label: {translation}")
        
        # Return the translation in the response
        return jsonify({"translation": translation}), 200
    
    except Exception as e:
        logger.error(f"Error in /translate: {e}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
