from flask import Flask, jsonify, request
from flask_cors import CORS
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import base64
import warnings
import os
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)

# Suppress TensorFlow warnings
warnings.filterwarnings("ignore", category=UserWarning, message="No training configuration found in the save file")

app = Flask(__name__)

# Enable CORS for specific origins
CORS(app, supports_credentials=True, origins=[
    "https://salinterpret.vercel.app",
    "https://salinterpret-2373231f0ed4.herokuapp.com"
])

# Model and label paths
model_path = os.environ.get('MODEL_PATH', 'Model/keras_model.h5')
labels_path = os.environ.get('LABELS_PATH', 'Model/labels.txt')

# Initialize the classifier and hand detector
try:
    classifier = Classifier(model_path, labels_path)
    detector = HandDetector(maxHands=1)
    logging.info("Classifier and HandDetector initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing Classifier or HandDetector: {e}")
    exit(1)  # Exit the application if initialization fails

# Constants
offset = 20
imgSize = 300
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I/J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y/Z"]

# Default route for the root URL
@app.route("/")
def home():
    return "Welcome to Flask!"

# Translation function
def translate_image(img):
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Ensure cropping is within bounds
        h_img, w_img, _ = img.shape
        x1, y1 = max(0, x - offset), max(0, y - offset)
        x2, y2 = min(w_img, x + w + offset), min(h_img, y + h + offset)
        imgCrop = img[y1:y2, x1:x2]

        # Prepare white image and resize cropped hand
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        aspectRatio = h / w
        try:
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            return labels[index]
        except Exception as e:
            logging.error(f"Error during translation: {e}")
            return ''
    else:
        return ''

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "https://salinterpret.vercel.app"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response

@app.route('/translate', methods=['POST', 'OPTIONS'])
def translate_asl():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'preflight successful'}), 200

    try:
        # Validate file existence
        file = request.files.get('image')
        if not file:
            return jsonify({'error': 'No image file provided'}), 400

        # Convert the uploaded file to an image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Get the translation
        translation = translate_image(img)

        # Encode the image for the response
        _, buffer = cv2.imencode('.jpg', img)
        img_str = base64.b64encode(buffer).decode('utf-8')

        # Return the response
        return jsonify({'img': img_str, 'translation': translation}), 200

    except Exception as e:
        logging.error(f"Translation error: {e}")
        return jsonify({'error': f"Server error: {e}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'running'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
