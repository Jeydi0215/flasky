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
CORS(app, supports_credentials=True, origins=["https://salinterpret.vercel.app", "https://salinterpret-2373231f0ed4.herokuapp.com"])

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

# Constants
offset = 20
imgSize = 300
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I/J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y/Z"]

# Translation function
def translate_image(img):
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            if imgCrop.size > 0:
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            else:
                return ''
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        translation = labels[index]
    else:
        translation = ''
    return translation

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

    # Return JSON response directly to the frontend
    return jsonify({'img': img_str, 'translation': translation})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
