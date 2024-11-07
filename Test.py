import requests
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

# Suppress TensorFlow warnings
warnings.filterwarnings("ignore", category=UserWarning, message="No training configuration found in the save file")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define the path for the model and labels
model_path = os.environ.get('MODEL_PATH', 'Model/keras_model.h5')  # Default to 'Model/keras_model.h5' if not set
labels_path = os.environ.get('LABELS_PATH', 'Model/labels.txt')  # Default to 'Model/labels.txt' if not set

# Initialize the classifier and hand detector
classifier = Classifier(model_path, labels_path)
detector = HandDetector(maxHands=1)

# Constants
offset = 20
imgSize = 300
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I/J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y/Z"]

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

@app.route('/translate', methods=['POST'])
def translate_asl():
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

    # Send the translation and image data to Vercel frontend (Salinterpret)
    vercel_url = 'https://salinterpret.vercel.app/Translation'  # Replace with your correct endpoint

    payload = {
        'image': img_str,
        'translation': translation
    }

    try:
        response = requests.post(vercel_url, json=payload)
        response.raise_for_status()  # Raise an exception if the request failed
        app.logger.info(f'Successfully sent data to Vercel, response: {response.status_code}')
    except requests.exceptions.RequestException as e:
        app.logger.error(f'Error sending data to Vercel: {e}')

    return jsonify({'img': img_str, 'translation': translation})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
