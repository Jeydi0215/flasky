from flask import Flask, jsonify, request
from flask_cors import CORS
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import base64
import os
import warnings

# Suppress TensorFlow warnings
warnings.filterwarnings("ignore", category=UserWarning, message="No training configuration found in the save file")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define the base directory
base_dir = os.path.dirname(os.path.abspath(__file__))
print("Base directory:", base_dir)  # Print base directory for debugging

# Construct paths to the model and labels file
model_path = os.path.join(base_dir, 'venv', 'Model', 'keras_model.h5')
labels_path = os.path.join(base_dir, 'venv', 'Model', 'labels.txt')

# Check if the model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"The model file was not found at the specified path: {model_path}")

# Initialize hand detector and classifier
try:
    detector = HandDetector(maxHands=1)
    classifier = Classifier(model_path, labels_path)
except Exception as e:
    print(f"Error initializing models: {e}")  # Log initialization error
    raise

offset = 20
imgSize = 300
labels = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I/J", "K",
    "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
    "V", "W", "X", "Y/Z"
]

def translate_image(img):
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape
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
    if 'image' not in request.files:
        return jsonify({'img': '', 'translation': 'No image uploaded'})

    # Get the image from the request
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'img': '', 'translation': 'Invalid image'})

    # Translate the image
    translation = translate_image(img)

    # Encode the image back to base64
    _, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'img': img_str, 'translation': translation})

if __name__ == '__main__':
    # Use the PORT environment variable if it exists, otherwise default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
