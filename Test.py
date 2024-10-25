from flask import Flask, jsonify, request
from flask_cors import CORS
import cv2
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model
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
model_path = os.path.join(base_dir, 'Model', 'keras_model.h5')  # Adjust this if necessary
labels_path = os.path.join(base_dir, 'Model', 'labels.txt')      # Adjust this if necessary

# Check if the model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"The model file was not found at the specified path: {model_path}")

# Load and compile the model
classifier = load_model(model_path)
classifier.compile(optimizer='adam',  # Use the same optimizer as during training
                   loss='categorical_crossentropy',  # Use the same loss function
                   metrics=['accuracy'])  # Add any metrics you want to track

# Initialize hand detector
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300
labels = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I/J", "K",
    "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
    "V", "W", "X", "Y/Z"
]

def translate_image(img):
    """Processes the input image and predicts the corresponding ASL letter."""
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

@app.route('/')
def index():
    """Welcome route."""
    return "Welcome to the ASL Translation Service! Use the /translate endpoint to send images."

@app.route('/translate', methods=['POST'])
def translate_asl():
    """Endpoint for translating ASL images."""
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'translation': 'No image data received'}), 400

    # Decode the base64 image
    img_data = base64.b64decode(data['image'])
    img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'translation': 'Invalid image'}), 400

    # Translate the image
    translation = translate_image(img)

    return jsonify({'translation': translation})

if __name__ == '__main__':
    # Ensure the correct port is used on deployment
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=True)
