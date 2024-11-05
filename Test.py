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

# Initialize Flask app
app = Flask(__name__)

# Enable CORS to allow requests from your React frontend on Vercel
CORS(app, resources={r"/*": {"origins": ["https://salinterpret.vercel.app"]}})

# Define base directory to load model and labels
base_dir = os.path.dirname(os.path.abspath(__file__))

# Paths to model and labels
model_path = os.path.join(base_dir, 'Model', 'keras_model.h5')
labels_path = os.path.join(base_dir, 'Model', 'labels.txt')

# Check if model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

# Load the ASL translation model
classifier = load_model(model_path, compile=False)

# Initialize hand detector from cvzone
detector = HandDetector(maxHands=1)

# Define constants for image processing
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

        # Create a white canvas for resized hand image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        aspectRatio = h / w

        # Resize and center the cropped image
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

        # Get prediction from the model
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
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image data received'}), 400

        # Decode the base64 image
        img_data = base64.b64decode(data['image'])
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Invalid image'}), 400

        # Translate the image to ASL
        translation = translate_image(img)

        return jsonify({'translation': translation})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Ensure the port matches what Render assigns
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)  # Set debug=False for production
