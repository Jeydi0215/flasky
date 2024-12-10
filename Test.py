from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
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

# Enable CORS for specific origin
CORS(app, resources={r"/*": {"origins": "https://salinterpret.vercel.app"}})

# Paths to model and labels
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'Model', 'keras_model.h5')

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

# Load the model and initialize hand detector
classifier = load_model(model_path, compile=False)
detector = HandDetector(maxHands=1)

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
        
        imgSize = 300
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - 20:y + h + 20, x - 20:x + w + 20]

        aspectRatio = h / w
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

        prediction = classifier.predict(np.expand_dims(imgWhite, axis=0))
        index = np.argmax(prediction)
        translation = labels[index]
    else:
        translation = ''
    return translation

@app.route('/translate', methods=['POST', 'OPTIONS'])
@cross_origin(origin='https://salinterpret.vercel.app')  # Ensure CORS for this endpoint
def translate_asl():
    """Endpoint for translating ASL images."""
    if request.method == 'OPTIONS':
        # Return a response for the preflight request
        response = jsonify({'status': 'Preflight handled'})
        response.headers.add("Access-Control-Allow-Origin", "https://salinterpret.vercel.app")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
        response.headers.add("Access-Control-Allow-Methods", "POST,OPTIONS")
        return response

    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image data received'}), 400

        img_data = base64.b64decode(data['image'])
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Invalid image'}), 400

        translation = translate_image(img)
        response = jsonify({'translation': translation})
        response.headers.add("Access-Control-Allow-Origin", "https://salinterpret.vercel.app")
        return response

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=True)
