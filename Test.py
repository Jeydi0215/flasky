from flask import Flask, jsonify
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
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Define the path for the model dynamically based on the environment
# You can either use absolute paths or relative paths based on where your app is running
model_path = os.environ.get('MODEL_PATH', 'Model/keras_model.h5')  # Default to 'Model/keras_model.h5' if not set
labels_path = os.environ.get('LABELS_PATH', 'Model/labels.txt')  # Default to 'Model/labels.txt' if not set

# Use environment variables for Heroku or local paths
classifier = Classifier(model_path, labels_path)

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

@app.route('/translate', methods=['GET'])
def translate_asl():
    success, img = cap.read()
    if not success:
        return jsonify({'img': '', 'translation': ''})

    translation = translate_image(img)

    _, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'img': img_str, 'translation': translation})

if __name__ == '__main__':
    app.run(debug=True)
