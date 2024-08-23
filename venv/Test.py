from flask import Flask, jsonify, request
from flask_cors import CORS
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import base64
import warnings

# Suppress TensorFlow warning
warnings.filterwarnings("ignore", category=UserWarning, message="No training configuration found in the save file")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)  # Allow detection of 2 hands
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I/J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y/Z"]
current_phrase = []  

def translate_image(img):
    hands, img = detector.findHands(img)
    translations = []
    
    # Debugging: Print number of hands detected
    print(f"Number of hands detected: {len(hands)}")
    
    for hand in hands:
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        if imgCrop.size == 0:
            translations.append('')
            continue

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

        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        translations.append(labels[index])

    return translations

@app.route('/translate', methods=['GET'])
def translate_asl():
    success, img = cap.read()
    if not success:
        return jsonify({'img': '', 'translations': [], 'current_phrase': ''})

    translations = translate_image(img)
    current_phrase.extend(translations)

    _, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'img': img_str, 'translations': translations, 'current_phrase': ' '.join(current_phrase)})

@app.route('/clear', methods=['POST'])
def clear_phrase():
    global current_phrase
    current_phrase = []
    return jsonify({'message': 'Phrase cleared', 'current_phrase': ''})

if __name__ == '__main__':
    app.run(debug=True)
