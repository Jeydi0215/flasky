from flask import Flask, jsonify
from flask_cors import CORS
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import base64
import warnings
import signal
import sys
import time

# Suppress TensorFlow warning
warnings.filterwarnings("ignore", category=UserWarning, message="No training configuration found in the save file")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Signal handler to release video capture
def signal_handler(sig, frame):
    cap.release()  # Release the video capture
    sys.exit(0)    # Exit the program

signal.signal(signal.SIGINT, signal_handler)

# Initialize hand detector and classifier
detector = HandDetector(maxHands=1)
classifier = Classifier(
    r"C:\Users\PC\OneDrive\Desktop\asl\flask-server\venv\Model\keras_model.h5", 
    r"C:\Users\PC\OneDrive\Desktop\asl\flask-server\venv\Model\labels.txt"
)

offset = 20
imgSize = 300
labels = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I/J", "K", 
    "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", 
    "V", "W", "X", "Y/Z"
]

# Initialize variables for timing and hand detection
last_translation_time = 0
last_detected_hand = None

def translate_image_with_label(img):
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
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        # Debug: Print the prediction result and index
        print("Prediction: ", prediction)
        print("Index: ", index)
        print("Label: ", labels[index])

        # Draw label on bounding box
        label = labels[index]
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Green text above the box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green bounding box
        
        return label, img
    return '', img

@app.route('/translate', methods=['GET'])
def translate_asl():
    global last_translation_time, last_detected_hand

    current_time = time.time()
    success, img = cap.read()
    if not success:
        return jsonify({'img': '', 'translation': ''})

    hands, _ = detector.findHands(img)
    if hands:
        # Check if 5 seconds have passed since the last translation
        if current_time - last_translation_time >= 5:
            hand_data = hands[0]['bbox']  # Current hand bounding box
            if hand_data != last_detected_hand:  # Only translate if the hand is new
                translation, img = translate_image_with_label(img)
                if translation:
                    last_detected_hand = hand_data  # Update last detected hand
                    last_translation_time = current_time  # Update the last translation time
            else:
                translation = ''  # Ignore repeated hand
        else:
            translation = ''  # Ignore until 5 seconds pass
    else:
        translation = ''  # No hand detected

    # Encode the frame for display
    _, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'img': img_str, 'translation': translation})

if __name__ == '__main__':
    app.run(debug=True)
