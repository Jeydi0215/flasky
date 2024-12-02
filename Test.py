import os
import pygame
import threading
import cv2
import numpy as np
from pygame.locals import QUIT
from tensorflow.keras.models import load_model
import tensorflow as tf

# Define paths for the model and labels
MODEL_PATH = os.path.join(os.getcwd(), "Model", "keras_model.h5")
LABELS_PATH = os.path.join(os.getcwd(), "Model", "labels.txt")

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("ASL Translator")

# Fonts and Colors
font = pygame.font.Font(None, 36)
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)

# Webcam setup (using OpenCV)
cap = cv2.VideoCapture(0)

# Load TensorFlow model and labels
model = load_model(MODEL_PATH)
with open(LABELS_PATH, "r") as f:
    labels = f.read().splitlines()

# Variables
translation = "Translation will appear here..."
running = True
is_translating = False

# Function to process frame and predict translation
def process_frame(frame):
    global translation
    try:
        # Preprocess frame for the model
        frame_resized = cv2.resize(frame, (224, 224))
        frame_normalized = frame_resized / 255.0
        frame_expanded = np.expand_dims(frame_normalized, axis=0)

        # Model prediction
        predictions = model.predict(frame_expanded)
        predicted_label = labels[np.argmax(predictions)]
        translation = predicted_label
    except Exception as e:
        translation = f"Error: {str(e)}"

# Threaded translation to avoid blocking the UI
def capture_and_translate():
    global is_translating
    while is_translating:
        ret, frame = cap.read()
        if ret:
            # Start a thread to process the frame
            threading.Thread(target=process_frame, args=(frame,)).start()

# Main loop
while running:
    screen.fill(white)
    
    # Event Handling
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:  # Toggle translation on SPACE key
                is_translating = not is_translating
                if is_translating:
                    threading.Thread(target=capture_and_translate, daemon=True).start()

    # Webcam Feed
    ret, frame = cap.read()
    if ret:
        # Convert OpenCV frame (BGR) to Pygame surface (RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)
        frame_surface = pygame.surfarray.make_surface(frame)
        screen.blit(frame_surface, (50, 50))  # Position the video feed

    # Display Translation
    translation_text = font.render(f"Translation: {translation}", True, black)
    screen.blit(translation_text, (50, 500))

    # Instructions
    instructions = font.render("Press SPACE to start/stop translating.", True, red)
    screen.blit(instructions, (50, 550))

    pygame.display.flip()

# Cleanup
cap.release()
pygame.quit()
