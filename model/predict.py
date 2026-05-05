import tensorflow as tf
import numpy as np
import cv2

# Load model
import os

model_path = os.path.join(os.path.dirname(__file__), "model.h5")
model = tf.keras.models.load_model(model_path)

# Class labels (IMPORTANT: same order as training)
class_names = [
    'apple_scab',
    'bacterial_spot',
    'early_blight',
    'healthy',
    'late_blight',
    'leaf_mold',
    'septoria'
]

# Advanced remedies system
remedies = {
    "apple_scab": {
        "remedy": ["Apply fungicide spray", "Remove infected leaves"],
        "prevention": ["Avoid overhead watering", "Ensure good air circulation"]
    },
    "bacterial_spot": {
        "remedy": ["Use copper-based spray", "Remove infected parts"],
        "prevention": ["Avoid wet leaves", "Use disease-free seeds"]
    },
    "early_blight": {
        "remedy": ["Apply neem oil", "Remove affected leaves"],
        "prevention": ["Rotate crops", "Maintain soil health"]
    },
    "healthy": {
        "remedy": ["No action needed"],
        "prevention": ["Maintain proper watering and sunlight"]
    },
    "late_blight": {
        "remedy": ["Apply fungicide", "Improve drainage"],
        "prevention": ["Avoid excess moisture", "Ensure spacing"]
    },
    "leaf_mold": {
        "remedy": ["Reduce humidity", "Use fungicide"],
        "prevention": ["Avoid overcrowding", "Improve ventilation"]
    },
    "septoria": {
        "remedy": ["Remove infected leaves", "Apply fungicide"],
        "prevention": ["Avoid overhead watering", "Keep leaves dry"]
    }
}

def predict_image(image_path):
    # Read image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))

    # Normalize
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    predictions = model.predict(img)
    class_index = np.argmax(predictions)
    confidence = np.max(predictions)

    disease = class_names[class_index]
    info = remedies[disease]

    return disease, confidence, info


# Test block
if __name__ == "__main__":
    disease, conf, info = predict_image("test.jpg")

    print("\nDisease:", disease)
    print("Confidence:", round(conf * 100, 2), "%")

    print("\nRemedy:")
    for r in info["remedy"]:
        print("-", r)

    print("\nPrevention:")
    for p in info["prevention"]:
        print("-", p)