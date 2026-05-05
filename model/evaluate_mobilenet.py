import sys
import os

# Make parent folder visible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
from utils.data_loader import load_data

# Load trained MobileNet model
model = tf.keras.models.load_model("model/model.h5")

# Load data (same as training)
train_data, val_data = load_data("data")

# Evaluate on validation set
loss, acc = model.evaluate(val_data)

print("\n✅ MobileNet Validation Accuracy:", round(acc * 100, 2), "%")