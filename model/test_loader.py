import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.data_loader import load_data

train, val = load_data("../data")

print("Classes:", train.class_indices)