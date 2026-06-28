#!/usr/bin/env python3
"""
First Inference Exercise
========================
Goal: Load a pre-trained model, feed it an image, get a prediction.
That's all AI inference is: input → model → output.

We're using MobileNetV2 — a lightweight image classifier trained on
ImageNet (1000 categories: dogs, cats, cars, planes, etc).
"""

import numpy as np
from PIL import Image
import urllib.request
import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Arguments for inference")
    parser.add_argument("-i", "--image_file", type=str, default="test_dog.jpg", help="Image file to run inference on")
    parser.add_argument("-m", "--model", type=str, default="mobilenet_v2.tflite", help="Model to run inference on")
    return parser.parse_args()

if __name__ == "main":
    args = parse_arguments()
    IMAGE_PATH = args.image_file
    
# ============================================================
# Step 1: Download a test image
# ============================================================
# Let's classify a photo of a dog
IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/1200px-YellowLabradorLooking_new.jpg"
IMAGE_PATH = "test_dog.jpg"

if not os.path.exists(IMAGE_PATH):
    print("Downloading test image...")
    urllib.request.urlretrieve(IMAGE_URL, IMAGE_PATH)
    print(f"Saved to {IMAGE_PATH}")

# ============================================================
# Step 2: Load the pre-trained model
# ============================================================
# TensorFlow downloads the model weights automatically (~14MB)
# These weights are what the model "learned" during training.
# We skip training entirely — someone else already did that.
import sys
import pprint
tflite_model = "mobilenet_v2.tflite"
if (len(sys.argv) > 1): 
    tflite_model = sys.argv[1]
print("\nLoading tflite model...")
import tensorflow as tf
interpreter = tf.lite.Interpreter(model_path=tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_index = input_details[0]["index"]
output_index = output_details[0]["index"]

'''print("input details")
print("="*40)
for item in input_details:
    print("Name:", item["name"])
    pprint.pprint(item)
print("="*40)
print("output details")
print("="*40)
for item in output_details:
    print("Name:", item["name"])
    pprint.pprint(item)
print("="*40)
'''

img = Image.open(IMAGE_PATH)
img = img.resize((224, 224))
img_array = np.array(img, dtype=np.float32)
img_array /= 127.5
img_array -= 1.0
print(f"Image array shape: {img_array.shape}, dtype: {img_array.dtype}")

img_batch = np.expand_dims(img_array, axis=0)
#print(f"Batch shape: {img_batch.shape}")

interpreter.set_tensor(input_index, img_batch)
import time
start = time.perf_counter()
interpreter.invoke()
tflite_results = interpreter.get_tensor(output_index)
elapsed = (time.perf_counter() - start) * 1000
print(f"Inference time: {elapsed:.6f} ms")

#print(f'\n\nModel is prediciting {tflite_results}')
sorted_indices = np.argsort(-tflite_results)
#print(tflite_results)
import json
IMAGNET_FILE = "/Users/ashwin/.keras/models/imagenet_class_index.json"
mag_vals = {}
with open(IMAGNET_FILE) as f:
    mag_vals = json.load(f)

print("\n" + "=" * 50)
print("TOP 5 PREDICTIONS:")
print("=" * 50)
for i in range(0, 5):
    indx = sorted_indices[0][i]
    confidence = tflite_results[0][indx]
    [imagenet_id, label] = mag_vals[str(indx)]
    bar = "█" * int(confidence * 50)
    print(f"  {i+1}. {label:20s} {confidence*100:5.1f}%  {bar}")
