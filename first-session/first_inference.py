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
print("\nLoading MobileNetV2 model...")
import tensorflow as tf
model = tf.keras.applications.MobileNetV2(weights='imagenet')
print("Model loaded!")

# Quick look at what the model expects:
# Input:  224x224 pixel image, 3 color channels (RGB), pixel values scaled
# Output: 1000 probabilities (one per ImageNet category)
print(f"Input shape:  {model.input_shape}")   # (None, 224, 224, 3)
print(f"Output shape: {model.output_shape}")  # (None, 1000)

# ============================================================
# Step 3: Prepare the image
# ============================================================
# The model expects a specific format. This is "preprocessing":
# 1. Resize to 224x224 (model was trained on this size)
# 2. Convert to numpy array
# 3. Scale pixel values (model-specific normalization)
# 4. Add a batch dimension (model expects batches of images)

print("\nPreprocessing image...")
img = Image.open(IMAGE_PATH)
img = img.resize((224, 224))           # Resize to 224x224
img_array = np.array(img)              # Convert to numpy array (224, 224, 3)
print(f"Image array shape: {img_array.shape}, dtype: {img_array.dtype}")

# MobileNetV2 expects pixel values in [-1, 1] range
img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

# Add batch dimension: (224, 224, 3) → (1, 224, 224, 3)
# "1" means we're classifying 1 image at a time
img_batch = np.expand_dims(img_array, axis=0)
print(f"Batch shape: {img_batch.shape}")

# ============================================================
# Step 4: Run inference
# ============================================================
# This is the magic line. Everything else is just setup.
# model.predict() runs the image through all the neural network
# layers and outputs 1000 probabilities.

print("\nRunning inference...")
import time

start = time.time()
predictions = model.predict(img_batch)
elapsed = (time.time() - start) * 1000

print(f"Inference time: {elapsed:.1f} ms")
print(f"Output shape: {predictions.shape}")  # (1, 1000)

# ============================================================
# Step 5: Interpret the results
# ============================================================
# The output is 1000 numbers. Each one is the probability that
# the image belongs to that category. We want the top 5.

decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=5)[0]

print("\n" + "=" * 50)
print("TOP 5 PREDICTIONS:")
print("=" * 50)
for i, (imagenet_id, label, confidence) in enumerate(decoded):
    bar = "█" * int(confidence * 50)
    print(f"  {i+1}. {label:20s} {confidence*100:5.1f}%  {bar}")

print("\n" + "=" * 50)
print("WHAT JUST HAPPENED:")
print("=" * 50)
print("""
1. We loaded a pre-trained model (MobileNetV2)
   - 3.4 million parameters (weights)
   - Trained on 1.2 million images by Google
   - We didn't train anything — just used their work

2. We preprocessed an image
   - Resized to 224x224 (what the model expects)
   - Normalized pixel values to [-1, 1]

3. We ran inference (model.predict)
   - Image flows through ~50 layers of math
   - Each layer extracts features (edges → shapes → objects)
   - Output: 1000 probabilities

4. On your i.MX8MQ board, you'd do the same thing
   but using TFLite (lighter runtime) and a quantized
   model (INT8 instead of FP32 = 4x smaller + faster)
""")

# ============================================================
# Bonus: Save the model as TFLite (preview of Week 3)
# ============================================================
print("=" * 50)
print("BONUS: Converting to TFLite (what runs on edge)")
print("=" * 50)

# Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save it
tflite_path = "mobilenet_v2.tflite"
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

original_size = os.path.getsize(tflite_path) / (1024 * 1024)
print(f"TFLite model saved: {tflite_path} ({original_size:.1f} MB)")

# Now quantize to INT8 (what you'd actually deploy on i.MX8MQ)
converter_q = tf.lite.TFLiteConverter.from_keras_model(model)
converter_q.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized = converter_q.convert()

tflite_q_path = "mobilenet_v2_quantized.tflite"
with open(tflite_q_path, 'wb') as f:
    f.write(tflite_quantized)

quantized_size = os.path.getsize(tflite_q_path) / (1024 * 1024)
print(f"Quantized model:    {tflite_q_path} ({quantized_size:.1f} MB)")
print(f"Size reduction:     {(1 - quantized_size/original_size)*100:.0f}%")

print(f"""
Next steps:
1. Try changing the image — download any image and point IMAGE_PATH to it
2. Watch 3Blue1Brown "Neural Networks" to understand the layers
3. Next exercise: run the TFLite model instead of the full Keras model
4. Then: cross-compile TFLite for aarch64 and run on the board
""")