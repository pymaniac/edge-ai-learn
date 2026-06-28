#!/usr/bin/env python3
"""
First Inference Exercise
========================
Goal: Load a pre-trained model, feed it an image, get a prediction.
That's all AI inference is: input → model → output.

We're using MobileNetV2 — a lightweight image classifier trained on
ImageNet (1000 categories: dogs, cats, cars, planes, etc).
"""


import argparse
import numpy as np
from PIL import Image
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(description="Arguments for inference")
    parser.add_argument("-i", "--image_file", type=str, default="test_dog.jpg", help="Image file to run inference on")
    parser.add_argument("-m", "--model", type=str, default="mobilenet_v2.tflite", help="Model to run inference on")
    parser.add_argument("-l", "--labels_file", type=str, default="labels.txt", help="Full filename of labels file")
    return parser.parse_args()

def load_model(tflite_model, img_batch):
    # ============================================================
    # Step 2: Load the pre-trained model
    # ============================================================
    # TensorFlow downloads the model weights automatically (~14MB)
    # These weights are what the model "learned" during training.
    # We skip training entirely — someone else already did that.
    import pprint
    print("\nLoading tflite model...")
    import tflite_runtime.interpreter as tflite

    interpreter = tflite.Interpreter(model_path=tflite_model)
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

    interpreter.set_tensor(input_index, img_batch)
    import time
    start = time.perf_counter()
    interpreter.invoke()
    tflite_results = interpreter.get_tensor(output_index)
    elapsed = (time.perf_counter() - start) * 1000
    print(f"Inference time: {elapsed:.6f} ms")
    return tflite_results

def process_image(image_file):
    img = Image.open(image_file)
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.uint8)
    img_batch = np.expand_dims(img_array, axis=0)
    print(f"Image array shape: {img_array.shape}, dtype: {img_array.dtype}")
    print(f"Batch shape: {img_batch.shape}")
    return img_batch

def run_prediction(tflite_results, labels_file):
    with open(labels_file) as f:
        labels = f.readlines()

    top5 = np.argsort(-tflite_results[0].astype(int))[:5]
    print("\n" + "=" * 50)
    print("TOP 5 PREDICTIONS:")
    print("=" * 50)
    for i, idx in enumerate(top5):
        confidence = tflite_results[0][idx] * 0.00390625
        label = labels[idx].strip()
        #print(f"  {i+1}. {label:30s} {confidence*100:.1f}%")
        bar = "█" * int(confidence * 50)
        print(f"  {i+1}. {label:30s} {confidence*100:5.1f}%  {bar}")

if __name__ == "__main__":
    args = parse_arguments()
    print(args)
    image_batch = process_image(args.image_file)
    tflite_results = load_model(args.model, image_batch)
    run_prediction(tflite_results, args.labels_file)
    sys.exit(0)
    
