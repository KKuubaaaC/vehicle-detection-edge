"""
Inference script for car detection - ONNX optimized
Test: python inference.py --image test.jpg --model models/car_detection_model.onnx
"""
import argparse
import time
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont


def preprocess_image(image_path, size=300):
    """Load and preprocess image"""
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # Resize
    image_resized = image.resize((size, size))
    
    # Normalize
    img_array = np.array(image_resized).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    
    # CHW format + batch dimension
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array, original_size, image


def predict(onnx_path, image_path, threshold=0.5):
    """Run inference"""
    # Load model
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    
    # Preprocess
    img_array, original_size, original_image = preprocess_image(image_path)
    
    # Inference
    start = time.time()
    class_pred, box_pred, conf_pred = session.run(None, {input_name: img_array})
    inference_time = (time.time() - start) * 1000
    
    # Postprocess
    exp_scores = np.exp(class_pred - np.max(class_pred))
    probs = exp_scores / exp_scores.sum()
    
    predicted_class = np.argmax(probs)
    class_confidence = float(probs[0][predicted_class])
    obj_confidence = float(conf_pred[0][0])
    confidence = class_confidence * obj_confidence
    
    result = None
    if confidence > threshold and predicted_class > 0:
        # Convert box from center format to corners
        cx, cy, w, h = box_pred[0]
        xmin = int((cx - w/2) * original_size[0])
        ymin = int((cy - h/2) * original_size[1])
        xmax = int((cx + w/2) * original_size[0])
        ymax = int((cy + h/2) * original_size[1])
        
        result = {
            'class': 'car',
            'confidence': confidence,
            'box': [xmin, ymin, xmax, ymax],
            'inference_time_ms': inference_time
        }
    
    return result, original_image, inference_time


def draw_result(image, result):
    """Draw bounding box on image"""
    if result is None:
        return image
    
    draw = ImageDraw.Draw(image)
    xmin, ymin, xmax, ymax = result['box']
    
    # Draw box
    draw.rectangle([xmin, ymin, xmax, ymax], outline='red', width=3)
    
    # Draw label
    label = f"Car: {result['confidence']:.2%}"
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    draw.text((xmin, ymin - 25), label, fill='red', font=font)
    
    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Car detection inference')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--model', default='models/car_detection_model.onnx', help='Path to ONNX model')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--output', default='result.jpg', help='Output image path')
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    print(f"Processing image: {args.image}")
    
    result, image, inference_time = predict(args.model, args.image, args.threshold)
    
    if result:
        print(f"\n✓ Car detected!")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Box: {result['box']}")
        print(f"  Inference time: {inference_time:.2f}ms")
        
        # Draw and save
        result_image = draw_result(image, result)
        result_image.save(args.output)
        print(f"  Saved to: {args.output}")
    else:
        print(f"\n No car detected (threshold: {args.threshold})")
        print(f"  Inference time: {inference_time:.2f}ms")