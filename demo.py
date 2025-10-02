import os
import time
import argparse
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


class CarDetector:
    
    
    def __init__(self, model_path, image_size=300):
        self.image_size = image_size
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        
    def preprocess(self, image):
        """Preprocess image for inference"""
        original_size = image.size
        image_resized = image.resize((self.image_size, self.image_size))
        
        # ImageNet normalization
        img_array = np.array(image_resized).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean) / std
        
        # CHW format + batch dimension
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, original_size
    
    def predict(self, image_path, threshold=0.5):
        """Run inference on image"""
        image = Image.open(image_path).convert('RGB')
        img_array, original_size = self.preprocess(image)
        
        # Inference
        start = time.time()
        class_pred, box_pred, conf_pred = self.session.run(None, {self.input_name: img_array})
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
        
        return result, image, inference_time
    
    def benchmark(self, image_path, runs=100):
        """Benchmark inference performance"""
        print(f"\nBenchmarking ({runs} runs)...")
        
        image = Image.open(image_path).convert('RGB')
        img_array, _ = self.preprocess(image)
        
        # Warmup
        for _ in range(10):
            self.session.run(None, {self.input_name: img_array})
        
        # Benchmark
        times = []
        for _ in range(runs):
            start = time.time()
            self.session.run(None, {self.input_name: img_array})
            times.append((time.time() - start) * 1000)
        
        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'p50': np.percentile(times, 50),
            'p95': np.percentile(times, 95),
            'p99': np.percentile(times, 99)
        }


def visualize_result(image, result, output_path):
    """Draw bounding box and save result"""
    if result is None:
        return
    
    draw = ImageDraw.Draw(image)
    xmin, ymin, xmax, ymax = result['box']
    
    # Draw box
    draw.rectangle([xmin, ymin, xmax, ymax], outline='red', width=3)
    
    # Draw label
    label = f"Car: {result['confidence']:.1%}"
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    bbox = draw.textbbox((xmin, ymin - 25), label, font=font)
    draw.rectangle(bbox, fill='red')
    draw.text((xmin, ymin - 25), label, fill='white', font=font)
    
    image.save(output_path)


def main():
    parser = argparse.ArgumentParser(
        description='Car Detection Demo - Edge-Optimized Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--image', required=True, help='Input image path')
    parser.add_argument('--model', default='models/car_detection_model.onnx', help='ONNX model path')
    parser.add_argument('--threshold', type=float, default=0.5, help='Detection threshold')
    parser.add_argument('--output', default='result.jpg', help='Output image path')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  Car Detection Demo - Edge-Optimized Inference")
    print("=" * 60)
    
    # Check files
    if not os.path.exists(args.model):
        print(f"\nError: Model not found at {args.model}")
        return
    
    if not os.path.exists(args.image):
        print(f"\nError: Image not found at {args.image}")
        return
    
    model_size = os.path.getsize(args.model) / (1024 * 1024)
    print(f"\nModel: {args.model} ({model_size:.1f} MB)")
    print(f"Image: {args.image}")
    
    # Initialize detector
    print("\nInitializing detector...")
    detector = CarDetector(args.model)
    
    # Run inference
    print("\nRunning inference...")
    result, image, inference_time = detector.predict(args.image, args.threshold)
    
    # Display results
    print("\n" + "=" * 60)
    print("  Results")
    print("=" * 60)
    
    if result:
        print(f"  Car detected!")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Bounding box: {result['box']}")
        print(f"  Inference time: {inference_time:.2f}ms")
        
        visualize_result(image, result, args.output)
        print(f"  Saved to: {args.output}")
    else:
        print(f"  No detection (threshold: {args.threshold})")
        print(f"  Inference time: {inference_time:.2f}ms")
    
    # Benchmark
    if args.benchmark:
        print("\n" + "=" * 60)
        print("  Performance Benchmark")
        print("=" * 60)
        
        stats = detector.benchmark(args.image, runs=100)
        
        print(f"\nInference time statistics (100 runs):")
        print(f"  Mean:   {stats['mean']:.2f}ms")
        print(f"  Std:    {stats['std']:.2f}ms")
        print(f"  Min:    {stats['min']:.2f}ms")
        print(f"  Max:    {stats['max']:.2f}ms")
        print(f"  P50:    {stats['p50']:.2f}ms")
        print(f"  P95:    {stats['p95']:.2f}ms")
        print(f"  P99:    {stats['p99']:.2f}ms")
        print(f"\nThroughput: {1000/stats['mean']:.1f} FPS")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()