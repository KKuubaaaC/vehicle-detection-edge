"""Vehicle detection inference on Raspberry Pi using ONNX Runtime.

This module provides real-time vehicle detection optimized for edge devices.
It uses ONNX Runtime for efficient inference on Raspberry Pi hardware with
performance monitoring and error handling.

Typical usage example:

    detector = VehicleDetector('models/vehicle_detector.onnx')
    results = detector.detect('input.jpg', confidence_threshold=0.6)
    detector.visualize_and_save(results, 'output.jpg')
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import cv2
import numpy as np
import onnxruntime as ort


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VehicleDetector:
    """Vehicle detection using ONNX model on Raspberry Pi.
    
    This class handles model loading, preprocessing, inference, and
    post-processing for vehicle detection on edge devices.
    
    Attributes:
        model_path: Path to the ONNX model file.
        session: ONNX Runtime inference session.
        input_size: Expected input size for the model (width, height).
        mean: Mean values for normalization.
        std: Standard deviation values for normalization.
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        input_size: Tuple[int, int] = (300, 300),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ) -> None:
        """Initializes the VehicleDetector with ONNX model.
        
        Args:
            model_path: Path to the ONNX model file.
            input_size: Model input size as (width, height). Defaults to (300, 300).
            mean: RGB mean values for normalization. Defaults to ImageNet means.
            std: RGB std values for normalization. Defaults to ImageNet stds.
            
        Raises:
            FileNotFoundError: If model file doesn't exist.
            RuntimeError: If ONNX Runtime session creation fails.
        """
        self.model_path = Path(model_path)
        self.input_size = input_size
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        logger.info(f"Loading ONNX model from: {self.model_path}")
        self.session = self._create_session()
        logger.info("Model loaded successfully")
        
    def _create_session(self) -> ort.InferenceSession:
        """Creates ONNX Runtime inference session with optimizations.
        
        Returns:
            Configured ONNX Runtime inference session.
            
        Raises:
            RuntimeError: If session creation fails.
        """
        try:
            # Configure session options for Raspberry Pi
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            sess_options.intra_op_num_threads = 4  # Optimize for RPi 4 cores
            
            # Create session with CPU execution provider
            session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=['CPUExecutionProvider']
            )
            
            logger.info(f"Available providers: {session.get_providers()}")
            return session
            
        except Exception as e:
            logger.error(f"Failed to create ONNX session: {e}")
            raise RuntimeError(f"ONNX session creation failed: {e}")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocesses image for model input.
        
        Args:
            image: Input image in BGR format (OpenCV default).
            
        Returns:
            Preprocessed image tensor ready for inference.
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        image_resized = cv2.resize(image_rgb, self.input_size)
        
        # Normalize: [0, 255] -> [0, 1] -> standardize
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_normalized = (image_normalized - self.mean) / self.std
        
        # Transpose to CHW format and add batch dimension
        image_transposed = np.transpose(image_normalized, (2, 0, 1))
        image_batched = np.expand_dims(image_transposed, axis=0)
        
        return image_batched.astype(np.float32)
    
    def detect(
        self,
        image_path: Union[str, Path],
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4
    ) -> Dict[str, Union[np.ndarray, float, List]]:
        """Runs vehicle detection on input image.
        
        Args:
            image_path: Path to input image file.
            confidence_threshold: Minimum confidence score for detections.
                Defaults to 0.5.
            nms_threshold: IoU threshold for Non-Maximum Suppression.
                Defaults to 0.4.
                
        Returns:
            Dictionary containing:
                - 'boxes': Bounding boxes as [x1, y1, x2, y2] in original image coords
                - 'scores': Confidence scores for each detection
                - 'image': Original input image
                - 'inference_time': Time taken for inference in seconds
                - 'fps': Frames per second
                
        Raises:
            FileNotFoundError: If image file doesn't exist.
            ValueError: If image cannot be loaded.
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        original_h, original_w = image.shape[:2]
        logger.info(f"Processing image: {image_path} ({original_w}x{original_h})")
        
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Run inference with timing
        start_time = time.time()
        
        try:
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: input_tensor})
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise RuntimeError(f"Model inference failed: {e}")
        
        inference_time = time.time() - start_time
        fps = 1.0 / inference_time if inference_time > 0 else 0
        
        logger.info(f"Inference time: {inference_time*1000:.2f}ms | FPS: {fps:.2f}")
        
        # Post-process outputs
        boxes, scores = self._postprocess(
            outputs,
            original_w,
            original_h,
            confidence_threshold,
            nms_threshold
        )
        
        logger.info(f"Detected {len(boxes)} vehicles")
        
        return {
            'boxes': boxes,
            'scores': scores,
            'image': image,
            'inference_time': inference_time,
            'fps': fps,
            'image_size': (original_w, original_h)
        }
    
    def _postprocess(
        self,
        outputs: List[np.ndarray],
        original_w: int,
        original_h: int,
        confidence_threshold: float,
        nms_threshold: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Post-processes model outputs to extract detections.
        
        Args:
            outputs: Raw model outputs.
            original_w: Original image width.
            original_h: Original image height.
            confidence_threshold: Minimum confidence for keeping detections.
            nms_threshold: IoU threshold for NMS.
            
        Returns:
            Tuple of (boxes, scores):
                - boxes: Array of shape (N, 4) with [x1, y1, x2, y2] coordinates
                - scores: Array of shape (N,) with confidence scores
        """
        # Extract predictions (format depends on model output)
        # Assuming output format: [batch, num_boxes, 6] (x1, y1, x2, y2, conf, class)
        predictions = outputs[0][0]  # Remove batch dimension
        
        # Filter by confidence
        mask = predictions[:, 4] > confidence_threshold
        filtered_predictions = predictions[mask]
        
        if len(filtered_predictions) == 0:
            return np.array([]), np.array([])
        
        # Extract boxes and scores
        boxes = filtered_predictions[:, :4]
        scores = filtered_predictions[:, 4]
        
        # Convert normalized coordinates to absolute
        boxes[:, [0, 2]] *= original_w  # x coordinates
        boxes[:, [1, 3]] *= original_h  # y coordinates
        
        # Apply Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            confidence_threshold,
            nms_threshold
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            boxes = boxes[indices]
            scores = scores[indices]
        else:
            boxes = np.array([])
            scores = np.array([])
        
        return boxes, scores
    
    def visualize_and_save(
        self,
        results: Dict,
        output_path: Union[str, Path],
        box_color: Tuple[int, int, int] = (0, 255, 0),
        text_color: Tuple[int, int, int] = (255, 255, 255),
        thickness: int = 2
    ) -> None:
        """Visualizes detections and saves result image.
        
        Args:
            results: Detection results from detect() method.
            output_path: Path to save output image.
            box_color: BGR color for bounding boxes. Defaults to green.
            text_color: BGR color for text. Defaults to white.
            thickness: Line thickness for boxes. Defaults to 2.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        image = results['image'].copy()
        boxes = results['boxes']
        scores = results['scores']
        
        # Draw detections
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box.astype(int)
            
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), box_color, thickness)
            
            # Draw label with confidence
            label = f"Vehicle: {score:.2f}"
            label_size, _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                1
            )
            
            # Background for text
            cv2.rectangle(
                image,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                box_color,
                -1
            )
            
            # Draw text
            cv2.putText(
                image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                text_color,
                1
            )
        
        # Add performance metrics
        fps_text = f"FPS: {results['fps']:.2f}"
        time_text = f"Time: {results['inference_time']*1000:.2f}ms"
        
        cv2.putText(
            image,
            fps_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )
        cv2.putText(
            image,
            time_text,
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )
        
        # Save image
        cv2.imwrite(str(output_path), image)
        logger.info(f"Results saved to: {output_path}")
    
    def detect_video(
        self,
        video_path: Union[str, Path],
        output_path: Union[str, Path],
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        show_preview: bool = False
    ) -> None:
        """Runs detection on video file.
        
        Args:
            video_path: Path to input video file.
            output_path: Path to save output video.
            confidence_threshold: Minimum confidence for detections.
            nms_threshold: IoU threshold for NMS.
            show_preview: Whether to show real-time preview. Defaults to False.
        """
        video_path = Path(video_path)
        output_path = Path(output_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
        
        # Create video writer
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_count = 0
        total_time = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Preprocess
                input_tensor = self.preprocess(frame)
                
                # Inference
                start_time = time.time()
                input_name = self.session.get_inputs()[0].name
                outputs = self.session.run(None, {input_name: input_tensor})
                inference_time = time.time() - start_time
                total_time += inference_time
                
                # Post-process
                boxes, scores = self._postprocess(
                    outputs, width, height,
                    confidence_threshold, nms_threshold
                )
                
                # Draw detections
                for box, score in zip(boxes, scores):
                    x1, y1, x2, y2 = box.astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"Vehicle: {score:.2f}"
                    cv2.putText(frame, label, (x1, y1-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Add FPS counter
                current_fps = 1.0 / inference_time
                cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Write frame
                out.write(frame)
                
                # Show preview
                if show_preview:
                    cv2.imshow('Detection', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Progress
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    avg_fps = frame_count / total_time
                    logger.info(f"Progress: {progress:.1f}% | Avg FPS: {avg_fps:.1f}")
        
        finally:
            cap.release()
            out.release()
            if show_preview:
                cv2.destroyAllWindows()
        
        avg_fps = frame_count / total_time
        logger.info(f"Video processing complete!")
        logger.info(f"Processed {frame_count} frames | Avg FPS: {avg_fps:.2f}")
        logger.info(f"Output saved to: {output_path}")


def main() -> None:
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Vehicle detection on Raspberry Pi using ONNX',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/vehicle_detector.onnx',
        help='Path to ONNX model file'
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Path to input image'
    )
    parser.add_argument(
        '--video',
        type=str,
        help='Path to input video'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/output.jpg',
        help='Path to save output'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='Confidence threshold for detections'
    )
    parser.add_argument(
        '--nms',
        type=float,
        default=0.4,
        help='NMS IoU threshold'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show real-time preview (video only)'
    )
    
    args = parser.parse_args()
    
    if not args.image and not args.video:
        parser.error("Either --image or --video must be specified")
    
    try:
        # Initialize detector
        detector = VehicleDetector(args.model)
        
        if args.image:
            # Run detection on image
            results = detector.detect(
                args.image,
                confidence_threshold=args.confidence,
                nms_threshold=args.nms
            )
            
            # Visualize and save
            detector.visualize_and_save(results, args.output)
            
            # Print summary
            print(f"\n{'='*50}")
            print(f"Detection Summary:")
            print(f"{'='*50}")
            print(f"Vehicles detected: {len(results['boxes'])}")
            print(f"Inference time: {results['inference_time']*1000:.2f}ms")
            print(f"FPS: {results['fps']:.2f}")
            print(f"Output saved: {args.output}")
            print(f"{'='*50}\n")
        
        elif args.video:
            # Run detection on video
            detector.detect_video(
                args.video,
                args.output,
                confidence_threshold=args.confidence,
                nms_threshold=args.nms,
                show_preview=args.show
            )
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == '__main__':
    main()