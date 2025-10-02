# Vehicle Detection Edge

Real-time vehicle detection system optimized for edge devices using SSD architecture with MobileNetV2 backbone. Designed for deployment on resource-constrained hardware such as Raspberry Pi 4.

## Overview

This project implements a complete pipeline for training, exporting, and deploying a lightweight object detection model for vehicle detection. The system achieves 85% validation accuracy while maintaining real-time performance on edge devices through ONNX Runtime optimization.

## Technical Stack

### Core Technologies
- **PyTorch 2.0+** - Deep learning framework for model training
- **MobileNetV2** - Lightweight CNN backbone pretrained on ImageNet
- **SSD (Single Shot Detector)** - Object detection architecture
- **ONNX Runtime** - Optimized inference engine for edge deployment
- **OpenCV** - Image processing and visualization
- **NumPy** - Numerical computations

### Development Tools
- **Jupyter Notebook** - Interactive model training
- **pytest** - Unit and integration testing
- **flake8** - Code linting and style checking
- **Docker** - Containerized deployment
- **GitHub Actions** - CI/CD pipeline

## Architecture

### Model Design
- **Backbone**: MobileNetV2 (pretrained on ImageNet)
- **Detection Head**: SSD with three prediction layers
  - Classification head for object detection
  - Bounding box regression head
  - Confidence scoring head
- **Input**: 300x300 RGB images
- **Output**: Bounding boxes with confidence scores

### Training Pipeline
1. Data preprocessing from CSV annotations
2. Transfer learning with frozen MobileNetV2 layers
3. Multi-task loss optimization (classification + localization + confidence)
4. Adam optimizer with gradient clipping
5. Early stopping based on validation loss
6. Model checkpointing for best performance

### Deployment Pipeline
1. Model export to ONNX format
2. ONNX Runtime optimization for CPU inference
3. Quantization for reduced model size
4. Docker containerization for edge deployment

## Performance

| Device | Framework | FPS | Accuracy | Model Size |
|--------|-----------|-----|----------|------------|
| Raspberry Pi 4 | ONNX Runtime | 15.3 | 85% | 4.8 MB |
| Desktop CPU | ONNX Runtime | 42.7 | 85% | 4.8 MB |
| Desktop GPU | PyTorch | 120+ | 85% | 4.8 MB |