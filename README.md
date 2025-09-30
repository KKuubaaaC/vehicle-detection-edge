Car Detection Pipeline - SSD MobileNetV2
Project Overview
Deep learning pipeline for car detection built with PyTorch. Uses SSD architecture with MobileNetV2 backbone trained on Kaggle Car Object Detection dataset. Achieves 85 percent validation accuracy with complete training and inference pipeline.
What It Does
Detects cars in images using single shot detection. Takes an image as input and outputs bounding box coordinates plus confidence score. Trained end-to-end from raw CSV annotations to production model with automated evaluation and Kaggle submission generation.
Technical Implementation
Built custom SSD detector with three prediction heads: object classification, bounding box regression, and objectness confidence. Uses pretrained MobileNetV2 for feature extraction with transfer learning. Multi-task loss function combines classification cross-entropy, localization smooth L1, and confidence binary cross-entropy.
Training pipeline includes gradient clipping for stability, early stopping to prevent overfitting, and learning rate scheduling with ReduceLROnPlateau. Model checkpoints saved automatically with best validation loss. Complete data preprocessing handles CSV parsing, train-val split, and image normalization.
Stack
PyTorch for model training and inference. Torchvision for pretrained weights and transforms. Pandas for CSV handling. NumPy for numerical operations. Matplotlib for visualization and result plotting.