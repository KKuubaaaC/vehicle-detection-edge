import torch
from model import MobileNetV2SSD

# Load PyTorch model
model = MobileNetV2SSD(num_classes=2)
checkpoint = torch.load('models/best_model.pth', map_location='cpu')

if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.eval()

# Export to ONNX
dummy_input = torch.randn(1, 3, 300, 300)

torch.onnx.export(
    model,
    dummy_input,
    'models/car_detection_model.onnx',
    input_names=['input'],
    output_names=['class_output', 'box_output', 'confidence_output'],
    opset_version=11
)

print("Model exported to models/car_detection_model.onnx")