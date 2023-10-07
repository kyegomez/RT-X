import json
from PIL import Image
import torch
from torchvision import transforms

from efficientnet_pytorch import EfficientNet

model = EfficientNet.from_pretrained("efficientnet-b0")

# Preprocess image
tfms = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
img = tfms(Image.open("img.jpeg")).unsqueeze(0)
print(img.shape)  # torch.Size([1, 3, 224, 224])

# Classify
model.eval()
with torch.no_grad():
    outputs = model(img)
