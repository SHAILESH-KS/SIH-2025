import torch
import torch.nn as nn
import os
from torchvision import models, transforms, datasets
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(BASE_DIR, "data")
model_path = os.path.join(BASE_DIR, "models", "best_model.pth")
test_image_path = os.path.join(BASE_DIR, "test.jpg")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = datasets.ImageFolder(data_path)

class_names = dataset.classes

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))

model.load_state_dict(torch.load(model_path, map_location=device))

model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

image = Image.open(test_image_path).convert("RGB")

image = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(image)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    confidence, predicted = torch.max(probabilities, 1)

print("Prediction:", class_names[predicted.item()])
print("Confidence:", confidence.item()*100)