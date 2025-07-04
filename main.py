import torch
from torchvision import models, transforms
from PIL import Image

# Load a pre-trained model
model = models.resnet50(pretrained=True)
model.eval()

# Transform the input image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Load and preprocess image
img = Image.open("dog-puppy-on-garden-royalty-free-image-1586966191.jpg")
input_tensor = transform(img).unsqueeze(0)

# Make prediction
with torch.no_grad():
    output = model(input_tensor)
    probs = torch.nn.functional.softmax(output[0], dim=0)

# Load ImageNet labels
import requests
labels = requests.get("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt").text.splitlines()

# Show top 5 results
top5 = torch.topk(probs, 5)
for i in range(5):
    print(f"{labels[top5.indices[i]]}: {top5.values[i].item():.2f}")
