import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os

# ---------------- CONFIG ----------------
MODEL_PATH = "mobilenetv2_plant_disease_final.pth"
CLASSES_PATH = "classes.json"
IMAGE_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Array of test images
TEST_IMAGES = [
    "images/apple_scab.JPG",
    "images/yellow.JPG",
    "images/brown_spot.JPG",
    "images/healthy.JPG",
    "images/leaf_blight.JPG",
    "images/black.jpg",
    "images/download (3).jpg",
]

TEST_INDEX = 6  # change this to 0, 1, 2 ...
# ---------------------------------------

# Load classes
with open(CLASSES_PATH, "r") as f:
    classes = json.load(f)

# Load model
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    len(classes)
)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["weights"], strict=True)
model.to(DEVICE)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

def predict(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {"error": f"Invalid image: {e}"}

    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        idx = probs.argmax().item()

    return {
        "image": image_path,
        "disease": classes[idx],
        "confidence": round(float(probs[idx]), 4)
    }

if __name__ == "__main__":
    if TEST_INDEX >= len(TEST_IMAGES):
        print(f"Invalid TEST_INDEX {TEST_INDEX}. Max index is {len(TEST_IMAGES)-1}")
    else:
        test_image = TEST_IMAGES[TEST_INDEX]
        if os.path.exists(test_image):
            print(predict(test_image))
        else:
            print(f"Test image not found: {test_image}")
