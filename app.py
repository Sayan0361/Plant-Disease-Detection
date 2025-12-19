import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import streamlit as st

# ---------------- CONFIG ----------------
MODEL_PATH = "mobilenetv2_plant_disease_final.pth"
CLASSES_PATH = "classes.json"
IMAGE_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------------------------

@st.cache_resource
def load_model():
    with open(CLASSES_PATH, "r") as f:
        classes = json.load(f)

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        len(classes)
    )

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["weights"], strict=True)
    model.to(DEVICE)
    model.eval()

    return model, classes


transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])


def predict_image(image, model, classes):
    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        idx = probs.argmax().item()

    return classes[idx], float(probs[idx])


# ---------------- UI ----------------
st.set_page_config(page_title="Plant Disease Detection", layout="centered")

st.title("🌱 Plant Disease Detection")
st.write("Upload a plant leaf image to detect disease")

uploaded_file = st.file_uploader(
    "Upload leaf image",
    type=["jpg","JPG", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    model, classes = load_model()

    if st.button("Predict"):
        disease, confidence = predict_image(image, model, classes)

        st.success(f"**Disease:** {disease}")
        st.info(f"**Confidence:** {confidence:.2%}")
