# 🌿 Plant Disease Detection – Inference Package

A lightweight, production-ready deep learning inference package for detecting plant diseases using MobileNetV2. Quickly identify plant leaf diseases with high accuracy and confidence scores.

**Training Notebook:** [Google Colab](https://colab.research.google.com/drive/1R2M_OP9iR7Dy2gRCQd64O5wu3Ro3mw66?usp=sharing) 

**Dataset:** [Dataset Link](https://drive.google.com/drive/folders/15RZjNqS7th4i-dZzF69HXWEsHZSmZrqs?usp=sharing)

---

## 📦 Requirements

- **Python 3.9 or higher**
- **torch** (PyTorch deep learning framework)
- **torchvision** (PyTorch vision utilities)
- **pillow** (Image processing library)


---

## 🚀 Installation

```bash
pip install torch torchvision pillow
```


---

## ⚡ Quick Start

### Basic Usage

1. **Place your plant leaf image** in the project folder (or in an `images/` subfolder)

2. **Edit `inference.py`** and update the test image paths:
   ```python
   TEST_IMAGE_1 = "images/your_image.jpg"
   TEST_IMAGE_2 = "images/another_image.jpg"
   ```

3. **Run the inference script:**
   ```bash
   python inference.py
   ```

4. **Get predictions** in JSON format with disease class and confidence score

---

## 📁 Project Structure

```
plant_disease_inference/
│
├── inference.py                          # Main inference script
├── mobilenetv2_plant_disease_final.pth   # Pre-trained model weights
├── classes.json                          # Disease class mappings
├── README.md                             # This file
├── .gitignore                            # Git ignore configuration
│
└── images/                               # sample images for testing
    ├── apple_scab.JPG
    └── yellow.JPG
```

---

## 🔍 Usage Guide

### Using the `predict()` Function

You can import and use the prediction function in your own Python code:

```python
from inference import predict

# Single image prediction
result = predict("path/to/your/image.jpg")

print(f"Disease: {result['disease']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Batch Processing

Process multiple images:

```python
from inference import predict
import os

image_folder = "images/"
for image_file in os.listdir(image_folder):
    if image_file.endswith(('.jpg', '.JPG', '.png', '.PNG')):
        result = predict(os.path.join(image_folder, image_file))
        print(f"{image_file}: {result['disease']} ({result['confidence']:.2%})")
```

### Running from Command Line

```bash
python inference.py
```

The script will attempt to process test images and display results.

---

## 📤 Output Format

The prediction function returns a dictionary with:

```python
{
    "disease": "Apple Scab",           # Predicted disease class
    "confidence": 0.9876               # Confidence score (0.0 to 1.0)
}
```





