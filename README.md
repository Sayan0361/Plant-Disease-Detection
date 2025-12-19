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
- **streamlit** (Web UI framework)


---

## 🚀 Installation

```bash
pip install torch torchvision pillow streamlit
```


---

## ⚡ Quick Start

### Web UI with Streamlit (Recommended)

1. **Ensure all dependencies are installed:**
   ```bash
   pip install torch torchvision pillow streamlit
   ```

2. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

3. **The app will:**
   - Open in your default browser (usually at `http://localhost:8501`)
   - Display an interactive web interface
   - Allow you to upload plant leaf images
   - Show predictions with confidence scores in real-time

4. **To stop the app:** Press `Ctrl+C` in your terminal

### Basic Usage with inference.py

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
├── app.py                                # Streamlit web application
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

### Using the Streamlit Web App (app.py)

The `app.py` file provides an interactive web interface for disease detection:

**Features:**
- 📤 Upload plant leaf images via a user-friendly interface
- 🎯 Get instant predictions with disease classification
- 📊 View confidence scores for predictions
- 🖼️ Image preview before and after processing

**How to use:**
1. Run: `streamlit run app.py`
2. Select "Upload leaf image" button
3. Choose a plant leaf image (JPG, JPEG, or PNG)
4. Click "Predict" to get the disease classification and confidence

---

## 📸 Demo

Here's an example of the Streamlit app in action:

![Plant Disease Detection Demo](https://imgur.com/abc123.png)

The app successfully identifies plant diseases with high confidence. In this example, it detected **bacterial_leaf_blight** with **99.88% confidence**.

---

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





