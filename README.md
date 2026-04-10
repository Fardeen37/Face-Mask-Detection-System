<div align="center">

# 🛡️ MaskGuard AI
### Face Mask Detection System

*An intelligent real-time face mask detection system using Computer Vision, OpenCV, and TensorFlow (MobileNetV2)*

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-Notebook-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</div>

---

## 📌 Overview

**MaskGuard AI** is a deep learning-powered face mask detection system that automatically identifies whether individuals in images or video streams are wearing face masks. Built using **MobileNetV2** transfer learning and **OpenCV** for face detection, the system provides real-time classification with an accessible **Gradio** web interface.

Originally developed as a Computer Vision academic project, MaskGuard AI is designed to address real-world public health compliance monitoring challenges — applicable across hospitals, airports, offices, and high-traffic public spaces.

---

## 🚀 Features

- ✅ **Real-Time Detection** — Works on live webcam feeds and CCTV streams
- ✅ **High Accuracy Classification** — `With Mask` / `Without Mask` via MobileNetV2
- ✅ **Face Detection Pipeline** — Haar Cascade Classifiers for robust face localization
- ✅ **Transfer Learning** — Pre-trained ImageNet weights for fast, efficient training
- ✅ **Interactive Interface** — Browser-accessible Gradio UI for non-technical users
- ✅ **Annotated Output** — Bounding boxes + classification labels on detected faces
- ✅ **Lightweight Architecture** — Runs on standard hardware without dedicated GPU

---

## 🧰 Tech Stack

| Component | Tool / Library |
|---|---|
| Language | Python 3.8+ |
| Deep Learning | TensorFlow / Keras (MobileNetV2) |
| Face Detection | Haar Cascade Classifier (OpenCV) |
| Image Processing | OpenCV, NumPy |
| Dataset Handling | Pandas, NumPy, ImageDataGenerator |
| Interface | Gradio / OpenCV VideoCapture |
| GPU Compute | Google Colab (NVIDIA T4) |
| Evaluation | scikit-learn (Confusion Matrix, Classification Report) |
| Visualisation | Matplotlib, Seaborn |

---

## 📁 Repository Structure

```
Face-Mask-Detection-System/
│
├── MaskGuard_AI.ipynb          # Main Colab notebook (full pipeline)
├── README.md                   # Project documentation
└── MaskGuard_AI_Feasibility_Study.pdf   # Feasibility Study Report
```

> **Note:** This project is delivered as a single self-contained Google Colab notebook. All steps — from dataset loading and preprocessing to model training, evaluation, and the Gradio demo — are included in `MaskGuard_AI.ipynb`.

---

## ▶️ Getting Started

### Run on Google Colab (Recommended)

1. Clone or download this repository
2. Open `MaskGuard_AI.ipynb` in [Google Colab](https://colab.research.google.com/)
3. Set runtime to **GPU** → `Runtime > Change runtime type > T4 GPU`
4. Run all cells sequentially — the notebook handles everything end to end

### Run Locally

```bash
# Clone the repo
git clone https://github.com/Fardeen37/Face-Mask-Detection-System.git
cd Face-Mask-Detection-System

# Install dependencies
pip install tensorflow opencv-python numpy pandas matplotlib seaborn scikit-learn gradio

# Open the notebook
jupyter notebook MaskGuard_AI.ipynb
```

---

## 🧠 Model Architecture

The system uses **MobileNetV2** as the feature extraction backbone with custom classification layers on top:

```
Input Image (224×224×3)
        │
  MobileNetV2 Base (ImageNet weights, frozen)
        │
  GlobalAveragePooling2D
        │
  Dense(128, ReLU) + Dropout(0.5)
        │
  Dense(2, Softmax)
        │
  Output: [With Mask | Without Mask]
```

Face detection is handled by **OpenCV Haar Cascade Classifiers** prior to model inference, ensuring the classifier only processes valid face regions.

---

## 📊 Pipeline

```
Raw Input (Image / Video Frame)
        │
  ┌─────▼─────┐
  │   OpenCV  │  ← Face Detection (Haar Cascade)
  └─────┬─────┘
        │  Cropped Face ROIs
  ┌─────▼──────────┐
  │  MobileNetV2   │  ← Mask Classification
  └─────┬──────────┘
        │  Prediction + Confidence
  ┌─────▼─────┐
  │ Annotated │  ← Bounding Box + Label Overlay
  │  Output   │
  └───────────┘
```

---

## 📈 Evaluation Metrics

The model is evaluated using:
- **Accuracy** on validation set
- **Confusion Matrix** (via Seaborn heatmap)
- **Classification Report** — Precision, Recall, F1-Score per class

---

## 👥 Team

| Name | Role | Responsibilities |
|---|---|---|
| **Data Fardeen** | Code Development | Dataset processing, model training, algorithm implementation |
| **Amina Asghar** | Presentation Design | PowerPoint preparation and organisation |
| **Iman Humayun** | Feasibility Study | Feasibility analysis and documentation |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ by **Data Fardeen**, **Amina Asghar** & **Iman Humayun**  
SZABIST Islamabad — Computer Vision Project

</div>
