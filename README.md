<div align="center">

# 🛡️ MaskGuard AI
### Face Mask Detection System

*An intelligent, real-time face mask detection system using Computer Vision, OpenCV, and Deep Learning (MobileNetV2)*

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-Notebook-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</div>

---

## 📌 Overview

**MaskGuard AI** is a comprehensive face mask detection system that automatically identifies whether individuals in static images, live webcam feeds, or uploaded video files are wearing face masks correctly, incorrectly, or not at all. Built with a dual-model architecture, the system provides real-time classification through an accessible, multi-tab **Gradio** web interface.

Designed to address real-world public health compliance monitoring challenges, MaskGuard AI offers a scalable, automated solution applicable across hospitals, airports, offices, and high-traffic public spaces.

---

## 🆕 What's New in v3.0 (Final Release)

> This version is a substantial upgrade from the mid-project HOG-only SVM baseline. Here is everything that changed:

| Feature | Mid-Project (v1.0) | Final Release (v3.0) |
|---|---|---|
| Classifier | HOG + SVM only | MobileNetV2 CNN + Fused SVM (switchable) |
| Feature Vector | HOG only (~1764 dims) | HOG + Gabor + HSV fused (~1796 dims) |
| Dimensionality Reduction | None | PCA (100 components) |
| Face Detection | 4 cascades simultaneously | 2 optimised cascades + CLAHE |
| Class Imbalance Handling | None | Weighted loss + balanced SVM |
| Explainability | None | CBIR visual retrieval tab |
| Low-Light Support | None | Auto-Night Mode (adaptive gamma LUT) |
| Coverage Estimation | None | Morphological mask coverage % |
| Video Processing | None | Frame-by-frame + annotated export |
| Interface | Single tab | Four-tab Gradio UI |

---

## 🚀 Key Features

- ✅ **Dual-Model Architecture** — Switch between MobileNetV2 CNN and HOG+Gabor+PCA+SVM at runtime without restarting
- ✅ **Real-Time Detection** — Continuous classification on live webcam feeds at ~10 FPS
- ✅ **Three-Class Output** — Detects `With Mask`, `Without Mask`, and `Mask Worn Incorrectly`
- ✅ **CBIR Explainability** — Retrieves top-3 most visually similar training patches to explain every prediction
- ✅ **Auto-Night Mode** — Adaptive gamma correction via LUT for reliable detection in low-light environments
- ✅ **Mask Coverage %** — Morphological HSV segmentation calculates exact mask coverage per face
- ✅ **Video Processing** — Upload any video, get annotated output + full compliance summary report
- ✅ **Transfer Learning** — Pretrained ImageNet weights with partial layer freezing for fast Colab training
- ✅ **Lightweight Runtime** — Full inference runs on CPU after training, no dedicated GPU required
- ✅ **Zero-Install Access** — Gradio generates a public shareable link on every launch

---

## 🧰 Tech Stack

| Component | Tool / Library |
|---|---|
| **Language** | Python 3.8+ |
| **Deep Learning** | PyTorch / torchvision (MobileNetV2) |
| **Classical ML** | scikit-learn (SVM, PCA, StandardScaler) |
| **Face Detection** | Haar Cascade Classifiers (OpenCV) |
| **Image Processing** | OpenCV, NumPy, scikit-image (HOG, Gabor) |
| **Similarity Retrieval** | SciPy (Cosine Distance for CBIR) |
| **Dataset Handling** | Kaggle API, Custom PASCAL VOC XML Parser |
| **Interface** | Gradio 4.44.0+ |
| **GPU Compute** | Google Colab (NVIDIA T4 via CUDA) |
| **Evaluation** | scikit-learn (Confusion Matrix, Classification Report) |
| **Visualisation** | Matplotlib |

---

## 📁 Repository Structure

```text
Face-Mask-Detection-System/
│
├── MaskGuard_AI.ipynb                    # Main Colab notebook (full end-to-end pipeline)
├── README.md                             # Project documentation
└── MaskGuard_AI_Feasibility_Study.pdf    # Feasibility Study and Architecture Report
```

> **Note:** This project is delivered as a highly modular, self-contained Google Colab notebook. All steps — from dataset downloading via Kaggle API and preprocessing, to dual-model training, evaluation, and the Gradio UI launch — are included in `MaskGuard_AI.ipynb`.

---

## ▶️ Getting Started

### Run on Google Colab (Recommended)

1. Clone or download this repository
2. Open `MaskGuard_AI.ipynb` in [Google Colab](https://colab.research.google.com/)
3. Set runtime to **GPU** → `Runtime > Change runtime type > T4 GPU`
4. Upload your `kaggle.json` file when prompted to automatically fetch the dataset (~150 MB)
5. Run all cells sequentially — the final cell launches the Gradio interface with a public link

### Run Locally

```bash
# Clone the repository
git clone https://github.com/Fardeen37/Face-Mask-Detection-System.git
cd Face-Mask-Detection-System

# Install required dependencies
pip install torch torchvision opencv-python-headless numpy pandas matplotlib scikit-learn scikit-image scipy gradio tqdm lxml

# Launch the notebook
jupyter notebook MaskGuard_AI.ipynb
```

---

## 🧠 Model Architecture

MaskGuard AI supports two classification pipelines selectable at runtime via a Gradio radio button:

### 1. Deep Learning Path — MobileNetV2

```text
Input Image (128x128x3)
        │
  MobileNetV2 Base (ImageNet weights, first 14 layers frozen)
        │
  Dropout (0.4) → Linear (256, ReLU) → Dropout (0.3)
        │
  Linear (3) — Output Layer
        │
  Output: [With Mask | Without Mask | Mask Worn Incorrectly]
```

**Training config:** AdamW (lr=5e-4), CosineAnnealingLR (T_max=20), WeightedCrossEntropyLoss [1.0, 2.5, 2.0], 20 epochs, batch size 32, NVIDIA T4 GPU

### 2. Classical ML Path — Fused SVM

```text
Cropped Face ROI
        │
  HOG (~1764 dims) + Gabor (32 dims) + HSV Histogram (512 dims)
        │
  Fused Vector (~1796 dims)
        │
  StandardScaler → PCA (100 components)
        │
  SVM (RBF Kernel, C=10, class_weight=balanced, probability=True)
        │
  Output: [With Mask | Without Mask | Mask Worn Incorrectly]
```

> The SVM pipeline is always active for CBIR explainability regardless of which model is selected for live detection.

---

## 📊 System Pipeline

```text
Raw Input (Image / Video Frame / Webcam Stream)
        │
  ┌─────▼───────────────────────┐
  │   Auto-Night Mode           │  ← Adaptive gamma LUT if mean brightness < 80
  └─────┬───────────────────────┘
        │
  ┌─────▼───────────────────────┐
  │   CLAHE Preprocessing       │  ← Contrast-limited adaptive histogram equalisation
  └─────┬───────────────────────┘
        │
  ┌─────▼───────────────────────┐
  │   Haar Cascade Detection    │  ← Alt2 (primary) + Default (fallback)
  └─────┬───────────────────────┘
        │  Cropped Face ROIs
  ┌─────▼───────────────────────┐
  │   Classification Stage      │  ← MobileNetV2 OR Fused SVM (runtime switch)
  └─────┬───────────────────────┘
        │  Prediction + Confidence Score
  ┌─────▼───────────────────────┐
  │   Morphological Analysis    │  ← HSV segmentation → mask coverage %
  └─────┬───────────────────────┘
        │
  ┌─────▼───────────────────────┐
  │   Annotated Output          │  ← Bounding box + class label + coverage text
  └─────────────────────────────┘
```

---

## 🔍 CBIR Explainability

The CBIR tab makes every prediction transparent and verifiable:

```text
Query Face Image
        │
  HSV Histogram (512 dims) + Gabor Descriptor (32 dims)
        │
  Cosine Distance via scipy.cdist
        │
  Compared against 4,072-patch training index
        │
  Top-3 Most Similar Patches Retrieved
        │
  Visual Panel: Query | Match 1 | Match 2 | Match 3
  (each labelled with class name and similarity %)
```

> Non-technical supervisors can verify any classification decision visually with zero knowledge of the underlying model.

---

## 🌙 Auto-Night Mode

Automatically activates when ambient lighting is poor:

- Measures mean brightness of each frame
- If mean brightness < 80 out of 255: computes gamma proportionally
- Gamma range: **1.1** (at threshold boundary) to **2.5** (very dark frames)
- Applied via `cv2.LUT` before face detection runs
- Bypassed entirely on well-lit frames to avoid overhead

---

## 📈 Evaluation and Metrics

- **Classification Report** — Precision, Recall, F1-Score per class via scikit-learn
- **Confusion Matrix** — 3x3 grid across With Mask, Without Mask, Incorrect Mask
- **Class Imbalance Fix** — Weighted loss (CNN) and `class_weight=balanced` (SVM) ensure minority classes are not ignored
- **Baseline Comparison** — Final v3.0 shows significant F1 improvement on minority classes over mid-project HOG-only baseline

---

## 🖥️ Gradio Interface — Four Tabs

| Tab | Description |
|---|---|
| **Webcam** | Live continuous detection at ~10 FPS with night mode toggle and sensitivity slider |
| **Image** | Upload static image, get annotated output + per-face confidence and coverage report |
| **Video** | Upload video file, configure frame skip (1 to 5), download annotated video + compliance summary |
| **CBIR** | Upload any face image to get prediction + top-3 visually similar training examples displayed |

---

## 👥 Team and Contributions

| Name | Role | Responsibilities |
|---|---|---|
| **Data Fardeen** | Code Development | Full system implementation: dataset pipeline, HOG/Gabor/PCA feature engineering, CBIR index, MobileNetV2 fine-tuning, morphological analysis, Auto-Night Mode, Gradio interface |
| **Iman Humayun** | Feasibility Study | Conducted feasibility analysis and prepared the comprehensive feasibility study documentation |
| **Amina Asghar** | Presentation Design | Prepared and organised the PowerPoint presentation and project demonstration slides |

---

## 📄 License

This project is open-source and available under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ by **Data Fardeen**, **Amina Asghar** & **Iman Humayun**

*SZABIST Islamabad — Computer Vision Lab Final Project*

</div>
