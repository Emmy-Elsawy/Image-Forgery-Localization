# Hybrid AI-Based Localization for Image Forgery Detection

![Status](https://img.shields.io/badge/Status-Accepted%20for%20Publication-green) 
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%2F%20Keras-orange)

Official implementation of the research paper: **"Hybrid AI-Based Localization for Image Forgery Detection."** This project utilizes a multi-channel deep learning approach to identify and localize digital image manipulations at the pixel level.

---

## 🔍 Visual Overview
![GUI Interface]([image6.png])
*Figure 1: The system interface showing the original image and the predicted forgery mask.*

---

## 🚀 Key Research Contributions

* **6-Channel Input Fusion:** Our architecture moves beyond standard RGB by fusing **RGB**, **Error Level Analysis (ELA)**, and **Noise Residuals** into a unified $256 \times 256 \times 6$ input tensor to capture forensic artifacts invisible to the human eye.
* **Custom U-Net Architecture:** Utilizes an **EfficientNet-B0** backbone adapted via "model surgery" to process multi-modal forensic data while maintaining pre-trained feature extraction capabilities.
* **Hybrid Focal-Dice Loss:** Specialized loss function implementation to solve the extreme class imbalance between forged and authentic pixels.

---

## 🛠 Project Workflow


### 1. Preprocessing
- Generation of **ELA (Error Level Analysis)** maps to detect compression inconsistencies.
- Extraction of **Noise Residuals** to identify local noise variance caused by splicing.
- Tensor stacking into a 6-channel format.

### 2. Model Training
- **Backbone:** EfficientNet-B0
- **Optimizer:** Adam
- **Metric:** Dice Coefficient & IoU (Intersection over Union)

---

## 💻 Tech Stack
- **Languages:** Python
- **Libraries:** TensorFlow, Keras, OpenCV, NumPy, Matplotlib
- **Tools:** Jupyter Notebook, Git

---

## 📝 Publication Status
This work has been **officially accepted for publication (2026)**. The full citation and DOI link will be updated here following the official release by the publisher.

---

## 🤝 Authors
* **Emmy El-sawy** - *Researcher & Developer* - [@Emmy-Elsawy](https://github.com/Emmy-Elsawy)
* **Abdelrahman El-essawi** - *Researcher & Developer* - [@abdulr2005](https://github.com/abdulr2005)

---
© 2026 Emmy El-Sawy & Abdelrahman El-essawi. This project is part of ongoing research in AI Forensics and Computer Vision.
