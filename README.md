# 🧠 Neural Network Convolutional U-Net

[![Streamlit App](https://img.shields.io/badge/Streamlit-Deployed-brightgreen?logo=streamlit)](link-do-seu-app)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Deployed-brightgreen?logo=streamlit)](link-do-seu-app)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.x-D00000?logo=keras)](https://keras.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-%23white?logo=opencv)](https://opencv.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-%23yellow?logo=matplotlib)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12.x-76B900?logo=seaborn)](https://seaborn.pydata.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.x-150458?logo=pandas)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.x-013243?logo=numpy)](https://numpy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikitlearn)](https://scikit-learn.org/)
[![CUDA](https://img.shields.io/badge/CUDA-enabled-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)

![Lung Cancer Awareness Symbol](https://github.com/RafaelGallo/Neural-Network-Convolutional-U-Net/blob/main/img/007.png?raw=true)


## 📌 Overview

Hospitals and healthcare providers face significant challenges in the timely and accurate diagnosis of COVID-19 and other respiratory diseases, particularly when resources are strained and demand for imaging is high. Chest CT scans are critical for detecting lung infections, but manual interpretation is both time-consuming and dependent on the availability of experienced radiologists. This bottleneck can delay patient triage and treatment, potentially impacting outcomes during outbreaks or in under-resourced settings.

**This project delivers a deep learning–based pipeline for the automatic segmentation of lung lesions in CT and X-ray images, leveraging a U-Net convolutional neural network architecture.** By automating the detection and quantification of lung abnormalities—including those caused by COVID-19—the model aims to accelerate diagnosis, standardize assessments, and support radiologists in clinical decision-making.

**Key benefits include:**

* **Faster diagnosis:** Enabling rapid triage and early intervention for patients with severe lung involvement.
* **Consistency:** Providing standardized and reproducible segmentation, reducing inter-observer variability.
* **Resource optimization:** Assisting in prioritizing critical cases for optimal allocation of hospital resources.
* **Scalability:** Deployable in multiple clinical environments, especially where radiology expertise is limited.

This AI-powered approach is designed to integrate seamlessly with existing clinical workflows, offering fast, accurate, and reproducible segmentation of infected lung regions to improve patient care.


## 🚀 Pipeline

1. **Preprocessing:** Images/masks resized and normalized
2. **Model:** U-Net (optionally VGG16 encoder)
3. **Training:** Binary crossentropy, metrics = Accuracy, Dice, IoU
4. **Evaluation:** Results overlayed for inspection
5. **Deployment:** Streamlit app for interactive predictions

## 📊 Metrics

* **Pixel accuracy:** Correctly classified pixels vs total
* **Dice coefficient:** Mask overlap (0 = no, 1 = perfect)
* **IoU (Intersection over Union):** Intersection/union of predicted & real masks

## 🖼️ Example Results

### CT Scan Example

| Original Image              | Predicted Mask (Probability) | Predicted Mask Overlay     |
| --------------------------- | ---------------------------- | -------------------------- |
| ![Original](https://github.com/RafaelGallo/Neural-Network-Convolutional-U-Net/blob/main/img/001.png?raw=true) | ![Predicted](https://github.com/RafaelGallo/Neural-Network-Convolutional-U-Net/blob/main/img/002.png?raw=true) | ![Overlay](https://github.com/RafaelGallo/Neural-Network-Convolutional-U-Net/blob/main/img/003.png?raw=true) |

### X-ray Example

| Original X-ray Image     | Ground Truth Mask     | Predicted Mask Overlay     |
| ------------------------ | --------------------- | -------------------------- |
| ![X-ray](https://github.com/RafaelGallo/Neural-Network-Convolutional-U-Net/blob/main/img/004.png?raw=true) | ![GT](https://github.com/RafaelGallo/Neural-Network-Convolutional-U-Net/blob/main/img/005.png?raw=true) | ![Overlay](https://github.com/RafaelGallo/Neural-Network-Convolutional-U-Net/blob/main/img/006.png?raw=true) |

## 📊 Segmentation Metrics

```
Acurácia pixel a pixel: 0.9644
Dice coefficient: 0.5996
IoU: 0.4283
```

> *See the Streamlit app for metric explanations and visualizations.*


## 🧠 Model Architecture

The model uses a U-Net CNN, specialized for medical segmentation.
**Example model summary:**

![Model Summary](https://github.com/RafaelGallo/Neural-Network-Convolutional-U-Net/blob/main/img/003.png?raw=true)

## 🚦 How to Use

```bash
git clone https://github.com/your-user/neural-network-convolutional-u-net.git
cd neural-network-convolutional-u-net
pip install -r requirements.txt
streamlit run app.py
```

* Put your trained model in `models/unet_2_segmentacao.h5`
* Upload a CT or X-ray image on the Streamlit interface
* See the predicted mask, overlay, and segmentation metrics

## 📂 Datasets

* [COVID-19 CT Scan Lesion Segmentation](https://www.kaggle.com/datasets/andrewmvd/covid19-ct-scan-lesion-segmentation-dataset)
* [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)

## ✨ Credits

* Model: [U-Net paper (Ronneberger et al.)](https://arxiv.org/abs/1505.04597)
* Paper: [U-Net (Olaf Ronneberger, Philipp Fischer, and Thomas Brox)](https://arxiv.org/pdf/1505.04597)
* Streamlit app and code: [Rafael Gallo](https://github.com/RafaelGallo)
* Datasets: Publicly available on Kaggle [Dataset]()
