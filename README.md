#  Alzheimer’s Disease Classification using CNN

This project focuses on classifying **Alzheimer’s disease stages** from **MRI brain images** using a **Convolutional Neural Network (CNN)**.  
The model is trained to categorize MRI scans into **four stages** of dementia based on the OASIS dataset.

---

## 🚀 Overview

Early diagnosis of Alzheimer’s can significantly improve treatment and care.  
This project leverages **deep learning** to identify the level of dementia from MRI images.

The four classes are:
-  **Non Demented**
-  **Very Mild Demented**
-  **Mild Demented**
-  **Moderate Demented**

The trained model (`snapshot_1.hdf5`) can be directly used for prediction and deployment.

---

## 📁 Project Structure

├── Alzheimers.py                 # Training pipeline for MRI image classification.

├── alzheimer_classification.py   # Dataset preprocessing & model comparison (OASIS CSV).

├── cnn.py                        # CNN model architecture definition.

├── predict.py                    # Basic prediction and visualization.

├── predict2.py                   # Enhanced prediction with preprocessing.

├── snapshot_1.hdf5               # Final trained CNN model.

└── README.md                     # Project documentation.




---

##  Dataset Information

### 🧾 1. OASIS Dataset (CSV)
Used for **initial model comparisons** and data exploration.  
Contains attributes like age, gender, MMSE score, and clinical dementia rating.

### 🧠 2. MRI Image Dataset
Used for **CNN training and final classification**.  
Each MRI image belongs to one of four dementia stages.

> The dataset was preprocessed using grayscale conversion, normalization, and resizing (176×176).

---

##  Model Architecture

Defined in `cnn.py`, the model is a deep **Convolutional Neural Network** with:
- Convolutional layers with ReLU activation  
- MaxPooling for feature reduction  
- Dropout for regularization  
- Dense layers for classification  

Optimizer: `Adam`  
Loss Function: `Categorical Crossentropy`  
Metrics: `Accuracy`

---

##  Training

Run the main training script:
```bash
python Alzheimers.py
```

The model is trained on MRI images and saved as:
```
snapshot_1.hdf5
```

You can monitor training accuracy and loss in the console or add TensorBoard for better visualization.
