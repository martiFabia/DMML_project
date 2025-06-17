# Predicting Student Dropout in Higher Education Using Supervised Learning

This project aims to develop a predictive model that classifies students into three possible academic outcomes: **Graduate**, **Enrolled**, or **Dropout**.<br>
The goal is to support higher education institutions in identifying students at risk and promoting timely and personalized interventions.

## 📚 Overview

- **Problem**: Student dropout is a critical issue with long-term personal and institutional consequences. Predicting student outcomes can improve academic support systems and reduce attrition rates.
- **Approach**: We use supervised machine learning techniques on real-world educational data to build a robust classification model, integrated with explainability methods and a graphical user interface (GUI).

## 📊 Dataset

The dataset was sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) and includes:
- 4,424 student records
- 37 attributes per student (demographic, socioeconomic, academic)
- Target variable: `Graduate`, `Enrolled`, `Dropout`

## 🔍 Methodology

- **Preprocessing**: Handling categorical and numerical features, standardization, and one-hot encoding.
- **Feature Engineering**: Creation of informative features (e.g., pass rates, weighted grades, parental education score).
- **Handling Imbalance**: SMOTE oversampling for minority classes.
- **Models Used**: Random Forest, CatBoost, XGBoost, LightGBM, SVM, Gradient Boosting, Decision Tree.
- **Model Selection**: Hyperparameter tuning via HalvingGridSearchCV with 5-fold cross-validation.
- **Evaluation**: Macro F1-score, Balanced Accuracy, ROC AUC.

## 🧠 Explainability

We apply **SHAP (SHapley Additive Explanations)** to:
- Understand the global importance of features across the dataset
- Interpret individual predictions through waterfall plots
- Compare model transparency and consistency

## 🖥️ GUI

A Python-based **Graphical User Interface** (built with Tkinter) allows users to:
- Input student data manually
- View predictions and class probabilities
- Access local SHAP explanations for each prediction

## 📂 Project Structure
├── data/ # Raw and cleaned datasets <br>
├── models/ # Saved trained models<br>
├── shap_output/ # SHAP values <br>
├── notebook/ # Jupyter notebooks<br>
├── utils/ # Preprocessing and feature engineering modules<br>
├── results/ # Model comparison results <br>
├── app.py/ # GUI application<br>
└── README.md


## 📄 Documentation

For a detailed explanation of the methodology, results, and model explainability, refer to the full project documentation:  
➡️ [`Documentation_FABIANI.pdf`](./Documentation_FABIANI.pdf)
