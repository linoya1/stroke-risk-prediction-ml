# 🧠 Stroke Prediction – Data Mining Project (OU 20595, Maman 21)

This repository contains my solution to **Maman 21** – the first part of the final project in the *Data Mining (20595)* course at the Open University of Israel.
The task focuses on **predicting stroke risk** using clinical and demographic data, applying **classification and prediction methods**.

---

## 🎯 Project Goal

The objective was to **predict whether a patient is at risk of stroke** based on features such as age, hypertension, heart disease, glucose level, BMI, smoking habits, and more.

Specific goals:

* Identify variables most strongly associated with stroke risk.
* Train and compare at least **two classification models**.
* Evaluate models using **accuracy, precision, recall, F1, ROC AUC**.
* Derive practical conclusions for medical risk prediction.

---

## 📂 Dataset

* Source: [Kaggle – Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
* 5,110 patient records
* Target variable: `stroke` (binary: 1 = stroke, 0 = no stroke)
* Contains both clinical (hypertension, heart disease, glucose, BMI) and demographic (age, gender, marital status, work type, residence, smoking) features.

---

## ⚙️ Methodology

The work was implemented in **Python (PyCharm)** using `pandas`, `scikit-learn`, `imblearn`, and `matplotlib`.

### 🔹 Data Preparation

* Removed irrelevant column (`id`)
* Handled missing values in `bmi`, `smoking_status`, `gender`
* Clipped outliers in glucose and BMI
* Encoded categorical variables (One-Hot & Frequency encoding)
* Normalized numeric features with `MinMaxScaler`
* Engineered additional features:

  * `glucose_bmi_ratio`, `bmi_age_product`, `glucose_age_ratio`
  * `age_hypertension`, `is_senior`, `heart_senior_interaction`
  * `smoke_age_interaction`, `age_squared`, etc.

### 🔹 Handling Class Imbalance

* Only \~5% positive stroke cases
* Applied **SMOTETomek** to balance the dataset

### 🔹 Models Used

1. **Decision Tree (C4.5/Entropy)**

   * `max_depth=4`, `min_samples_leaf=40`, `class_weight="balanced"`
2. **Random Forest**

   * `n_estimators=150`, `min_samples_leaf=2`, class weighting `{0:1, 1:3}`

---

## 📊 Results

* Both models achieved **accuracy above 90%**, with Random Forest showing higher recall and robustness.
* Decision Tree: provided interpretability and clear decision rules.
* Random Forest: superior predictive power, better ROC AUC, but less explainable.

Confusion matrices and detailed metrics (Precision, Recall, F1, Specificity, False Negatives) were computed.

---

## ▶️ Running the Project

Clone the repo and run the script:

```bash
python Q2.py
```

Requirements:

```txt
pandas
numpy
scikit-learn
imblearn
matplotlib
```

---

## 📖 Notes

* This project is the **first stage (Maman 21)** of the final assignment.
* In the next stage (**Maman 22**), additional methods are applied: **clustering and neural networks**.
* The code was developed and tested in **PyCharm**, with simple run configuration.

---

## 📜 License

All rights reserved. This project was submitted as part of the academic requirements for course 20595 (*Data Mining*), Open University of Israel.

---
