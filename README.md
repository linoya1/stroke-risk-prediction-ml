# 🧠 Stroke Prediction – Data Mining Project (OU 20595)

This repository contains my **Data Mining assignments (Maman 21 + Maman 22)** for the Open University course *Data Mining (20595)*.  
The task focuses on **predicting stroke risk** using clinical and demographic data, applying **classification and prediction methods**.
The project extends the previous work on the **Healthcare Stroke Prediction Dataset** and includes both **predictive modeling** and **unsupervised analysis**.

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

# Stroke Prediction – Data Mining Project (OU 20595)

This repository contains my **Data Mining assignments (Maman 21 + Maman 22)** for the Open University course *Data Mining (20595)*.  
The project extends the previous work on the **Healthcare Stroke Prediction Dataset** and includes both **predictive modeling** and **unsupervised analysis**.

---

## 📖 Maman 22 – Extended Analysis

After building the predictive models in Maman 21, Maman 22 focused on **knowledge discovery** using two main approaches:

### 1. Association Rule Mining (Apriori)
- Applied **Apriori algorithm** with thresholds:  
  - `min_support = 0.4`  
  - `min_confidence = 0.6`  
- Dataset preprocessing included:  
  - Discretization of `age`, `bmi`, `glucose` into clinical groups.  
  - Feature merging (e.g., *Obese + Overweight → High*).  
  - Binary feature engineering (e.g., `is_old`, `married_old`, `urban_and_private`).  
- Extracted **54 strong rules** with **Lift > 1**, focusing only on **stroke=1** population.  
- Rules highlight combinations of clinical and demographic risk factors correlated with stroke.

### 2. Clustering Analysis
- Used **KMeans** on normalized features after outlier filtering (Isolation Forest).  
- Evaluated clustering with:  
  - **SSE (Sum of Squared Errors)**  
  - **Silhouette Score**  
- Optimal solution found around **k=10 clusters**, balancing compactness and separation.  
- Outputs include:  
  - Stroke rate per cluster.  
  - Final centroids table.  
  - Clustered instances distribution.  
  - Visualizations (feature–stroke correlations, scatter plots, stacked bars).

### 3. Neural Networks (Deep Learning)

In addition to rule mining and clustering, we implemented a **feed-forward neural network** (file: `noiro6.py`) to improve stroke prediction.

#### ⚙️ Implementation
- Framework: **TensorFlow / Keras**
- Architecture:
  - Input layer: numeric + one-hot encoded categorical features
  - Hidden layers: Dense layers with **ReLU** activation
  - Output layer: single neuron with **Sigmoid** activation (binary classification)
- Training:
  - Optimizer: **Adam**
  - Loss: **Binary Crossentropy**
  - Class imbalance handled using **class weights**
- Evaluation metrics:
  - **Accuracy**
  - **Precision**
  - **Recall (Sensitivity)**
  - **Specificity**
  - **ROC AUC**

#### 📊 Results
- The model achieved accuracy > 90% with improved recall compared to baseline tree models.
- ROC AUC demonstrated good separability between stroke vs. non-stroke cases.
- Graphs of **loss curves**, **ROC curves**, and **confusion matrices** are included in the full report (`maman22.pdf`).

#### 🚀 Technologies Used
- **Python 3.10**
- **TensorFlow / Keras**
- **scikit-learn** (data preprocessing, metrics)
- **imbalanced-learn** (SMOTETomek in earlier stages)
- **matplotlib** / **seaborn** (visualizations)

---

## 📂 Code Files (Maman 22)

- `22q1t.py` – Association Rules with Apriori.  
- `noiro6.py` – Neural network baseline for stroke classification.  
- `blistroke1.py`, `wstroke1.py` – Additional experiments and preprocessing flows.  

---

## 📊 Example Results

- **Apriori**: Found rules such as  
  *{is_old=1, high_glucose=1} ⇒ {stroke=1}* with strong support and confidence.  
- **Clustering**: Certain clusters showed stroke prevalence > 25%, while others had near 0%.  
- These findings validate the usefulness of combining **rule-based insights** with **unsupervised learning**.

---

## 📝 Notes

- All PDF deliverables (`maman21.pdf`, `maman22.pdf`) are included in the **private repository** with full tables, graphs, and answers.  
- This public version provides code, structure, and documentation only.


