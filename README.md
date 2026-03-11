# 🧠 Stroke Prediction – Data Mining Project

## 📊 Project Overview

This project explores clinical and demographic data in order to identify **patterns associated with stroke risk**.

The work combines multiple **Data Mining techniques** including:

* Exploratory Data Analysis (EDA)
* Feature Engineering
* Supervised Learning
* Association Rule Mining
* Clustering Analysis
* Neural Networks (Deep Learning)

The goal is to analyze the dataset from multiple perspectives and uncover meaningful insights about stroke risk factors.

---

# 🎯 Project Goal

The objective of this project is to **predict whether a patient is at risk of stroke** based on medical and demographic attributes.

The analysis focuses on:

* Identifying variables strongly associated with stroke
* Building predictive classification models
* Discovering hidden patterns using association rules
* Identifying patient groups using clustering
* Exploring deep learning approaches for prediction

---

# 📂 Dataset

Dataset source:

Stroke Prediction Dataset (Kaggle)

The dataset contains **5,110 patient records** with attributes such as:

* Age
* Gender
* Hypertension
* Heart disease
* Average glucose level
* BMI
* Smoking status
* Work type
* Residence type

The target variable:

**stroke**

```
1 → stroke occurred  
0 → no stroke
```

⚠ The dataset is highly **imbalanced**, with stroke cases representing a small minority of observations.

# 📊 Dataset Exploration

### Stroke Class Distribution

<img width="741" height="572" alt="image" src="https://github.com/user-attachments/assets/d56d231f-7b13-4846-97c9-13f87f64540e" />

The dataset is highly imbalanced, with stroke cases representing a small fraction of observations.

---

### Age Distribution

<img width="677" height="482" alt="image" src="https://github.com/user-attachments/assets/ac213c80-199d-4a99-b814-256752f076e9" />

Age is one of the most important predictors for stroke risk.

---

### Glucose Level Distribution

<img width="751" height="502" alt="image" src="https://github.com/user-attachments/assets/c94a99c1-5a66-40ca-a004-2b6c9fa502fd" />

Higher glucose levels appear more frequently among stroke cases.

---

### BMI Distribution

<img width="733" height="455" alt="image" src="https://github.com/user-attachments/assets/ef48dcfb-cdb7-4470-b5f1-a63b66dc2f5d" />

---

# 🧹 Data Preprocessing

Before training models, the dataset required several preprocessing steps.

### Cleaning and Handling Missing Values

The following actions were performed:

* Removed irrelevant column (`id`)
* Handled missing values in **BMI**
* Cleaned inconsistent entries in **smoking_status**
* Removed rare gender category (`Other`)

### Feature Scaling

Numeric features were normalized using:

```
MinMaxScaler
```

### Feature Engineering

Additional features were created to capture interactions between variables.

Examples include:

* `glucose_bmi_ratio`
* `bmi_age_product`
* `glucose_age_ratio`
* `age_hypertension`
* `is_senior`
* `heart_senior_interaction`
* `age_squared`

These features help machine learning models detect more complex patterns.

<img width="838" height="833" alt="image" src="https://github.com/user-attachments/assets/60d28d56-d124-4d15-bbd8-9db00140bb35" />
<img width="847" height="426" alt="image" src="https://github.com/user-attachments/assets/d8985027-c60b-4d24-bee9-85895d5be48f" />
<img width="837" height="352" alt="image" src="https://github.com/user-attachments/assets/c5dbc367-5bc8-4cae-8138-7d5739140633" />

---

# 📊 Exploratory Data Analysis

Exploratory analysis was performed to understand variable distributions and correlations.

Visualizations included:

* Age distribution
* BMI distribution
* Glucose distribution
* Correlation heatmap

These graphs helped identify possible relationships between variables and stroke risk.

<img width="446" height="863" alt="image" src="https://github.com/user-attachments/assets/e9cfcfdd-af46-4c52-813a-8d3a69f97bdc" />

<img width="592" height="436" alt="image" src="https://github.com/user-attachments/assets/dc9332ec-15ca-480e-9c3f-fd2e7432816c" />

---

# 🌳 Supervised Learning Models

Two machine learning models were implemented to predict stroke risk.

## Decision Tree

Decision trees provide interpretable rules that explain how predictions are made.

Advantages:

* Transparent model
* Easy to interpret
* Captures non-linear relationships


<img width="862" height="380" alt="image" src="https://github.com/user-attachments/assets/15a30610-58ff-4308-b94c-b121eafcdf09" />


## Random Forest

Random Forest improves predictive accuracy by combining multiple decision trees.

Advantages:

* Reduced overfitting
* Higher predictive performance
* Robust to noise in data

Model evaluation metrics included:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC AUC

<img width="851" height="388" alt="image" src="https://github.com/user-attachments/assets/25541a6a-58d0-4178-a74a-7e473bbc589f" />

<img width="788" height="115" alt="image" src="https://github.com/user-attachments/assets/9e1aef62-c5c7-49fd-b7c2-7a221978d192" />

---

# 🔗 Association Rule Mining (Apriori)

To discover hidden relationships between features, the **Apriori algorithm** was applied.

Parameters used:

```
min_support = 0.4
min_confidence = 0.6
```

The dataset was discretized and transformed into categorical groups before applying the algorithm.

Examples of preprocessing steps:

* Age grouped into categories
* BMI grouped into weight categories
* Binary features created (e.g. `is_old`, `married_old`)

The analysis produced **54 association rules** with **Lift > 1**, focusing on the population where stroke occurred.

These rules reveal combinations of medical and demographic factors associated with stroke risk.

<img width="1155" height="738" alt="image" src="https://github.com/user-attachments/assets/52c27738-8813-485a-a1bd-1c2158da7863" />
<img width="995" height="157" alt="image" src="https://github.com/user-attachments/assets/ff8be2dc-cf53-4fb4-a272-a782bf7c7206" />

---

# 🧩 Clustering Analysis

Unsupervised learning techniques were used to identify **groups of patients with similar characteristics**.

Algorithm used:

```
K-Means
```

### Determining the Number of Clusters

Several evaluation methods were used:

* SSE (Sum of Squared Errors)
* Silhouette Score
* Elbow Method

The optimal number of clusters was found around:

```
k = 10
```
<img width="1032" height="600" alt="image" src="https://github.com/user-attachments/assets/0e6d9638-e705-4656-9f60-96c6012ed259" />


Clusters were analyzed based on:

* cluster size
* centroid values
* stroke rate inside each cluster

Some clusters showed significantly higher stroke prevalence.

<img width="807" height="536" alt="image" src="https://github.com/user-attachments/assets/cc7d8eb4-4683-4287-9cff-96465a359ec6" />
<img width="1135" height="556" alt="image" src="https://github.com/user-attachments/assets/4b8eae5f-00cb-4294-a5b6-e487780b9aca" />

<img width="802" height="508" alt="image" src="https://github.com/user-attachments/assets/f5aad728-c5b0-407c-955d-b4e48c7d2013" />
<img width="1141" height="562" alt="image" src="https://github.com/user-attachments/assets/ee3db9ac-2119-482d-af60-589edefa4e48" />

<img width="967" height="536" alt="image" src="https://github.com/user-attachments/assets/939ab1f1-d4f6-46e7-b6f1-eb1c98170046" />

---

# 🧠 Neural Network Model

A neural network model was implemented to explore deep learning approaches for stroke prediction.

Implementation:

```
PyTorch
```

### Model Architecture

The neural network includes:

* Input layer with engineered features
* Hidden layers with **LeakyReLU activation**
* Batch Normalization
* Dropout regularization
* Output layer with **Sigmoid activation**

### Handling Imbalanced Data

Because stroke cases are rare, the model uses:

```
Focal Loss
```

instead of standard binary cross-entropy.

Focal Loss focuses learning on difficult samples and improves performance on minority classes.

### Hyperparameter Optimization

Several configurations were tested including:

* hidden layer sizes
* learning rate
* batch size
* loss parameters

The final configuration was selected based on validation performance.

<img width="967" height="550" alt="image" src="https://github.com/user-attachments/assets/a1976698-fddd-494c-899b-aae46bdbeb7c" />
<img width="925" height="687" alt="image" src="https://github.com/user-attachments/assets/846a2ccb-fa9b-470e-9df0-60013558bc7e" />

---

# 📊 Model Evaluation

The neural network was evaluated using multiple metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC AUC
* Specificity

Training progress was monitored using **loss curves**.

<img width="727" height="723" alt="image" src="https://github.com/user-attachments/assets/cfa9e123-f333-4141-aa6b-f9c5e8175b70" />
<img width="767" height="615" alt="image" src="https://github.com/user-attachments/assets/9ffe0d78-a3e7-4e8c-8096-9b32a7b8b6c3" />

---

# 🔄 Project Pipeline

```
Dataset
   │
   ▼
Data Cleaning
   │
   ▼
Feature Engineering
   │
   ▼
Exploratory Data Analysis
   │
   ▼
Machine Learning Models
   ├─ Decision Tree
   └─ Random Forest
   │
   ▼
Association Rules
   └─ Apriori Algorithm
   │
   ▼
Clustering
   └─ KMeans
   │
   ▼
Neural Network (PyTorch)
   │
   ▼
Model Evaluation
   │
   ▼
Insights
```

---

# 📂 Project Structure

```
Stroke-Prediction-Data-Mining-Project

dataset
│ healthcare-dataset-stroke-data.csv

classification
│ Q2.py

association_rules
│ 22q1t.py

clustering
│ blistroke1.py
│ wstroke1.py

neural_network
│ noiro6.py

README.md
```

### Code Files Description

**Q2.py**

Classification models including Decision Tree and Random Forest.

**22q1t.py**

Association rule mining using the Apriori algorithm.

**blistroke1.py / wstroke1.py**

Clustering analysis using K-Means and cluster evaluation metrics.

**noiro6.py**

Neural network implementation using PyTorch including focal loss and hyperparameter tuning.

---

## ⚙️ Environment Setup

This project was developed in **Python 3** using **PyCharm** as the development environment.

Required libraries:

- pandas
- numpy
- matplotlib
- scikit-learn
- imbalanced-learn
- mlxtend
- torch
- seaborn

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/linoya1/Stroke-Prediction-Data-Mining-Project.git
cd Stroke-Prediction-Data-Mining-Project
```

Install the required packages:

```bash
pip install pandas numpy matplotlib scikit-learn imbalanced-learn mlxtend torch seaborn
```

---

## ▶️ Running the Project

Run each module separately according to the analysis you want to reproduce.

### Classification

```bash
python classification/Q2.py
```

### Association Rules

```bash
python association_rules/22q1t.py
```

### Clustering

```bash
python clustering/blistroke1.py
```

or

```bash
python clustering/wstroke1.py
```

### Neural Network

```bash
python neural_network/noiro6.py
```

# 📊 Key Insights

The analysis revealed several patterns:

* Age is strongly associated with stroke risk
* High glucose levels frequently appear in stroke cases
* BMI interacts with other medical conditions
* Certain clusters contain significantly higher stroke rates
* Deep learning models can improve recall for minority classes

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- Mlxtend
- PyTorch
- K-Means Clustering
- Apriori Algorithm

---
## Academic Context

This project was developed as part of a Data Mining course at the Open University of Israel.

The repository presents the implementation, experiments, and analysis performed during the project, adapted for portfolio and learning purposes.

---
# ⚠ Academic Integrity Notice

This repository is shared for educational and portfolio purposes only.

Students currently taking similar courses should not copy or submit this work as their own.
