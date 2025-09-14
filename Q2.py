
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, matthews_corrcoef, precision_recall_curve,
    average_precision_score, confusion_matrix
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from imblearn.combine import SMOTETomek
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# --- שלב 1: טעינה וניקוי ---
df = pd.read_csv('healthcare-dataset-stroke-data.csv')
df = df[df['gender'] != 'Other']
df.drop(columns=['id'], inplace=True)
df['bmi'] = df['bmi'].fillna(df['bmi'].mean())
df['smoking_status'] = df['smoking_status'].fillna(df['smoking_status'].mode()[0])
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

# --- טיפול בקיצוניים ---
for col in ['avg_glucose_level', 'bmi']:
    low = df[col].quantile(0.01)
    high = df[col].quantile(0.99)
    df[col] = np.clip(df[col], low, high)

# --- Encoding ---
for col in ['smoking_status', 'work_type']:
    freq = df[col].value_counts(normalize=True)
    df[col + '_freq'] = df[col].map(freq)
    df.drop(columns=[col], inplace=True)

df = pd.get_dummies(df, columns=['Residence_type', 'ever_married'], drop_first=True)

# --- Feature Engineering ---
df['glucose_bmi_ratio'] = df['avg_glucose_level'] / df['bmi']
df['age_hypertension'] = df['age'] * df['hypertension']
df['glucose_age_ratio'] = df['avg_glucose_level'] / df['age']
df['bmi_age_product'] = df['bmi'] * df['age']
df['is_senior'] = (df['age'] > 60).astype(int)
df['heart_senior_interaction'] = df['heart_disease'] * df['is_senior']
df['bmi_hypertension'] = df['bmi'] * df['hypertension']
df['age_squared'] = df['age'] ** 2
df['glucose_heart'] = df['avg_glucose_level'] * df['heart_disease']
df['smoke_age_interaction'] = df['smoking_status_freq'] * df['age']
df['young_without_risk'] = (
    (df['age'] < 0.6) &
    (df['hypertension'] == 0) &
    (df['heart_disease'] == 0) &
    (df['avg_glucose_level'] < 0.3) &
    (df['bmi'] < 0.4)
).astype(int)

X = df.drop('stroke', axis=1)
y = df['stroke']

# --- נירמול ---
scaler = MinMaxScaler()
X[X.columns] = scaler.fit_transform(X)

# --- בחירת תכונות חשובות כולל ידנית ---
mi = mutual_info_classif(X, y)
top_features = pd.Series(mi, index=X.columns).sort_values(ascending=False).head(15).index.tolist()
if 'young_without_risk' not in top_features:
    top_features[-1] = 'young_without_risk'
X = X[top_features]

# --- חלוקה ואיזון ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
smt = SMOTETomek(random_state=42)
X_train_res, y_train_res = smt.fit_resample(X_train, y_train)

# --- הגדרת המודלים ---
dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=40, class_weight='balanced', criterion='entropy', random_state=42)
rf = RandomForestClassifier(n_estimators=150, min_samples_leaf=2, class_weight={0: 1, 1: 3}, random_state=42)

# --- אימון Decision Tree ---
dt.fit(X_train_res, y_train_res)
y_proba_dt = dt.predict_proba(X_test)[:, 1]
prec_dt, rec_dt, thr_dt = precision_recall_curve(y_test, y_proba_dt)
f1_dt = 2 * (prec_dt * rec_dt) / (prec_dt + rec_dt + 1e-10)
best_thr_dt = thr_dt[np.argmax(f1_dt)]
y_pred_dt = (y_proba_dt >= best_thr_dt).astype(int)

# --- אימון Random Forest ---
rf.fit(X_train_res, y_train_res)
y_proba_rf = rf.predict_proba(X_test)[:, 1]
prec_rf, rec_rf, thr_rf = precision_recall_curve(y_test, y_proba_rf)
f1_rf = 2 * (prec_rf * rec_rf) / (prec_rf + rec_rf + 1e-10)
best_thr_rf = 0.32
y_pred_rf = (y_proba_rf >= best_thr_rf).astype(int)

# --- פונקציות הדפסה ---
def model_weighted_avg(name, y_true, y_pred, y_proba=None, threshold=None):
    report = classification_report(y_true, y_pred, output_dict=True)
    weighted = report['weighted avg']
    return {
        "Model": name,
        "Threshold": round(threshold, 2) if threshold is not None else "N/A",
        "Accuracy": round(accuracy_score(y_true, y_pred), 3),
        "Precision": round(weighted['precision'], 3),
        "Recall": round(weighted['recall'], 3),
        "F1 Score": round(weighted['f1-score'], 3),
        "ROC AUC": round(roc_auc_score(y_true, y_proba), 3) if y_proba is not None else "N/A"
    }

dt_w = model_weighted_avg("Decision Tree", y_test, y_pred_dt, y_proba_dt)
rf_w = model_weighted_avg("Random Forest", y_test, y_pred_rf, y_proba_rf, best_thr_rf)

print("\n=== Summary Table: Weighted Avg (Like WEKA) ===")
print(f"{'Model':<18}{'Threshold':<10}{'Accuracy':<10}{'Precision':<12}{'Recall':<10}{'F1 Score':<10}{'ROC AUC'}")
print("-" * 70)
for r in [dt_w, rf_w]:
    print(f"{r['Model']:<18}{r['Threshold']:<10}{r['Accuracy']:<10}{r['Precision']:<12}{r['Recall']:<10}{r['F1 Score']:<10}{r['ROC AUC']}")

# --- מטריצת בלבול לכל מודל ---
for name, y_pred, y_proba in [('Decision Tree', y_pred_dt, y_proba_dt), ('Random Forest', y_pred_rf, y_proba_rf)]:
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    print(f"\n=== {name} Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))
    print(f"Correctly Classified Instances: {(y_test == y_pred).sum()} ({round((y_test == y_pred).mean() * 100, 2)}%)")
    print(f"Incorrectly Classified Instances: {(y_test != y_pred).sum()} ({round((y_test != y_pred).mean() * 100, 2)}%)")
    print(f"Specificity: {round(specificity, 3)}")
    print(f"False Negatives (חולי שבץ שפספסנו): {fn}")
