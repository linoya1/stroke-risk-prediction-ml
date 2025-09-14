import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

# קריאה וניקוי
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
df = df.dropna(subset=["stroke"])
df = df[df["gender"] != "Other"].reset_index(drop=True)
df["bmi"] = SimpleImputer(strategy="mean").fit_transform(df[["bmi"]])
df["smoking_status"] = df["smoking_status"].fillna("Unknown")

# שמירת stroke בצד לצורך ניתוח
y = df["stroke"].copy()

# קידודים בסיסיים
df = pd.get_dummies(df, columns=["ever_married", "Residence_type"], drop_first=True)
df = pd.get_dummies(df, columns=["gender"], drop_first=True)
df = pd.get_dummies(df, columns=["work_type"], drop_first=True)
df = pd.get_dummies(df, columns=["smoking_status"], drop_first=True)
df = df.drop(columns=["id", "stroke", "work_type_Self-employed", "smoking_status_formerly smoked"])  # הסרת עמודת המטרה

# הנדסת תכונות נבחרות בלבד לפי המתאם עם stroke
df["age_squared"] = df["age"] ** 2
df["is_senior"] = (df["age"] > 60).astype(int)
df["bmi_age_product"] = df["bmi"] * df["age"]
df["glucose_heart"] = df["avg_glucose_level"] * df["heart_disease"]
df["age_hypertension"] = df["age"] * df["hypertension"]
df["heart_senior_interaction"] = df["heart_disease"] * df["is_senior"]
df["bmi_hypertension"] = df["bmi"] * df["hypertension"]
df["glucose_bmi_ratio"] = df["avg_glucose_level"] / df["bmi"]
df["senior_diabetic_interaction"] = df["is_senior"] * df["avg_glucose_level"]
df["hypertension_bmi_product"] = df["bmi"] * df["hypertension"]
# df["smoking_self_emp"] = df["smoking_status_formerly smoked"] * df["work_type_Self-employed"]
df["smoking_self_emp"] = df["is_senior"] * df["ever_married_Yes"]


# בחירת רק התכונות המשמעותיות
selected_features = [
    'age', 'age_squared', 'is_senior', 'bmi', 'bmi_age_product',
    'glucose_heart', 'age_hypertension', 'heart_senior_interaction',
    'heart_disease', 'avg_glucose_level', 'hypertension',
    'bmi_hypertension', 'glucose_bmi_ratio',
    'ever_married_Yes', 'Residence_type_Urban', 'gender_Male',
    'senior_diabetic_interaction',
    'hypertension_bmi_product',
    'smoking_self_emp'
]
df = df[selected_features]

# בדיקת עמודות
print("האם עמודת stroke הוסרה לחלוטין?", 'stroke' not in df.columns)
print("עמודות שמוזנות לקלאסטרינג:")
print(df.columns.tolist())

# נירמול
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df)

# סינון outliers
clf = IsolationForest(contamination=0.01, random_state=42)
outliers = clf.fit_predict(X_scaled)
df = df[outliers == 1].reset_index(drop=True)
y = y[outliers == 1].reset_index(drop=True)
X_scaled = X_scaled[outliers == 1]


# הרצת KMeans
k = 10
kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df["cluster_no_stroke"] = clusters
df["stroke"] = y
df["squared_error_no_stroke"] = np.min(kmeans.transform(X_scaled)**2, axis=1)

from sklearn.metrics import silhouette_score
sil = silhouette_score(X_scaled, clusters)
print(f" Silhouette Score: {sil:.4f}")

# ניתוח
print("============================")
print(" ניתוח ללא עמודת stroke")
print("============================")
sse = kmeans.inertia_
print(f" SSE ללא stroke: {sse:.2f}")

counts = df["cluster_no_stroke"].value_counts().sort_index()
for i, c in counts.items():
    percent = 100 * c / len(df)
    print(f"Cluster {i}: {c} ({percent:.1f}%)")

stroke_rates = df.groupby("cluster_no_stroke")["stroke"].mean()
print("\nStroke rate per cluster:")
for cluster_id, rate in stroke_rates.items():
    print(f"Cluster {cluster_id}: {rate:.5f} ({rate*100:.2f}%)")


sse_per_cluster = df.groupby("cluster_no_stroke")["squared_error_no_stroke"].sum().round(2)
print("\n📉 SSE לפי אשכול:")
print(sse_per_cluster)

# שמירת התוצאה
df[["cluster_no_stroke", "stroke"]].to_csv("clusters_without_stroke_filtered.csv", index=False)

# גרף מתאם (רשות)
corr = df.drop(columns=["cluster_no_stroke", "stroke", "squared_error_no_stroke"]).corrwith(df["stroke"]).sort_values(ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x=corr.values, y=corr.index)
plt.title("מתאם תכונות עם stroke")
plt.xlabel("מתאם")
plt.tight_layout()
plt.show()

def print_clustered_instances_table(cluster_labels):
    from collections import Counter
    counts = Counter(cluster_labels)
    total = len(cluster_labels)
    print("\nClustered Instances")
    for cluster in sorted(counts):
        percent = round(100 * counts[cluster] / total)
        print(f"{cluster:<2} {counts[cluster]:<4} ({percent}%)")



def print_final_centroids(X_scaled, cluster_labels, feature_names):
    import pandas as pd
    df = pd.DataFrame(X_scaled, columns=feature_names)
    df["cluster"] = cluster_labels
    df_mean = df.groupby("cluster").mean().T
    full_data_means = df.drop(columns="cluster").mean()
    df_mean.insert(0, "Full Data", full_data_means)
    print("\nFinal cluster centroids (mean values per feature):")
    print(df_mean.round(4).to_string())

print_clustered_instances_table(kmeans.labels_)
# print_initial_centroids(kmeans, df.columns)  # רק אם init="random"
print_final_centroids(X_scaled, kmeans.labels_, selected_features)

# 🔍 Elbow Method - מציאת k אופטימלי לפי SSE
sse_list = []
K_range = range(2, 21)  # בדיקה עבור K מ-2 עד 20

for k in K_range:
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    sse_list.append(kmeans.inertia_)

# ציור הגרף
plt.figure(figsize=(10, 6))
plt.plot(K_range, sse_list, marker='o')
plt.xticks(K_range)
plt.xlabel("Number of Clusters (K)")
plt.ylabel("SSE (Sum of Squared Errors)")
plt.title("Elbow Method – Optimal K based on SSE")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
