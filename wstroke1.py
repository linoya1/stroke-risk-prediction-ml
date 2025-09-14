import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import seaborn as sns

# קריאה וניקוי
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
df = df.dropna(subset=["stroke"])
df = df[df["gender"] != "Other"].reset_index(drop=True)
df["bmi"] = SimpleImputer(strategy="mean").fit_transform(df[["bmi"]])
df["smoking_status"] = df["smoking_status"].fillna("Unknown")

# קידודים
df = pd.get_dummies(df, columns=["ever_married", "Residence_type", "gender"], drop_first=True)
df = df.drop(columns=["work_type", "smoking_status", "id"])

# הנדסת תכונות
df["glucose_bmi_ratio"] = df["avg_glucose_level"] / df["bmi"]
df["bmi_age_product"] = df["bmi"] * df["age"]
df["age_hypertension"] = df["age"] * df["hypertension"]
df["is_senior"] = (df["age"] > 60).astype(int)
df["heart_senior_interaction"] = df["heart_disease"] * df["is_senior"]
df["bmi_hypertension"] = df["bmi"] * df["hypertension"]
df["age_squared"] = df["age"] ** 2
df["glucose_heart"] = df["avg_glucose_level"] * df["heart_disease"]
df["senior_diabetic_interaction"] = df["is_senior"] * df["avg_glucose_level"]
df["hypertension_bmi_product"] = df["bmi"] * df["hypertension"]
df["smoking_self_emp"] = df["is_senior"] * df["ever_married_Yes"]

# קלט ותיוג
X = df.drop(columns=["stroke"])
y = df["stroke"]

# נירמול
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# סינון Outliers
clf = IsolationForest(contamination=0.01, random_state=42)
outliers = clf.fit_predict(X_scaled)
df = df[outliers == 1].reset_index(drop=True)
y = y[outliers == 1].reset_index(drop=True)
X_scaled = X_scaled[outliers == 1]

# הרצת KMeans
k = 10
kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
df["cluster_with_stroke"] = kmeans.fit_predict(X_scaled)
df["stroke"] = y
df["squared_error_with_stroke"] = np.min(kmeans.transform(X_scaled)**2, axis=1)

# פלט
print("======================")
print("📊 ניתוח עם עמודת stroke")
print("======================")
print(f"✅ SSE עם stroke: {kmeans.inertia_:.2f}")
print(f"✅ Silhouette עם stroke: {silhouette_score(X_scaled, df['cluster_with_stroke']):.4f}")

counts = df["cluster_with_stroke"].value_counts().sort_index()
print("\n🧮 מופעים בכל אשכול:")
for i, c in counts.items():
    percent = 100 * c / len(df)
    print(f"Cluster {i}: {c} ({percent:.1f}%)")

print("\n🔎 שיעור מקרי שבץ בכל אשכול:")
print(df.groupby("cluster_with_stroke")["stroke"].mean().round(5))

print("\n📉 SSE לפי אשכול:")
print(df.groupby("cluster_with_stroke")["squared_error_with_stroke"].sum().round(2))

# גרף מתאם
plt.figure(figsize=(10, 6))
corr = df.drop(columns=["cluster_with_stroke", "squared_error_with_stroke"]).corr()["stroke"].drop("stroke").sort_values(ascending=False)
sns.barplot(x=corr.values, y=corr.index)
plt.title("מתאם תכונות עם stroke")
plt.xlabel("מתאם")
plt.tight_layout()
plt.show()

def print_clustered_instances_table_with_target(cluster_labels):
    from collections import Counter
    counts = Counter(cluster_labels)
    total = len(cluster_labels)
    print("\nClustered Instances")
    for cluster in sorted(counts):
        percent = round(100 * counts[cluster] / total)
        print(f"{cluster:<2} {counts[cluster]:<4} ({percent}%)")

def print_stroke_rate_by_cluster(df, cluster_col="cluster_with_stroke", stroke_col="stroke"):
    print("\nStroke rate per cluster:")
    rates = df.groupby(cluster_col)[stroke_col].mean()
    for cluster, rate in rates.items():
        print(f"Cluster {cluster:<2}: {rate:.5f} ({rate * 100:.2f}%)")


def print_final_centroids_with_target(X_scaled, cluster_labels, feature_names):
    import pandas as pd
    df = pd.DataFrame(X_scaled, columns=feature_names)
    df["cluster"] = cluster_labels
    df_mean = df.groupby("cluster").mean().T
    full_data_means = df.drop(columns="cluster").mean()
    df_mean.insert(0, "Full Data", full_data_means)
    print("\nFinal cluster centroids (mean values per feature):")
    print(df_mean.round(4).to_string())

print("============================")
print(" Clustering Results With Target Feature:")
print("============================")

print(f" Silhouette Score: {silhouette_score(X_scaled, df['cluster_with_stroke']):.4f}")
print(f" SSE with stroke: {kmeans.inertia_:.2f}")

print_clustered_instances_table_with_target(kmeans.labels_)
print_stroke_rate_by_cluster(df, cluster_col="cluster_with_stroke", stroke_col="stroke")
print_final_centroids_with_target(X_scaled, kmeans.labels_, X.columns)

def plot_instance_vs_class_improved(df, cluster_col="cluster_with_stroke", class_col="stroke"):
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import LabelEncoder

    # ממיינים את התצפיות לפי class כך ש-stroke=1 יופיעו מימין
    df_sorted = df.sort_values(by=class_col).reset_index(drop=True)

    # קידוד של הערכים (לציר Y) – למשל 0 → No Stroke, 1 → Stroke
    y_labels = df_sorted[class_col].apply(lambda v: "Stroke" if v == 1 else "No Stroke")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_labels)

    # ציור
    plt.figure(figsize=(12, 6))
    scatter = plt.scatter(
        x=range(len(df_sorted)),
        y=y_encoded,
        c=df_sorted[cluster_col],
        cmap="tab20",
        s=[30 if val == 1 else 10 for val in df_sorted[class_col]],  # Stroke בגודל גדול
        alpha=0.7
    )

    plt.title("Instance Distribution by Stroke (colored by Cluster)")
    plt.xlabel("Sorted Instance Index (stroke=1 to the right)")
    plt.ylabel(class_col)
    plt.yticks(ticks=range(len(le.classes_)), labels=le.classes_)
    plt.colorbar(scatter, label="Cluster")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

plot_instance_vs_class_improved(df, cluster_col="cluster_with_stroke", class_col="stroke")


def plot_stacked_bar_class_distribution(df, cluster_col="cluster_with_stroke", class_col="stroke"):
    import matplotlib.pyplot as plt
    import pandas as pd

    # טבלת צירוף – כמה מופעים מכל class בכל cluster
    crosstab = pd.crosstab(df[cluster_col], df[class_col])

    # צביעה
    colors = ["skyblue", "salmon"] if crosstab.shape[1] == 2 else None

    # ציור
    crosstab.plot(kind='bar', stacked=True, figsize=(12, 6), color=colors)

    plt.title(f"{class_col} Distribution across Clusters (Stacked Bar)")
    plt.xlabel("Cluster")
    plt.ylabel("Number of Instances")
    plt.legend(title=class_col, labels=["No Stroke", "Stroke"])
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


plot_stacked_bar_class_distribution(df, cluster_col="cluster_with_stroke", class_col="stroke")


def plot_clusters_umap(X_scaled, cluster_labels):
    import matplotlib.pyplot as plt
    import umap.umap_ as umap
    import numpy as np
    import pandas as pd

    # הפחתת מימד באמצעות UMAP
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(X_scaled)

    # יצירת DataFrame לצורך חישוב מיקומי האשכולות
    df_umap = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"])
    df_umap["cluster"] = cluster_labels

    # ציור הנקודות
    plt.figure(figsize=(14, 7))
    scatter = plt.scatter(
        df_umap["UMAP1"], df_umap["UMAP2"],
        c=df_umap["cluster"],
        cmap="tab10",       # צבעים מובחנים
        s=20,               # גודל נקודות
        alpha=0.8,
        edgecolor='k', linewidth=0.1  # מסגרת שחורה דקה
    )

    # הוספת מספר האשכול במרכז כל מקבץ
    for cl in np.unique(cluster_labels):
        center = df_umap[df_umap["cluster"] == cl][["UMAP1", "UMAP2"]].mean()
        plt.text(center["UMAP1"], center["UMAP2"], str(cl),
                 fontsize=10, weight='bold', color='black',
                 ha='center', va='center')

    # תוספות גרפיות
    plt.title("UMAP Projection of Clusters (K=10)", fontsize=16)
    plt.xlabel("UMAP 1", fontsize=12)
    plt.ylabel("UMAP 2", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    cbar = plt.colorbar(scatter)
    cbar.set_label("Cluster", fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()

# קריאה לפונקציה
plot_clusters_umap(X_scaled, df["cluster_with_stroke"])

