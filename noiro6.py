#סעיף ד נסיון 2
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
set_seed()

# === ניקוי וטעינה ===
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
df = df.drop(['id'], axis=1)
df = df[df['gender'] != 'Other']
df['bmi'] = df['bmi'].fillna(df['bmi'].median())
df['smoking_status'] = df['smoking_status'].fillna(df['smoking_status'].mode()[0])
df['work_type'] = df['work_type'].map({'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': 3, 'Never_worked': 4}) / 4
df['smoking_status'] = df['smoking_status'].map({'never smoked': 0, 'formerly smoked': 1, 'smokes': 2, 'Unknown': 3}) / 3
df = pd.get_dummies(df, drop_first=True)
if 'Residence_type_Urban' in df.columns:
    df = df.drop(columns=['Residence_type_Urban'])
if 'work_type' in df.columns:
    df = df.drop(columns=['work_type'])

X = df.drop(columns=['stroke'])
y = df['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# === Focal Loss ===
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

# === רשת נוירונים גמישה ===
class StrokeNet(nn.Module):
    def __init__(self, input_dim, h1, h2):
        super(StrokeNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(h1, h2),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(h2, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

# === גריד פרמטרים ===
param_grid = {
    'hiddens': [(32, 16), (40, 14), (24, 12)],
    'batch_sizes': [32],
    'lrs': [0.001],
    'alphas': [1]
}

best_loss = float('inf')
best_model = None
best_params = None
best_train_losses = []
best_test_losses = []

# === אימון מודלים ===
for h1, h2 in param_grid['hiddens']:
    for batch_size in param_grid['batch_sizes']:
        for lr in param_grid['lrs']:
            for alpha in param_grid['alphas']:
                model = StrokeNet(X_train.shape[1], h1, h2)
                criterion = FocalLoss(alpha=alpha, gamma=2)
                optimizer = optim.Adam(model.parameters(), lr=lr)
                train_losses = []
                test_losses = []
                for epoch in range(30):
                    model.train()
                    epoch_losses = []
                    permutation = torch.randperm(X_train_tensor.size(0))
                    for i in range(0, X_train_tensor.size(0), batch_size):
                        indices = permutation[i:i+batch_size]
                        batch_X = X_train_tensor[indices]
                        batch_y = y_train_tensor[indices]
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                        epoch_losses.append(loss.item())
                    train_losses.append(np.mean(epoch_losses))
                    model.eval()
                    with torch.no_grad():
                        test_output = model(X_test_tensor)
                        test_loss = criterion(test_output, y_test_tensor).item()
                        test_losses.append(test_loss)
                if test_losses[-1] < best_loss:
                    best_loss = test_losses[-1]
                    best_model = model
                    best_params = (h1, h2, batch_size, lr, alpha)
                    best_train_losses = train_losses
                    best_test_losses = test_losses

# === מציאת סף אופטימלי
best_model.eval()
with torch.no_grad():
    probs = torch.sigmoid(best_model(X_test_tensor)).numpy()
fpr, tpr, thresholds = roc_curve(y_test, probs)
f1_scores = [f1_score(y_test, probs >= t) for t in thresholds]
best_thresh = thresholds[np.argmax(f1_scores)]
preds = (probs >= best_thresh).astype(int)

# === מדדים
cm = confusion_matrix(y_test, preds)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)

print(f"\n✅ Best Params: h1={best_params[0]}, h2={best_params[1]}, batch_size={best_params[2]}, lr={best_params[3]}, alpha={best_params[4]}")
print(f"Best Threshold (F1): {best_thresh:.3f}")
print(f"Final Loss: {best_loss:.4f}")
print(f"\n=== Confusion Matrix ===\n{cm}")
print(f"\n=== Classification Report ===\n{classification_report(y_test, preds)}")
print(f"ROC AUC: {roc_auc_score(y_test, probs):.6f}")
print(f"Specificity: {specificity:.4f}")
from sklearn.metrics import accuracy_score
print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")

# === מקרים חריגים: חולי שבץ שסווגו שגוי (False Negatives) ===
missed_cases = pd.DataFrame({
    "Actual": y_test.values.astype(bool),
    "Predicted": preds.astype(bool)
})
missed_cases['Correct'] = missed_cases['Actual'] == missed_cases['Predicted']

false_negatives = missed_cases[(missed_cases["Actual"] == True) & (missed_cases["Predicted"] == False)]
print("\n=== המקרים החריגים בהם בוצע סיווג שגוי לחולי שבץ (False Negatives) ===")
print(false_negatives.head(20))  # ניתן להגדיל את הכמות אם תרצי

# === מקרים חריגים: בריאים שסווגו כחולים (False Positives) ===
false_positives = missed_cases[(missed_cases["Actual"] == False) & (missed_cases["Predicted"] == True)]
print("\n=== המקרים החריגים בהם בוצע סיווג שגוי לבריאים (False Positives) ===")
pd.set_option('display.max_rows', None)
print(false_positives)
  # ניתן לשנות את המספר לפי הצורך


# === גרף loss משולב
plt.plot(best_train_losses, label="Training Loss")
plt.plot(best_test_losses, label="Test Loss")
plt.title("Model loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === שורות שסווגו שגוי ===
wrong_indices = np.where(y_test.values != preds)[0]
wrong_df = pd.DataFrame(X_test_scaled, columns=X.columns).iloc[wrong_indices].copy()
wrong_df["True_Label"] = y_test.values[wrong_indices]
wrong_df["Predicted"] = preds[wrong_indices]

print(f"\n=== סך הכל כלל השורות שסווגו שגוי: {len(wrong_df)} ===")
print(wrong_df[["True_Label", "Predicted"]].reset_index(drop=True).head(10))

missed_strokes = wrong_df[wrong_df["True_Label"] == 1]
print(f"\n🧠 חולי שבץ שלא זוהו (False Negatives): {len(missed_strokes)}")
print(missed_strokes.reset_index(drop=True).head(10))

print("\n=== ממוצעים של מאפיינים בין הסיווגים השגויים ===")
print(wrong_df.drop(columns=["True_Label", "Predicted"]).mean().sort_values(ascending=False))

with torch.no_grad():
    probs = torch.sigmoid(best_model(X_test_tensor)).numpy()

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# מחשוב ROC
fpr, tpr, thresholds = roc_curve(y_test, probs)
auc_score = roc_auc_score(y_test, probs)

# ציור הגרף
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.3f})", color='darkorange', linewidth=2)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # קו אלכסון - ניחוש רנדומלי
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity / Recall)")
plt.title("ROC Curve - Stroke Detection")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()
