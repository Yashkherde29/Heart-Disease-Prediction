import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("../dataset/heart.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ==========================================
# 2. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop("target", axis=1)
y = df["target"]


# ==========================================
# 3. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 4. FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==========================================
# 5. TRAIN MODEL
# ==========================================

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train_scaled, y_train)


# ==========================================
# 6. PREDICTION
# ==========================================

y_pred = model.predict(X_test_scaled)


# ==========================================
# 7. MODEL EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n================================")
print("MODEL RESULTS")
print("================================")

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ==========================================
# 8. SAVE MODEL
# ==========================================

joblib.dump(model, "heart_model.pkl")

joblib.dump(scaler, "scaler.pkl")

print("\n================================")
print("FILES SAVED")
print("================================")

print("heart_model.pkl")
print("scaler.pkl")