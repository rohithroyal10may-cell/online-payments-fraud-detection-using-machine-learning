import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

print("=" * 80)
print("TRAINING FRAUD DETECTION MODEL")
print("=" * 80)

dataset_path = os.path.join("code files", "balanced_dataset.csv")
print(f"\n[1] Loading dataset...")
df = pd.read_csv(dataset_path)
print(f"    Shape: {df.shape}")

X = df.drop("isFraud", axis=1)
y = df["isFraud"]

print(f"\n[2] Encoding string columns...")
label_encoders = {}
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le
    print(f"    Encoded: {col}")

print(f"\n[3] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"    Train: {X_train.shape} | Test: {X_test.shape}")

print(f"\n[4] Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"    Scaled successfully")

print(f"\n[5] Training model (this takes 5-10 minutes)...")
model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1, verbose=1)
model.fit(X_train_scaled, y_train)

print(f"\n[6] Evaluating model...")
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n{'='*80}")
print(f"RESULTS:")
print(f"  Accuracy:  {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall:    {rec:.4f}")
print(f"  F1 Score:  {f1:.4f}")
print(f"{'='*80}")

print(f"\n[7] Saving files...")
joblib.dump(model, os.path.join("code files", "fraud_model.pkl"))
joblib.dump(scaler, os.path.join("code files", "scaler.pkl"))
joblib.dump(label_encoders, os.path.join("code files", "encoders.pkl"))

print(f"\n✅ TRAINING COMPLETE!")