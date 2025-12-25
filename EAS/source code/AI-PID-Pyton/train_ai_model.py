import pandas as pd
import numpy as np
from xgboost import XGBRegressor  # <--- Ganti Import
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load Dataset
filename = 'pid_training_SMART_TIERED_20251221_134114.csv' 
print(f"📂 Loading data from {filename}...")

try:
    df = pd.read_csv(filename)
except FileNotFoundError:
    print("❌ File tidak ditemukan! Pastikan nama file CSV sudah benar.")
    exit()

# Pembersihan data
df = df.dropna(subset=['mae', 'setpoint', 'kp', 'ki', 'kd'])

# Menentukan Fitur (Input) dan Target (Output)
X = df[['setpoint', 'kp', 'ki', 'kd']]
y = df['mae']

# Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model AI (XGBoost)
print("🧠 Training AI Model (XGBoost Balanced)...")

model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluasi
predictions = model.predict(X_test)
mae_score = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"\n📊 Model Performance:")
print(f"   Mean Absolute Error: {mae_score:.2f} (Semakin kecil semakin akurat)")
print(f"   R2 Score: {r2:.2f} (Mendekati 1.0 berarti sangat akurat)")

# 6. Simpan Model
joblib.dump(model, 'pid_model.pkl')
print("\n✅ Model disimpan sebagai 'pid_model.pkl'")
print("Sekarang Anda bisa menggunakan 'ai_pid_tuner.py' untuk mencari PID terbaik!")