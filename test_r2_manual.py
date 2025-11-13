import numpy as np
from sklearn.metrics import r2_score

# Data aktual (y_true) dan hasil prediksi (y_pred)
y_true = np.array([3, -0.5, 2, 7, 9])
y_pred = np.array([2.5, 0.0, 2, 8, 9.7])

# --- 1. Hitung R² secara manual ---
y_mean = np.mean(y_true)
ss_res = np.sum((y_true - y_pred)**2)       # jumlah kuadrat residual
ss_tot = np.sum((y_true - y_mean)**2)       # total jumlah kuadrat
r2_manual = 1 - (ss_res / ss_tot)

# --- 2. Hitung R² menggunakan scikit-learn ---
r2_sklearn = r2_score(y_true, y_pred)

# --- Tampilkan hasil ---
print("R² (manual)       :", r2_manual)
print("R² (scikit-learn) :", r2_sklearn)
