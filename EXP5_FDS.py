import csv
import math
import random
import os
import numpy as np
 
os.system("")  # enables colours in VS Code / Windows terminal
GREEN, RED, YELLOW, CYAN, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[96m", "\033[0m"
 
FILE_NAME = "london_weather_data_1979_to_2023.csv"
 
# ---------------------------------------------------------
# Step 1: Read the dataset
# ---------------------------------------------------------
rows = []
with open(FILE_NAME) as f:
    for r in csv.DictReader(f):
        rows.append(r)
 
print(CYAN + "=== EXPERIMENT 5 : REGRESSION IMPUTATION ===" + RESET)
print("Total rows               :", len(rows))
 
# rows where TG is known (used to build model) and rows where TG is missing
known = [r for r in rows if r["TG"] != ""]
missing = [r for r in rows if r["TG"] == ""]
print("Rows with TG known       :", len(known))
print(YELLOW + "Rows with TG missing     : " + str(len(missing)) + RESET)
 
# ---------------------------------------------------------
# Step 2: Split known rows into training (80%) and testing (20%)
# (testing part is used only to check the accuracy)
# ---------------------------------------------------------
random.seed(1)
random.shuffle(known)
split = int(0.8 * len(known))
train = known[:split]
test = known[split:]
 
 
def get(data, col):
    return [float(r[col]) for r in data]
 
 
# ---------------------------------------------------------
# Step 3: Simple Linear Regression   y = w0 + w1*x
#   w1 = sum((x - xmean)*(y - ymean)) / sum((x - xmean)^2)
#   w0 = ymean - w1*xmean
# ---------------------------------------------------------
def simple_regression(x, y):
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    num = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    den = sum((a - x_mean) ** 2 for a in x)
    w1 = num / den
    w0 = y_mean - w1 * x_mean
    return w0, w1
 
 
# ---------------------------------------------------------
# Step 4: Multiple Linear Regression   b = (X^T X)^-1 X^T Y
# ---------------------------------------------------------
def multiple_regression(X, y):
    X = np.column_stack([np.ones(len(X)), X])      # add column of 1s for b0
    b = np.linalg.inv(X.T @ X) @ X.T @ np.array(y)
    return b
 
 
# ---------------------------------------------------------
# Step 5: Accuracy measures
# ---------------------------------------------------------
def accuracy(actual, predicted):
    n = len(actual)
    errors = [a - p for a, p in zip(actual, predicted)]
    mae = sum(abs(e) for e in errors) / n
    rmse = math.sqrt(sum(e * e for e in errors) / n)
    a_mean = sum(actual) / n
    ss_res = sum(e * e for e in errors)
    ss_tot = sum((a - a_mean) ** 2 for a in actual)
    r2 = 1 - ss_res / ss_tot
    return mae, rmse, r2
 
 
y_train = get(train, "TG")
y_test = get(test, "TG")
 
# ---------------- Simple Linear Regression ----------------
w0, w1 = simple_regression(get(train, "TN"), y_train)
slr_pred_test = [w0 + w1 * x for x in get(test, "TN")]
slr_mae, slr_rmse, slr_r2 = accuracy(y_test, slr_pred_test)
 
print(CYAN + "\n--- 1. SIMPLE LINEAR REGRESSION (TG from TN) ---" + RESET)
print(f"Equation : TG = {w0:.3f} + {w1:.3f} * TN")
print(f"MAE  = {slr_mae:.3f}   RMSE = {slr_rmse:.3f}   R2 = {slr_r2:.4f}")
 
# ---------------- Multiple Linear Regression ----------------
X_train = [[float(r["TN"]), float(r["TX"])] for r in train]
X_test = [[float(r["TN"]), float(r["TX"])] for r in test]
b = multiple_regression(X_train, y_train)
mlr_pred_test = [b[0] + b[1] * x[0] + b[2] * x[1] for x in X_test]
mlr_mae, mlr_rmse, mlr_r2 = accuracy(y_test, mlr_pred_test)
 
print(CYAN + "\n--- 2. MULTIPLE LINEAR REGRESSION (TG from TN and TX) ---" + RESET)
print(f"Equation : TG = {b[0]:.3f} + {b[1]:.3f} * TN + {b[2]:.3f} * TX")
print(f"MAE  = {mlr_mae:.3f}   RMSE = {mlr_rmse:.3f}   R2 = {mlr_r2:.4f}")
 
# ---------------------------------------------------------
# Step 6: Compare both models
# ---------------------------------------------------------
print(CYAN + "\n--- 3. COMPARISON ---" + RESET)
print(f"{'Model':<28}{'MAE':>8}{'RMSE':>8}{'R2':>9}")
print(f"{'Simple Linear (TN)':<28}{slr_mae:>8.3f}{slr_rmse:>8.3f}{slr_r2:>9.4f}")
print(f"{'Multiple Linear (TN, TX)':<28}{mlr_mae:>8.3f}{mlr_rmse:>8.3f}{mlr_r2:>9.4f}")
better = "Multiple Linear Regression" if mlr_rmse < slr_rmse else "Simple Linear Regression"
print(GREEN + "Better model (lower error): " + better + RESET)
 
# ---------------------------------------------------------
# Step 7: Predict the actual missing TG values (using the better model,
#         trained again on ALL known rows)
# ---------------------------------------------------------
X_all = [[float(r["TN"]), float(r["TX"])] for r in known]
y_all = get(known, "TG")
b = multiple_regression(X_all, y_all)
 
print(CYAN + "\n--- 4. PREDICTED MISSING TG VALUES (Multiple Regression) ---" + RESET)
print(f"{'DATE':<12}{'TN':>8}{'TX':>8}{'Predicted TG':>15}")
for r in missing:
    tn, tx = float(r["TN"]), float(r["TX"])
    r["TG"] = round(b[0] + b[1] * tn + b[2] * tx, 1)   # fill the missing value
    print(f"{r['DATE']:<12}{tn:>8.0f}{tx:>8.0f}{r['TG']:>15}")
 
print(GREEN + f"\nAll {len(missing)} missing TG values have been filled." + RESET)
print("(TG, TN, TX are in 0.1 degC, e.g. 115 = 11.5 degC)")
