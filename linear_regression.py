# Task 3: Linear Regression
# Author: Darshan
# Internship - AI & ML

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# 1. Generate Synthetic House Price Dataset
# -----------------------------
np.random.seed(42)
n = 200

SquareFeet = np.random.randint(500, 3500, n)
Bedrooms = np.random.randint(1, 5, n)
Age = np.random.randint(1, 30, n)

# Simple formula for Price with some noise
Price = 50000 + (SquareFeet * 50) + (Bedrooms * 10000) - (Age * 800) + np.random.randint(-20000, 20000, n)

df = pd.DataFrame({
    "SquareFeet": SquareFeet,
    "Bedrooms": Bedrooms,
    "Age": Age,
    "Price": Price
})

print("First 5 rows of dataset:")
print(df.head())

# -----------------------------
# 2. Simple Linear Regression (SquareFeet vs Price)
# -----------------------------
X = df[['SquareFeet']]   # feature
y = df['Price']          # target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_simple = LinearRegression()
model_simple.fit(X_train, y_train)

y_pred_simple = model_simple.predict(X_test)

print("\n--- Simple Linear Regression ---")
print("Coefficient (slope):", model_simple.coef_[0])
print("Intercept:", model_simple.intercept_)
print("R² score:", r2_score(y_test, y_pred_simple))

# Plot regression line
plt.figure(figsize=(8,6))
plt.scatter(X_test, y_test, color='blue', label="Actual")
plt.plot(X_test, y_pred_simple, color='red', linewidth=2, label="Predicted Line")
plt.xlabel("SquareFeet")
plt.ylabel("Price")
plt.title("Simple Linear Regression: SquareFeet vs Price")
plt.legend()
plt.show()

# -----------------------------
# 3. Multiple Linear Regression (SquareFeet, Bedrooms, Age vs Price)
# -----------------------------
X_multi = df[['SquareFeet','Bedrooms','Age']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X_multi, y, test_size=0.2, random_state=42)

model_multi = LinearRegression()
model_multi.fit(X_train, y_train)

y_pred_multi = model_multi.predict(X_test)

print("\n--- Multiple Linear Regression ---")
print("Coefficients:", model_multi.coef_)
print("Intercept:", model_multi.intercept_)

# Evaluation
mae = mean_absolute_error(y_test, y_pred_multi)
mse = mean_squared_error(y_test, y_pred_multi)
r2 = r2_score(y_test, y_pred_multi)

print("MAE:", mae)
print("MSE:", mse)
print("R² score:", r2)

# -----------------------------
# 4. Residual Plot (check fit)
# -----------------------------
residuals = y_test - y_pred_multi
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_pred_multi, y=residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("Residual Plot - Multiple Regression")
plt.show()
