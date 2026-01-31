# Supervised Machine Learning from Scratch 🧠

This repository contains clean, modular, and optimized implementations of fundamental Supervised Learning algorithms developed using **NumPy** and **Python**. The focus is on understanding the mathematical "under-the-hood" logic without relying on high-level libraries like Scikit-Learn.

[Image of supervised machine learning workflow diagram]

## 🚀 Implemented Algorithms

### 1. Linear Regression (Dual Approach)
* **Gradient Descent:** Iteratively minimizes Mean Squared Error (MSE) to find optimal weights.
* **Ordinary Least Squares (OLS):** Implemented using the **Normal Equation** for an exact analytical solution.

### 2. Logistic Regression
* A binary classification model using the **Sigmoid activation function**.
* Optimized using **Binary Cross-Entropy loss** and Gradient Descent.

### 3. K-Nearest Neighbors (KNN)
* **Classifier:** Assigns labels based on the majority vote of the nearest neighbors.
* **Regressor:** Predicts continuous values by averaging the $k$ nearest target values.
* **Distance Metric:** Efficiently calculated using **Vectorized Euclidean Distance**.

---

## 🛠️ Tech Stack & Skills
* **Language:** Python
* **Library:** NumPy (Vectorized operations for performance)
* **Concepts:** Matrix Calculus, Optimization, Gradient Descent, Linear Algebra.

---

## 📂 Project Structure
```text
Supervised_ML_Scratch_Implementation/
│
├── KNN_Classifier.ipynb        # K-Nearest Neighbors for Classification
├── KNN_Regressor.ipynb         # K-Nearest Neighbors for Regression
├── Linear_regression.ipynb     # Gradient Descent & OLS Implementation
└── Logistic_regression.ipynb   # Binary Classification from scratch
