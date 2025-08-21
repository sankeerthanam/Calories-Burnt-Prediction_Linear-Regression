

## 🔥 Calories Burnt Prediction (XGBoost vs Linear Regression)

📌 Project Overview

This project predicts Calories Burnt during exercise based on physiological and activity-related features.
We use two machine learning models:

1. XGBoost Regressor (tree-based, non-linear)
2. Linear Regression (simple linear baseline)

The goal is to compare their performances and build a predictive system.

---

📂 Dataset

We used two CSV files:
calories.csv → contains `User_ID` and `Calories` burnt
exercise.csv → contains `User_ID`, `Gender`, `Age`, `Height`, `Weight`, `Duration`, `Heart_Rate`, `Body_Temp`

These were merged into one dataset with features and target.

---

⚙️ Data Preprocessing

1. Merged `calories.csv` and `exercise.csv` on `User_ID`.
2. Encoded categorical feature:
   `Gender`: male → 0, female → 1
3. Removed unnecessary column `User_ID`.
4. Split data into training (80%) and testing (20%).
5. Scaled not needed (XGBoost and LR both handle numeric scales well).

---

📊 Exploratory Data Analysis

Duration, Heart Rate, and Body Temp showed strong positive correlation with calories burnt.
Distribution plots show natural spread across `Age`, `Height`, and `Weight`.
Heatmap of correlations confirmed main influencing features.

---

🧠 Models Used

1️⃣ XGBoost Regressor

* Tuned Hyperparameters:

  ```python
  model = XGBRegressor(
      n_estimators=200,
      learning_rate=0.1,
      max_depth=6,
      subsample=0.8,
      colsample_bytree=0.8,
      random_state=42
  )
  ```
* Captures **non-linear interactions** between features.
* Performs **cross-validation (5-fold)** for robust evaluation.

2️⃣ Linear Regression

* Simple baseline assuming Linear relationship.
* No feature engineering beyond categorical encoding.

---

📈 Results

| Model                 | Train R² | Test R² | MAE  | RMSE  |
| --------------------- | -------- | ------- | ---- | ----- |
| **XGBoost**           | 0.9996   | 0.9988  | 1.48 | 2.17  |
| **Linear Regression** | 0.9673   | 0.9669  | 8.38 | 11.41 |

✅ **XGBoost** outperforms Linear Regression by a huge margin.

* LR captures \~96% variance but struggles with non-linearities.
* XGBoost achieves near-perfect fit with very low error.

---

## 📉 Visualizations

* **Actual vs Predicted (Scatterplot)** → XGBoost aligns closely with the diagonal line, LR has more spread.
* **Residuals Distribution** → LR residuals wider, XGBoost residuals very small.
* **Feature Importance (XGBoost)** shows `Duration`, `Heart_Rate`, and `Body_Temp` are most influential.

---

## 🚀 Predictive System

We can now predict calories burnt for a new input:

```python
input_data = (0, 69, 179.0, 79.0, 5.0, 88.0, 38.7)
prediction = model.predict([input_data])
print("Predicted Calories Burnt:", prediction[0])
```

---

## 📌 Key Takeaways

* **Linear Regression** is a decent baseline but **misses non-linear feature interactions**.
* **XGBoost** captures complex relationships and provides **high accuracy** with low error.
* The project highlights the importance of **choosing the right algorithm** depending on data complexity.

---

## 🛠️ Tech Stack

* Python 🐍
* Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`




