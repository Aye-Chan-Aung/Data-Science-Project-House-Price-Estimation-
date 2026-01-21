# House Price Prediction - Advanced Regression Techniques

This project is a data science solution for the [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) Kaggle competition. The goal is to predict the final sales price of residential homes in Ames, Iowa, using 79 explanatory variables describing (almost) every aspect of the homes.

## 📌 Project Overview

This repository contains a Jupyter Notebook that demonstrates a comprehensive machine learning workflow, moving from data preprocessing and extensive feature engineering to building a complex Stacked Ensemble model.

**Key Achievements:**
* **Feature Engineering:** created new aggregate features and transformed categorical variables.
* **Advanced Modeling:** Implemented a **Stacked Averaged Model** class from scratch.
* **Ensemble Strategy:** Blended predictions from Stacking, XGBoost, and LightGBM.
* **Performance:** Achieved a Training RMSE of approximately **0.081** on the stacked model.

## 🛠️ Technologies Used

* **Python 3.11**
* **Data Manipulation:** `pandas`, `numpy`
* **Visualization & Stats:** `scipy` (Box-Cox, Skew)
* **Machine Learning:** `scikit-learn`, `xgboost`, `lightgbm`

## 📊 Methodology

### 1. Data Preprocessing
* **Outlier Removal:** Removed extreme outliers in `GrLivArea` to improve model robustness.
* **Target Transformation:** Applied `Log1p` transformation to `SalePrice` to normalize the distribution.
* **Missing Values:** Handled utilizing both domain knowledge (e.g., filling 'None' for missing garage/basement features) and neighborhood-based median imputation for `LotFrontage`.

### 2. Feature Engineering
* **New Features:** Created aggregate features such as `TotalSF` (Total Square Footage), `TotalBath`, and `TotalPorchSF`.
* **Boolean Flags:** Added binary features like `HasPool`, `Has2ndFloor`, `HasGarage`, etc.
* **Skewness Correction:** Applied Box-Cox transformation to numerical features with skewness > 0.75.
* **Encoding:** Mapped ordinal quality features (e.g., `ExterQual`, `KitchenQual`) to integers and One-Hot encoded remaining categorical variables.

### 3. Modeling Architecture
The final prediction is a weighted ensemble of three powerful components:

1.  **Stacked Averaged Models:**
    * **Base Models:** ENet, GBoost, KRR (Ridge), Lasso.
    * **Meta Model:** Ridge Regression.
    * *Mechanism:* Uses out-of-fold predictions from base models to train the meta-model.
2.  **XGBoost Regressor:** Tuned for gradient boosting.
3.  **LightGBM Regressor:** Tuned for efficiency and performance.

### 4. Final Ensemble
The final submission is generated using a weighted average:
$$Final Prediction = (0.70 \times Stacked) + (0.15 \times XGBoost) + (0.15 \times LightGBM)$$

## 📉 Results

The models achieved the following Root Mean Squared Error (RMSE) scores on the training data:

| Model | Training RMSE |
|-------|---------------|
| **Stacked Ensemble** | 0.08189 |
| **XGBoost** | 0.08501 |
| **LightGBM** | 0.06917 |
