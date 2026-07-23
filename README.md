# 🏠 Property Price Prediction & Real Estate Analytics

A Data Science and Machine Learning project designed to predict property prices in the United States real estate market. The project leverages Exploratory Data Analysis (EDA) to uncover key drivers of property valuation, performs rigorous data preprocessing and feature engineering, and builds regression models to optimize pricing strategies and support investment decisions.

---

## 🌟 Key Features

* **Exploratory Data Analysis (EDA):** In-depth exploration of property dimensions, locations, and structural characteristics to identify primary price determinants.
* **Data Anomaly & Outlier Cleaning:** Systematic identification and removal of zero-value anomalies (e.g., zero price or room counts) and extreme outliers using the IQR method.
* **Logarithmic Transformation:** Normalization of skewed numerical distributions (`price` and `sqft_living`) using log transformations ($\log(1+x)$) to satisfy regression modeling assumptions.
* **Custom Feature Engineering:** Creation of domain-specific ratios, interactions, and age metrics to boost predictive power.
* **Comparative Model Evaluation:** Benchmarking multiple regression algorithms (Linear Regression, Random Forest, XGBoost) using RMSE, MAE, and $R^2$ metrics.
* **Model Serialization:** Deployment-ready packaging of the best-performing pipeline into a production file (`.pkl`).

---

## 🏗️ Project Pipeline

1. **Data Understanding & Cleaning:**
   * Loads `train_data` (2,944 rows) and `public_test_data` (736 rows).
   * Filters out invalid records with 0 bedrooms, 0 bathrooms, or 0 price.
   * Applies logarithmic transformations ($\log(1+x)$) to `price` and `sqft_living` to normalize skewed distributions.
   * Removes outliers using IQR bounds, refining the dataset to 2,849 training rows and 708 test rows.
2. **Feature Engineering & Transformation:**
   * Creates structural attributes: `house_age`, `renovated` flag, and binned age categories (`new`, `mid`, `old`).
   * Groups rare geographical entries in `city` (< 30 entries) under an `'Others'` category.
   * Engineers domain ratios and interaction terms: `bed_bath_ratio`, `living_lot_ratio`, `bed_lot_ratio`, `age_living_ratio`, `since_renovated`, `age_renovated_interaction`, `sqft_above_ratio`, `is_waterfront`, `has_view`, `living_to_bed_ratio`, and `lot_to_bath_ratio`.
3. **Data Preprocessing & Finalization:**
   * Imputes missing values resulting from ratio features using column medians (numeric) and modes (categorical).
   * Scales numerical features using `StandardScaler`.
   * Splits training data into train (80%) and validation/test (20%) sets.
4. **Model Training & Selection:**
   * Trains Linear Regression, Random Forest, and XGBoost models.
   * Evaluates performance on test sets to select the optimal model for production.

---

## 📂 Outputs & Key Insights

The pipeline uncovers actionable insights into real estate pricing factors:

* **Key Price Drivers:**
  * **Living Area (`sqft_living`):** Strongest positive correlation with property price ($r = 0.68$).
  * **Above-Ground Area (`sqft_above`):** Significant secondary determinant ($r = 0.58$).
  * **Bathrooms (`bathrooms`):** Moderate positive relationship ($r = 0.50$).
* **Location Influence:**
  * Seattle leads dataset distribution with 1,033 properties, followed by Renton (181) and Bellevue (181).
* **Weak Correlates:**
  * Building age (`yr_built`), property condition (`condition`), and floor count (`floors`) exhibit relatively weak direct correlations with price, making interaction terms essential.

### Feature Correlation Summary

| Feature | Feature Type | Correlation ($r$) with Price | Impact / Insight |
| :--- | :--- | :--- | :--- |
| **`sqft_living`** | Numerical | **0.68** | Primary structural price driver |
| **`sqft_above`** | Numerical | **0.58** | Strong secondary living space determinant |
| **`bathrooms`** | Numerical | **0.50** | Moderate impact on valuation |
| **`yr_built` / `condition`** | Numerical | Low | Requires interaction/ratio engineering |

---

## 📊 Preprocessing & Model Highlights

### Code Highlight: Pipeline & Feature Scaling

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Log Transformation for Skewed Target & Feature
train_data['price_log'] = np.log1p(train_data['price'])
train_data['sqft_living_log'] = np.log1p(train_data['sqft_living'])

# Feature Engineering Examples
train_data['house_age'] = 2024 - train_data['yr_built']
train_data['bed_bath_ratio'] = train_data['bedrooms'] / train_data['bathrooms']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
```

### Model Performance Comparison

Evaluated on the test split:

| Model | MAE (USD) | RMSE (USD) | $R^2$ Score |
| :--- | :---: | :---: | :---: |
| **Linear Regression** | $52,940.44 | $75,707.60 | 0.93 |
| **XGBoost** | $2,307.04 | $6,120.28 | 1.00 |
| **Random Forest (Best)** | **$691.63** | **$2,052.87** | **1.00** |

> **Selected Model:** The **Random Forest Regressor** achieved superior accuracy with an MAE of **$691.63**, RMSE of **$2,052.87**, and an $R^2$ of **1.00**.

---

## 🛠️ Built With

* [Python 3.x](https://www.python.org/) - Programming Language
* [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/) - Data Cleaning & Feature Engineering
* [Scikit-learn](https://scikit-learn.org/) - Preprocessing, Standard Scaling & Machine Learning Models
* [XGBoost](https://xgboost.ai/) - Gradient Boosting Framework
* [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/) - Data Visualization
* [Joblib](https://joblib.readthedocs.io/) - Model Serialization (`.pkl`)

---

## 📌 Business Recommendations

* **Prioritize Living Area Adjustments:** Focus valuation strategies around total living area (`sqft_living`) and above-ground square footage (`sqft_above`), as these are the strongest drivers of property value.
* **Leverage Automated Pricing:** Integrate the trained Random Forest model into valuation workflows to automate baseline price estimations, reducing manual appraisal overhead.
* **Location-Based Pricing Strategies:** Tailor marketing and investment approaches to high-density hubs (e.g., Seattle, Bellevue) while applying localized adjustments for lower-density suburban markets.
