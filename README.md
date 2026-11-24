
# Linear Regression and Regularization Using Python

## Overview

This project demonstrates the implementation of Linear Regression and regularization techniques using Python, Pandas, NumPy, and Scikit-learn.

A synthetic car dataset is generated to predict car prices based on different features such as vehicle age, kilometers driven, engine capacity, and fuel type.

The project explores model evaluation, Ridge Regression, Lasso Regression, feature importance, and the effect of different train-test splits.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook

## Dataset Description

A synthetic dataset containing 120 records is generated using NumPy.

| Feature | Description |
|---|---|
| age_years | Age of the car in years |
| km_driven | Total kilometers driven |
| engine_cc | Engine capacity in CC |
| fuel_type | Fuel type represented numerically |
| noise_1 | Random noise feature |
| noise_2 | Random noise feature |
| price_lakhs | Target variable representing car price in lakhs |

## Tasks Performed

### Task 1: Linear Regression

- Generated a synthetic car price dataset.
- Separated independent variables (X) and the target variable (y).
- Split the dataset into 80% training data and 20% testing data.
- Applied StandardScaler for feature scaling.
- Implemented Linear Regression using `LinearRegression()`.
- Evaluated model performance using the R² score.

**Results:**
- Training R² Score: 0.6856
- Testing R² Score: 0.6365

### Task 2: Ridge Regression (L2 Regularization)

- Implemented Ridge Regression using `Ridge()`.
- Tested different alpha values to understand the effect of regularization.
- Compared training and testing R² scores.
- Selected the best alpha based on the highest testing R² score.

**Results:**

| Alpha | Train R² | Test R² |
|---|---:|---:|
| 0.01 | 0.6856 | 0.6365 |
| 1 | 0.6855 | 0.6405 |
| 10 | 0.6788 | 0.6637 |
| 100 | 0.5052 | 0.5549 |
| 500 | 0.2064 | 0.2335 |

**Best Alpha: 10**

### Task 3: Lasso Regression (L1 Regularization)

- Implemented Lasso Regression using `Lasso()`.
- Used an alpha value of 1.0.
- Evaluated the coefficients of individual features.
- Observed how L1 regularization reduces less important feature coefficients to zero.

**Feature Coefficients:**

| Feature | Coefficient |
|---|---:|
| age_years | -0.6307 |
| km_driven | -0.2007 |
| engine_cc | 1.0973 |
| fuel_type | 0.0000 |
| noise_1 | 0.0000 |
| noise_2 | 0.0000 |

### Task 4: Effect of Different Random Seeds

- Tested the Ridge Regression model using different random seeds.
- Used random states 42, 7, and 123.
- Compared testing R² scores to understand how different train-test splits affect model performance.

**Results:**

| Random Seed | Test R² |
|---|---:|
| 42 | 0.6637 |
| 7 | 0.6649 |
| 123 | 0.6026 |

## Libraries Used

```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
```

## Key Learnings

- Understanding Linear Regression and predictive modelling.
- Generating synthetic datasets using NumPy.
- Splitting datasets into training and testing sets.
- Applying StandardScaler for feature scaling.
- Understanding Ridge Regression (L2 Regularization).
- Understanding Lasso Regression (L1 Regularization).
- Comparing different alpha values and their impact on model performance.
- Analysing feature coefficients and identifying less useful features.
- Understanding the effect of random seeds on model evaluation.
- Evaluating machine learning models using the R² score.

## Conclusion

This project provided practical experience with Linear Regression and regularization techniques.

Ridge Regression achieved its best testing R² score of approximately 0.6637 at an alpha value of 10. Lasso Regression demonstrated how regularization can reduce feature coefficients to zero, helping identify less influential features.

The experiment with different random seeds also showed that model performance can vary depending on how the dataset is divided into training and testing sets.

Overall, the project demonstrates important concepts in regression modelling, regularization, feature selection, and model evaluation.
