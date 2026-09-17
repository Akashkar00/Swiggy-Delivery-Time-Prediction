# Swiggy Delivery Time Prediction

This project aims to predict the **delivery time of Swiggy food orders** using machine learning models based on rider details, weather, traffic conditions, and geolocation data.

---

## Problem Statement
Predict the food order delivery time based on real-world factors like rider attributes, vehicle type, weather, and traffic. This enables:
- Optimizing delivery operations
- Improving customer satisfaction
- Efficient resource allocation

---

## Business Use Case
- **Customer Satisfaction:** Deliver accurate ETAs and improve transparency.
- **Operational Efficiency:** Better resource and route management for riders.
- **Financial Impact:** Reduce compensation for delays, optimize fuel costs, and boost customer retention.

---

## Dataset
The dataset (`swiggy.csv`) contains:
- Rider details (age, ratings)
- Weather conditions
- Traffic density
- Delivery distance (calculated using geolocation)
- Delivery time in minutes

---

## ML Workflow

### 1. Data Preprocessing
- Handle missing values and outliers
- Feature engineering (delivery distance, time-based features)
- Encoding categorical variables

### 2. Train/Test Split
Chronological, not random: the model predicts an ETA for orders that
haven't happened yet, so it's tested on the most recent slice of orders
(the last 20% by `order_date`), not on a random sample that could put
same-day orders on both sides of the split.

### 3. Baseline
Before any model, a naive baseline (always predict the training set's
median delivery time) is evaluated on the same held-out rows as every
real model. Test MAE: **7.63 minutes** (n=45,502; run `train_model.py`
to reproduce). Any model has to beat this by a real margin to be worth
using.

### 4. Model Building
Models compared:
- **Linear Regression**
- **Random Forest Regressor**, lightly tuned with cross-validated random
  search (`RandomizedSearchCV`) rather than a single fit on default
  hyperparameters.
- **XGBoost** — trains on GPU automatically when one is available (e.g. a
  Colab GPU runtime), falls back to CPU otherwise.
- **LightGBM** — CPU only; GPU LightGBM needs a special build not
  guaranteed to be present on a stock Colab image.

All four reuse the same imputation + encoding pipeline, so they're
compared on identical features.

### 5. Model Evaluation
Metrics: MAE, RMSE, R² — each reported for both train and test so an
overfitting gap is visible, not hidden.

### 6. Model Tournament (A/B Testing)
Rather than eyeballing four models' aggregate metrics side by side,
`ab_test.py` runs a champion/challenger tournament: starting from Linear
Regression, each next model (Random Forest, then XGBoost, then LightGBM)
only takes the title if its per-row absolute errors on the *same* test
rows are significantly lower by a paired Wilcoxon signed-rank test — not
just numerically lower, which could be noise on this test set. The final
champion is what gets saved to `model.joblib`.

---

## Results

Trained on Colab (train n=36,069, test n=9,433, chronological split).

**Naive baseline** (predict training median): test MAE 7.63 min, test R² -0.00.

| Model | Train MAE | Test MAE | Train RMSE | Test RMSE | Train R² | Test R² |
|---|---|---|---|---|---|---|
| Linear Regression | 4.79 | 4.81 | 6.02 | 6.06 | 0.59 | 0.59 |
| Random Forest (tuned) | 2.18 | 3.32 | 2.77 | 4.21 | 0.91 | 0.80 |
| XGBoost (GPU, untuned) | 2.91 | 3.34 | 3.66 | 4.22 | 0.85 | 0.80 |
| LightGBM (untuned) | 3.09 | 3.35 | 3.88 | 4.24 | 0.83 | 0.80 |

All three tree models comfortably beat the naive baseline and Linear
Regression. Random Forest, XGBoost and LightGBM land within ~0.03 minutes
of each other on test MAE, and the tournament confirms none of that gap is
statistically real (see below) — so as trained here they're
interchangeable, not one clearly best.

**Tournament** (champion/challenger, paired Wilcoxon signed-rank test on
per-row absolute test error):

| Round | p-value | Result |
|---|---|---|
| Linear Regression vs Random Forest | 1.16e-292 | Random Forest wins (significant) |
| Random Forest vs XGBoost | 0.729 | No significant difference — champion holds |
| Random Forest vs LightGBM | 0.531 | No significant difference — champion holds |

**Final champion: Random Forest** — saved to `model.joblib`.

Worth flagging: Random Forest is the only model that got hyperparameter
search (`RandomizedSearchCV`, 8 candidates × 3-fold CV); XGBoost and
LightGBM ran with fixed, untuned hyperparameters. That's the likely reason
Random Forest edges the other two rather than a real algorithmic
advantage — a fair comparison would tune all three equally before
declaring a winner (see Future Improvements).

---

## Future Improvements
- Tune XGBoost and LightGBM with the same `RandomizedSearchCV` treatment
  Random Forest got — right now Random Forest's win in the tournament may
  just reflect that it's the only tuned model, not a real algorithmic edge
- Let XGBoost/LightGBM handle missing values natively instead of running
  them through the shared KNN-imputation step (would need each model on
  its own preprocessing branch, and a separate A/B test to check it's
  actually worth the added complexity vs. the shared pipeline)
- Add real-time traffic and weather APIs
- Deploy as a real-time prediction service

---

## Project Structure
```
├── swiggy.csv
├── Data Cleaning.ipynb
├── Food_Delivery_EDA.ipynb
├── data_clean_utils.py
├── swiggy_cleaned.csv
├── model_building.ipynb      # exploratory notebook: EDA + model comparison
├── model_utils.py            # chronological split, naive baseline, ordinal encoding
├── ab_test.py                # paired significance test between two models
├── train_model.py            # end-to-end script version of the notebook's modeling
├── model.joblib               # saved winning pipeline (generated by train_model.py)
└── tests/                    # pytest tests for data_clean_utils.py, model_utils.py, ab_test.py
```
