# Swiggy Delivery Time Prediction

Predicts how long a Swiggy food order will take to deliver, from rider
details, weather, traffic, order type and pickup/drop-off distance.

The focus is on evaluating models honestly, not just posting a high score:
a chronological train/test split, a naive baseline every model has to beat,
and a champion/challenger model tournament where a challenger only wins if
a paired significance test backs it up.

---

## Results

Chronological split: train n=36,069 (orders up to 2022-03-28), test n=9,433
(2022-03-29 to 2022-04-06). All models share the same imputation and encoding
pipeline, and the target is Box-Cox transformed for training and inverted for
scoring.

| Model | Train MAE | Test MAE | Test RMSE | Train R² | Test R² |
|---|---|---|---|---|---|
| Naive baseline (training median) | — | 7.63 | — | — | -0.00 |
| Linear Regression | 4.79 | 4.81 | 6.06 | 0.59 | 0.59 |
| Random Forest (tuned) | 2.18 | 3.32 | 4.21 | 0.91 | 0.80 |
| XGBoost (GPU, untuned) | 2.91 | 3.34 | 4.22 | 0.85 | 0.80 |
| LightGBM (untuned) | 3.09 | 3.35 | 4.24 | 0.83 | 0.80 |
| **TabPFN (10k-row training subsample)** | 2.79* | **3.19** | **4.02** | 0.86* | **0.82** |

MAE and RMSE are in minutes. \*TabPFN's train metrics are on the 10,000 rows it
was fit on, not the full training set.

**Tournament** (paired Wilcoxon signed-rank test on per-row absolute test
error, α = 0.05):

| Round | p-value | Outcome |
|---|---|---|
| Linear Regression vs Random Forest | 1.2e-292 | Random Forest takes the title |
| Random Forest vs XGBoost | 0.73 | No significant difference, champion holds |
| Random Forest vs LightGBM | 0.53 | No significant difference, champion holds |
| Random Forest vs TabPFN | 5.8e-13 | TabPFN takes the title |

**Champion: TabPFN.**

### Reading these results

- **The three tree models are statistically tied.** Random Forest, XGBoost and
  LightGBM finish within 0.03 minutes of each other, and the tournament
  can't separate them. Random Forest is also the only one of the three that
  got a hyperparameter search, so its small lead may come from tuning rather
  than from the algorithm.
- **TabPFN's win is real but small in practice.** It's significant
  (p = 5.8e-13) and it trained on 26,000 fewer rows than the other models, but
  the gain over Random Forest is about 0.13 minutes of MAE (roughly 8 seconds
  per order).
- **TabPFN is harder to deploy.** The tree models are self-contained sklearn
  pipelines. TabPFN needs a GPU for fast local inference, or a hosted API and
  a `TABPFN_TOKEN`. For production, Random Forest or XGBoost is the safer
  choice unless those 8 seconds matter.

---

## Methodology

1. **Cleaning** (`swiggy_delivery.data_cleaning`): normalises column names and
   string values, drops riders under 18 and invalid 6-star ratings, fixes
   coordinates, extracts date and time-of-day features, and computes
   haversine distance and distance bands.
2. **Chronological split** (`model_utils.time_based_split`): the latest 20% of
   order dates form the test set, and the cutoff falls on a date boundary so no
   single day is split across train and test. A random split would leak
   same-day conditions into training.
3. **Leakage control**: `pickup_time_minutes` is dropped because it isn't
   known when the ETA is predicted.
4. **Preprocessing**: mode or constant imputation, MinMax scaling, one-hot and
   ordinal encoding, then KNN imputation. The ordinal encoder keeps missing
   values as `NaN` rather than a `-999` sentinel, which KNN would treat as a
   real, far-away value.
5. **Models**: Linear Regression, Random Forest (`RandomizedSearchCV`, 8
   candidates × 3-fold), XGBoost (uses a GPU when one is detected), LightGBM
   (CPU), and TabPFN (pretrained tabular foundation model; its attention cost
   grows roughly quadratically with the number of rows, so training is capped
   at 10,000 seeded rows while evaluation uses the full test set).
6. **Tournament** (`swiggy_delivery.ab_test`): each challenger must beat the
   current champion on the same test rows with a significant paired test. A
   lower average error alone isn't enough.

---

## Project Structure

```
├── data/
│   ├── raw/swiggy.csv                  # original dataset
│   ├── interim/cleaned_data.csv        # exploratory cleaning output (notebook 01)
│   └── processed/swiggy_cleaned.csv    # modelling input (swiggy_delivery.data_cleaning)
├── notebooks/
│   ├── 01_data_cleaning.ipynb          # step-by-step cleaning exploration
│   ├── 02_eda.ipynb                    # exploratory data analysis
│   └── 03_model_building.ipynb         # modelling + tournament (run on Colab GPU)
├── src/swiggy_delivery/
│   ├── config.py                       # project paths
│   ├── data_cleaning.py                # raw → processed cleaning pipeline
│   ├── model_utils.py                  # chronological split, naive baseline, encoder
│   ├── ab_test.py                      # paired Wilcoxon model comparison
│   └── train.py                        # end-to-end training + tournament script
├── models/                             # trained artifacts (gitignored)
├── tests/                              # pytest unit tests
├── pyproject.toml
└── requirements.txt                    # flat dependency list for Colab
```

---

## Getting Started

### Local

```bash
git clone https://github.com/Akashkar00/Swiggy-Delivery-Time-Prediction.git
cd Swiggy-Delivery-Time-Prediction

uv venv && source .venv/bin/activate
uv pip install -e ".[notebooks,dev]"

pytest                                   # run unit tests
python -m swiggy_delivery.data_cleaning  # data/raw → data/processed
python -m swiggy_delivery.train          # train all models, save champion to models/
```

Training includes TabPFN, which needs PyTorch and is slow without a GPU.

### Google Colab

1. Open `notebooks/03_model_building.ipynb` from GitHub (the badge at the top
   of the notebook, or File → Open notebook → GitHub).
2. Runtime → Change runtime type → GPU.
3. If TabPFN asks for a token, add it as a Colab secret named `TABPFN_TOKEN`
   (key icon in the left sidebar). **Never paste the token into a cell**,
   because the notebook gets committed.
4. Runtime → Run all. The first cell clones the repo, installs requirements and
   sets the working directory.

---

## Future Improvements

- Give XGBoost and LightGBM the same hyperparameter search as Random Forest
  before reading anything into the tree-model rankings.
- Use TabPFN's ensembling extensions (`tabpfn-extensions`) to train on more
  than 10,000 rows.
- Let the gradient-boosted models handle missing values natively instead of
  going through the shared KNN imputer.
- Add a lightweight prediction API around the saved pipeline.
