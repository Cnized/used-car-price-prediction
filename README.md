# 🚗 Used Car Price Prediction

Predicting used car prices with gradient boosting — built from a messy, real-world Kaggle dataset with broken formatting, extreme skew, and high-cardinality categorical features.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Overview

This project predicts the sale price of used cars based on features like brand, model year, mileage, engine specs, and accident history. It's a from-scratch regression pipeline — no templates, no pre-built notebooks — built to practice the full ML workflow end to end: data cleaning, feature engineering, model tuning, and honest evaluation.

**Final result:** MAE ≈ **$12,771** on a held-out validation set (log-transformed target, XGBoost).

---

## 🧩 The Dataset

A raw scraped Kaggle used-car listings dataset — ~4,000 rows, 12 columns, and genuinely messy:

- `price` stored as text with currency symbols and commas (`"$21,500"`)
- `milage` stored as text with units (`"45,000 mi."`)
- `engine` a free-text description bundling horsepower, displacement, cylinder count, and fuel type into one string (`"300.0HP 3.7L V6 Cylinder Engine Flex Fuel Capability"`)
- `model` with **622 unique values** across ~4,000 rows — most appearing only once or twice
- A heavily right-skewed target: median price ~$31K, but a long tail up to $489K

None of this was clean going in — cleaning it was most of the actual work.

---

## 🛠️ What I Did

**🧹 Data cleaning**
- Parsed `price` and `milage` out of formatted strings into real numeric columns
- Extracted `horsepower` and `engine_size` from the free-text `engine` column using regex
- Converted `model_year` to a proper numeric/ordinal feature
- Trimmed extreme price outliers from the *training* set only (never touched validation — that has to stay an honest, untouched test of real-world performance)

**⚙️ Feature engineering**
- Categorical columns filtered by cardinality — low-cardinality features (`brand`, `fuel_type`, `transmission`) one-hot encoded; high-cardinality ones (`model`, raw `engine` text) excluded from naive encoding after testing showed they hurt more than helped
- Tested target encoding on `model` — reverted it after discovering the median car model appears only *once* in training data, making the encoding functionally identical to leaking the target
- Log-transformed the target (`price`) to correct for the dataset's heavy right-skew

**🤖 Modeling**
- `ColumnTransformer` + `Pipeline` for reproducible preprocessing (median/constant imputation depending on the column, one-hot encoding for categoricals)
- `XGBRegressor`, tuned via `RandomizedSearchCV` (max_depth, learning_rate, n_estimators, subsample, colsample_bytree)
- Also benchmarked CatBoost (handles high-cardinality categoricals natively) — comparable performance, no significant edge over the tuned XGBoost pipeline

---

## 📊 Results

| Stage | MAE |
|---|---|
| First working pipeline (raw, unfixed columns) | ~45,000 *(bug — log/real-scale mismatch)* |
| Bug fixed, log-transform working correctly | 22,226 |
| Outlier trimming | 22,091 |
| `milage` fixed from text → numeric | ~20,000 |
| High-cardinality columns excluded from one-hot | 14,900 |
| `constant` imputation over `median` for engine features | 14,900 |
| **Categorical dtype bug fixed** (`str` vs `object` mismatch was silently dropping all categorical features) | **12,771** |
| Hyperparameter search (RandomizedSearchCV) | 12,900 *(no further gain — manual tuning had already converged)* |

---

## 🧠 Why It Plateaus Around $12.7K

This isn't a case of "needs more tuning" — it's a structural limit of the dataset:

- **`model` has 622 unique values with a median of ~1 example each.** There isn't enough repetition per model for any encoding method (target, frequency, or native categorical handling) to learn a reliable price signal from it.
- **Used car pricing has real-world noise this dataset can't capture** — negotiation, seller type, regional demand, and condition beyond a binary accident flag all affect price and aren't in the data.

Getting meaningfully below this would need a richer dataset (more examples per model, condition scores, regional data) — not better modeling of what's already here.

---

## 🧰 Tech Stack

- **Python** — pandas, NumPy
- **scikit-learn** — pipelines, preprocessing, `RandomizedSearchCV`
- **XGBoost** — primary model
- **CatBoost** — benchmark comparison for native categorical handling

---

## 📁 Project Structure

```
used-car-pred-price/
├── data/
│   └── used_cars.csv
├── src/
│   └── car_price_pred.py
├── test.ipynb
├── venv/
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 Setup

```bash
git clone https://github.com/Cnized/used-car-price-prediction.git
cd used-car-pred-price

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Open `test.ipynb` or run `src/car_price_pred.py` to reproduce the results.

---

## 💡 Key Takeaways

- 🐛 Silent dtype bugs are dangerous — a `str` vs `object` mismatch quietly dropped every categorical feature from the pipeline for several iterations before being caught
- 🎯 Encoding technique matters more than raw modeling power on messy categorical data — target encoding is not a free win, it actively backfires on ultra-high-cardinality, low-frequency categories
- 📉 Knowing *why* a model plateaus is as valuable as pushing the number down further

---

## 👨‍💻 Author

💻 Built with ❤️ by Kian Kheiri N. ([@Cnized](https://github.com/Cnized))
