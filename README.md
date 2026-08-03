# 🚗 Used Car Price Prediction

Predicting used car prices with gradient boosting — built from a messy, real-world Kaggle dataset with broken formatting, extreme skew, and high-cardinality categorical features.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Overview

This project predicts the sale price of used cars based on features like brand, model year, mileage, engine specs, and accident history. It's a from-scratch regression pipeline — no templates, no pre-built notebooks — built to practice the full ML workflow end to end: data cleaning, feature engineering, model tuning, and honest evaluation.

Final result: MAE ≈ **$5,516** on a held-out validation set (log-transformed target, tuned XGBoost).

---

## 🧩 The Dataset

A raw scraped Kaggle used-car listings dataset — ~4,000 rows, 12 columns, and genuinely messy:

- `price` stored as text with currency symbols and commas (`"$21,500"`)
- `milage` stored as text with units (`"45,000 mi."`)
- `engine` a free-text description bundling horsepower, displacement, cylinder count, and fuel type into one string (`"300.0HP 3.7L V6 Cylinder Engine Flex Fuel Capability"`)
- `model` and other categorical features with high cardinality
- A heavily right-skewed target: median price ~$31K, but a long tail reaching several million dollars due to a few extreme listings

None of this was clean going in — cleaning it was most of the actual work.

---

## 🛠️ What I Did

**🧹 Data cleaning**
- Parsed `price` and `milage` out of formatted strings into real numeric columns
- Extracted `horsepower` and `engine_size` from the free-text `engine` column using regex
- Converted `model_year` to a proper numeric/ordinal feature
- Removed extreme price outliers (`price > $130K`) after analyzing the target distribution. These rare listings severely distorted MAE because there were too few examples for the model to learn from.


**⚙️ Feature engineering**
- Categorical features were processed using one-hot encoding with unknown category handling. High-cardinality features were tested, but the final pipeline focused on features that improved validation performance.- Tested target encoding on `model` — reverted it after discovering the median car model appears only *once* in training data, making the encoding functionally identical to leaking the target
- Log-transformed the target (`price`) to correct for the dataset's heavy right-skew
- Created additional numerical features:
  - `horsepower` extracted from engine text
  - `engine_size` extracted from engine text
  - `age` calculated from model year


**🤖 Modeling**
- `ColumnTransformer` + `Pipeline` for reproducible preprocessing (median/constant imputation depending on the column, one-hot encoding for categoricals)
- XGBRegressor, manually tuned after experimentation with tree depth, number of estimators, learning rate, regularization, and sampling parameters
- Also benchmarked CatBoost (handles high-cardinality categoricals natively) — comparable performance, no significant edge over the tuned XGBoost pipeline

---


## 📊 Results

The model improved through several iterations of debugging, data cleaning, and experimentation. Each step addressed a specific problem discovered during analysis.

| Stage                                      |        MAE | What Changed                                                                                                                                   |
| ------------------------------------------ | ---------: | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| First working pipeline                     |    ~45,000 | Initial model trained on raw formatted columns. Error was caused by incorrect evaluation scale (log predictions compared against real prices). |
| Log-transform bug fixed                    |    ~22,226 | Predictions were correctly converted back using `expm1()` before calculating MAE.                                                              |
| Initial preprocessing improvements         |    ~20,000 | Fixed price and mileage parsing, improved missing value handling, and extracted useful numerical information from messy columns.               |
| High-cardinality feature handling          |    ~14,900 | Adjusted categorical preprocessing after testing showed that naive one-hot encoding of extremely high-cardinality columns hurt generalization. |
| Categorical dtype issue fixed              |    ~12,771 | Fixed a silent preprocessing bug where categorical columns were not being correctly passed through the pipeline.                               |
| Price distribution analysis                |     ~9,400 | Removed a small number of extreme luxury listings that heavily distorted MAE due to the long-tail target distribution.                         |
| Optimized price filtering (`price < 130K`) |     ~6,300 | Found a better cutoff that removed problematic outliers while keeping almost all normal vehicle examples.                                      |
| Final XGBoost tuning                       | **~5,516** | Increased model capacity (`max_depth=4`, `n_estimators=3000`) after the dataset was cleaned, reducing underfitting.                            |

### Final Model Performance

```text
Train MAE:      ~3,142
Validation MAE: ~5,516
```

The final model achieved a validation MAE of approximately **$5.5K**, meaning the average prediction error is around five and a half thousand dollars on unseen vehicle listings.

The biggest improvements did not come from blindly tuning hyperparameters — they came from understanding the dataset:

* Fixing incorrect evaluation caused by log-scale predictions
* Cleaning malformed numerical features
* Extracting useful information from the engine text field
* Removing extreme price outliers
* Increasing model complexity only after reducing dataset noise


---

## 🧠 What Limited Performance Before Optimization

The biggest limitation was not model complexity but dataset quality:

- A small number of extreme luxury cars created a long price tail that heavily affected MAE.
- High-cardinality categorical features had many rare categories, making it difficult to learn reliable patterns.
- Real-world factors such as negotiation, location, seller type, and vehicle condition were not available.

    After handling the target distribution and improving feature extraction, the model reached a validation MAE of around $5.5K.
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
│   └── test.ipynb
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
