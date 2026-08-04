import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv('data/used_cars.csv')
data['price'] = data['price'].replace('[\\$,]', '', regex=True).astype(float)
data['milage'] = data['milage'].replace('[^0-9]', '', regex=True).astype(float)
data = data[data['price'] < 130000]

data['horsepower'] = data['engine'].str.extract(r'(\d+\.?\d*)\s*HP', expand=False).astype(float)
data['engine_size'] = data['engine'].str.extract(r'(\d+\.\d+)\s*L(?:iter)?', expand=False).astype(float)
data['age'] = data['model_year'].max() - data['model_year']
data = data.drop(columns=['engine'])
data['horsepower_missing'] = data['horsepower'].isna().astype(int)
data['engine_size_missing'] = data['engine_size'].isna().astype(int)


X = data.drop(['price'], axis=1)
y = data.price

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)
y_train_log = np.log1p(y_train)

num_vals = [c for c in X_train.columns if X_train[c].dtype in ['int64', 'float64']]
cat_vals = X_train.select_dtypes(include=['object', 'string']).columns.tolist()



'''transformers for number and categories'''
num_transformer = SimpleImputer(strategy='constant')

cat_transformer = Pipeline(steps=[
    ('Imputer', SimpleImputer(strategy='most_frequent')),
    ('OneHot', OneHotEncoder(handle_unknown='ignore'))
])

preproccessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_vals),
    ('cat', cat_transformer, cat_vals)
])



'''Defining our model
Tune values are obtained by test.
'''
model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=3000,
    learning_rate=0.03,
    max_depth=4,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=5,
    reg_alpha=0,
    random_state=42
)

final = Pipeline(steps=[
    ('preproccessor', preproccessor),
    ('model', model)
])

final.fit(X_train, y_train)
pred = final.predict(X_valid)
#pred = np.expm1(log_pred)

print('Mean Absolute Error: ',mean_absolute_error(y_valid, pred))