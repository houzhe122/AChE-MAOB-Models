import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
import warnings
import os
os.chdir()
warnings.filterwarnings("ignore")

evaluation_results = pd.DataFrame(columns=['MSE', 'MAE', 'R2', 'Best Params'])

# 1) read data
file_path = "RegressionData_MAO-B_RFE.csv"
df = pd.read_csv(file_path, index_col = 'ID')

X = df.drop(columns = ['Y'])
y = df['Y']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# GBDT model and hyperparameter optimization
model = GradientBoostingRegressor()
param_dist = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'max_depth': [3, 5, 7, 9],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 0.9, 1.0]
    }
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=50, cv=3,
                                       scoring='neg_mean_squared_error', random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)
best_gbdt_model = random_search.best_estimator_
best_params = random_search.best_params_

# predict
y_pred = best_gbdt_model.predict(X_test)

# calculate metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# save result
evaluation_results = evaluation_results._append({
    'MSE': mse,
    'MAE': mae,
    'R2': r2,
    'Best Params': best_params
    }, ignore_index=True)

print(evaluation_results)
evaluation_results.to_csv('GBDTresults_rfe.csv', index=False)
