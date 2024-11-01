import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score
import warnings
from imblearn.combine import SMOTETomek
import os
os.chdir()
warnings.filterwarnings("ignore")


evaluation_results = pd.DataFrame(
    columns=['Fingerprint', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC', 'MCC', 'Best Params'])

# 1) read data
file_path = "cla_data_rfe.xlsx"
df = pd.read_excel(file_path, sheet_name=None)

# hyperparameter grid
param_grid = {
    'num_leaves': [10,15,20,25,30,35],
    'learning_rate': [0.1, 0.01, 0.001],
    'n_estimators': [120,200,300,500]
}

for fp in df:
    X = df[fp].iloc[:, 2:]
    y = df[fp]['Y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    #  SMOTETomek
    smt = SMOTETomek(random_state=42)
    X_train_resampled, y_train_resampled = smt.fit_resample(X_train, y_train)
    lgbm_model = LGBMClassifier(class_weight='balanced')

    # hyperparameter research
    grid_search = GridSearchCV(estimator=lgbm_model, param_grid=param_grid,scoring='f1', refit='mcc', cv=5, n_jobs=-1)
    grid_search.fit(X_train_resampled, y_train_resampled)

    best_lgbm_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    best_lgbm_model.fit(X_train, y_train)
    y_pred = best_lgbm_model.predict(X_test)
    y_pred_prob = best_lgbm_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_prob)

    evaluation_results = pd.concat([evaluation_results, pd.DataFrame({
        'Fingerprint': [fp],
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1 Score': [f1],
        'AUC-ROC': [auc_roc],
        'MCC': [mcc],
        'Best Params': [best_params]
    })], ignore_index=True)

print(evaluation_results)
evaluation_results.to_csv('LGBM_results.csv', index=False)