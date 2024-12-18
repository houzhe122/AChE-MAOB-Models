import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score
import warnings
import os
from imblearn.combine import SMOTETomek

os.chdir()
warnings.filterwarnings("ignore")

evaluation_results = pd.DataFrame(
    columns=['Fingerprint', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC', 'MCC', 'Best Params'])

# 1) read data
file_path = "cla_data_rfe.xlsx"
df = pd.read_excel(file_path, sheet_name=None)

for fp in df:
    X = df[fp].iloc[:, 2:]
    y = df[fp]['Y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # balance data
    smt = SMOTETomek(random_state=42)
    X_train_resampled, y_train_resampled = smt.fit_resample(X_train, y_train)

    # KNN model and optimization
    knn = KNeighborsClassifier()
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='f1')
    grid_search.fit(X_train_resampled, y_train_resampled)  # 使用平衡后的数据进行训练
    best_knn_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # predict
    y_pred = best_knn_model.predict(X_test)
    y_pred_prob = best_knn_model.predict_proba(X_test)[:, 1]

    # calculate metric
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_prob)

    # save result
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

# print result
print(evaluation_results)
evaluation_results.to_csv('KNN_results.csv', index=False)
