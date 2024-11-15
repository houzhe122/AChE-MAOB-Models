import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
import os
os.chdir()
from sklearn.preprocessing import StandardScaler
from math import sqrt
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import train_test_split

def show_metrics(y_true, y_pred):
    R2 = r2_score(y_true, y_pred)
    MAE = mean_absolute_error(y_true, y_pred)
    RMSE = sqrt(mean_squared_error(y_true, y_pred))
    return [R2, MAE, RMSE]


def cal_h(tr_x, X):
    tr_arr = tr_x.values
    XTX_inv = np.linalg.pinv(np.dot(tr_arr.T, tr_arr))  
    X_arr = X.values
    h = np.einsum('ij,jk,ik->i', X_arr, XTX_inv, X_arr) 
    return h


def cal_performance_AD(df, h_):
    in_AD = (df['h'] <= h_) & (-3 <= (df['y_true'] - df['y_pred'])) & ((df['y_true'] - df['y_pred']) <= 3)
    coverage = in_AD.mean()  # 使用平均值计算覆盖率
    metrics = show_metrics(df['y_true'][in_AD], df['y_pred'][in_AD])
    metrics.append(coverage)
    return metrics

def cal_williams(tr_x, tr_y, tr_pred, te_x, te_y, te_pred, williams_path):
 
    tr_h = cal_h(tr_x, tr_x)
    h_ = 3 * (len(tr_x.columns) + 1) / len(tr_x)
    te_h = cal_h(tr_x, te_x)


    df_tr = pd.DataFrame({'h': tr_h, 'y_true': tr_y, 'y_pred': tr_pred})
    df_te = pd.DataFrame({'h': te_h, 'y_true': te_y, 'y_pred': te_pred})


    results_tr = cal_performance_AD(df_tr, h_)
    results_te = cal_performance_AD(df_te, h_)


    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 16
    plt.xlim([-0.2, 2])
    plt.ylim([-4, 4])
    plt.plot(tr_h, (df_tr['y_true'] - df_tr['y_pred']), 'bo', markerfacecolor='none', label='training set', alpha=0.5, markersize=5)
    plt.plot(te_h, (df_te['y_true'] - df_te['y_pred']), 'rx', label='test set', alpha=0.5, markersize=5)
    plt.axvline(h_, color='black', linestyle="--", lw=1)
    plt.axhline(-3, color='black', linestyle="--", lw=1)
    plt.axhline(3, color='black', linestyle="--", lw=1)
    plt.legend(loc='best')
    plt.xlabel("hi")
    plt.ylabel("Standardized residual")
    plt.savefig(williams_path, dpi=600, bbox_inches="tight")


    result_dict = {
        'h_': h_,
        'tr_coverage': results_tr[-1], 'tr_R2': results_tr[0], 'tr_MAE': results_tr[1], 'tr_RMSE': results_tr[2],
        'te_coverage': results_te[-1], 'te_R2': results_te[0], 'te_MAE': results_te[1], 'te_RMSE': results_te[2]
    }
    return result_dict


file_path = "RegressionData_ACHE_RFE.csv"
df = pd.read_csv(file_path, index_col = 'ID')

X = df.drop(columns = ['Y'])
y = df['Y']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_df = pd.DataFrame(X_train, columns=df.drop(columns=['Y']).columns)
X_test_df = pd.DataFrame(X_test, columns=df.drop(columns=['Y']).columns)

Best_para = {'subsample': 0.9, 'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_depth': 7, 'learning_rate': 0.1}

#svr = SVR(**Best_para)
#svr.fit(X_train, y_train)
#gbdt = GradientBoostingRegressor(**Best_para)
#gbdt.fit(X_train, y_train)LGBMRegressor(**Best_para)
best_model = GradientBoostingRegressor(**Best_para)
best_model.fit(X_train, y_train)

y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

williams_path = 'williams_plot.png'

result_dict = cal_williams(X_train_df, y_train, y_train_pred, X_test_df, y_test, y_test_pred, williams_path)

print(result_dict)