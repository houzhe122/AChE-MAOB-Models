import pandas as pd
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings("ignore")
import os
os.chdir()

# 1) import data
df_path = "RegressionData_AChE.csv"
df = pd.read_csv(df_path, index_col = 'ID')
X = df.drop(columns = ['Y','Smiles'])
y = df['Y']
X_original = X.copy()

# remove constant features
constant_features = [feat for feat in X.columns if X[feat].std() == 0]
X.drop(labels=constant_features, axis=1, inplace=True)

# remove near constant features
sel = VarianceThreshold(threshold=0.01)
sel.fit(X)
features_to_keep = X.columns[sel.get_support()]
X = sel.transform(X)
X = pd.DataFrame(X)

X.columns = features_to_keep

# Instantiate RFECV visualizer with a random forest regressor
rfe = RFE(RandomForestRegressor(),n_features_to_select= 50, step=1)
rfe.fit(X, y) # Fit the data to the visualizer

feature_mask = rfe.support_

X_selected = X.loc[:, feature_mask]
X_selected.index = X_original.index
X_selected.insert(0,"Y",y)
print(X_selected)
# save selected features
X_selected.to_csv('RegressionData_AChE_RFE50.csv', index=True)