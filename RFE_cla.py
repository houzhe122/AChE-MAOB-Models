import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import warnings
import os
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

warnings.filterwarnings("ignore")
os.chdir()

# build a new workbook
output_path = "cla_data.xlsx"
wb = Workbook()

# path
input_path = ""
# import data
xls = pd.ExcelFile(input_path)
sheet_names = xls.sheet_names

# iterate every sheet
for sheet in sheet_names:
    # read current sheet
    df = pd.read_excel(input_path, sheet_name=sheet, index_col='Smiles')

    X = df.drop(columns=['Y'])
    y = df['Y']

    # select features with classification model
    classifier = RandomForestClassifier(class_weight='balanced')
    rfe = RFE(estimator=classifier, n_features_to_select=50, step=1)
    rfe.fit(X, y)

    # get selected features
    feature_mask = rfe.support_
    X_selected = X.loc[:, feature_mask]
    X_selected.index = df.index
    X_selected.insert(0, "Y", y)

    # save selected features to different sheet
    ws = wb.create_sheet(title=sheet)
    for r in dataframe_to_rows(X_selected, index=True, header=True):
        ws.append(r)

del wb['Sheet']

wb.save(output_path)
print(f"feature selection completed，results saved to {output_path}。")

