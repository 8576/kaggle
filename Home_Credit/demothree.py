# -*- coding: utf-8 -*-

import xgboost as xgb
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', 103)
basepath = r'D:\kaggle\Home Credit\data'
del_columns = ["FLAG_DOCUMENT_2", "FLAG_DOCUMENT_3", "FLAG_DOCUMENT_4", "FLAG_DOCUMENT_5", "FLAG_DOCUMENT_6", "FLAG_DOCUMENT_7", "FLAG_DOCUMENT_8",
               "FLAG_DOCUMENT_9", "FLAG_DOCUMENT_10", "FLAG_DOCUMENT_11", "FLAG_DOCUMENT_12", "FLAG_DOCUMENT_13", "FLAG_DOCUMENT_14", "FLAG_DOCUMENT_15",
               "FLAG_DOCUMENT_16", "FLAG_DOCUMENT_17", "FLAG_DOCUMENT_18", "FLAG_DOCUMENT_19", "FLAG_DOCUMENT_20", "FLAG_DOCUMENT_21",
               ]

# 训练数据集准备
app = pd.read_csv(os.path.join(basepath,'application_train.csv'), header=0, sep=',')
bure = pd.read_csv(os.path.join(basepath,'bureau.csv'), header=0, sep=',').drop('SK_ID_BUREAU', 1)
bure = bure.drop_duplicates(subset=['SK_ID_CURR'], keep='first')
temp = pd.merge(app, bure, on=['SK_ID_CURR'], how='left')
data = temp.drop(columns=del_columns, axis=1)
data = data.fillna(np.random.rand())
# need_codes = ["NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_TYPE_SUITE",
#               "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_HOUSING_TYPE", "ORGANIZATION_TYPE", "WALLSMATERIAL_MODE",
#               "HOUSETYPE_MODE", "FONDKAPREMONT_MODE", "WEEKDAY_APPR_PROCESS_START", "OCCUPATION_TYPE", "NAME_FAMILY_STATUS"]

# data = data.drop(columns=del_columns, axis=1)

# testdata = pd.read_csv(os.path.join(basepath, 'application_test.csv'), sep=',', header=0)
# testdata = testdata.drop(columns=del_columns, axis=1)
# for item in need_codes:
#     testdata[item] = pd.Categorical(testdata[item]).codes
# testdata = testdata.fillna(1)
# x_test = testdata.iloc[:, 1:-1]
# idnum = testdata.iloc[:, 0]


for col in data.select_dtypes(include=['object']).columns:
    data[col] = pd.Categorical(data[col]).codes
x, y = data.iloc[:, 2:-1], data.iloc[:, 1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# 测试数据集准备
td = pd.read_csv(os.path.join(basepath, 'application_test.csv'), sep=',', header=0)
tempp = pd.merge(td, bure, on=['SK_ID_CURR'], how='left')
tdata = tempp.drop(columns=del_columns, axis=1)
tdata = tdata.fillna(np.random.rand())
for col in tdata.select_dtypes(include=['object']).columns:
    tdata[col] = pd.Categorical(tdata[col]).codes


round = 100
lr = 1
learning_rate = [lr] * int(round * 0.4) + [lr * 0.5] * int(round * 0.3) + [lr * 0.25] * int(round * 0.2) + [lr * 0.1] * int(round * 0.1)
train_data = xgb.DMatrix(x, label=y)
test_data = xgb.DMatrix(x_test, label=y_test)
watch_list = [(train_data, 'train'), (test_data, 'test')]
# params = {'max_depth': 4, 'eta': 1, 'silent': 1,
#           'objective': 'binary:logistic', 'num_class': 2,
#           'booster': 'gblinear'}

params = {'silent': 1,
          'objective': 'reg:logistic',
          'booster': 'gblinear'}
xgbmodel = xgb.train(params=params, dtrain=train_data, num_boost_round=round,
                     evals=watch_list)
# learning_rates=list(np.sort(np.logspace(-2, 0.01, round)))[::-1]













x_test = xgb.DMatrix(tdata.iloc[:, 1:-1])
idnum = tdata.iloc[:, 0]
print(train_data.num_row(), train_data.num_col())
for item in train_data.feature_names:
    print(item)
print('train data'.center(80, '*'))

print(x_test.num_row(), x_test.num_col())
for feature in x_test.feature_names:
    print(feature)
predict = xgbmodel.predict(x_test)
print(type(predict), type(idnum))
print(predict)
# predictfile = pd.DataFrame({'SK_ID_CURR': idnum, 'TARGET': predict})
# predictfile.to_csv(os.path.join(basepath, 'predict.csv'), index=False)