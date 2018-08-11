import pyhanlp

# # Config = pyhanlp.JClass("com.hankcs.hanlp.HanLP$Config")
# # Config.ShowTermNature = False
# CRFSegment = pyhanlp.JClass("com.hankcs.hanlp.seg.CRF.CRFSegment")
# segment = CRFSegment().enableCustomDictionary(False)
#
# print(segment.seg('小荷才露尖尖角'))

import pandas as pd
import xgboost as xgb
data = pd.DataFrame({'a': [23, 34], 'b':[12, 765]})
other = pd.DataFrame({'a': [23, 'a', 'ee'], 'b':[765, 'qw', 'rt']})
td = pd.merge(data, other, on=['a'],)
print(td)


# print(data.select_dtypes('object').columns)
# d = xgb.DMatrix(data)
# print(d.feature_names)
# print(d.feature_types)
# print(d.handle)
# print(d.num_col(), d.num_row())
#
# from sklearn.ensemble import AdaBoostClassifier

# AdaBoostClassifier()
