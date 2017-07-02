
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingRegressor
from sklearn import preprocessing
import xgboost as xgb
import gc
import re
import numpy as np
from sklearn.externals import joblib
from sklearn.ensemble import  RandomForestRegressor
from lightgbm import *
import numpy as np
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import f_classif
path='/vol6/home/hnu_hcq/xiecheng/'






train_set=pd.read_csv('cache/train_feature.csv')


train_index = train_set[['orderid','roomid']]
train_x = train_set.drop(['orderid',"orderlabel"], axis=1)  # .values
train_y = train_set["orderlabel"]


del train_index,train_set
gc.collect()

shuffle=False
if shuffle:
    id = np.random.permutation(train_y.size)
    train_x = train_x[id]
    train_y = train_y[id]

print(train_x.head())
print(train_y.head())


train_matrix = Dataset(train_x, label=train_y)
del train_x, train_y
gc.collect()


params={'boosting_type':'gbdt',
	    'objective': 'binary',
	    'metric':'auc',
	    'max_depth':8,
	    'num_leaves':80,
	    'lambda_l2':1,
	    'subsample':0.7,
	    'learning_rate': 0.03,
	    'feature_fraction':0.7,
	    'bagging_fraction':0.8,
	    'bagging_freq':5,
	    'num_threads':20
	    }


num_round = 10000


model = train(params, train_matrix, num_round,valid_sets=[train_matrix],early_stopping_rounds=200)  # zlp
model.save_model('lgb_model_0622_all_data_5000.model')

model=Booster(model_file='lgb_model_0622_all_data_5000.model')



df=pd.read_csv('cache/test_feature.csv')
test_x =df.drop(['orderid'], axis=1)
print(test_x.shape)



result=None
t=0
for k in (0.25,0.5,0.75,1):
    k=int(test_x.shape[0]*k)
    print(k)
    sub_test_x = test_x.iloc[t:k, :]
    t=k
    res = model.predict(sub_test_x.values)
    res = pd.DataFrame(res)
    res.columns = ["prob"]
    if result is None:
        result = res
        print(result.head())
    else:
       result = pd.concat([result, res], axis=0)



result["orderid"] = df["orderid"].values
result["predict_roomid"] = df["roomid"].values
result.to_csv("sub/all_result_lgb_alldata_5000.csv", index=None)
result = result.sort_values("prob")
del result["prob"]
result = result.drop_duplicates("orderid", keep="last")
result["orderid"] = result["orderid"].apply(lambda x: "ORDER_" + str(x))
result["predict_roomid"] = result["predict_roomid"].apply(lambda x: "ROOM_" + str(x))
result.to_csv("sub/submit_lgb_all_data_5000.csv", index=None)














