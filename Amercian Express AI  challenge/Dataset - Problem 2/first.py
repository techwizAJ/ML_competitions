# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 13:59:59 2018
@author: techwiz
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

header = pd.read_csv('header.csv')
sub = pd.read_csv('sample_submission.csv')

test_ = test.iloc[:,:]
X = train.iloc[:,:-1]
y = train.iloc[:,55]

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x_scaled = ss.fit_transform(X)
test_scaled = ss.transform(test_)
plt.matshow(train.corr())
plt.show()
sns.heatmap(train.corr(),annot=True,fmt='d')

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.25,random_state=1)
X_tr_scale , X_te_scale , y_tr_scale , y_te_scale = train_test_split(x_scaled,y,test_size=0.25,random_state=42)

from sklearn.ensemble import RandomForestClassifier
import xgboost as xg

clf = xg.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=0.5,
       colsample_bytree=1, gamma=0.1, learning_rate=0.09, max_delta_step=4,
       max_depth=1024, min_child_weight=0.5, missing=None, n_estimators=1000,
       n_jobs=-1, nthread=-1, objective='binary:logistic',
       random_state=42, reg_alpha=0.1, reg_lambda=0.7,
       scale_pos_weight=0.5, seed=None, silent=False, subsample=0.9,verbose=10)
clf.fit(X_train , y_train)
y_pred = clf.predict(X_test)

clf1 = xg.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=0.5,
       colsample_bytree=1, gamma=0.1, learning_rate=0.09, max_delta_step=4,
       max_depth=640, min_child_weight=0.5, missing=None, n_estimators=800,
       n_jobs=-1, nthread=-1, objective='binary:logistic',
       random_state=42, reg_alpha=0.1, reg_lambda=0.7,
       scale_pos_weight=0.5, seed=None, silent=True, subsample=0.9,verbose=10)
clf1.fit(X_train , y_train)
y_pred1 = clf1.predict(X_test)


clf2 =  xg.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=0.5,
       colsample_bytree=1, gamma=0.1, learning_rate=0.09, max_delta_step=4,
       max_depth=320, min_child_weight=0.5, missing=None, n_estimators=400,
       n_jobs=-1, nthread=-1, objective='binary:logistic',
       random_state=42, reg_alpha=0.1, reg_lambda=0.7,
       scale_pos_weight=0.5, seed=None, silent=True, subsample=0.9,verbose=10)
clf2.fit(X_train,y_train)
y_pred2 = clf2.predict(X_test)

from sklearn.metrics import confusion_matrix
cf = confusion_matrix(y_test,y_pred)
cf1 = confusion_matrix(y_test,y_pred1)
cf2 = confusion_matrix(y_test,y_pred2)

clf_ = xg.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=0.5,
       colsample_bytree=1, gamma=0.1, learning_rate=0.09, max_delta_step=4,
       max_depth=320, min_child_weight=0.5, missing=None, n_estimators=400,
       n_jobs=-1, nthread=None, objective='binary:logistic',
       random_state=42, reg_alpha=0.1, reg_lambda=0.7,
       scale_pos_weight=0.5, seed=None, silent=True, subsample=0.9)
clf_.fit(X,y)

pred = clf_.predict(test_)
sub1 = test[['0','1']]
sub1['1'] = pred
sub1.rename(index=str,columns={'0':'key','1':'score'},inplace=True)
sub1.to_csv('submission.csv',index=False)

from sklearn.externals import joblib
joblib.dump(clf_,'clf_.sav')


#2nd try with PCA and feature Engineering
from sklearn.decomposition import PCA
pca = PCA(n_components = 8)
pca_components = pca.fit_transform(x_scaled)
df_pca = pd.DataFrame(pca_components )
pca.explained_variance_ratio_
pca.explained_variance_

import xgboost as xg
clf_pca = xg.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=0.5,
       colsample_bytree=1, gamma=0.1, learning_rate=0.09, max_delta_step=4,
       max_depth=320, min_child_weight=0.5, missing=None, n_estimators=400,
       n_jobs=-1, nthread=None, objective='binary:logistic',
       random_state=42, reg_alpha=0.1, reg_lambda=0.7,
       scale_pos_weight=0.5, seed=None, silent=True, subsample=0.9)
clf_pca.fit(X_tr_scale,y_tr_scale)

y_pred_pca = clf_pca.predict(X_te_scale)
from sklearn.metrics import confusion_matrix
cf_pca = confusion_matrix(y_te_scale,y_pred_pca)