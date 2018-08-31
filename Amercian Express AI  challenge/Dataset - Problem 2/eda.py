# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 20:09:20 2018

@author: techwiz
"""

import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train.drop('55',axis=1,inplace=True)

df = pd.concat([train,test])

x_test = test.iloc[:,:]
X = train.iloc[:,:-1]
y = train.iloc[:,55]

from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures
def add_interactions(df):
    combos = list(combinations(list(df.columns),2))
    colnames = list(df.columns)+['_'.join(x) for x in combos]
    
    poly  = PolynomialFeatures(interaction_only=True , include_bias = False)
    df = poly.fit_transform(df)
    df = pd.DataFrame(df)
    df.columns = colnames
    
    noint_indicies = [i for i,x in enumerate(list((df==0).all())) if x]
    df = df.drop(df.columns[noint_indicies],axis=1)
    return df

df_new = add_interactions(train)

import sklearn.feature_selection as fs
select = fs.SelectKBest(score_func= fs.f_classif ,k=8)
selected_features = select.fit(X,y)
indices_selected = selected_features.get_support(indices=True)
colnames_selected = [X.columns[i] for i in indices_selected]

X_train_selected = X[colnames_selected]
X_test_selected = X[colnames_selected]