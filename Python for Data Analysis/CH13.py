#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 21:46:47 2021

@author: wangziwen
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import random
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score
import patsy


df = pd.read_csv('http://bit.ly/kaggletrain')
df = pd.read_csv('/Users/wangziwen/Documents/Python/Analysis/titanic/train.csv')
df.info()

df.Sex = df.Sex.astype('category')
dummies = pd.get_dummies(df.Sex,prefix = 'category')
df = df.join(dummies)
df[['Sex','category_female','category_male']]

"""
Patsy
"""
df.corr().Survived
df.corr().Fare
df.corr().Pclass

y,X = patsy.dmatrices('Survived ~ Fare + Age',df)
y
X
np.asarray(y)
np.asarray(X)

y,X = patsy.dmatrices('Survived ~ Fare + Age + 0',df)
X #no intercept

#ols
y,X = patsy.dmatrices('Survived ~ Fare + Age',df)
coef, resid, _, _ = np.linalg.lstsq(X,y)
coef 
coef = pd.Series(coef.squeeze(),index = X.design_info.column_names)
coef 

#data transform in Patsy
data = pd.DataFrame({
    'x0': [1,2,3,4,5],
    'x1': [0.01,-0.01,0.25,-4.1,0.],
    'y': [-1.5,0.,3.6,1.3,-2.]
    })
y,X = patsy.dmatrices('y ~ x0 + np.log(np.abs(x1) + 1)',data)
X

y,X = patsy.dmatrices('y ~ standardize(x0) +  center(x1)',data) #標準化、居中
X

new_data = pd.DataFrame({
    'x0':[6,7,8,9],
    'x1':[3.1,-0.5,0,2.3],
    'y': [1,2,3,4]})
new_X = patsy.build_design_matrices([X.design_info],new_data) #把Ｘ的資訊套到新的data裡面

y,X = patsy.dmatrices('y~I(x0+x1)',data) #variable sum
X

#category and pasty
df.Sex
y,X = patsy.dmatrices('Survived~Sex',df) #str to factor
X

y,X = patsy.dmatrices('Survived~Sex+0',df) #str to dummy
X

y,X = patsy.dmatrices('Survived~C(Pclass)',df) #num as cat
X

y,X = patsy.dmatrices('Survived~C(Pclass)+0',df)
X

df['Pclass'] = df['Pclass'].map({1:'A',
                                 2:'B',
                                 3:'C'   })

y,X = patsy.dmatrices('Survived~Sex+Pclass',df)
X

y,X = patsy.dmatrices('Survived~Sex+Pclass+ Sex:Pclass',df) #交互作用
X

"""
statsmodel
"""

#Estimating Linear Models
def dnorm(mean, variance, size=1): 
    if isinstance(size, int):
        size = size,
    return mean + np.sqrt(variance) * np.random.randn(*size)

# For reproducibility
np.random.seed(12345)
N=100
X = np.c_[dnorm(0, 0.4, size=N),dnorm(0, 0.6, size=N),dnorm(0, 0.2, size=N)] 
eps = dnorm(0, 0.1, size=N)
beta = [0.1, 0.3, 0.5]
y = np.dot(X, beta) + eps
X[:5]

X_model = sm.add_constant(X)
X_model[:5] #add constant(Beta_0)

#OLS
model = sm.OLS(y,X)
results = model.fit()
results
results.params
results.summary()

#rename
data = pd.DataFrame(X, columns=['col0', 'col1', 'col2'])
data['y'] = y
data[:5]
results = smf.ols('y ~ col0 + col1 + col2', data=data).fit()
results.params
results.tvalues
results.summary()

results.predict(data[:5])

#titanic term
X_model = sm.add_constant(df.Pclass)
X_model 
y = df.Fare
model = sm.OLS(y,X) #ValueError: endog and exog matrices are different sizes

results = smf.ols('Fare ~ Pclass', data=df).fit()
results.params
results.tvalues
results.summary()

results.predict(df.Pclass[:5])
df.Fare[:5]

#Estimating Time Series Processes
init_x = 4
values = [init_x, init_x] 
N=1000
b0=0.8
b1 = -0.4
noise = dnorm(0, 0.1, N) 
for i in range(N):
    new_x = values[-1] * b0 + values[-2] * b1 + noise[i] 
    values.append(new_x)

MAXLAGS = 5
model = sm.tsa.AR(values) #AR model
results = model.fit(MAXLAGS)
results.summary()
results.params

"""
scikit-learn
"""
train = pd.read_csv('/Users/didi/Documents/Python/train.csv')
test = pd.read_csv('/Users/didi/Documents/Python/test.csv')

train.isnull().sum() #Age177 Cabin687 Embarked2
test.isnull().sum() #Age86 Fare1 Cabin327

#clean
impute_value = train['Age'].median()
train['Age'] = train['Age'].fillna(impute_value)
test['Age'] = test['Age'].fillna(impute_value)

train['IsFemale'] = (train['Sex'] == 'female').astype(int)
test['IsFemale'] = (test['Sex'] == 'female').astype(int)

#NumPy arrays
train.corr().Pclass
predictors = ['Pclass', 'IsFemale', 'Age'] 
X_train = train[predictors].values
X_test = test[predictors].values
y_train = train['Survived'].values
X_train[:5]
y_train[:5]

#model
model = LogisticRegression()

model.fit(X_train, y_train) #fit with train
y_predict = model.predict(X_test) #predict with test
#If you had the true values for the test dataset
(y_true == y_predict).mean() #error metric

#CV
model_cv = LogisticRegressionCV(10) 
model_cv.fit(X_train, y_train)

model = LogisticRegression(C=10)
scores = cross_val_score(model, X_train, y_train, cv=4) #To do cross-validation by hand
scores





























