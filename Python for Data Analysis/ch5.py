# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
"""
read
"""
df = pd.read_csv('http://bit.ly/kaggletrain')
df.info() #891*12

"""
5.1 DF
"""
df.head()
df_c = df.head(5)
col = list(df.columns)
pd.DataFrame(df_c, columns = sorted(col))
df_c['Age']
df_c.Survived
df_c['times'] = '30'
df_c.T
df_c.values

"""
5.2 Basic
"""
df_c.drop(['times'],axis =1)
df_c.drop(['times'],axis = 'columns')
df_c.drop([0],axis = 0)

#select columns
df_c['Age'] 
df_c[['Age','times']] #multiple

#slice
df_c[:2]
df_c ==1

df_c = df.copy()
df_c.Fare[df_c.Fare>30] = 'top'
df_c.Fare 

#loc&iloc
df_c.loc[0,['PassengerId','Survived']]
df_c.loc[0:3,['PassengerId','Survived']]
df_c.loc[0:3,'PassengerId':'Pclass'] #use : need to drop[]
df_c.iloc[0,[0,1]]
df_c.iloc[0:3,[0,1]]
df_c.iloc[0:3,0:3]
df_c.iloc[0:3,0:3][df_c.PassengerId > 1]

# cul
df_c.Fare + df_c.PassengerId
df_c.Fare - df_c.PassengerId
df_c.Fare.add(df_c.PassengerId)
1/df_c.Fare
df_c.Fare.rdiv(1)
np.abs(df_c.PassengerId - df_c.Fare)

f = lambda x: x.max() - x.min()
df_c.loc[:,['Fare','PassengerId']].apply(f)
f = lambda x: '%.2f' % x
df_c.Fare.rdiv(1).map(f)

df_c.Fare.sum()
df_c.loc[:,['Fare','Age']].sum() 
df_c.loc[:,['Fare','Age']].sum(axis = 'columns') # also can use +
df_c.loc[:,['Fare','Age']].mean(axis = 'columns') # skipna = False
df_c.Fare.idxmax() # max value by index
df_c.Fare.idxmin()
df_c.Fare.iloc[[258,179]]
df_c.select_dtypes(exclude = 'object').describe()
df_c.Cabin.unique()
df_c.Sex.value_counts(sort = True)
df_c.Sex.isin(['male','man'])
# rank
df_c.sort_index(axis=0)
df_c.sort_index(axis=1)
df_c.sort_index(axis=1, ascending = False)
df_c.sort_values (by = 'Fare')
df_c.sort_values (by = ['Fare','PassengerId'])
df_c.rank()
df_c.rank(method = 'max')
df_c.rank(method = 'first') # first not supported for non-numeric data
df_c.select_dtypes(exclude = 'object').rank(method = 'first')
df_c.rank(axis = 'columns')

# corr
df_c['Age'].corr(df_c['Fare'])
df_c.select_dtypes(exclude='object').corr()
df_c.select_dtypes(exclude='object').cov()
df_c.select_dtypes(exclude='object').corrwith(df_c.Age)




