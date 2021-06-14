#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 13:48:09 2021

@author: wangziwen
"""

import pandas as pd
import numpy as np

df = pd.read_csv('http://bit.ly/kaggletrain')
df.info()

"""
Category
"""
df.Sex.unique()
df.Sex.value_counts()

#num to str
values = pd.Series([0,1,0,0]*2)
dim = pd.Series(['apple','orange'])
dim.take(values)

df.Pclass.unique()
dim = pd.Series(['N','A','B','C'])
dim.take(df.Pclass)

#type of category
df.Sex.astype('category')
df.Sex.astype('category').values
type(df.Sex.astype('category').values)
df.Sex.astype('category').values.categories 
df.Sex.astype('category').values.codes 

df.Sex = df.Sex.astype('category')
df.info()
df['Sex_C'] = df.Sex.values.codes
df[['Sex','Sex_C']] #male 1, female 0

#agg to category 
pd.qcut(df.Age,4) #四分位
pd.qcut(df.Age,4,labels = ['A1','A2','A3','A4']) 
df.Fare.groupby(pd.qcut(df.Age,4,labels = ['A1','A2','A3','A4'])).agg(['count','min','max'])
df.Fare.groupby(pd.qcut(df.Age,4,labels = ['A1','A2','A3','A4'])).agg(['count','min','max']).reset_index()

#category method
df.Sex
df.Sex.cat.codes #cat
df.Sex.cat.categories
df.Sex.cat.set_categories(['female','male','kid'])

df.Sex.value_counts()
df.Sex.cat.set_categories(['female','male','kid']).value_counts()

df.Sex[df.Sex.isin(['female','kid'])]
df[df.Sex.isin(['female','kid'])]

df.Sex[df.Sex.isin(['female'])].cat.remove_unused_categories()

#dummy
pd.get_dummies(df.Sex)

"""
Apply High Level Groupby
"""
df[['Sex','Age']]
df.Age.groupby(df.Sex).mean()

df.Age.groupby(df.Sex).transform(lambda x: x.mean())
df.Age.groupby(df.Sex).transform('mean')

df.Age.groupby(df.Sex).transform(lambda x: x*2)
df.Age.groupby(df.Sex).transform(lambda x: x.rank(ascending = False)) #rank by group

def normalize(x):
    return(x - x.mean()) / x.std()

df.Age.groupby(df.Sex).transform(normalize)
df.Age.groupby(df.Sex).apply(normalize) #same here
(df.Age - df.Age.groupby(df.Sex).transform('mean'))  / df.Age.groupby(df.Sex).transform('std') 

#time resampling by group
N = 15
times = pd.date_range('2021-03-01 00:00', freq='1min', periods=N)
df = pd.DataFrame({'time': times,'value': np.arange(N)})
df

df.set_index('time').resample('5min').count()

df2 = pd.DataFrame({'time': times.repeat(3),
                    'key': np.tile(['a', 'b', 'c'], N),
                    'value': np.arange(N * 3.)})
df2[:7]

time_key = pd.Grouper(freq = '5min')
resampled = (df2.set_index('time').groupby(['key', time_key]).sum())
resampled
resampled.reset_index()



















