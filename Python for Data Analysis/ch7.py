#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 20:16:40 2021

@author: wangziwen
"""
import pandas as pd
#from numpy import nan as NA

"""
read
"""
df = pd.read_csv('http://bit.ly/kaggletrain')
df.info() #Age,Cabin,Embarked na

"""
missing value
"""
#check
df.isnull()

df.Age[df.Age.notnull()] #same as the coding below
df.Age.dropna()
df.dropna(how = 'all') #how='all' means that only delete the row which is na in all col
df.dropna(how = 'all',axis=1) #only delete the column that only includes na
df.dropna(thresh=12) #cannot used for series
#thresh:保留至少有n个非NaN数据的行/列

#fill
df.fillna(0)

df.Age.mean() #29.69911764705882
df.Cabin.value_counts()
df.Embarked.value_counts() #S
df.fillna({'Age':30,'Cabin':'Missing','Embarked':'S'})

df.Age.fillna(df.Age.mean())

"""
Transform
"""
#duplicate
df.duplicated()
df.duplicated().value_counts()

df.drop_duplicates()
df.drop_duplicates(['Pclass','Sex']) #default that deduplicate by first col
df.drop_duplicates(['Pclass','Sex'],keep='last') #change to deduplicate by last col

#function
sex_to_title = {
    'male': 'MR',
    'female': 'MRS'
    }
df['title'] = df.Sex.map(sex_to_title)

#replace
df.Age 
df.Age.replace(22,'A')
df.Age.replace([22,38],'A')
df.Age.replace([22,38],['A','B'])
df.Age.replace({22:'A',38:'B'})

#rename index
df.head(5)
df.head(5).rename(index = {0:'A',
                           1:'B',
                           2:'C',
                           3:'D',
                           4:'E'
                           })

f = lambda x: x.lower()
df.head(5).rename(index = {0:'A',
                           1:'B',
                           2:'C',
                           3:'D',
                           4:'E'
                           }).index.map(f) #error?

#discrete
df.Age.describe()
df['Age_Group'] = pd.cut(df.Age,[0,15,30,45,60,75],labels = ['kids','adults','parents','elder_1','elder_2'])
df.Age_Group
df.loc[:,['Age','Age_Group']]
df.loc[:,['Age','Age_Group']][df.Age_Group=='kids']
df.Age_Group.value_counts()

df['Fare_Group'] = pd.cut(df.Fare,4,precision=2) #十進位到兩位 ＃cut to 5 cat
df.Fare_Group
df['Fare_Group'] = pd.cut(df.Fare,4,precision=2,right = False) #變成左開右閉
df.Fare_Group
df.Fare_Group.value_counts()
df['Fare_Group'] = pd.qcut(df.Fare,4)
df.Fare_Group.value_counts() #qcut can sliced data into same length!
df['Fare_Group'] = pd.qcut(df.Fare,[0,0.1,0.5,0.9,1]) #define percentile by myself 
df.Fare_Group.value_counts()

#filter
df.Age[df.Age>60]
df.select_dtypes(exclude = ['object','category'])[(df.select_dtypes(exclude = ['object','category'])>60).any(1)]

#dummy
df.loc[:5,['Age','Sex']]
pd.get_dummies(df['Sex'])
pd.get_dummies(df['Sex'],prefix = 'key')
df[['Age']].join(pd.get_dummies(df['Sex'],prefix = 'key'))
#有一個比較複雜的例子，之後可以練習！

"""
String
"""
#pd.str.string分析的東西





























