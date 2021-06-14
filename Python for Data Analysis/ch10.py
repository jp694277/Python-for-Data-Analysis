#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 11:40:40 2021

@author: wangziwen
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm

"""
Data Aggregation and Group Operations
"""

"""
GroupBy Mechanics
"""
df = pd.read_csv('http://bit.ly/kaggletrain')
df = pd.read_csv('/Users/wangziwen/Documents/Python/Analysis/titanic/train.csv')
df.info()
grouped = df['Fare'].groupby(df['Sex'])
grouped #no vaulue
grouped.mean()

means = df['Fare'].groupby([df['Sex'], df['Parch']]).mean()
means
means.unstack()
df.groupby(['Sex','Parch'])['Fare'].mean()
df.groupby(['Sex','Parch']).Fare.mean()

df[['Fare','Age']].groupby([df['Sex'], df['Parch']]).mean()
df[['Fare','Age']].groupby([df['Sex'], df['Parch']]).mean().unstack()
df[['Fare','Age']].groupby([df['Sex'], df['Parch']]).mean().unstack()
df.groupby(['Sex', 'Parch']).size() #計數包含Nan

#Iterating Over Groups
for name, group in df.groupby('Sex'):  
    print(name)
    print(group)
#name = key; group = value

for name, group in df.groupby('Parch'):  
    print(name)
    print(group)

for (k1,k2), group in df.groupby(['Sex','Parch']):  
    print((k1,k2))
    print(group)

#dict
dict(list(df.groupby('Sex')))
dict(list(df.groupby('Sex')))['female']

#group by types
df.dtypes
grouped = df.groupby(df.dtypes, axis = 1)
for dtypes, group in grouped:
    print(dtypes)
    print(group)
    
# choose subsets by col
df.groupby('Sex')['Fare'].mean()
df.groupby('Sex')[['Fare']].mean()

df['Fare'].groupby(df['Sex']).mean()
df[['Fare']].groupby(df['Sex']).mean()

df.groupby(['Sex','Parch'])['Fare'].mean()
df.groupby(['Sex','Parch'])[['Fare']].mean()

# group by dict and series
people = pd.DataFrame(np.random.randn(5,5),
                      columns = ['a','b','c','d','e'],
                      index = ['Joe','Steve','Wes','Jim','Travis'])
people.iloc[2:3,[1,2]] = np.nan 
people

mapping = {'a':'red','b':'red','c':'blue',
            'd':'blue','e':'red','f':'orange'
    }
people.groupby(mapping,axis = 1).sum()

map_series = pd.Series(mapping)
people.groupby(map_series,axis = 1).sum()

# group by function
people.groupby(len).sum() #len(index)
keylist = ['one','one','one','two','two']
people.groupby([len,keylist]).sum()

# group by index
columns = pd.MultiIndex.from_arrays([['US','US','US','JP','JP'],
                                     [1,3,5,1,3]],
                                    names = ['cty','tenor'])
hier_df = pd.DataFrame(np.random.randn(4,5), columns = columns)
hier_df.groupby(level = 'cty', axis = 1).count()

"""
Data Aggregation
"""
df.groupby('Sex')['Fare'].quantile(0.9)
df.groupby('Sex').describe()
df.groupby('Sex')['Fare'].describe()
df.groupby('Sex')['Fare'].describe().unstack()

def peak_to_peak(arr):
    return arr.max() - arr.min()
df.groupby('Sex')[['Age','Fare']].agg(peak_to_peak)

df.groupby('Sex')['Fare'].agg('mean')
df.groupby('Sex')['Fare'].agg(['mean','std',peak_to_peak])
df.groupby('Sex')['Fare'].agg([('Exp','mean'),('Sd','std'),('R',peak_to_peak)]) #rename

df.groupby(['Sex','Parch'])['Fare'].agg(['count','mean','max'])
df.groupby('Sex')[['Fare','Age']].agg(['count','mean','max'])
df.groupby('Sex')[['Fare','Age']].agg(['count','mean','max'])['Fare']
df.groupby('Sex',as_index=False)[['Fare','Age']].agg(['count','mean','max'])['Fare']

"""
Apply
"""
def top(df,n=5,column = 'Fare'):
    return df.sort_values(by = column)[-n:]

top(df,n=6)
df.groupby('Sex').apply(top)
df.groupby('Sex').apply(top)['Fare']
df.groupby(['Sex','Parch']).apply(top,n=1,column = 'Age')['Age']

#group key
df.groupby('Sex').apply(top)['Fare']
df.groupby('Sex',group_keys=False).apply(top)['Fare']

#cut and quantile
pd.cut(df.Age,4)

def get_stats(group):
    return {'min':group.min(), 'max': group.max(),
            'count':group.count(),'mean':group.mean()
        }
df.Fare.groupby(pd.cut(df.Age,4)).apply(get_stats)
df.Fare.groupby(pd.cut(df.Age,4)).apply(get_stats).unstack()

pd.qcut(df.Age,4)
pd.qcut(df.Age,4,labels = False)
df.Fare.groupby(pd.qcut(df.Age,10,labels = False)).apply(get_stats) #quantile
df.Fare.groupby(pd.qcut(df.Age,10,labels = False)).apply(get_stats).unstack()

#fill in NA
df.info()
df.Age.fillna(df.Age.mean())

states = ['Ohio','New York','Vermont','Florida','Oregon','Nevada','California','Idaho']
group_key = ['East'] * 4 + ['West'] * 4
data = pd.Series(np.random.randn(8),index = states)
data[['Vermont','Nevada','Idaho']] = np.nan
data.groupby(group_key).mean()
fill_mean = lambda g: g.fillna(g.mean())
data.groupby(group_key).apply(fill_mean)

fill_values = {'East':0.5,'West':-1}
fill_func = lambda g:g.fillna(fill_values[g.name])
data.groupby(group_key).apply(fill_func)

#random sample and sort
suits = ['H', 'S', 'C', 'D']
card_val = (list(range(1, 11)) + [10] * 3) * 4
base_names = ['A'] + list(range(2, 11)) + ['J', 'K', 'Q']
cards = []
for suit in ['H', 'S', 'C', 'D']:
    cards.extend(str(num) + suit for num in base_names)
    
deck = pd.Series(card_val, index=cards)
def draw(deck, n=5):
    return deck.sample(n)
draw(deck)  #random sample

get_suit = lambda card: card[-1] # last letter is suit
deck.groupby(get_suit).apply(draw, n=2) #random sample by group

deck.groupby(get_suit, group_keys=False).apply(draw, n=2)

#weighted avg and corr
#wavg
get_wavg = lambda g:np.average(g['Fare'],weights = g['Age'])
df_c = df.copy()
df_c.Age = df_c.Age.fillna(df.Age.mean())
df_c.groupby('Sex').apply(get_wavg)

#corr
Fare_corr = lambda x:x.corrwith(x['Fare'])
Fare_corr(df)
df.groupby('Sex').apply(Fare_corr)

#LR by group
def regress(data,xvar,yvar):
    Y = data[yvar]
    X = data[xvar]
    X['intercept'] = 1.
    result = sm.OLS(Y,X).fit()
    return result.params

df.groupby('Sex').apply(regress,['Pclass'],'Fare')

"""
PV Table and Crosstab 
"""  
#pv  
df.pivot_table(index=['Sex']) #avg
df.pivot_table(index=['Sex','Pclass'])
df.pivot_table(index=['Sex','Pclass'],columns = 'Parch')

df.pivot_table(index=['Sex','Pclass'],margins=True) #all
df.pivot_table(['Fare'],index=['Sex','Pclass'],columns = 'Parch',margins=True)

df.pivot_table(index=['Sex','Pclass'],margins=True,aggfunc=len) #aggfunc=len -> freq
df.pivot_table(['Fare'],index=['Sex','Pclass'],columns = 'Parch',margins=True,aggfunc=len)

df.pivot_table(index=['Sex','Pclass'],margins=True,aggfunc='mean')
df.pivot_table(index=['Sex','Pclass'],margins=True,aggfunc='mean',fill_value=0) #fillna

#crosstab
pd.crosstab(df.Sex,df.Pclass,margins = True)
pd.crosstab([df.Sex,df.Parch],df.Pclass,margins = True)


