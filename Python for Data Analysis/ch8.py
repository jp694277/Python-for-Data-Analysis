#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 15:24:08 2021

@author: wangziwen
"""
import pandas as pd
import numpy as np

"""
read
"""
df = pd.read_csv('http://bit.ly/kaggletrain')
df.info() #Age,Cabin,Embarked na

"""
layer
"""
#multiple index
data = pd.Series(np.random.randn(9),
                 index = [['a','a','a','b','b','c','c','d','d']
                     ,[1,2,3,1,3,1,2,2,3]])
data.index
data['b']
data['b'][1]
data[['b','d']]
data.unstack()
data.unstack().stack()

#multiple index x multiple columns
data = pd.DataFrame(np.arange(12).reshape((4,3)),
                 index = [['a','a','b','b'],[1,2,1,2]],
                 columns = [['Ohio','Ohio','Colorado'],
                            ['Green','Red','Green']]
    )
data['Ohio']
data

#sort
data.index.names = ['key1','key2']
data.columns.names = ['state','color']
data
data.swaplevel('key1','key2') #接收兩個層級
data.sort_index(level = 1)
data.swaplevel(0,1).sort_index(level = 0)

#cul
data.sum(level = 'key2')
data.sum(level = 'key1')
data.sum(level = 'color',axis = 1)

#index
data = pd.DataFrame(
    {'a':range(7),
     'b':range(7,0,-1),
     'c':['one','one','one','two','two','two','two'],
     'd':[0,1,2,0,1,2,3]
     }
    )
data
data2 = data.set_index(['c','d'])
data2
data.set_index(['c','d'],drop = False)
data2.reset_index()
del data2

"""
merge&join
"""
df1 = pd.DataFrame({
    'key': ['b','b','a','c','a','a','b'],
    'data1':range(7)
    }
    )

df2 = pd.DataFrame({
    'key':['a','b','d'],
    'data2':range(3)
    }
    )
df1 #2*7
df2 #2*3

#merge
pd.merge(df1,df2) #因為沒有指定key，所以會自動把重疊的當成key
pd.merge(df1,df2,on = 'key')
pd.merge(df2,df1,on = 'key')

df3 = pd.DataFrame({
    'lkey': ['b','b','a','c','a','a','b'],
    'data1':range(7)
    }
    )

df4 = pd.DataFrame({
    'rkey':['a','b','d'],
    'data2':range(3)
    }
    )

pd.merge(df3,df4,left_on = 'lkey', right_on='rkey')

pd.merge(df1,df2,how = 'outer')

df1 = pd.DataFrame({
    'key': ['b','b','a','c','a','b'],
    'data1':range(6)
    }
    )

df2 = pd.DataFrame({
    'key':['a','b','a','b','d'],
    'data2':range(5)
    }
    )
pd.merge(df1,df2,on = 'key', how = 'left')

left = pd.DataFrame(
    {
    'key1': ['foo','foo','bar'],
    'key2':['one','two','one'],
    'lval':[1,2,3]
    }
    )

right = pd.DataFrame(
    {
    'key1': ['foo','foo','bar','bar'],
    'key2':['one','one','one','two'],
    'lval':[4,5,6,7]
    }
    )
pd.merge(left,right,on=['key1','key2'],how='outer')
pd.merge(left,right,on = 'key1')

"""
after work
"""
left = pd.DataFrame(
    {'key1':['foo','foo','bar'],
     'key2':['one','two','one'],
     'lval':[1,2,3]
     })

right = pd.DataFrame(
     {'key1':['foo','foo','bar','bar'],
     'key2':['one','two','one','two'],
     'lval':[4,5,6,7]
     }
    )

pd.merge(left, right,on=['key1','key2'],how = 'outer')
pd.merge(left, right,on = 'key1')
pd.merge(left,right,on = 'key1',suffixes = ('_left','_right'))

#merge on index
left = pd.DataFrame(
    {'key':['a','b','a','a','b','c'],
     'value': range(6)
     })

right = pd.DataFrame(
     {'group_val':[3.5,7],
     },index = ['a','b']
    )
pd.merge(left, right,left_on = 'key',right_index=True)
pd.merge(left, right,left_on = 'key',right_index=True,how = 'outer')

#multiple index
left = pd.DataFrame(
    {'key1':['Ohio','Ohio','Ohio','Nevada','Neveda'],
     'key2':[2000,2001,2002,2001,2002],
     'data': np.arange(5.)
     })

right = pd.DataFrame(
     np.arange(12).reshape((6,2)),
     index = [['Nevada','Nevada','Ohio','Ohio','Ohio','Ohio'],
                [2001,2000,2000,2000,2001,2002]],
       columns = ['event1','event2']
    )
pd.merge(left, right,left_on = ['key1','key2'],right_index=True)
pd.merge(left, right,left_on = ['key1','key2'],right_index=True,how='outer')

left = pd.DataFrame(
    [[1.,2.],[3.,4.],[5.,6.]],
     index = ['a','b','c'],
     columns=['Ohio','Nevada']
     )

right = pd.DataFrame(
     [[7.,8.],[9.,10.],[11.,12.],[13.,14.]],
     index = ['b','c','d','e'],
     columns=['Missourt','Alabama']
    )



dt = pd.DataFrame(
   [[7.,8.],[9.,10.],[11.,12.],[16.,17.]],
   index = ['a','c','e','f'],
   columns = ['New York','Oregon']
    )
left.join([right,another])
left.join([right,another],how = 'outer')

#merge by columns
arr = np.arange(12).reshape((3,4))
np.concatenate([arr,arr],axis = 1)

s1 = pd.Series([0,1],index=['a','b'])
s2 = pd.Series([2,3,4], index = ['c','d','e'])
s3 = pd.Series([5,6], index=['f','g'])
pd.concat([s1,s2,s3])
pd.concat([s1,s2,s3],axis = 1)
s4 = pd.concat([s1,s3])
pd.concat([s1,s4],axis = 1)
pd.concat([s1,s4],axis = 1,join = 'inner')

results = pd.concat([s1,s1,s3],keys = ['one','two','three'])
results.unstack()
pd.concat([s1,s2,s3],axis = 1,keys = ['one','two','three'])

df1 = pd.DataFrame(np.arange(6).reshape(3,2),
                   index = ['a','b','c'],
                   columns = ['one','two'])

df2 = pd.DataFrame(5+np.arange(4).reshape(2,2),
                   index = ['a','c'],
                   columns = ['three','four'])

pd.concat([df1,df2],axis = 1,keys = ['level1','level2'])
pd.concat({
    'level1':df1,
    'level2':df2
    },axis = 1)

pd.concat([df1,df2],axis = 1,keys= ['level1','level2'],names=['upper','lower'])

df1 = pd.DataFrame(np.random.randn(3,4),
                   columns = ['a','b','c','d'])

df2 = pd.DataFrame(np.random.randn(2,3),
                   columns = ['b','d','a'])

pd.concat([df1,df2],ignore_index=True)

#combining data with overlap
a = pd.Series([np.nan, 2.5, np.nan, 3.5, 4.5, np.nan],
              index=['f', 'e', 'd', 'c', 'b', 'a'])
b = pd.Series(np.arange(len(a), dtype=np.float64),
              index=['f', 'e', 'd', 'c', 'b', 'a'])
b[-1] = np.nan
np.where(pd.isnull(a),b,a) #Return elements chosen from x or y depending on condition
b[:-2].combine_first(a[2:]) #left join and fill in na

df1 = pd.DataFrame({'a': [1., np.nan, 5., np.nan],
                    'b': [np.nan, 2., np.nan, 6.],
                    'c': range(2, 18, 4)})
df2 = pd.DataFrame({'a': [5., 4., np.nan, 3., 7.],
                    'b': [np.nan, 3., 4., 6., 8.]})
df1.combine_first(df2) ##left join and fill in na

"""
reshaping and pivot
"""
data = pd.DataFrame(np.arange(6).reshape((2, 3)),
                    index=pd.Index(['Ohio', 'Colorado'], name='state'),
                    columns=pd.Index(['one', 'two', 'three'],name='number'))
data.stack()
data.stack().unstack()
data.stack().unstack(0) #unstack by row
data.stack().unstack('state')

s1 = pd.Series([0, 1, 2, 3], index=['a', 'b', 'c', 'd'])
s2 = pd.Series([4, 5, 6], index=['c', 'd', 'e'])
data2 = pd.concat([s1, s2], keys=['one', 'two'])
data2.unstack()
data2.unstack().stack()
data2.unstack().stack(dropna = False)

df = pd.DataFrame({'left': data.stack(), 'right': data.stack() + 5},
                  columns=pd.Index(['left', 'right'], name='side'))
df.unstack('state')
df.unstack('state').stack('side')

# Pivoting “Long” to “Wide” Format
data = pd.read_csv('/Users/wangziwen/Documents/Python/Analysis/titanic/python-for-data-analytics-master/ch08/macrodata.csv')
data.head()
data.info()
periods = pd.PeriodIndex(year=data.year, quarter=data.quarter,name='date')

columns = pd.Index(['realgdp', 'infl', 'unemp'], name='item')
data = data.reindex(columns=columns)
data.index = periods.to_timestamp('D', 'end') #結尾的間隔是date
# periods.to_timestamp('D', 'start')
ldata = data.stack().reset_index().rename(columns={0: 'value'})

pivoted = ldata.pivot('date', 'item', 'value') #index,columns,values
pivoted

ldata['value2'] = np.random.randn(len(ldata))
ldata[:10]

pivoted = ldata.pivot('date', 'item') #如果value大於一個，那麼可以用下面的方法
pivoted[:5]
pivoted['value'][:5]

unstacked = ldata.set_index(['date', 'item']).unstack('item')
unstacked[:7]

# Pivoting “Wide” to “Long” Format
df = pd.DataFrame({'key': ['foo', 'bar', 'baz'],
                   'A': [1, 2, 3],
                   'B': [4, 5, 6],
                   'C': [7, 8, 9]})
df

melted = pd.melt(df, ['key']) #'key' = group
melted

reshaped = melted.pivot('key', 'variable', 'value')
reshaped

reshaped.reset_index()

pd.melt(df, id_vars=['key'], value_vars=['A', 'B']) #only print 'A' & 'B'
pd.melt(df, value_vars=['A', 'B', 'C']) #not print 'key'
pd.melt(df, value_vars=['key', 'A', 'B']) 







