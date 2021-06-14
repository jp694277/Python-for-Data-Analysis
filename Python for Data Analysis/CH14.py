#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 21:46:47 2021

@author: wangziwen
"""
import pandas as pd
import json
from collections import defaultdict
from collections import Counter
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

"""
USA.gov Data from Bitly
"""
path = '/Users/didi/Documents/Python/python-for-data-analytics-master/ch02/usagov_bitly_data2012-03-16-1331923249.txt'
open(path).readline()

#althogh it's txt file, we should import it as json file
records = [json.loads(line) for line in open(path)]
records[0]
len(records)
records[0:2]

#Counting Time Zones in Pure Python
time_zones = [rec['tz'] for rec in records] #error because of nan of 'tz'
time_zones = [rec['tz'] for rec in records if 'tz' in rec] 
time_zones[:10]

#count time zones
def get_counts(sequence): 
    counts = {} 
    for x in sequence: 
        if x in counts:
            counts[x] += 1 
        else:
            counts[x] = 1 
    return counts

def get_counts2(sequence): 
    counts = defaultdict(int) # values will initialize to 0 
    for x in sequence:
        counts[x] += 1 
    return counts

counts = get_counts(time_zones) 
counts['America/New_York']
len(time_zones)

counts = get_counts2(time_zones) 
counts['America/New_York']
len(time_zones)

#top 10 time zones and their counts,
def top_counts(count_dict, n=10):
    value_key_pairs = [(count, tz) for tz, count in count_dict.items()] 
    value_key_pairs.sort() 
    return value_key_pairs[-n:]

top_counts(counts)

#可以直接用Counter
counts = Counter(time_zones)
counts.most_common(10)

#Counting Time Zones with pandas
frame = pd.DataFrame(records)
frame.info()

frame['tz'][:10]
frame['tz'].value_counts()
frame['tz'].value_counts()[:10]

frame['tz'].isnull().sum()
clean_tz = frame['tz'].fillna('Missing')#120
clean_tz[clean_tz == ''] = 'Unknown' 
tz_counts = clean_tz.value_counts() 
tz_counts[:10]

#plot
subset = tz_counts[:10]
sns.barplot(y=subset.index, x=subset.values)

#information about the browser, device, or application
frame['a'][1]
frame['a'][50]
frame['a'][51]
frame['a'][51][:50] # long line

results = pd.Series([x.split()[0] for x in frame.a.dropna()]) #擷取最前面
results[:5]
results.value_counts()[:8]

cframe = frame[frame.a.notnull()] #frame.a.dropna()
cframe['os'] = np.where(cframe['a'].str.contains('Windows'),'Windows', 'Not Windows')
cframe['os'][:5]
by_tz_os = cframe.groupby(['tz', 'os'])
agg_counts = by_tz_os.size().unstack().fillna(0) 
agg_counts[:10]

# Use to sort in ascending order
indexer = agg_counts.sum(1).argsort()
indexer[:10]
count_subset = agg_counts.take(indexer[-10:]) 
count_subset

agg_counts.sum(1).nlargest(10)

# Rearrange the data for plotting
count_subset = count_subset.stack() 
count_subset.name = 'total'
count_subset = count_subset.reset_index() 
count_subset[:10]

#plot
sns.barplot(x='total', y='tz', hue='os', data=count_subset)

def norm_total(group):
    group['normed_total'] = group.total / group.total.sum() 
    return group

results = count_subset.groupby('tz').apply(norm_total)
sns.barplot(x='normed_total', y='tz', hue='os', data=results)

g = count_subset.groupby('tz')
results2 = count_subset.total / g.total.transform('sum')

"""
MovieLens 1M Dataset
"""
# Make display smaller
pd.options.display.max_rows = 10
unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
users = pd.read_table('/Users/wangziwen/Documents/Python/Analysis/titanic/python-for-data-analytics-master/ch02/movielens/users.dat', sep='::',header=None, names=unames)
rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table('/Users/wangziwen/Documents/Python/Analysis/titanic/python-for-data-analytics-master/ch02/movielens/ratings.dat', sep='::',header=None, names=rnames)
mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table('/Users/wangziwen/Documents/Python/Analysis/titanic/python-for-data-analytics-master/ch02/movielens/movies.dat', sep='::',header=None, names=mnames)

users[:5] #age and occupation are int
ratings[:5]
movies[:5]

users.info()
ratings.info()
movies.info()

#merge users and ratings
pd.merge(users,ratings)
#merge all
data = pd.merge(pd.merge(users,ratings),movies)
data
data.iloc[0]

#ratings group by sex
mean_ratings = data.pivot_table('rating',index = 'title',columns = 'gender',aggfunc='mean')
mean_ratings[:5]
mean_ratings.loc['Zed & Two Noughts, A (1985)',:]

#想要過濾掉少於250個評分的電影
ratings_by_title = data.groupby('title').size()
ratings_by_title[:10]
ratings_by_title.index
ative_titles = ratings_by_title.index[ratings_by_title >= 250]
ative_titles
mean_ratings = mean_ratings.loc[ative_titles]

#sort by female
mean_ratings.sort_values(by = 'F', ascending=False)

#找出男女評價最分歧的電影
mean_ratings['diff'] = mean_ratings['M'] - mean_ratings['F']
mean_ratings.sort_values(by = 'diff')[:10] #girls love much than boys
mean_ratings.sort_values(by = 'diff')[::-1][:10] #boys love much than girls

#用標準差來衡量分歧
#Standard deviation of rating grouped by title
rating_std_by_title = data.groupby('title')['rating'].std() # Filter down to active_titles
rating_std_by_title = rating_std_by_title.loc[ative_titles]
rating_std_by_title.sort_values(ascending = False)[:10]

"""
US Baby Names 1880–2010
"""
#可以用這個方式來show前十行
!head -n 10 /Users/wangziwen/Documents/Python/Analysis/titanic/python-for-data-analytics-master/ch02/names/yob1880.txt
names1880 = pd.read_csv('/Users/wangziwen/Documents/Python/Analysis/titanic/python-for-data-analytics-master/ch02/names/yob1880.txt',names=['name', 'sex', 'births'])

names1880.groupby('sex').births.sum()

#因為數據集很多，所以應該把所有數據集匯總成一個，且標上年份
years = range(1880,2011)
pieces = []
columns = ['name','sex','births']

for year in years:
    path = '/Users/wangziwen/Documents/Python/Analysis/titanic/python-for-data-analytics-master/ch02/names/yob%d.txt' % year
    frame = pd.read_csv(path,names = columns)
    frame['year'] = year
    pieces.append(frame)

pieces
names = pd.concat(pieces,ignore_index=True)
names

#pv table group by year and sex
total_births = names.pivot_table('births',index  ='year', columns = 'sex', aggfunc = sum)
total_births.tail()
total_births.plot(title = 'Total births by sex and year')

#該名字在該年度該性別的出生佔比
def add_prop(group):
    group['prop'] = group.births / group.births.sum()
    return group 
names = names.groupby(['year','sex']).apply(add_prop)
names
names.groupby(['year','sex']).prop.sum()

#top1000
def get_top1000(group):
    return group.sort_values(by = 'births', ascending = False)[:1000]

grouped = names.groupby(['year','sex'])
top1000 = grouped.apply(get_top1000)
top1000

top1000.reset_index(inplace = True,drop = True)
top1000

#analyze names trend
boys = top1000[top1000.sex=='M']
girls = top1000[top1000.sex=='F']
total_births = top1000.pivot_table('births', index = 'year',columns='name',aggfunc = sum)
total_births.info()

subset = total_births[['John','Harry','Mary','Marilyn']]
subset.plot(subplots = True, figsize = (12,10), grid = False, title = 'Number of births per year')

#命名多樣性
table = top1000.pivot_table('prop',index = 'year', columns = 'sex', aggfunc = sum)
table.plot(title = 'Sum of table.1000 prop by year and sex', yticks = np.linspace(0,1.2,13),
           xticks = range(1880,2020,10))


df = boys[boys.year ==2010]
df
prop_cumsum = df.sort_values(by = 'prop',ascending=False).prop.cumsum()
prop_cumsum.values.searchsorted(0.5)

df = boys[boys.year ==1900]
in1900 = df.sort_values(by = 'prop',ascending=False).prop.cumsum()
in1900.values.searchsorted(0.5) + 1

def get_quantile_count(group, q=0.5):
    group = group.sort_values(by='prop', ascending=False) 
    return group.prop.cumsum().values.searchsorted(q) + 1

diversity = top1000.groupby(['year', 'sex']).apply(get_quantile_count)
diversity = diversity.unstack('sex')
diversity.plot(title = 'Number of popular names in top 50%')    

#最後一個字母革命
get_last_letter = lambda x: x[-1] 
last_letters = names.name.map(get_last_letter) 
last_letters.name = 'last_letter'
table = names.pivot_table('births', index=last_letters,columns=['sex', 'year'], aggfunc=sum)
table

#選出幾個有代表性的年份
subtable = table.reindex(columns=[1910, 1960, 2010], level='year')
subtable.head()
subtable.sum()

letter_prop = subtable / subtable.sum()
letter_prop

fig, axes = plt.subplots(2, 1, figsize=(10, 8))
letter_prop['M'].plot(kind='bar', rot=0, ax=axes[0], title='Male')
letter_prop['F'].plot(kind='bar', rot=0, ax=axes[1], title='Female',legend=False)

letter_prop = table / table.sum()
dny_ts = letter_prop.loc[['d', 'n', 'y'], 'M'].T
dny_ts.head()
dny_ts.plot()    

#男生名字變成女生名字
all_names = pd.Series(top1000.name.unique())
lesley_like = all_names[all_names.str.lower().str.contains('lesl')]
lesley_like

filtered = top1000[top1000.name.isin(lesley_like)]
filtered.groupby('name').births.sum()

table = filtered.pivot_table('births', index='year',columns='sex', aggfunc='sum')
table = table.div(table.sum(1), axis=0) #標準化
table.tail()
table.plot(style={'M': 'k-', 'F': 'k--'})

"""
USDA Food Database
"""
db = json.load(open('/Users/wangziwen/Documents/Python/Analysis/titanic/python-for-data-analytics-master/ch07/foods-2011-10-03.json'))
len(db)

db[0].keys()
db[0]['nutrients']
db[0]['nutrients'][0]

nutrients = pd.DataFrame(db[0]['nutrients'])
nutrients[:7]

info_keys = ['description','group','id','manufacturer']
info = pd.DataFrame(db,columns = info_keys)
info[:5]
info.info()
info.group.value_counts()[:10]

nutrients.duplicated().sum() #duplicated num
nutrients = nutrients.drop_duplicates()

#rename
col_mapping = {'description': 'food',
               'group': 'fgroup'
    }
info = info.rename(columns = col_mapping, copy = False)
col_mapping = {'description': 'nutrient',
               'group': 'nutgroup'
    }
nutrients = nutrients.rename(columns = col_mapping, copy = False)
nutrients.info()

#merge

ndata = pd.merge(nutrients,info, on = 'id',how = 'outer')
result = ndata.groupby(['nutrient', 'fgroup'])['value'].quantile(0.5)
result['Zinc, Zn'].sort_values().plot(kind='barh')

by_nutrient = ndata.groupby(['nutgroup', 'nutrient']) 
get_maximum = lambda x: x.loc[x.value.idxmax()]
get_minimum = lambda x: x.loc[x.value.idxmin()]
max_foods = by_nutrient.apply(get_maximum)[['value', 'food']]

# make the food a little smaller
max_foods.food = max_foods.food.str[:50]

"""
2012 Federal Election Commission Database
"""
fec = pd.read_csv('/Users/wangziwen/Documents/Python/Analysis/titanic/python-for-data-analytics-master/ch09/P00000001-ALL.csv')
fec.info()
fec.iloc[123456]

#candidate
unique_cands = fec.cand_nm.unique()
unique_cands

#party
parties = {'Bachmann, Michelle': 'Republican',
               'Cain, Herman': 'Republican',
               'Gingrich, Newt': 'Republican',
               'Huntsman, Jon': 'Republican',
               'Johnson, Gary Earl': 'Republican',
               'McCotter, Thaddeus G': 'Republican',
               'Obama, Barack': 'Democrat',
               'Paul, Ron': 'Republican',
               'Pawlenty, Timothy': 'Republican',
               'Perry, Rick': 'Republican',
               "Roemer, Charles E. 'Buddy' III": 'Republican',
               'Romney, Mitt': 'Republican',
               'Santorum, Rick': 'Republican'}
fec.cand_nm[123456:123461]
fec.cand_nm[123456:123461].map(parties)
fec['party'] = fec.cand_nm.map(parties)
fec.party.value_counts()

#貢獻金額
(fec.contb_receipt_amt > 0).value_counts()
fec = fec[fec.contb_receipt_amt>0]
fec_mrbo = fec[fec.cand_nm.isin(['Obama, Barack','Romney, Mitt'])]

#contb by occupation and employer
fec.contbr_occupation.value_counts()
occ_mapping = {
       'INFORMATION REQUESTED PER BEST EFFORTS' : 'NOT PROVIDED',
       'INFORMATION REQUESTED' : 'NOT PROVIDED',
       'INFORMATION REQUESTED (BEST EFFORTS)' : 'NOT PROVIDED',
       'C.E.O.': 'CEO'
}

# If no mapping provided, return x
f = lambda x: occ_mapping.get(x, x) 
fec.contbr_occupation = fec.contbr_occupation.map(f)
fec.contbr_occupation.value_counts()

emp_mapping = {
       'INFORMATION REQUESTED PER BEST EFFORTS' : 'NOT PROVIDED',
       'INFORMATION REQUESTED' : 'NOT PROVIDED',
       'SELF' : 'SELF-EMPLOYED',
       'SELF EMPLOYED' : 'SELF-EMPLOYED',
}
    # If no mapping provided, return x
f = lambda x: emp_mapping.get(x, x) 
fec.contbr_employer = fec.contbr_employer.map(f)
fec.contbr_employer.value_counts()

by_occupation = fec.pivot_table('contb_receipt_amt',
                                index='contbr_occupation',
                                columns='party', aggfunc='sum')
by_occupation
over_2mm = by_occupation[by_occupation.sum(1) > 2000000]
over_2mm
over_2mm.plot(kind='barh')

def get_top_amounts(group, key, n=5):
    totals = group.groupby(key)['contb_receipt_amt'].sum() 
    return totals.nlargest(n)

grouped = fec_mrbo.groupby('cand_nm')
grouped.apply(get_top_amounts, 'contbr_occupation', n=7)
grouped.apply(get_top_amounts, 'contbr_employer', n=10)

#Bucketing Donation Amounts
bins = np.array([0, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000])
labels = pd.cut(fec_mrbo.contb_receipt_amt, bins)
labels
grouped = fec_mrbo.groupby(['cand_nm', labels])
grouped.size().unstack(0)

bucket_sums = grouped.contb_receipt_amt.sum().unstack(0)
normed_sums = bucket_sums.div(bucket_sums.sum(axis=1), axis=0)
normed_sums
normed_sums[:-2].plot(kind='barh')

#Donation Statistics by State
grouped = fec_mrbo.groupby(['cand_nm', 'contbr_st'])
totals = grouped.contb_receipt_amt.sum().unstack(0).fillna(0)
totals = totals[totals.sum(1) > 100000]
totals[:10]

percent = totals.div(totals.sum(1), axis=0)
percent[:10]



















