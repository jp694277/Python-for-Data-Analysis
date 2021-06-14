#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 19:07:25 2021

@author: wangziwen
"""

from datetime import datetime
from datetime import timedelta
from dateutil.parser import parse
from scipy.stats import percentileofscore
from pandas.tseries.offsets import Hour, Minute
from pandas.tseries.offsets import Day, MonthEnd
import matplotlib as plt
import pandas as pd
import numpy as np
import pytz

"""
Date  and Ｔime Category & Tool
"""
datetime.now()
datetime.now().year, datetime.now().month, datetime.now().day

delta = datetime(2011, 1, 7) - datetime(2008, 6, 24, 8, 15)
delta #timedelta 代表時間差
delta.days
delta.seconds

datetime(2011, 1, 7)
datetime(2011, 1, 7) + timedelta(12)
datetime(2011, 1, 7) - 2 * timedelta(12)

"""
Converting Between String and Datetime
"""
#strftime
datetime(2021, 3, 30)
str(datetime(2021, 3, 30)) #transform to normal format
datetime(2021, 3, 30).strftime('%Y-%m-%d')
datetime(2021, 3, 30).strftime('%Y-%m-%d-%w')

#strptime
datetime.strptime('2021-03-30' , '%Y-%m-%d') # = datetime(2021, 3, 30)
str(datetime.strptime('2021-03-30' , '%Y-%m-%d'))
datestrs = ['3/30/2021', '4/30/2021']
[datetime.strptime(x, '%m/%d/%Y') for x in datestrs]
[str(datetime.strptime(x, '%m/%d/%Y')) for x in datestrs]

parse('2021-03-30') # = datetime(2021, 3, 30)
parse('Mar 30, 1997 5:07 AM') 
parse('30/3/2021',dayfirst = True)  
# In international locales, day appearing before month is very common, 
# so can pass dayfirst=True to indicate this

datestrs = ['2021-03-30 12:00:00', '2021-04-30 00:00:00']
pd.to_datetime(datestrs)

pd.to_datetime(datestrs + [None])
pd.to_datetime(datestrs + [None])[2]
pd.isnull(pd.to_datetime(datestrs + [None]))

"""
Time Series Basics
"""
dates = [datetime(2021, 1, 2), datetime(2021, 1, 5),
         datetime(2021, 1, 7), datetime(2021, 1, 8),
         datetime(2021, 1, 10), datetime(2021, 1, 12)]
ts = pd.Series(np.random.randn(6), index=dates)
ts
ts.index #DatetimeIndex
ts + ts[::2]
ts.index.dtype
ts.index[0]

#Indexing, Selection, Subsetting
#index
ts[ts.index[2]]
ts['1/10/2021']
ts['20210110']

longer_ts = pd.Series(np.random.randn(1000), 
                      index=pd.date_range('1/1/2020', periods=1000))
longer_ts #periods = 1000 Days
longer_ts['2021']
longer_ts['2021-03']
longer_ts['2021/03']
longer_ts['3-2021']
longer_ts['202103'] #wrong
longer_ts['20210330'] #but cannot use longer_ts['202103']

#slice
ts[datetime(2021, 1, 7):]
ts['1/6/2021':'1/11/2021']
ts.truncate(after='1/9/2021')
ts

dates = pd.date_range('1/1/2018', periods=100, freq='W-WED')
long_df = pd.DataFrame(np.random.randn(100, 4), 
                       index = dates,
                       columns = ['Colorado','Texas','New York', 'Ohio'])
long_df.loc['2019-03']

#Time Series with Duplicate Indices
dates = pd.DatetimeIndex(['1/1/2021', '1/2/2021', '1/2/2021', '1/2/2021', '1/3/2021'])
dup_ts = pd.Series(np.arange(5), index=dates)
dup_ts.index.is_unique #False
dup_ts['1/3/2021'] # not duplicated
dup_ts['1/2/2021'] # duplicated

dup_ts.groupby(level=0).mean() #group by row
dup_ts.groupby(level=0).count()

"""
Date Ranges, Frequencies, and Shifting
"""
ts
ts.resample('D')

#Generating Date Ranges
pd.date_range('2021-04-01', '2021-06-01') #date_range
pd.date_range(start='2021-04-01', periods=20)
pd.date_range(end='2021-06-01', periods=20)
pd.date_range('2021-01-01', '2021-12-01', freq='2D') #BM = BusinessMonthEnd
pd.date_range('2021-01-01', '2021-12-01', freq='BM') #BM = BusinessMonthEnd
pd.date_range('2021-01-01', '2021-12-01', freq='W-Mon')
pd.date_range('2021-03-30 12:56:31', periods=5)
pd.date_range('2021-03-30 12:56:31', periods=5, normalize=True)

#Frequencies and Date Offsets
Hour()
Hour(4)
pd.date_range('2021-01-01', '2021-01-03 23:59', freq='4h')
Hour(2) + Minute(30)
pd.date_range('2021-01-01', periods=10, freq='1h30min')
rng = pd.date_range('2021-01-01', '2021-09-01', freq='WOM-3FRI') #WeekOfMonth
list(rng)

#Shifting (Leading and Lagging) Data
ts = pd.Series(np.random.randn(4),index=pd.date_range('1/1/2021', periods=4, freq='M'))
ts
ts.shift(2) #時間向前或向後移
ts.shift(-2)

ts.shift(2, freq='M')
ts.shift(3, freq='D')
ts.shift(1, freq='90T')

now = datetime(2021, 3, 30)
now + 3 * Day()
now + MonthEnd()
now + MonthEnd(2) #Next MonthEnd
offset = MonthEnd()
offset.rollforward(now)
offset.rollback(now)

ts = pd.Series(np.random.randn(20),
               index=pd.date_range('3/30/2021', periods=20, freq='4d'))
ts
ts.groupby(offset.rollforward).mean()
ts.resample('M').mean()

"""
Time Zone Handling
"""
pytz.common_timezones[-5:]
pytz.timezone('Asia/Taipei')
pytz.timezone('UTC')

#Time Zone Localization and Conversion
rng = pd.date_range('3/30/2021 9:30', periods=6, freq='D')  
ts = pd.Series(np.random.randn(len(rng)), index=rng)
print(ts.index.tz) #time zone field

pd.date_range('3/30/2021 9:30', periods=10, freq='D', tz='UTC')
ts
ts_utc = ts.tz_localize('UTC')
ts_utc.index
print(ts_utc.index.tz)

ts_utc.tz_convert('Asia/Taipei')
ts_tw = ts.tz_localize('Asia/Taipei')
ts_tw.tz_convert('UTC')
ts_tw.tz_convert('Europe/Berlin')

ts.index.tz_localize('America/New_York')

#Operations with Time Zone−Aware Timestamp Objects
pd.Timestamp('2021-03-30 04:00') 
pd.Timestamp('2021-03-30 04:00').tz_localize('utc')
pd.Timestamp('2021-03-30 04:00').tz_localize('utc').tz_convert('America/New_York')
pd.Timestamp('2021-03-30 04:00', tz='America/New_York')

pd.Timestamp('2021-03-30 04:00').tz_localize('utc').value 
#store a UTC timestamp value as nanoseconds since (January 1, 1970)
pd.Timestamp('2021-03-30 04:00').tz_localize('utc').tz_convert('America/New_York').value

pd.Timestamp('2021-03-30 04:00', tz='US/Eastern')
pd.Timestamp('2021-03-30 04:00', tz='US/Eastern') + Hour()
pd.Timestamp('2021-03-30 04:00', tz='US/Eastern') + 2 * Hour()

#Operations Between Different Time Zones
rng = pd.date_range('3/30/2012 9:30', periods=10, freq='B') 
ts = pd.Series(np.random.randn(len(rng)), index=rng)
ts
ts1 = ts[:7].tz_localize('Europe/London')
ts2 = ts1[2:].tz_convert('Europe/Moscow')
result = ts1 + ts2
result.index #different time zones are combined, the result will be UTC

"""
Periods and Period Arithmetic
"""
pd.Period(2020, freq='A-DEC')
pd.Period(2020, freq='A-DEC') + 2
pd.Period(2020, freq='A-DEC') - 3
pd.Period(2020, freq='A-DEC') - pd.Period(1997, freq='A-DEC')

rng = pd.period_range('2021-01-01', '2021-06-30', freq='M')
rng
pd.Series(np.random.randn(6), index = rng)

values = ['2001Q3', '2002Q2', '2003Q1']
index = pd.PeriodIndex(values, freq='Q-DEC')
index

#Period Frequency Conversion
pd.Period('2020', freq='A-DEC')
pd.Period('2020', freq='A-DEC').asfreq('M', how='start')
pd.Period('2020', freq='A-DEC').asfreq('M', how='end')
 
pd.Period('2020', freq='A-JUN')
pd.Period('2020', freq='A-JUN').asfreq('M', 'start')
pd.Period('2020', freq='A-JUN').asfreq('M', 'end') 
 
pd.Period('Aug-2020', 'M')
pd.Period('Aug-2020', 'M').asfreq('A-JUN')

rng = pd.period_range('2018', '2021', freq='A-DEC') 
ts = pd.Series(np.random.randn(len(rng)), index=rng)

ts.asfreq('M', how='start')
ts.asfreq('B', how='end')

#Quarterly Period Frequencies
pd.Period('2020Q4', freq='Q-JAN')
pd.Period('2020Q4', freq='Q-JAN').asfreq('D', 'start')
pd.Period('2020Q4', freq='Q-JAN').asfreq('D', 'end')

#季度倒數第二個工作日的下午四點
(pd.Period('2020Q4', freq='Q-JAN').asfreq('B', 'e') - 1).asfreq('T', 's') + 16 * 60 
((pd.Period('2020Q4', freq='Q-JAN').asfreq('B', 'e') - 1).asfreq('T', 's') + 16 * 60).to_timestamp()

rng = pd.period_range('2020Q3', '2021Q4', freq='Q-JAN')
ts = pd.Series(np.arange(len(rng)), index=rng)
ts

new_rng = (rng.asfreq('B', 'e') - 1).asfreq('T', 's') + 16 * 60 
ts.index = new_rng.to_timestamp()
ts

#Converting Timestamps to Periods (and Back)
rng = pd.date_range('2021-01-01', periods=3, freq='M')
ts = pd.Series(np.random.randn(3), index=rng)
ts
ts.to_period()

rng = pd.date_range('1/29/2021', periods=6, freq='D')
ts2  = pd.Series(np.random.randn(6), index=rng)
ts2 
ts2.to_period('M')
ts2.to_period()
ts2.to_period().to_timestamp(how='end')

#Creating a PeriodIndex from Arrays
data = pd.read_csv('/Users/didi/Documents/Python/python-for-data-analytics-master/ch08/macrodata.csv')
data.head(5)
data.year 
data.quarter

index = pd.PeriodIndex(year=data.year, quarter=data.quarter,freq='Q-DEC')
index
data.index = index
data.infl

"""
Resampling and Frequency Conversion
"""
rng = pd.date_range('2021-01-01', periods=100, freq='D') 
ts = pd.Series(np.random.randn(len(rng)), index=rng)
ts
ts.resample('M').mean()
ts.resample('M', kind='period').mean()

#Downsampling
rng = pd.date_range('2021-01-01', periods=12, freq='T') 
ts = pd.Series(np.arange(12), index=rng)
ts
ts.resample('5min', closed='right').sum() #00:00:00 sum(00~05)
ts.resample('5min', closed='right', label='right').sum() #加上label較為直觀

ts.resample('5min', closed='right',label='right', loffset='-1s').sum()

ts.resample('5min').ohlc() #Open-High-Low-Close (OHLC) resampling

#Upsampling and Interpolation
frame = pd.DataFrame(np.random.randn(2, 4), 
                     index=pd.date_range('1/1/2021', periods=2,freq='W-WED'), 
                     columns=['Colorado', 'Texas', 'New York', 'Ohio'])
frame
df_daily = frame.resample('D').asfreq()
df_daily
 
frame.resample('D').ffill() 
frame.resample('D').ffill(limit=2) 
#only fill a certain number of periods forward to limit how
#far to continue using an observed value
frame.resample('W-THU').ffill()

#Resampling with Periods
frame = pd.DataFrame(np.random.randn(24, 4), 
                     index=pd.period_range('1-2020', '12-2021',freq='M'),
                     columns=['Colorado', 'Texas', 'New York', 'Ohio'])
frame[:5]
annual_frame = frame.resample('A-DEC').mean()
annual_frame

annual_frame.resample('Q-DEC').ffill()
annual_frame.resample('Q-DEC', convention='end').ffill() 
#convention argument defaults to 'start' but can also be 'end'
annual_frame.resample('Q-MAR').ffill()

"""
Moving Window Functions
"""
close_px_all = pd.read_csv('/Users/didi/Documents/Python/python-for-data-analytics-master/ch11/stock_px.csv', parse_dates=True, index_col=0)
close_px = close_px_all[['AAPL', 'MSFT', 'XOM']]
close_px = close_px.resample('B').ffill()

close_px.AAPL.plot()
close_px.AAPL.rolling(250).mean().plot()
#rolling:similarly to resample and groupby
#rolling(250) grouping over a 250-day

appl_std250 = close_px.AAPL.rolling(250, min_periods=10).std() 
#min_periods最少需要有值的观测点的数量
#https://baijiahao.baidu.com/s?id=1622798772654712959&wfr=spider&for=pc
appl_std250[:12]
appl_std250.plot()

expanding_mean = appl_std250.expanding().mean()
close_px.rolling(60).mean().plot(logy=True)

close_px.rolling('20D').mean()

#Exponentially Weighted Functions
aapl_px = close_px.AAPL['2006':'2007']
ma60 = aapl_px.rolling(30, min_periods=20).mean()
ewma60 = aapl_px.ewm(span=30).mean() #ew移動平均線

ma60.plot(style='k--', label='Simple MA')
ewma60.plot(style='k-', label='EW MA')
plt.legend()

#Binary Moving Window Functions
spx_px = close_px_all['SPX']
spx_rets = spx_px.pct_change() #百分比變化
returns = close_px.pct_change()

corr = returns.AAPL.rolling(125, min_periods=100).corr(spx_rets)
corr.plot()

corr = returns.rolling(125, min_periods=100).corr(spx_rets) 
corr.plot()

#User-Defined Moving Window Functions
score_at_2percent = lambda x: percentileofscore(x, 0.02) 
result = returns.AAPL.rolling(250).apply(score_at_2percent) 
result.plot()



















