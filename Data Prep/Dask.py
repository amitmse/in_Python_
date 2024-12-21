####################################################################################################################
######## DASK #######
###################################################################################################################

from pathlib import Path
import pyreadstat
import dask
import dask.dataframe as dd
from dask.delayed import delayed
from dask.distributed import Client
import dask.datasets
from dask_sql import Context
from datetime import datetime 
import numpy as np
import pandas as pd
###################################################
import os
### to Display sample #############################
import pandas as pd
pd.options.display.precision = 2
pd.options.display.max_rows = 10
###################################################
dask.config.set({"dataframe.convert-string": False})
client = Client()
client
###################################################

# Go to code and select MARKDOWN and then ### space text. To execute CLTR + enter
# Change Directory
os.chdir(r"C:\Users\test")
#os.listdir()
os.getcwd()

# Import CSV
A1 = dd.read_csv('A1.CSV', assume_missing=True)
A2 = dd.read_csv('A2.CSV', assume_missing=True)
A1.head()
#A1.shape[0].compute()

# Export to CSV
A1.compute().to_csv (r'export_dataframe.csv', index = None, header=True)
A1.head(10)

# Read SAS file
sas = pd.read_sas('segment_unbiased_pd.sas7bdat')
sas.head()

# Convert Dask to Pandas
pd_A1 = A1.compute()
type(pd_A1)

# Convert Pandas to Dask
ddf = dd.from_pandas(pd_A1) #, npartitions=1)
type(pd_A1)
ddf.head()

# Covert Dask to Parquet
A1.to_parquet("A1_parquet.parquet")
A1.to_parquet("parquet/A1_parquet_test") # define folder name

# Covert Parquet to Dask
A1_par = dd.read_parquet("parquet/A1_parquet_test")
A1_par.head()
A11_par = dd.read_parquet("A1_parquet.parquet")
A11_par.head()

# Rename variable
ddf = ddf.rename(columns = {'key':'rename_key', 'var1':'rename_var1'}).compute()
ddf.head()
type(ddf)

# Count of missing
A1.isna().sum().compute()

# Sort and Remove Duplicate
A2= (A2.sort_values('key', ascending=False))
A1_nodup=A1.drop_duplicates('key', keep='last')
#A2.head()

# Append
append = dd.concat([A1, A2])
append.head()
#result.head(10)
#type(result)

# Merge : Left, Right, Outer, Inner
left = dd.merge(A1, A2, on=['key'], how='left')
right = dd.merge(A1, A2, on=['key'], how='right')
inner = dd.merge(A1, A2, on=['key'], how='inner')
outer = dd.merge(A1, A2, on=['key'], how='outer')
outer.head()

# Cross Join or Cartesian Product
# Performing the cross join cartesian product
A1['cross_key'] = 1
A2['cross_key'] = 1
cross = dd.merge(A1, A2, on=['cross_key'], how='outer')
cross.head()

# panda cross join
#pd.merge(A1, A2, on='cross_key').drop('cross_key', axis=1) 
#convert dash to panda
#pd_test = test.compute() 
#test.compute().to_csv (r'export_dataframe.csv', index = None, header=True)
#test.head(10)

# create variable, variable transformation
A1['multiply'] = (9990.0 * A1['var1'])
A1['division'] = A1['multiply']/10
A1['logarithm_base10'] = np.log10(A1['multiply']) # base 10
A1['natural_log'] = np.log(A1['multiply']) # natural logarithm
#A1['div'] = (A1['d']).div(10)
A1.head()

# Overall: sum, count, min, max, mean
cross.var2.sum().compute()
cross.var2.count().compute()
cross.var2.min().compute()
cross.var2.max().compute()
cross.var2.mean().compute()

# Group by sum, count, min, max, mean
freq=cross.groupby(['key_x','key_y'])['var2'].sum().reset_index()
freq.head()
freq = cross.groupby(['key_x','key_y'])['var2'].agg(['sum','count', 'min', 'max', 'mean']).reset_index()
freq.head()
freq = cross.groupby('key_x').aggregate({'var1':'count','var2':['min','max']}).reset_index()
freq.head()

# SQL query
# create a context to register tables
c = Context()
test = dd.read_csv('A1.csv', assume_missing=True)
c.create_table("sample_test",test)

# execute a SQL query; the result is a "lazy" Dask dataframe
result = c.sql(""" SELECT * FROM sample_test """)
# actually compute the query...
result.compute()

# Date Time: Conversion, Date Range, Extract (D M Y), Next Date, Date Difference, Rename
#Present Date
#print(datetime.now())
from dask.dataframe import DataFrame
ddf = DataFrame.from_dict({"datetime": ["2013-01-01 09:10:12", "2022-01-02 09:10:12", "2022-01-03 09:10:12"], "patients": [16, 19, 11]}, npartitions=1)
#Convert into date
ddf['date1']  = dd.to_datetime(ddf['datetime']).dt.date
ddf['date']   = dd.to_datetime(ddf['date1'], dayfirst=True)
#Extarct day month year
ddf['year']   = ddf['date'].dt.year
ddf['month']  = ddf['date'].dt.month
ddf['day']    = ddf['date'].dt.day
ddf['day_nm'] = ddf['date'].dt.day_name()
#Convert into date
ddf['cnvt_date2']  = dd.to_datetime(ddf[['year', 'month', 'day']])
# Next date 
ddf['nxt_date_m'] = ddf['date'] + pd.tseries.offsets.DateOffset(months=5)
#ddf['nxt_date_d'] = ddf['date'] + pd.tseries.offsets.DateOffset(days=17)
#ddf['nxt_date_y'] = ddf['date'] + pd.tseries.offsets.DateOffset(years=1)
ddf['new_date_d'] = ddf['date'] + ddf['day'].astype('timedelta64[D]')
ddf['bck_date_y'] = ddf['date'] - pd.tseries.offsets.DateOffset(years=1)
#ddf['new_date_m'] = ddf['date'] + ddf['month'].astype('timedelta64[M]')
#ddf['new_date_y'] = ddf['date'] + ddf['month'].astype('timedelta64[Y]')
#ddf['c']= ddf['date'] + ddf['day'].apply(pd.offsets.Day)
# date difference
ddf['month_diff_nxt'] = ddf['nxt_date_m'].dt.to_period('M').astype('int64') - ddf['date'].dt.to_period('M').astype('int64')
#ddf['day_diff'] = ddf['nxt_date_d'].dt.to_period('D').astype('int64') - ddf['date'].dt.to_period('D').astype('int64')
#ddf['year_diff'] = ddf['nxt_date_y'].dt.to_period('Y').astype('int64') - ddf['date'].dt.to_period('Y').astype('int64')
ddf.head()

#Present Date
#print(datetime.now())
'''

https://pandas.pydata.org/docs/user_guide/timeseries.html
## Start of month freq='MS'       -> 01Jan2024
## End   of month freq='ME'    -> 31Jan2024
## week           freq='W'
### Date range 
DT_Range = print(pd.date_range("2018-01-01", periods=3, freq="MS"), #month 
                 pd.date_range("2018-01-01", periods=3, freq="d"), #day
                 pd.date_range("2018-01-01", periods=3, freq="YE") #year
                )

### Date range between two dates
import datetime
start       = datetime.datetime(2011, 1, 1)
end         = datetime.datetime(2012, 1, 1)
index_month = pd.date_range(start, end, freq='ME')
index_day   = pd.date_range(start, end, freq='D')
index_year  = pd.date_range(start, end, freq='YE')
'''

df           = pd.DataFrame({'datetime': ["2013-01-01 09:10:12", "2022-01-02 09:10:12", "2022-01-03 09:10:12"], 'patients': [16, 19, 11]})
#Convert into date
df['date1']  = pd.to_datetime(df['datetime']).dt.date
df['date']   = pd.to_datetime(df['date1'], dayfirst=True)
#Extarct day month year
df['year']   = df['date'].dt.year
df['month']  = df['date'].dt.month
df['day']    = df['date'].dt.day
df['day_nm'] = df['date'].dt.day_name()
#Convert into date
df['cnvt_date2']  = pd.to_datetime(df[['year', 'month', 'day']])
# Next date 
df['nxt_date_m'] = pd.to_datetime(df['date1']) + pd.DateOffset(months=5)
#df['nxt_date_d'] = pd.to_datetime(df['date1']) + pd.DateOffset(days=10)
#df['nxt_date_y'] = pd.to_datetime(df['date1']) + pd.DateOffset(years=1)
# back date 
df['bck_date_m'] = pd.to_datetime(df['date1']) - pd.DateOffset(months=18)
#df['bck_date_d'] = pd.to_datetime(df['date1']) - pd.DateOffset(days=1)
#df['bck_date_y'] = pd.to_datetime(df['date1']) - pd.DateOffset(years=1)
# date difference
df['month_diff_nxt'] = df['nxt_date_m'].dt.to_period('M').astype('int64') - df['date'].dt.to_period('M').astype('int64')
#df['day_diff'] = df['bck_date_m'].dt.to_period('D').astype('int64') - df['date'].dt.to_period('D').astype('int64')
#df['year_diff'] = df['bck_date_m'].dt.to_period('Y').astype('int64') - df['date'].dt.to_period('Y').astype('int64')
df.rename(columns = {'month_diff_nxt':'rename'}, inplace = True)
df

######################################################################################################################################
#### End of Dask
######################################################################################################################################

# pyarrow fastparquet
#python - How to select and run model from dash dropdown menu and update confusion matrix figure? - Stack Overflow
#sqlite - Issue with Dropdown pulling data from SQL in Python Dash - Stack Overflow
#What are some alternatives to Dask? - StackShare
#Scale your pandas workflow by changing a single line of code — Modin 0.32.0+0.g3e951a6.dirty documentation




