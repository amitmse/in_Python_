from __future__ import print_function
import subprocess
import sys
import pydotplus
import os
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from IPython.display import Image
import matplotlib.pylab as plt
from time import time
from operator import itemgetter
from scipy.stats import randint
from collections import defaultdict
import pickle

####################################################################
os.chdir(r"C:\Users\AMIT\Google Drive\Study\ML\06.Random_Forest")
input_data = pd.read_csv('IV_Data.csv')
df= input_data[:]
#####################################################################
dependent_variable_name = 'reservation_n'
df['day']=pd.to_datetime(df['day'])
df['month']=pd.to_datetime(df['month'])
df['year']=pd.to_datetime(df['year'])
###################################################################

def contents(input_data=None):
    #############
    contents = pd.DataFrame(input_data.dtypes,columns = ['type'])
    contents = contents.reset_index()
    contents.rename(columns = {'index':'variable'}, inplace = True)
    #############
    cnt = pd.DataFrame(input_data.count(),columns = ['count'])
    cnt = cnt.reset_index()
    cnt.rename(columns = {'index':'variable'}, inplace = True)
    contents = pd.merge(contents, cnt, how='left', on=['variable'])
    ##############
    miss = pd.DataFrame(input_data.isnull().sum(),columns = ['missing'])
    miss = miss.reset_index()
    miss.rename(columns = {'index':'variable'}, inplace = True)
    contents = pd.merge(contents, miss, how='left', on=['variable'])
    ##############
    unique = pd.DataFrame(input_data.nunique(),columns = ['unique_values'])
    unique = unique.reset_index()
    unique.rename(columns = {'index':'variable'}, inplace = True)
    contents = pd.merge(contents, unique, how='left', on=['variable'])
    ################
    #print (contents)
    return contents

##############################################################################
def var_type(input):
    if input['type'] == 'datetime64[ns]':
        val = 'NA'
    #elif input['unique_values'] == input_data.shape[0]:
    #    val = 'NA'
    elif input['unique_values'] <10: 
        val = 'char'
    elif input['type'] in ['object','bool']: 
        val = 'char'
    else: 
        val = 'num'
    return val

#########################################################################################################
def calculation_info_val_num_var(input_data=None, var=None, target=None):
    temp = pd.DataFrame(input_data, columns=[target, var])
    temp.rename(columns = {target:'response', var:'score'}, inplace = True) 
    temp['non_response'] = 1 - temp['response'] #temp.response
    #DEFINE 10 BUCKETS WITH EQUAL SIZE
    try:
        temp['bucket'] = pd.qcut(temp.score, 10)
    except:
        temp['Rank'] = temp["score"].rank(method='first')
        temp['bucket'] = pd.qcut(temp.Rank, 10)
 
    #GROUP THE DATA FRAME BY BUCKETS
    grouped = temp.groupby('bucket', as_index = False)
    ####################################################################
    #CREATE A SUMMARY DATA FRAME
    df_info_val_n= pd.DataFrame()
    df_info_val_n['categery'] = grouped.min().score.round(decimals=2).astype(str) + ' - ' + grouped.max().score.round(decimals=2).astype(str)
    df_info_val_n['variable'] = var
    df_info_val_n['count']   = grouped.sum().response + grouped.sum().non_response
    df_info_val_n['target'] = grouped.sum().response
    df_info_val_n = df_info_val_n[['variable', 'categery', 'count', 'target' ]]
    df_info_val_n['non_target'] = grouped.sum().non_response
    df_info_val_n['pct_count'] = (df_info_val_n['count']/df_info_val_n['count'].sum())
    df_info_val_n['pct_target'] = (df_info_val_n['target']/df_info_val_n['target'].sum())
    df_info_val_n['pct_non_target']= (df_info_val_n['non_target']/df_info_val_n['non_target'].sum())
    df_info_val_n['target_rate'] = (df_info_val_n['target'] / df_info_val_n['count'])
    df_info_val_n['woe'] = np.log( df_info_val_n['pct_non_target'] / df_info_val_n['pct_target'] )
    df_info_val_n = df_info_val_n.replace([np.inf, -np.inf], 0).dropna(axis=1)
    df_info_val_n['info_value'] = ( df_info_val_n['pct_non_target'] - df_info_val_n['pct_target'])*df_info_val_n['woe']
    return df_info_val_n

###################################################################################################
def calculation_info_val_char_var(input_data=None, var=None, target=None):
    df_info_val_c = pd.pivot_table(input_data, index=[var], values=[target],aggfunc=[np.sum,len],fill_value=0)
    df_info_val_c = df_info_val_c.reset_index()
    df_info_val_c.columns = (df_info_val_c.columns.map('|'.join).str.strip('|'))
    df_info_val_c.columns = df_info_val_c.columns.str.split('|').str[0]
    df_info_val_c.rename(columns = {var:'categery','sum':'target', 'len':'count'}, inplace = True)
    df_info_val_c['variable'] = var
    df_info_val_c = df_info_val_c[['variable', 'categery', 'count', 'target' ]]
    df_info_val_c['non_target'] =   df_info_val_c['count']  - df_info_val_c['target']
    df_info_val_c['pct_count']  = ( df_info_val_c['count']  / df_info_val_c['count'].sum() )
    df_info_val_c['pct_target'] = ( df_info_val_c['target'] / df_info_val_c['target'].sum() )
    df_info_val_c['pct_non_target'] = (df_info_val_c['non_target'] / df_info_val_c['non_target'].sum() )
    df_info_val_c['target_rate'] = (df_info_val_c['target'] / df_info_val_c['count'])
    df_info_val_c['woe'] = np.log( df_info_val_c['pct_non_target'] / df_info_val_c['pct_target'] )
    df_info_val_c = df_info_val_c.replace([np.inf, -np.inf], 0).dropna(axis=1)
    df_info_val_c['info_value'] = ( df_info_val_c['pct_non_target'] - df_info_val_c['pct_target'])*df_info_val_c['woe']
    return df_info_val_c

#################################################################################
contents = contents(input_data=df)
contents['treatment'] = contents.apply(var_type, axis=1)
contents

char_list = contents[contents['treatment'] == 'char'].loc[:,'variable'].values.tolist()
num_list = contents[contents['treatment'] == 'num'].loc[:,'variable'].values.tolist()
NA_list = contents[contents['treatment'] == 'NA'].loc[:,'variable'].values.tolist()

char_list.remove('reservation_c')
char_list.remove(dependent_variable_name)

char_list.append(dependent_variable_name)
num_list.append(dependent_variable_name)

df_char = df.loc[:, char_list]
df_num = df.loc[:, num_list]

char_list.remove(dependent_variable_name)
num_list.remove(dependent_variable_name)

##################################################################################################
info_val_num = None
for var in num_list:
    df_info_val_n=calculation_info_val_num_var(input_data=df_num, var=var, target=dependent_variable_name)
    info_val_num = pd.concat([df_info_val_n,info_val_num])
###################################################################################################
info_val_char = None
for var in char_list:
    df_info_val_c=calculation_info_val_char_var(input_data=df_char, var=var, target=dependent_variable_name)
    info_val_char = pd.concat([df_info_val_c,info_val_char])
######################################################################################################
## categery level information value
info_val_all = None
info_val_all = pd.concat([info_val_all,info_val_char])
info_val_all = pd.concat([info_val_all,info_val_num])
##################
info_val_all['pct_count'] = (info_val_all['pct_count']).apply('{0:.2%}'.format)
info_val_all['pct_target'] = (info_val_all['pct_target']).apply('{0:.2%}'.format)
info_val_all['pct_non_target'] = (info_val_all['pct_non_target']).apply('{0:.2%}'.format)
info_val_all['target_rate'] = (info_val_all['target_rate']).apply('{0:.2%}'.format)
info_val_all['info_value'] = info_val_all['info_value'].round(decimals=2)
#################################################################################################
# Overall Information value
info_val_overall = pd.pivot_table(info_val_all, index=['variable'], values=['info_value'],aggfunc=[np.sum],fill_value=0)
info_val_overall = info_val_overall.reset_index()
info_val_overall.columns = (info_val_overall.columns.map('|'.join).str.strip('|'))
info_val_overall.columns = info_val_overall.columns.str.split('|').str[0]
info_val_overall.rename(columns = {'sum':'Information_value'}, inplace = True)
info_val_overall
#################################################################################################

