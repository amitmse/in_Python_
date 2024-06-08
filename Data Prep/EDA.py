###################################################################################################################################
#### Exploratory Data Analysis (EDA) provides basic distribution of data i.e., count, missing, unique, sum, mean, STD, percentile
###################################################################################################################################
import os
import csv
import sys
import numpy as np
from datetime import datetime
from numpy.lib.recfunctions import append_fields
import matplotlib.mlab as mlab
from math import sqrt
import pandas as pd
import prettytable as pt
from pandas import DataFrame, read_csv
from scipy import stats

################################################################################################
#current dir	
	os.getcwd()
#change dir
	os.chdir('D:\\Training\\Python\\Logistic_Model_Development_Process')
#############################################################################################################################################		
## Test on client data ######################################################################################################################
	#Prepare format of variable to read raw file. Keep the same order for variable name as it is raw file.
	client_raw_dtype = np.dtype([( 'vid' , 'S100' ), ( 'session' , 'float64' ), 	( 'session_start_server' , 'float64' ),	( 'max_page_count' , 'float64' ),	( 'max_page_count_before_purchase' , 'float64' ),	( 'max_page_count_before_purchase_flag' , 'float64' ),	( 'os_cat' , 'S10' ),	( 'dd' , 'S1' ),	( 'country' , 'S20' ),	( 'browser_cat' , 'S5' ),	( 'Property_Brand' , 'S4' ),	( 'nop_last_visit' , 'float64' ),	( 'no_of_visits_last_7_days' , 'float64' ),	( 'no_of_purchases_last_7_days' , 'f3' ),	( 'page_01_category' , 'S100' ),	( 'page_02_category' , 'S100' ),	( 'page_03_category' , 'S100' ),	( 'page_04_category' , 'S100' ),	( 'page_05_category' , 'S100' ),	( 'page_06_category' , 'S100' ),	( 'page_07_category' , 'S100' ),	( 'page_08_category' , 'S100' ),	( 'page_09_category' , 'S100' ),	( 'page_10_category' , 'S100' ),	( 'page_11_category' , 'S100' ),	( 'page_12_category' , 'S100' ),	( 'page_13_category' , 'S100' ),	( 'page_14_category' , 'S100' ),	( 'page_15_category' , 'S100' ),	( 'page_16_category' , 'S100' ),	( 'page_17_category' , 'S100' ),	( 'page_18_category' , 'S100' ),	( 'page_19_category' , 'S100' ),	( 'page_20_category' , 'S100' ),	( 'reservation' , 'float64' ),	( 'client_loyalty_flag' , 'float64' ),	( 'client_loyalty_Ever_flag' , 'float64' ),	( 'random_number' , 'float64' )])
	#Apply the above format to convert in numpy data.  keep dataset name as "df"
	df = np.genfromtxt('client_Model_Dev_Val_Data.csv', delimiter=',', names=True, missing_values='nan', dtype = client_raw_dtype)
	
###########################################################################################################################################################
############# Do Not change below code. Only run it
###########################################################################################################################################################

def overall_eda(df):
#### EDA for all data type
	#Data type for Columns
	eda_data_type = np.dtype([('var', 'S100'),('type', 'S5'),( 'total' , 'float64' ), ( 'missing' , 'float64' ), ( 'unique' , 'float64' ), ( 'sum' , 'float64' ), ( 'min' , 'S12' ), ( 'pct_01' , 'float64' ), ( 'pct_05' , 'float64' ), ( 'pct_10' , 'float64' ), ( 'pct_20' , 'float64' ), ( 'pct_30' , 'float64' ), ( 'pct_40' , 'float64' ), ( 'pct_50' , 'float64' ), ( 'pct_60' , 'float64' ), ( 'pct_70' , 'float64' ), ( 'pct_80' , 'float64' ), ( 'pct_90' , 'float64' ), ( 'pct_95' , 'float64' ), ( 'pct_99' , 'float64' ), ( 'max' , 'S12' ), ( 'mean' , 'float64' ), ( 'median' , 'float64' ), ( 'std' , 'float64' ), ( 'mode' , 'float64' )])
	#Create empty array for final out of EDA 
	eda_output 	  = np.array([], dtype=eda_data_type)
	#Generate EDA for numeric , date and string columns
	for var in df.dtype.names:
		#EDA for numeric
		if df[var].dtype.char not in ('S', 'M'):
			category, count = np.unique(df[var],return_counts=True)
			total      = df[var].size
			missing    = np.isnan(df[var]).sum()
			unique     = np.unique(df[var]).size
			sum        = np.nansum(df[var])
			min        = np.nanmin(df[var])
			pct_01     = np.nanpercentile(df[var],1)
			pct_05     = np.nanpercentile(df[var],5)
			pct_10     = np.nanpercentile(df[var],10)
			pct_20     = np.nanpercentile(df[var],20)
			pct_30     = np.nanpercentile(df[var],30)
			pct_40     = np.nanpercentile(df[var],40)
			pct_50     = np.nanpercentile(df[var],50)
			pct_60     = np.nanpercentile(df[var],60)
			pct_70     = np.nanpercentile(df[var],70)
			pct_80     = np.nanpercentile(df[var],80)
			pct_90     = np.nanpercentile(df[var],90)
			pct_95     = np.nanpercentile(df[var],95)
			pct_99     = np.nanpercentile(df[var],99)
			max        = np.nanmax(df[var])
			mean       = np.nanmean(df[var])
			median     = np.nanmedian(df[var])
			std        = np.nanstd(df[var],ddof=1)
			mode   	   = category[np.argmax(count)]
			#dump all metrics in an array with data type
			eda_num = np.array([(var,'Num',total,missing,unique,sum,min,pct_01,pct_05,pct_10,pct_20,pct_30,pct_40,pct_50,pct_60,pct_70,pct_80,pct_90,pct_95,pct_99,max,mean,median,std,mode)],dtype=eda_data_type)
			#append in final array
			eda_output = np.hstack((eda_output, eda_num))
		#EDA for Date
		if df[var].dtype.char == 'M':
			date_test = np.array([], '<M8[s]')
			for i in df[var]:
				if (i != np.datetime64('1970-01-01T05:30:00+0530')):
					date_test = np.hstack((date_test, i))
			total 	= df[var].size
			missing = df[var].size - date_test.size
			unique 	= np.unique(df[var]).size
			min 	= np.datetime_as_string(np.min(date_test.astype('M8[D]')))
			max 	= np.datetime_as_string(np.max(date_test.astype('M8[D]')))
			#dump all metrics in an array with data type
			eda_date = np.array([(var,'Date',total,missing,unique,'nan',min,'nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan',max,'nan','nan','nan','nan')],dtype=eda_data_type)
			#append in final array
			eda_output = np.hstack((eda_output, eda_date))
		#EDA for string
		if df[var].dtype.char == 'S':
			total 	= df[var].size
			missing = np.sum((df[var]) == '')
			unique 	= np.unique(df[var]).size
			#dump all metrics in an array with data type
			eda_string = np.array([(var,'Str',total,missing,unique,'nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan')],dtype=eda_data_type)
			#append in final array
			eda_output = np.hstack((eda_output, eda_string))
			
	#EDA output Write in csv
		os.remove('overall_eda_output.csv') if os.path.exists('overall_eda_output.csv') else None
		###same as above
		###	if os.path.isfile('eda_output.csv'):
		###		os.remove('eda_output.csv')
		eda_output.sort(order=('type','var'))
		mlab.rec2csv(eda_output, 'overall_eda_output.csv')
		
#############################################################################################################################################		
overall_eda(df)
#############################################################################################################################################
