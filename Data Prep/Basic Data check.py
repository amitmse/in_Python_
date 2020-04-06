####### Install Library ############################################################

pip install sas7bdat

####### Import Library ############################################################
from sas7bdat import SAS7BDAT
import pandas as pd
import numpy as np
import os
import pandasql as ps

###### Change working directory ###################################################
	work = r'C:\Users\AMIT'		# Don't put "\" in the last ('C:\Users\AMIT\')
	os.chdir(work)
	os.getcwd()

###### Import data #################################################################

# Import data : SAS file
	sas_data_path=r'C:\Users\AMIT\final.sas7bdat'
	reader = SAS7BDAT(sas_data_path) 
	df = reader.to_data_frame()

# Import data : CSV
	df = pd.read_csv('train.csv', header=0)

# Import data : excel (work on both xlsx & xls)
	xlsx = r'C:\Users\AMIT\Dev1_Model_Data.xlsx'
	df = pd.read_excel(xlsx)
	
# Import data : Text 
	txt = r'C:\Users\AMIT\Dev1_Model_Data.txt' # space delimiter (csv)
	df = pd.read_table(txt, delim_whitespace=True) #df = pd.read_csv(txt, sep='	')

	txt = r'C:\Users\AMIT\Dev1_Model_Data_v2.txt' # comma delimiter
	df = pd.read_csv(txt , sep=',' , header=None)

# Convert data : numpy to pandas
	df=pd.DataFrame(data_in_numpy)

# Convert data : pandas to numpy
	data_in_numpy=df.as_matrix()
	data_in_numpy=df.values

# Convert : Pandas Series to Dataframe
	contents = pd.DataFrame(df.dtypes,columns = ['type'])
	#contents['variable'] = cont.index #index as column
	contents=contents.reset_index()
	contents.rename(columns = {'index':'variable'}, inplace = True)
	
# Convert : Pandas to List
	df.values.tolist()
	
###### Export data #################################################################
# in CSV
	df.to_csv('file1.csv') 

# in Excel
	df.to_excel('File2.xlsx')
	df.to_excel('File3.xls')
# in TXT
	df.to_csv('file2.txt',  header=True, index=False, sep=',')
	
######### Basic data checks ########################################################

# contents
	df.info()
	# Pandas data type [object, int64, float64, bool, datetime64]  [timedelta, category]
		df.dtypes 
		dict(df.dtypes)
		list((df.dtypes[df.dtypes == np.object]).index) # String column
		df.select_dtypes(include=np.number).columns.tolist() # Numeric Variable list
	# Number of Unique values
		np.unique(df['reservation_n']).size
		df['reservation_n'].nunique()
		df.nunique()
	# Proc contents
		contents = pd.DataFrame(df.dtypes,columns = ['type'])
		contents = contents.reset_index()
		contents.rename(columns = {'index':'variable'}, inplace = True)
		unique = pd.DataFrame(df.nunique(),columns = ['unique_values'])
		unique = unique.reset_index()
		unique.rename(columns = {'index':'variable'}, inplace = True)
		contents = pd.merge(contents, unique, how='left', on=['variable'])
		contents
	
# No. of Row & Column
	df.shape

# Variable list 		
	df.columns

## Print sample 
	# Print 20 obs
		df.head()
	# Print 20 obs
		df.head(20)
	# Print selected columns
		df[['reservation', 'dd']].head(10)	
	# Print last obs
		df.tail(20)
		


# Numeric variable distribution
	df.describe()

# Freq
	# Single
		df['Sex'].value_counts() 
		df.groupby('Sex').size()
	# Cross Freq
		a.groupby(['time_period', 'hh']).hh.count().unstack().fillna(0)

# Mean
	# single variable
		mean = df['float_col'].mean()
	# aggregate
		df1.mean()
		pd.pivot_table(df, index= 'time_period',  values= "Factor_Value" , aggfunc=np.mean)
		pd.pivot_table(df, index=['time_period'], values=['Factor_Value'], aggfunc={'Factor_Value':len,'Factor_Value':[np.sum, np.mean]},fill_value=0)
		pd.pivot_table(df, index=['time_period'], columns=['3rd_dimension_existing_var'], values=['Factor_Value'], aggfunc={'Factor_Value':len,'Factor_Value':[np.sum, np.mean]},fill_value=0)

# Global Variable
	global test_var	# test_var is a global variable
	
# Random number
	df['Random_score'] = np.random.randint(0,1000,size=(len(df),1))
	df['Random_variable'] = np.random.uniform(size=df.shape[0])
	
# Sampling 
	df1 = df.sample(frac =.7) 

# Rename Variable
	dev.rename(columns = {score_variable:'score'}, inplace = True)
	df.rename(columns={ df.columns[1]: "score" })
	
####### Missing ##################################################################		

# Total missing count
	df.isnull().sum().sum()
# Missing count by variables
	df.isnull().sum()
# Missing count by variables in pandas data frame
	pd.DataFrame(df.isnull().sum().to_dict().items(), columns=['var','missing'])
# Missing count by row
	df.apply(lambda x: sum(x.isnull().values), axis = 1)
# Fill Missing		
	df3['float_col'].fillna(mean)
# Drop Missing		
	df.dropna()
# subset data without missing
	df1 = df[~df['Value_FINAL_l4'].isnull()]
	df1 = df.dropna(axis=0, subset=['Value_FINAL_l4'])
	
	# Missing row but not NAN 
	nan_value = float("NaN")
	df1.replace("", nan_value, inplace=True)
	df1.dropna(subset = ['Metrics'], inplace=True)

# subset data only for missing
	df2 = df[df['Value_FINAL_l4'].isnull()] 

######### Create/Drop variable ###################################################
# Drop a column		
	input_data=input_data.drop('reservation',1)

# Create new variable
	a['c'] = a.apply(lambda row: row.stand_factor_value + row.Factor_Value, axis=1)
	a['d'] = a['c'] - (20.1 * df['Factor_Value'])
	a['e'] = np.where(a['c']>=338, 'yes', 'no')
	a['f'] = [1500 if x >=338 else 800 for x in a['c']]
	a['g'] = np.where(a['c']>=338, 1,0)

	event_dictionary ={'Music' : 1500, 'Poetry' : 800, 'Comedy' : 1200} 
	df['Price'] = df['Event'].map(event_dictionary)
	
# variable category
	def ff(row):
	  if row['c'] > 339: 
		val = 0
	  elif row['c'] in [999,888]: 
		val = 1
	  else: 
		val = -1
	  return val
	a['hh'] = a.apply(ff, axis=1)

# Generate sequence number from Index
	df['flag']=df.index
	
# Generate condition based sequence number i.e. monthly customer information 
	df['constant'] = 1
	df['sn'] = df.groupby(['customer'])['constant'].cumsum()
	
####### Bin of a variable assigned to another variable ############################
ser, bins = pd.qcut(df["final_score"], 10, retbins=True, labels=False)
df['binned'] = pd.cut(df['final_score'], bins) # add catogery in this 
df.head()
	
######## Subset data ###############################################################
#Filter column		
	df.loc[:, 'City']				# .loc is to access row and column together		
	df.loc[:, ['City', 'State']]
# Few row and columns
	a=df.loc[0:5,['time_period','stand_factor_value']]	 
	b=df.loc[0:5,['time_period','Factor_Value']]
# in between all columns
	c=df.loc[0:2,'repayment':'portfolio_code']
# Based on condition
	x = df[train_df['A'] == 3]

######### Append / Index ########################################################
# Append data
	old_data_frame = pd.concat([old_data_frame,new_record])

# Set index by using columns  
	df1 = df.set_index('two')

# Reset Index		
	df=df.reset_index()

# Sort
	a.sort_values(by=['Factor_Value'], inplace=True, ascending=True)
	a.sort_values(by=['Factor_Value'], inplace=True, ascending=False)

###### Duplicate #################################################################
# Duplicate daatset 
	df_copy = df.copy()
	
# Check Duplicate
	a['time_period'].duplicated().any()
	
# Count of duplicate
	a.duplicated(subset=['time_period'], keep='first').sum()
	
# Row level duplicate
	a.duplicated(subset=None, keep='first').sum()
	
# Drop duplicate by column
	a.drop_duplicates(subset =["time_period"], keep = False, inplace = True)
	b.drop_duplicates(subset =["time_period"], keep = False, inplace = True) 
	
# Drop rows where all data is the same
	my_dataframe = my_dataframe.drop_duplicates()

# Drop column
	a.drop(['Discounted_Price','elderly','Price'],1)
	
######## Lag, Lead, First dot and Last Dot ###################################################
# Lag 
	df.reservation.shift())
# Lead
	df.reservation.shift(-1))
	
# First dot and Last Dot
	df['flag'] = ((df.reservation != df.reservation.shift()) | (df.reservation != df.reservation.shift(-1))).astype(int)
	df['flag'] = np.where((df.reservation != df.reservation.shift()) | (df.reservation != df.reservation.shift(-1)), 1, 0)
	df.loc[ df.groupby('reservation',as_index=False).nth([0,-1]).index, 'flag' ] = 1

# Compare with previous value
	df.dd.eq(df.dd.shift())
	df.dd == df.dd.shift()
	
######## Merge ####################################################################
# Merge (inner,left,right,outer) https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html
	c = pd.merge(a, b, how='left', on=['time_period'],indicator='Key in dataset')
# pd.merge(left, right, how='inner', on=None, left_on=None, right_on=None,left_index=False, 
# right_index=False, sort=True,suffixes=('_x', '_y'), copy=True, indicator=False,validate=None)
# Merge multiple dataset https://stackoverflow.com/questions/44327999/python-pandas-merge-multiple-dataframes
# compile the list of dataframes you want to merge
	d = [a, b , c ]
	f = reduce(lambda left,right: pd.merge(left,right,on=['time_period'],how='left'), d)
		
######## Transpose ###################################################################
# using pivot
	a.pivot(index='time_period', columns='hh', values='Factor_Value').reset_index()
	pd.pivot_table(a, values = 'Factor_Value', index=['time_period'], columns = 'hh').reset_index()
# Transpose and aggregate
	pd.pivot_table(a, index= 'time_period', columns='hh' , values= "Factor_Value" , aggfunc=np.mean)
	pd.pivot_table(a, index=['time_period'], columns=['hh'], values=['Factor_Value'],aggfunc={'Factor_Value':len,'Factor_Value':[np.sum, np.mean]},fill_value=0)
		
######## Correlation ################################################################
# 2 variables
	df['counter'].corr(df['Factor_Value'])
# All variables
	a=df.corr()
	
####### SQL #########################################################################
q1 = """SELECT reservation, count(*) as total FROM df group by 1"""
print(ps.sqldf(q1, locals()))



######### Proc contents ##############################################################
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

contents = contents(input_data=df)

def var_type(input):
    if input['type'] == 'datetime64[ns]':
        val = 'NA'    
    elif input['unique_values'] <10: 
        val = 'char'
    elif input['type'] in ['object','bool']: 
        val = 'char'
    else: 
        val = 'num'
    return val

contents['treatment'] = contents.apply(var_type, axis=1)
contents

####### Lift table/KS #################################################################
def generate_lift_table(input_data=None, dependent_variable=None, score_variable=None, high_score_for_bad=False):
    ## Generate lift table, ks table
    temp = pd.DataFrame(input_data, columns=[dependent_variable, score_variable])
    temp.rename(columns = {dependent_variable:'response', score_variable:'score'}, inplace = True) 
    temp['non_response'] = 1 - temp['response'] #temp.response
    #DEFINE 10 BUCKETS WITH EQUAL SIZE
    try:
        temp['bucket'] = pd.qcut(temp.score, 10)
    except:
        temp['Rank'] = temp["score"].rank(method='first')
        temp['bucket'] = pd.qcut(temp.Rank, 10)
        #temp=temp.drop('Rank',1)
        #temp['bucket'] = pd.qcut(temp.score,len(temp.score.dropna()),duplicates='drop')
        
    #GROUP THE DATA FRAME BY BUCKETS
    grouped = temp.groupby('bucket', as_index = False)
    
    ####################################################################
    #CREATE A SUMMARY DATA FRAME
    agg1= pd.DataFrame()
    agg1['min_scr'] = grouped.min().score
    agg1['max_scr'] = grouped.max().score
    agg1['total'] = agg1['total'] = grouped.sum().response + grouped.sum().non_response
    agg1['pct_total'] = (agg1.total/agg1.total.sum())
    agg1['non_response'] = grouped.sum().non_response
    agg1['pct_non_response']= (agg1.non_response/agg1.non_response.sum())
    agg1['response'] = grouped.sum().response
    agg1['pct_response'] = (agg1.response/agg1.response.sum()).apply('{0:.2%}'.format)
    agg1['bad_rate'] = (agg1.response / agg1.total).apply('{0:.2%}'.format)
    agg1['odds']= (agg1.non_response / agg1.response).apply('{0:.2f}'.format)
    ##################################################################
    
    #SORT THE DATA FRAME BY SCORE
    if high_score_for_bad == True:
        lift_table = (agg1.sort_values(by = 'min_scr', ascending=False)).reset_index(drop = True)
    else:
        lift_table = (agg1.sort_values(by = 'min_scr', ascending=True)).reset_index(drop = True)
    
    lift_table['cum_response'] = lift_table.response.cumsum()
    lift_table['cum_non_response'] = lift_table.non_response.cumsum()
    lift_table['cum_pct_response'] = (lift_table.cum_response/lift_table.response.sum())
    lift_table['cum_pct_non_response']= (lift_table.cum_non_response/lift_table.non_response.sum()).apply('{0:.2%}'.format)
    #CALCULATE KS STATISTIC
    lift_table['ks'] = np.round(((lift_table.cum_non_response/lift_table.non_response.sum()) - (lift_table.cum_response/lift_table.response.sum()))*100,2).abs()
    #DEFINE A FUNCTION TO FLAG MAX KS
    flag = lambda x: '<----' if x == lift_table.ks.max() else ''
    #FLAG OUT MAX KS
    lift_table['max_ks'] = lift_table.ks.apply(flag)
    #accuracy ratio (AR) using ROC method 
    lift_table['AUC_by_ROC'] = ((lift_table['cum_pct_response'] + lift_table.cum_pct_response.shift().fillna(0))*lift_table['pct_non_response'])/2
    #accuracy ratio (AR) using CAP method
    lift_table['AUC_by_CAP'] = ((lift_table['cum_pct_response'] + lift_table.cum_pct_response.shift().fillna(0))*lift_table['pct_total'])/2
    # AR is same from ROC and CAP method. Using only ROC for final AR 
    lift_table['AR']= ((2*lift_table.AUC_by_ROC.sum())-1)
    lift_table['AR']= (lift_table['AR']).apply('{0:.2%}'.format)
    lift_table['AUC_by_ROC']= (lift_table['AUC_by_ROC']).apply('{0:.2%}'.format)
    lift_table['AUC_by_CAP']= (lift_table['AUC_by_CAP']).apply('{0:.2%}'.format)
    lift_table['cum_pct_response'] = (lift_table['cum_pct_response']).apply('{0:.2%}'.format)
    lift_table['pct_non_response']= (lift_table['pct_non_response']).apply('{0:.2%}'.format)
    lift_table['pct_total'] = (lift_table['pct_total']).apply('{0:.2%}'.format)
    lift_table['min_scr'] = lift_table['min_scr'].round(decimals=4)
    lift_table['max_scr'] = lift_table['max_scr'].round(decimals=4)
    return lift_table
    
temp=generate_lift_table(input_data=df, dependent_variable='reservation', score_variable='prob',high_score_for_bad=True)
temp

### PSI ##########################################################################################
def PSI(development_data=None , validation_data=None, score_variable=None, number_of_bins=10 ):
	####### Dev ##################################################
	dev = pd.DataFrame(development_data, columns=[score_variable])
	dev.rename(columns = {score_variable:'score'}, inplace = True)
	#Create score bin
	ser, score_bins = pd.qcut(dev['score'], number_of_bins, retbins=True, labels=range(0,number_of_bins))
	#Apply above score bin on dev data 
	dev['binned'] = pd.cut(dev['score'], score_bins,labels=range(0,number_of_bins))
	dev_grouped = dev.groupby('binned', as_index = False)
	#CREATE A SUMMARY DATA FRAME
	dev_agg= pd.DataFrame()
	dev_agg['Group'] = dev_grouped.min().binned
	dev_agg['Min_Score_Dev'] = dev_grouped.min().score
	dev_agg['Max_Score_Dev'] = dev_grouped.max().score
	dev_agg['Total_Dev'] = dev_grouped.count().score
	dev_agg['Pct_Total_Dev'] = dev_agg.Total_Dev/dev_agg.Total_Dev.sum()
	####### Val ##################################################
	#Apply above same score bin on val data
	val = pd.DataFrame(validation_data, columns=[score_variable])
	val.rename(columns = {score_variable:'score'}, inplace = True)
	val['binned'] = pd.cut(val['score'], score_bins,labels=range(0,number_of_bins))
	val_grouped = val.groupby('binned', as_index = False)
	#CREATE A SUMMARY DATA FRAME
	val_agg= pd.DataFrame()
	val_agg['Group'] = val_grouped.min().binned
	val_agg['Min_Score_Val'] = val_grouped.min().score
	val_agg['Max_Score_Val'] = val_grouped.max().score
	val_agg['Total_Val'] = val_grouped.count().score
	val_agg['Pct_Total_Val'] = val_agg.Total_Val/val_agg.Total_Val.sum()
	###########################################################
	dev_val_agg = pd.merge(dev_agg, val_agg, how='left', on=['Group'],indicator='Key in dataset')
	dev_val_agg['PSI'] = ((dev_agg['Pct_Total_Dev'] - val_agg['Pct_Total_Val'])*(np.log(dev_agg['Pct_Total_Dev'] / val_agg['Pct_Total_Val'])))
	dev_val_agg=dev_val_agg.drop(['Group','Min_Score_Val','Max_Score_Val','Key in dataset'],1)
	total=(pd.DataFrame((dev_val_agg.apply(np.sum)).values, index=(dev_val_agg.apply(np.sum)).keys()).T).assign(Min_Score_Dev=None, Max_Score_Dev=None)
	dev_val_agg=dev_val_agg.append(total, ignore_index=True)
	dev_val_agg['PSI'] = (dev_val_agg['PSI']).apply('{0:.1%}'.format)
	dev_val_agg['Pct_Total_Dev'] =(dev_val_agg['Pct_Total_Dev']).apply('{0:.0%}'.format)
	dev_val_agg['Pct_Total_Val'] =(dev_val_agg['Pct_Total_Val']).apply('{0:.0%}'.format)
	#############################################################
	return dev_val_agg

dev=PSI(development_data=df , validation_data=df1, score_variable='final_score', number_of_bins=10 )

###  ##########################################################################################


# Another way to define argument in function
#def calc_cumulative_gains(df: pd.DataFrame, actual_col: str, predicted_col:str, probability_col:str):

#notmiss = df[['binned']][df.binned.notnull()]
#notmiss.isnull().sum()
#miss = df1[['X','Y']][df1.X.isnull()]
#notmiss = df1[['X','Y']][df1.X.notnull()]

	


#####################################################################################
# End Of Code
#####################################################################################
