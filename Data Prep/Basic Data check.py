####### Install Library ############################################################

pip install sas7bdat

####### Import Library ############################################################
from sas7bdat import SAS7BDAT
import pandas as pd
import numpy as np
import os

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
	
###### Export data #################################################################
# in CSV
	df.to_csv('file1.csv') 

# in Excel
	df.to_excel('File2.xlsx')
	df.to_excel('File3.xls')
# in TXT
	df.to_csv('file2.txt',  header=True, index=False, sep=',')
	
######### Basic data checks ########################################################

# No. of Row & Column
	df.shape

# Variable list 		
	df.columns

## Print sample 
	# Print 20 obs
		df.head()
	# Print 20 obs
		df.head(20)

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
		pd.pivot_table(a, index= 'time_period', columns='hh' , values= "Factor_Value" , aggfunc=np.mean)
		pd.pivot_table(a, index=['time_period'], columns=['hh'], values=['Factor_Value'],aggfunc={'Factor_Value':len,'Factor_Value':[np.sum, np.mean]},fill_value=0)

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
#Drop Missing		
	df.dropna()
#subset data without missing
	df1 = df[~df['Value_FINAL_l4'].isnull()]
	df1 = df.dropna(axis=0, subset=['Value_FINAL_l4'])
#subset data only for missing
	df2 = df[df['Value_FINAL_l4'].isnull()] 

######### Create/Drop variable ###################################################
#Drop a column		
	input_data=input_data.drop('reservation',1)

# Create new variable
	a['c'] = a.apply(lambda row: row.stand_factor_value + row.Factor_Value, axis=1)
	a['d'] = a['c'] - (20.1 * df['Factor_Value'])
	a['e'] = np.where(a['c']>=338, 'yes', 'no')
	a['f'] = [1500 if x >=338 else 800 for x in a['c']]
	a['g'] = np.where(a['c']>=338, 1,0)

	event_dictionary ={'Music' : 1500, 'Poetry' : 800, 'Comedy' : 1200} 
	df['Price'] = df['Event'].map(event_dictionary)
	
#variable category
	def ff(row):
	  if row['c'] > 339: 
		val = 0
	  elif row['c'] > 338: 
		val = 1
	  else: 
		val = -1
	  return val
	a['hh'] = a.apply(ff, axis=1)

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
	df.reset_index()

# Sort
	a.sort_values(by=['Factor_Value'], inplace=True, ascending=True)
	a.sort_values(by=['Factor_Value'], inplace=True, ascending=False)


###### Duplicate #################################################################
# Check Duplicate
	a['time_period'].duplicated().any()
#Count of duplicate
	a.duplicated(subset=['time_period'], keep='first').sum()
#Row level duplicate
	a.duplicated(subset=None, keep='first').sum()
# Drop duplicate by column
	a.drop_duplicates(subset =["time_period"], keep = False, inplace = True)
	b.drop_duplicates(subset =["time_period"], keep = False, inplace = True) 
# Drop rows where all data is the same
	my_dataframe = my_dataframe.drop_duplicates()

# Drop column
	a.drop(['Discounted_Price','elderly','Price'],1)

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
		
#####################################################################################
# End Of Code
#####################################################################################
