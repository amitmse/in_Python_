####################################################################################################################

#######################################################################
### Get list of file with owner name, file size #######################
#######################################################################
import os
import win32api
import win32con
import win32security
import csv
import pandas as pd
import time
#######################################################################
def list_of_files(data_location, output_location, output):
    ##### Change Directory #############################################
    os.getcwd()
    os.chdir(data_location)
    os.listdir()   
    ##### Get list of files from folder and sub-folder ##################################
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(data_location):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]

    #### Add owner of file, size of file and last access timestamp ######################
    Final = list()
    for filename in listOfFiles:
        #name, domain, type = win32security.LookupAccountSid(None, win32security.GetFileSecurity(filename, win32security.OWNER_SECURITY_INFORMATION).GetSecurityDescriptorOwner())
        Final.append(tuple([filename]+[int(os.path.getsize(filename))]+[time.strftime('%Y-%m-%d', time.localtime(os.path.getatime(filename)))]))

    #### Convert data in pandas and export in csv #######################################    
    labels = ['File', 'Size in bytes', 'Last Access Time']
    df = pd.DataFrame.from_records(Final, columns=labels)
    df.rename(columns = {'File':'Full_File_location'}, inplace = True)
    df['file_name'] = df['Full_File_location'].str.split('\\').str[-1]
    df['location'] = df.apply(lambda row : row['Full_File_location'].replace(str(row['file_name']), ''), axis=1)
    df['File_extension'] = df['file_name'].str.split('.').str[-1]
    df = df[['file_name', 'location', 'Full_File_location', 'File_extension', 'Size in bytes', 'Last Access Time']]

    ##### Save CSV file ###########################
    os.chdir(output_location)
    #output  = 'List_of_Files_Data_Tokenization.csv'
    df.to_csv(output, encoding='utf-8', index=False)
    df.head()
    ##### END of Function ############################################################################

##################################################################################################################

#### Call Function ###############################################################################################
if name == '__main__':
    data_location=r"C:\Users"
    output_location=r"C:\Users\"
    output  = 'List_of_Files.csv'
    list_of_files(data_location, output_location, output)

#################################################################################################################

##############################################################################################################

### Get list of file with owner name, file size
	import os
	import win32api
	import win32con
	import win32security
	import csv
	import pandas as pd
	import time

# Change below
	os.getcwd() #Current directory
	dirName = r"C:\Users\"
	os.chdir(dirName)
	os.listdir()
	output  = 'test.csv'

#Get list of files from folder and sub-folder
	listOfFiles = list()
	for (dirpath, dirnames, filenames) in os.walk(dirName):
		listOfFiles += [os.path.join(dirpath, file) for file in filenames]
#Add owner of file, size of file and last access timestamp   
	Final = list()
	for filename in listOfFiles:
		name, domain, type = win32security.LookupAccountSid(None, win32security.GetFileSecurity(filename, win32security.OWNER_SECURITY_INFORMATION).GetSecurityDescriptorOwner())
		Final.append(tuple([filename]+[name.encode("utf-8")]+[int(os.path.getsize(filename))]+[time.strftime('%Y-%m-%d', time.localtime(os.path.getatime(filename)))]))

#Convert data in pandas and export in csv
	labels = ['File', 'Owner - ID', 'Size in bytes', 'Last Access Time']
	df = pd.DataFrame.from_records(Final, columns=labels)
	df.to_csv(output, encoding='utf-8', index=False)
	df.head()
	/******* Zip a file/folder  using Python ***********************************************************************/
	os.chdir(r'C:\Users\')

-------------------------------

#Zip a file

	import os

	import zipfile

	jungle_zip = zipfile.ZipFile('jungle.zip', 'w')

	jungle_zip.write('9781441996121-c2.pdf', compress_type=zipfile.ZIP_DEFLATED)

	jungle_zip.close()

-------------------------------

#Zip All PDF Files

	import os
	import zipfile
	fantasy_zip = zipfile.ZipFile('archive.zip', 'w')
	for folder, subfolders, files in os.walk(r'C:\Users'):
		for file in files:
			if file.endswith('.pdf'):
				fantasy_zip.write(os.path.join(folder, file), os.path.relpath(os.path.join(folder,file), r'C:\Users\Downloads\Copy'), compress_type = zipfile.ZIP_DEFLATED)    
	fantasy_zip.close()

-------------------------------

#Zip a folder

	import shutil
	os.chdir(r'C:\Users')
	# Copy is the folder which is getting zipped
	shutil.make_archive('filename', 'zip', 'Copy')

-------------------------------

#Zip a Folder

	import os,zipfile
	# Change the directory where you want your new zip file to be
	os.chdir(r'C:\Users')
	zf = zipfile.ZipFile('myfile.zip','w')
	# Copy is the folder which is getting zipped
	for dirnames,folders,files in os.walk('Copy'):
		zf.write('Copy')
		for file in files:
			zf.write(os.path.join('Copy',file))	
	zf.close()

	---------------------------------------

	import os
	import sys
	import zipfile
	source_dir = r'C:\Users'  
	dest_dir = r'C:\Users'
	os.chdir(dest_dir)
	def csv_files(source_dir):
		for filename in os.listdir(source_dir):
			if filename.endswith('.sas7bdat'):
				yield filename
				#print (filename)

	#csv_files(r'C:\Users')

	for csv_filename in csv_files(source_dir):
		file_root = os.path.splitext(csv_filename)[0]
		zip_file_name = file_root + '.zip'
		zip_file_path = os.path.join(dest_dir, zip_file_name)
		with zipfile.ZipFile(zip_file_path, mode='w') as zf:
			zf.write(csv_filename, compress_type=zipfile.ZIP_DEFLATED)

---------------------------------------
#Zip individual file. Folder and sub-folders
	# https://stackoverflow.com/questions/43881491/how-to-recursively-zip-multiple-folders-as-individual-zip-files
	import os
	import zipfile
	start_path = r'C:\Users'
	file_type = ".sas7bdat"
	def zipdir(start_path):
		dir_count = 0
		file_count = 0
		for (path,dirs,files) in os.walk(start_path):
			print('Directory: {:s}'.format(path))
			dir_count += 1
			for file in files:
				if file.endswith(file_type): 
					file_path = os.path.join(path, file)
					print('\nAttempting to zip: \'{}\''.format(file_path))
					with zipfile.ZipFile(file_path + '.zip', 'w', zipfile.ZIP_DEFLATED) as ziph:
						ziph.write(file_path, file)
					print('Done')
					file_count += 1
		print('\nProcessed {} files in {} directories.'.format(file_count,dir_count))
		
	if name == '__main__':
		zipdir(start_path)

-------------------------------------------------------------------------------------------

#Unzip a file
	import zipfile
	zip_ref = zipfile.ZipFile(r'C:\Users', 'r')
	zip_ref.extractall(r'C:\Users')
	zip_ref.close()

---------------------------------------

#Unzip a folder

	import os, zipfile
	dir_name = 'C:\Users'
	extension = ".zip"
	os.chdir(dir_name) # change directory from working dir to dir with files
	for item in os.listdir(dir_name): 			# loop through items in dir
		if item.endswith(extension): 			# check for ".zip" extension
			file_name = os.path.abspath(item) 	# get full path of files
			zip_ref = zipfile.ZipFile(file_name)    # create zipfile object
			zip_ref.extractall(dir_name) 		# extract file to dir
			zip_ref.close() 			# close file
			#os.remove(file_name) 			# delete zipped file

---------------------------------------------------------

#Unzip .7z file
	from pyunpack import Archive
	import os
	#Not requredimport patool
	os.chdir(r'C:\Users')
	Archive('qc.7z').extractall("")
----------------------------------------------------------

#Unzip .7z folder

	import os
	from pyunpack import Archive
	dir_name = r'C:\Users'
	extension = ".7z"
	os.chdir(dir_name) # change directory from working dir to dir with files
	for item in os.listdir(dir_name): 			# loop through items in dir
		if item.endswith(extension): 			# check for ".zip" extension
			file_name = os.path.abspath(item) 	# get full path of files
			Archive(file_name).extractall("")

------------------------------------------------------------------------------------------

#Copy data from one dir to another dir
	import shutil
	shutil.copy2(r'C:\Users\Downloads\9781441996121-c2.pdf', r'C:\Users\Downloads\test\9781441996121-c2.pdf')
---------------------------------------

#Copy a folder one dir to another dir
	from distutils.dir_util import copy_tree
	---------------------------------------
	fromDirectory = r'C:\Users\Downloads\Copy'
	toDirectory = r'C:\Users\Downloads\test'
	copy_tree(fromDirectory, toDirectory)

------ Lambda ------------------------------------------------------------------------------------

	#Lambda function is a small anonymous function. It can take any number of arguments, but can only have one expression.
	#lambda 	arguments 		: expression
	#x 		= 	lambda a, b 	: a * b
	#print(x(5, 6))
	miss 	= df1[['X','Y']][df1.X.isnull()]
	notmiss = df1[['X','Y']][df1.X.notnull()]
	“from sklearn.feature_selection import RFE
	rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=5)”

------------------------------------------------------------------------------------------

UTF-8 encoding	: # -*- coding: encoding -*-
detect errors	: PyChecker to detect errors in python code.
Find bugs	: pylint and pyflakes.
Decorator	: use @ symbol.
Join function	: ''.join(['John','Ray'])
Pass		: It is just a placeholder and doesn't execute any code or command.
Docstring	: To adding comments or summarizing a piece of code in Python. '__doc__' 
code debugging	: pdb
###################################################################################################################

###################################################################################################################
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

##########################################################

# pyarrow fastparquet
#python - How to select and run model from dash dropdown menu and update confusion matrix figure? - Stack Overflow
#sqlite - Issue with Dropdown pulling data from SQL in Python Dash - Stack Overflow
#What are some alternatives to Dask? - StackShare
#Scale your pandas workflow by changing a single line of code — Modin 0.32.0+0.g3e951a6.dirty documentation
