#####################################################################################################################
## 1. This Python code will read the data from different location based on certain condition for same month.
## 2. Add all link variable
## 3. Create a csv file with two columns i.e. key variable and summation of all link variables
#####################################################################################################################

#### python libraries ###############################################################################################
import os
import pandas as pd
import numpy as np
#########################################################################

##### Use Input ####################################################################
## Provide Path for folder for input & output
input_folder 	= 'C:/Users/amit.kumar/Downloads/Project'
output_folder 	= 'C:/Users/amit.kumar/Downloads/Project'
## Key variable which will be used in merging all dataset
key_var 		= 'dc_engn_ref_nb'
## Path for sample data to get the data type of key variable and other variable. This data will have only  two columns i.e. key varibale and link variable
path_for_sample_data = 'C:/Users/amit.kumar/Downloads/Project/revoke1-24book1-24/N1_/cp06.csv'
## store folder name. Folder name should start with "Link/revoke"
## Input folder. check the name in right hand side in string format i.e. 'revoke1-24book1-24'
revoke1_24book1_24 		= 'revoke1-24book1-24'
revoke1_24book25_36 	= 'revoke1-24book25-36'
revoke1_24book37_48 	= 'revoke1-24book37-48'
revoke1_24book49_999 	= 'revoke1-24book49-999'
revoke25_36book25_36 	= 'revoke25-36book25-36'
revoke25_36book37_48 	= 'revoke25-36book37-48'
revoke25_36book49_999 	= 'revoke25-36book49-999'
revoke37_48book37_48 	= 'revoke37-48book37-48'
revoke37_48book49_999 	= 'revoke37-48book49-999'

## output folder. check the name in right hand side in string format i.e. 'Link24'
Link24 	= 'Link24'
Link24z = 'Link24z'
Link36 	= 'Link36'
Link36z = 'Link36z'
Link48 	= 'Link48'
Link48z = 'Link48z'

##### Create List of folders#######################
## input_folder_level_1 = "Link24"
## change current path
## os.chdir( input_folder + '/' + input_folder_level_1 )
## os.getcwd()
## list of sub folders
## subdirectories_level_1 = os.listdir(input_folder)
## store monthly folder name 
## subdirectories_level_2 = os.listdir(input_folder + '/' +  list_input_folder_level_1[1])
## Dictionary for folder name
## list_output_folder_level_1 	= [i for i in subdirectories_level_1 if "Link".upper() in i.upper()]
## list_input_folder_level_1	= [i for i in subdirectories_level_1 if "revoke".upper() in i.upper()]
## dic_subdirectories_level_2 = { int((a.replace('N','')).replace('_','')):a for a in os.listdir(input_folder + '/' +  list_input_folder_level_1[1])}
## print Dictionary for folder name :  for key, value in dic_subdirectories_level_2.iteritems(): print key, '\t', value

## Please change the "revoke1_24book1_24" in below code if folder structure ('N1_', 'N2_', 'N3_', 'N4_', 'N5_', 'N6_'........) is not available in that location and provide the correct one.
dic_subdirectories_level_2 = { int((a.replace('N','')).replace('_','')):a for a in os.listdir(input_folder + '/' +  revoke1_24book1_24)}

########################################################################################################

##### Create List of csv files  ########################################################################
## Raw file name without extension  ###############
def get_file_names(directory):
		# http://stackoverflow.com/questions/3207219/how-to-list-all-files-of-a-directory-in-python
		raw_csv_file_name_list = []
		for root, directories, files in os.walk(directory):
			for filename in files:
				raw_csv_file_name_list.append(filename.replace('.csv',''))
		return raw_csv_file_name_list

## Unique file name			
raw_csv_file_name = list(set(get_file_names(input_folder)))

## get the format of key / link variable
sample_data=pd.read_csv(path_for_sample_data)
sample_data_columns=list(sample_data.columns)
sample_data_columns.remove(key_var)
#########################################################################

#########################################################################
### read csv files and merge all and create a summation variable from all link variables and then save it to 6 different folders in respective months

## 1. Link24: 
for file_nm in range(len(raw_csv_file_name)):

		for folder_nm in dic_subdirectories_level_2.keys():
		
				## read csv through pandas. If data not available then replace with empty data to avoid any error
				## Empty data:  pd.DataFrame(columns=[key_var], dtype='int64')
				try: 	f_24_1 = pd.read_csv(input_folder+'/'+revoke1_24book1_24+'/'+dic_subdirectories_level_2[folder_nm]+'/'+raw_csv_file_name[file_nm]+'.csv')
				except: f_24_1 = pd.DataFrame(np.empty(0, dtype=[(key_var, sample_data[key_var].dtypes), (raw_csv_file_name[file_nm], sample_data[sample_data_columns[0]].dtypes)]))

				try: 	f_24_2 = pd.read_csv(input_folder+'/'+revoke1_24book25_36+'/'+dic_subdirectories_level_2[folder_nm]+'/'+raw_csv_file_name[file_nm]+'.csv')
				except: f_24_2 = pd.DataFrame(np.empty(0, dtype=[(key_var, sample_data[key_var].dtypes), (raw_csv_file_name[file_nm], sample_data[sample_data_columns[0]].dtypes)]))

				try: 	f_24_3 = pd.read_csv(input_folder+'/'+revoke1_24book37_48+'/'+dic_subdirectories_level_2[folder_nm]+'/'+raw_csv_file_name[file_nm]+'.csv')
				except: f_24_3 = pd.DataFrame(np.empty(0, dtype=[(key_var, sample_data[key_var].dtypes), (raw_csv_file_name[file_nm], sample_data[sample_data_columns[0]].dtypes)]))

				try: 	f_24_4 = pd.read_csv(input_folder+'/'+revoke1_24book49_999+'/'+dic_subdirectories_level_2[folder_nm]+'/'+raw_csv_file_name[file_nm]+'.csv')
				except: f_24_4 = pd.DataFrame(np.empty(0, dtype=[(key_var, sample_data[key_var].dtypes), (raw_csv_file_name[file_nm], sample_data[sample_data_columns[0]].dtypes)]))

				## Merge all data 
				temp_1 = pd.merge(pd.merge(pd.merge(f_24_1,f_24_2,on=key_var,how = 'outer'),f_24_3,on=key_var,how = 'outer'),f_24_4,on=key_var,how = 'outer')
				## Change column name as its same
				temp_1.columns=[key_var, 'a1', 'a2', 'a3', 'a4']
				## save column name to add all column
				sum_var_list=list(temp_1)
				## Remove key var from summation var list name
				sum_var_list.remove(key_var)
				## Sum all columns
				temp_1 = temp_1.fillna(0)
				temp_1[raw_csv_file_name[file_nm]] = temp_1[sum_var_list].sum(axis=1)
				## temp_1[raw_csv_file_name[file_nm]] = temp_1.fillna(0)['a1'] + temp_1.fillna(0)['a2'] + temp_1.fillna(0)['a3'] + temp_1.fillna(0)['a4']
				## Drop rest of columns as won't need them
				temp_1 = temp_1.drop(sum_var_list,1)
				## Create directory if does not exists
				if not os.path.exists(output_folder+'/'+Link24+'/'+dic_subdirectories_level_2[folder_nm]): os.makedirs(output_folder+'/'+Link24+'/'+dic_subdirectories_level_2[folder_nm])
				## Save the data to link folders
				temp_1.to_csv(output_folder+'/'+Link24+'/'+dic_subdirectories_level_2[folder_nm]+'/'+raw_csv_file_name[file_nm]+'.csv',index=False)
				## delete all temporary data set
				del f_24_1, f_24_2, f_24_3, f_24_4, temp_1
				del sum_var_list
#########################################################################

#########################################################################
## 2. Link24z: 
for file_nm in range(len(raw_csv_file_name)):

		for folder_nm in dic_subdirectories_level_2.keys():
		
				try: 	f_24_1 = pd.read_csv(input_folder+'/'+revoke1_24book1_24+'/'+dic_subdirectories_level_2[folder_nm]+'/'+raw_csv_file_name[file_nm]+'.csv')
				except: f_24_1 = pd.DataFrame(np.empty(0, dtype=[(key_var, sample_data[key_var].dtypes), (raw_csv_file_name[file_nm], sample_data[sample_data_columns[0]].dtypes)]))

				## Save the data to link folders
				if not os.path.exists(output_folder+'/'+Link24z+'/'+dic_subdirectories_level_2[folder_nm]): os.makedirs(output_folder+'/'+Link24z+'/'+dic_subdirectories_level_2[folder_nm])	
				f_24_1.to_csv(output_folder+'/'+Link24z+'/'+dic_subdirectories_level_2[folder_nm]+'/'+raw_csv_file_name[file_nm]+'.csv',index=False)
				## delete all temporary data set
				del f_24_1
#########################################################################

#########################################################################
## 3. Link36: 
for file_nm in range(len(raw_csv_file_name)):

		for folder_nm in dic_subdirectories_level_2.keys():
		
				try: 	f_24_1 = pd.read_csv(input_folder+'/'+revoke1_24book1_24+'/'+dic_subdirectories_level_2[folder_nm]+'/'+raw_csv_file_name[file_nm]+'.csv')
				except: f_24_1 = pd.DataFrame(np.empty(0, dtype=[(key_var, sample_data[key_var].dtypes), (raw_csv_file_name[file_nm], sample_data[sample_data_columns[0]].dtypes)]))

				try: 	f_24_2 = pd.read_csv(input_folder+'/'+revoke1_24book25_36+'/'+dic_subdirectories_level_2[folder_nm]+'/'+raw_csv_file_name[file_nm]+'.csv')
				except: f_24_2 = pd.DataFrame(np.empty(0, dtype=[(key_var, sample_data[key_var].dtypes), (raw_csv_file_name[file_nm], sample_data[sample_data_columns[0]].dtypes)]))

				try: 	f_24_3 = pd.read_csv(input_folder+'/'+revoke1_24book37_48+'/'+dic_subdirectories_level_2[folder_nm]+'/'+raw_csv_file_name[file_nm]+'.csv')
				except: f_24_3 = pd.DataFrame(np.empty(0, dtype=[(key_var, sample_data[key_var].dtypes), (raw_csv_file_name[file_nm], sample_data[sample_data_columns[0]].dtypes)]))

				try: 	f_24_4 = pd.read_csv(input_folder+'/'+revoke1_24book49_999+'/'+dic_subdirectories_level_2[folder_nm]+'/'+raw_csv_file_name[file_nm]+'.csv')
				except: f_24_4 = pd.DataFrame(np.empty(0, dtype=[(key_var, sample_data[key_var].dtypes), (raw_csv_file_name[file_nm], sample_data[sample_data_columns[0]].dtypes)]))
				
				try: 	f_24_5 = pd.read_csv(input_folder+'/'+revoke25_36book25_36+'/'+dic_subdirectories_level_2[folder_nm]+'/'+raw_csv_file_name[file_nm]+'.csv')
				except: f_24_5 = pd.DataFrame(np.empty(0, dtype=[(key_var, sample_data[key_var].dtypes), (raw_csv_file_name[file_nm], sample_data[sample_data_columns[0]].dtypes)]))

				try: 	f_24_6 = pd.read_csv(input_folder+'/'+revoke25_36book37_48+'/'+dic_subdirectories_level_2[folder_nm]+'/'+raw_csv_file_name[file_nm]+'.csv')
				except: f_24_6 = pd.DataFrame(np.empty(0, dtype=[(key_var, sample_data[key_var].dtypes), (raw_csv_file_name[file_nm], sample_data[sample_data_columns[0]].dtypes)]))

				try: 	f_24_7 = pd.read_csv(input_folder+'/'+revoke25_36book49_999+'/'+dic_subdirectories_level_2[folder_nm]+'/'+raw_csv_file_name[file_nm]+'.csv')
				except: f_24_7 = pd.DataFrame(np.empty(0, dtype=[(key_var, sample_data[key_var].dtypes), (raw_csv_file_name[file_nm], sample_data[sample_data_columns[0]].dtypes)]))				
				
				## Merge all data 
				temp_1 = pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(f_24_1,f_24_2,on=key_var,how = 'outer'),f_24_3,on=key_var,how = 'outer'),f_24_4,on=key_var,how = 'outer'),f_24_5,on=key_var,how = 'outer'),f_24_6,on=key_var,how = 'outer'),f_24_7,on=key_var,how = 'outer')
				
				## Change column name as its same
				temp_1.columns=[key_var, 'a1', 'a2', 'a3', 'a4','a5','a6','a7']
				## save column name to add all column
				sum_var_list=list(temp_1)
				## Remove key var from summation var list name
				sum_var_list.remove(key_var)
				## Sum all columns
				temp_1 = temp_1.fillna(0)
				temp_1[raw_csv_file_name[file_nm]] = temp_1[sum_var_list].sum(axis=1)
				## temp_1[raw_csv_file_name[file_nm]] = temp_1.fillna(0)['a1'] + temp_1.fillna(0)['a2'] + temp_1.fillna(0)['a3'] + temp_1.fillna(0)['a4'] + temp_1.fillna(0)['a5'] + temp_1.fillna(0)['a6'] + temp_1.fillna(0)['a7']
				## Drop rest of columns as won't need them
				temp_1 = temp_1.drop(sum_var_list,1)
				## Save the data to link folders
				if not os.path.exists(output_folder+'/'+Link36+'/'+dic_subdirectories_level_2[folder_nm]): os.makedirs(output_folder+'/'+Link36+'/'+dic_subdirectories_level_2[folder_nm])
				temp_1.to_csv(output_folder+'/'+Link36+'/'+dic_subdirectories_level_2[folder_nm]+'/'+raw_csv_file_name[file_nm]+'.csv',index=False)
				## delete all temporary data set
				del f_24_1, f_24_2, f_24_3, f_24_4, f_24_5, f_24_6, f_24_7, temp_1
				del sum_var_list
#########################################################################

#########################################################################
## 4. Link36z: 
for file_nm in range(len(raw_csv_file_name)):

		for folder_nm in dic_subdirectories_level_2.keys():
		
				try: 	f_24_1 = pd.read_csv(input_folder+'/'+revoke1_24book1_24+'/'+dic_subdirectories_level_2[folder_nm]+'/'+raw_csv_file_name[file_nm]+'.csv')
				except: f_24_1 = pd.DataFrame(np.empty(0, dtype=[(key_var, sample_data[key_var].dtypes), (raw_csv_file_name[file_nm], sample_data[sample_data_columns[0]].dtypes)]))

				try: 	f_24_2 = pd.read_csv(input_folder+'/'+revoke1_24book25_36+'/'+dic_subdirectories_level_2[folder_nm]+'/'+raw_csv_file_name[file_nm]+'.csv')
				except: f_24_2 = pd.DataFrame(np.empty(0, dtype=[(key_var, sample_data[key_var].dtypes), (raw_csv_file_name[file_nm], sample_data[sample_data_columns[0]].dtypes)]))
				
				try: 	f_24_3 = pd.read_csv(input_folder+'/'+revoke25_36book25_36+'/'+dic_subdirectories_level_2[folder_nm]+'/'+raw_csv_file_name[file_nm]+'.csv')
				except: f_24_3 = pd.DataFrame(np.empty(0, dtype=[(key_var, sample_data[key_var].dtypes), (raw_csv_file_name[file_nm], sample_data[sample_data_columns[0]].dtypes)]))
				
				## Merge all data 
				temp_1 = pd.merge(pd.merge(f_24_1,f_24_2,on=key_var,how = 'outer'),f_24_3,on=key_var,how = 'outer')
												
				## Change column name as its same
				temp_1.columns=[key_var, 'a1', 'a2', 'a3']
				## save column name to add all column
				sum_var_list=list(temp_1)
				## Remove key var from summation var list name
				sum_var_list.remove(key_var)
				## Sum all columns
				temp_1 = temp_1.fillna(0)
				temp_1[raw_csv_file_name[file_nm]] = temp_1[sum_var_list].sum(axis=1)
				## temp_1[raw_csv_file_name[file_nm]] = temp_1.fillna(0)['a1'] + temp_1.fillna(0)['a2'] + temp_1.fillna(0)['a3']
				## Drop rest of columns as won't need them
				temp_1 = temp_1.drop(sum_var_list,1)
				## Save the data to link folders
				if not os.path.exists(output_folder+'/'+Link36z+'/'+dic_subdirectories_level_2[folder_nm]): os.makedirs(output_folder+'/'+Link36z+'/'+dic_subdirectories_level_2[folder_nm])
				temp_1.to_csv(output_folder+'/'+Link36z+'/'+dic_subdirectories_level_2[folder_nm]+'/'+raw_csv_file_name[file_nm]+'.csv',index=False)
				## delete all temporary data set
				del f_24_1, f_24_2, f_24_3, temp_1
				del sum_var_list
#########################################################################

#########################################################################
## 5. Link48: 
for file_nm in range(len(raw_csv_file_name)):

		for folder_nm in dic_subdirectories_level_2.keys():
		
				try: 	f_24_1 = pd.read_csv(input_folder+'/'+revoke1_24book1_24+'/'+dic_subdirectories_level_2[folder_nm]+'/'+raw_csv_file_name[file_nm]+'.csv')
				except: f_24_1 = pd.DataFrame(np.empty(0, dtype=[(key_var, sample_data[key_var].dtypes), (raw_csv_file_name[file_nm], sample_data[sample_data_columns[0]].dtypes)]))

				try: 	f_24_2 = pd.read_csv(input_folder+'/'+revoke1_24book25_36+'/'+dic_subdirectories_level_2[folder_nm]+'/'+raw_csv_file_name[file_nm]+'.csv')
				except: f_24_2 = pd.DataFrame(np.empty(0, dtype=[(key_var, sample_data[key_var].dtypes), (raw_csv_file_name[file_nm], sample_data[sample_data_columns[0]].dtypes)]))

				try: 	f_24_3 = pd.read_csv(input_folder+'/'+revoke1_24book37_48+'/'+dic_subdirectories_level_2[folder_nm]+'/'+raw_csv_file_name[file_nm]+'.csv')
				except: f_24_3 = pd.DataFrame(np.empty(0, dtype=[(key_var, sample_data[key_var].dtypes), (raw_csv_file_name[file_nm], sample_data[sample_data_columns[0]].dtypes)]))

				try: 	f_24_4 = pd.read_csv(input_folder+'/'+revoke1_24book49_999+'/'+dic_subdirectories_level_2[folder_nm]+'/'+raw_csv_file_name[file_nm]+'.csv')
				except: f_24_4 = pd.DataFrame(np.empty(0, dtype=[(key_var, sample_data[key_var].dtypes), (raw_csv_file_name[file_nm], sample_data[sample_data_columns[0]].dtypes)]))
				
				try: 	f_24_5 = pd.read_csv(input_folder+'/'+revoke25_36book25_36+'/'+dic_subdirectories_level_2[folder_nm]+'/'+raw_csv_file_name[file_nm]+'.csv')
				except: f_24_5 = pd.DataFrame(np.empty(0, dtype=[(key_var, sample_data[key_var].dtypes), (raw_csv_file_name[file_nm], sample_data[sample_data_columns[0]].dtypes)]))

				try: 	f_24_6 = pd.read_csv(input_folder+'/'+revoke25_36book37_48+'/'+dic_subdirectories_level_2[folder_nm]+'/'+raw_csv_file_name[file_nm]+'.csv')
				except: f_24_6 = pd.DataFrame(np.empty(0, dtype=[(key_var, sample_data[key_var].dtypes), (raw_csv_file_name[file_nm], sample_data[sample_data_columns[0]].dtypes)]))

				try: 	f_24_7 = pd.read_csv(input_folder+'/'+revoke25_36book49_999+'/'+dic_subdirectories_level_2[folder_nm]+'/'+raw_csv_file_name[file_nm]+'.csv')
				except: f_24_7 = pd.DataFrame(np.empty(0, dtype=[(key_var, sample_data[key_var].dtypes), (raw_csv_file_name[file_nm], sample_data[sample_data_columns[0]].dtypes)]))				

				try: 	f_24_8 = pd.read_csv(input_folder+'/'+revoke37_48book37_48+'/'+dic_subdirectories_level_2[folder_nm]+'/'+raw_csv_file_name[file_nm]+'.csv')
				except: f_24_8 = pd.DataFrame(np.empty(0, dtype=[(key_var, sample_data[key_var].dtypes), (raw_csv_file_name[file_nm], sample_data[sample_data_columns[0]].dtypes)]))				

				try: 	f_24_9 = pd.read_csv(input_folder+'/'+revoke37_48book49_999+'/'+dic_subdirectories_level_2[folder_nm]+'/'+raw_csv_file_name[file_nm]+'.csv')
				except: f_24_9 = pd.DataFrame(np.empty(0, dtype=[(key_var, sample_data[key_var].dtypes), (raw_csv_file_name[file_nm], sample_data[sample_data_columns[0]].dtypes)]))				
								
				## Merge all data 
				temp_1 = pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(f_24_1,f_24_2,on=key_var,how = 'outer'),f_24_3,on=key_var,how = 'outer'),f_24_4,on=key_var,how = 'outer'),f_24_5,on=key_var,how = 'outer'),f_24_6,on=key_var,how = 'outer'),f_24_7,on=key_var,how = 'outer'),f_24_8,on=key_var,how = 'outer'),f_24_9,on=key_var,how = 'outer')
				
				## Change column name as its same
				temp_1.columns=[key_var, 'a1', 'a2', 'a3', 'a4','a5','a6','a7','a8','a9']
				## save column name to add all column
				sum_var_list=list(temp_1)
				## Remove key var from summation var list name
				sum_var_list.remove(key_var)
				## Sum all columns
				temp_1 = temp_1.fillna(0)
				temp_1[raw_csv_file_name[file_nm]] = temp_1[sum_var_list].sum(axis=1)
				## temp_1[raw_csv_file_name[file_nm]] =  temp_1.fillna(0)['a1'] + temp_1.fillna(0)['a2'] + temp_1.fillna(0)['a3'] + temp_1.fillna(0)['a4'] + temp_1.fillna(0)['a5'] + temp_1.fillna(0)['a6'] + temp_1.fillna(0)['a7'] + temp_1.fillna(0)['a8'] + temp_1.fillna(0)['a9']
				## Drop rest of columns as won't need them
				temp_1 = temp_1.drop(sum_var_list,1)
				## Save the data to link folders
				if not os.path.exists(output_folder+'/'+Link48+'/'+dic_subdirectories_level_2[folder_nm]): os.makedirs(output_folder+'/'+Link48+'/'+dic_subdirectories_level_2[folder_nm])
				temp_1.to_csv(output_folder+'/'+Link48+'/'+dic_subdirectories_level_2[folder_nm]+'/'+raw_csv_file_name[file_nm]+'.csv',index=False)
				## delete all temporary data set
				del f_24_1, f_24_2, f_24_3, f_24_4, f_24_5, f_24_6, f_24_7, f_24_8, f_24_9, temp_1
				del sum_var_list
#########################################################################

#########################################################################
## 6. Link48z:
for file_nm in range(len(raw_csv_file_name)):

		for folder_nm in dic_subdirectories_level_2.keys():
		
				try: 	f_24_1 = pd.read_csv(input_folder+'/'+revoke1_24book1_24+'/'+dic_subdirectories_level_2[folder_nm]+'/'+raw_csv_file_name[file_nm]+'.csv')
				except: f_24_1 = pd.DataFrame(np.empty(0, dtype=[(key_var, sample_data[key_var].dtypes), (raw_csv_file_name[file_nm], sample_data[sample_data_columns[0]].dtypes)]))

				try: 	f_24_2 = pd.read_csv(input_folder+'/'+revoke1_24book25_36+'/'+dic_subdirectories_level_2[folder_nm]+'/'+raw_csv_file_name[file_nm]+'.csv')
				except: f_24_2 = pd.DataFrame(np.empty(0, dtype=[(key_var, sample_data[key_var].dtypes), (raw_csv_file_name[file_nm], sample_data[sample_data_columns[0]].dtypes)]))

				try: 	f_24_3 = pd.read_csv(input_folder+'/'+revoke1_24book37_48+'/'+dic_subdirectories_level_2[folder_nm]+'/'+raw_csv_file_name[file_nm]+'.csv')
				except: f_24_3 = pd.DataFrame(np.empty(0, dtype=[(key_var, sample_data[key_var].dtypes), (raw_csv_file_name[file_nm], sample_data[sample_data_columns[0]].dtypes)]))
				
				try: 	f_24_4 = pd.read_csv(input_folder+'/'+revoke25_36book25_36+'/'+dic_subdirectories_level_2[folder_nm]+'/'+raw_csv_file_name[file_nm]+'.csv')
				except: f_24_4 = pd.DataFrame(np.empty(0, dtype=[(key_var, sample_data[key_var].dtypes), (raw_csv_file_name[file_nm], sample_data[sample_data_columns[0]].dtypes)]))

				try: 	f_24_5 = pd.read_csv(input_folder+'/'+revoke25_36book37_48+'/'+dic_subdirectories_level_2[folder_nm]+'/'+raw_csv_file_name[file_nm]+'.csv')
				except: f_24_5 = pd.DataFrame(np.empty(0, dtype=[(key_var, sample_data[key_var].dtypes), (raw_csv_file_name[file_nm], sample_data[sample_data_columns[0]].dtypes)]))

				try: 	f_24_6 = pd.read_csv(input_folder+'/'+revoke37_48book37_48+'/'+dic_subdirectories_level_2[folder_nm]+'/'+raw_csv_file_name[file_nm]+'.csv')
				except: f_24_6 = pd.DataFrame(np.empty(0, dtype=[(key_var, sample_data[key_var].dtypes), (raw_csv_file_name[file_nm], sample_data[sample_data_columns[0]].dtypes)]))				

				## Merge all data 
				temp_1 = pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(f_24_1,f_24_2,on=key_var,how = 'outer'),f_24_3,on=key_var,how = 'outer'),f_24_4,on=key_var,how = 'outer'),f_24_5,on=key_var,how = 'outer'),f_24_6,on=key_var,how = 'outer')
				
				## Change column name as its same
				temp_1.columns=[key_var, 'a1', 'a2', 'a3', 'a4','a5','a6']
				## save column name to add all column
				sum_var_list=list(temp_1)
				## Remove key var from summation var list name
				sum_var_list.remove(key_var)
				## Sum all columns
				temp_1 = temp_1.fillna(0)
				temp_1[raw_csv_file_name[file_nm]] = temp_1[sum_var_list].sum(axis=1)
				## temp_1[raw_csv_file_name[file_nm]] = temp_1.fillna(0)['a1'] + temp_1.fillna(0)['a2'] + temp_1.fillna(0)['a3'] + temp_1.fillna(0)['a4'] + temp_1.fillna(0)['a5'] + temp_1.fillna(0)['a6']
				## Drop rest of columns as won't need them
				temp_1 = temp_1.drop(sum_var_list,1)
				## Save the data to link folders
				if not os.path.exists(output_folder+'/'+Link48z+'/'+dic_subdirectories_level_2[folder_nm]): os.makedirs(output_folder+'/'+Link48z+'/'+dic_subdirectories_level_2[folder_nm])
				temp_1.to_csv(output_folder+'/'+Link48z+'/'+dic_subdirectories_level_2[folder_nm]+'/'+raw_csv_file_name[file_nm]+'.csv',index=False)
				## delete all temporary data set
				del f_24_1, f_24_2, f_24_3, f_24_4, f_24_5, f_24_6, temp_1
				del sum_var_list
				
#########################################################################
## END OF CODE ##########################################################
#########################################################################
				