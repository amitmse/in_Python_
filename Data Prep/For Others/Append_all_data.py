#####################################################################################################################
## 1. This Python code will read the data from different location.
## 2. Append all data and append in one CSV file
#####################################################################################################################

#### python libraries ###############################################################################################
import os
import pandas as pd
import numpy as np
#########################################################################

##### Use Input ####################################################################
## Provide Path for folder for input & output
input_folder 	= 'C:/Users/amit.kumar/Downloads/Append_all_files/Raw_data'
output_folder 	= 'C:/Users/amit.kumar/Downloads/Append_all_files/Output'

##### List of folders#######################
## Please change the "revoke1_24book1_24" in below code if folder structure ('N1_', 'N2_', 'N3_', 'N4_', 'N5_', 'N6_'........) is not available in that location and provide the correct one.
dic_subdirectories_level_2 = { int((a.replace('X',''))):a for a in os.listdir(input_folder )}
dic_subdirectories_level_2 = {0:'X1', 1:'X2'}
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
#########################################################################

for file_nm in range(len(raw_csv_file_name)):
		for folder_nm in dic_subdirectories_level_2.keys():

				try: 	raw_file = pd.read_csv(input_folder+'/'+dic_subdirectories_level_2[folder_nm]+'/'+raw_csv_file_name[file_nm]+'.csv')
				except: raw_file = None
				if raw_file is not None:
						raw_file = raw_file.drop(raw_file.index[0])
						if folder_nm == 1 :
								base = raw_file
								del raw_file
						elif folder_nm < 12:
								base = pd.concat([base, raw_file])
								del raw_file
						else:
								base = pd.concat([base, raw_file])
								del raw_file
		if base is not None:
			## Create directory if does not exists
			if not os.path.exists(output_folder): os.makedirs(output_folder)
			## Save the data to link folders
			base.to_csv(output_folder+'/'+raw_csv_file_name[file_nm]+'.csv',index=False)
			## delete all temporary data set
			del base
				
#########################################################################

#########################################################################
## END OF CODE ##########################################################
#########################################################################
				