
#### python libraries ###############################################################################################
import os
import pandas as pd
import numpy as np
import networkx as nx
import operator
#########################################################################

##### Use Input ####################################################################
## Provide Path for folder for input & output
input_folder 	= 'C:/Users/amit.kumar/Downloads/Append_all_files/Raw_data'
output_folder 	= 'C:/Users/amit.kumar/Downloads/Append_all_files/Output'
summary         =  pd.dataframe(np.empty(0,dtype=[('variable','s100'),('IV','float64')]))
subset_folder 	= 'C:/Users/amit.kumar/Downloads/Append_all_files/Output'
key_var			= 'scrable_key'

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

special_values_folder = 'C:\\Users\\amit.kumar\\Google Drive\\Study\\Python\\Kalu\\Python code\\Variable_reduction\\special_values'
special_value_file_name = list(set(get_file_names(special_values_folder)))
all_special_value = {}
for file_nm in range(len(special_value_file_name)):
		special_value = dict()
		special_value_data=pd.read_csv(special_values_folder+'/'+special_value_file_name[file_nm]+'.csv',names=['variable','type','value']).dropna()
		special_value=pd.Series(special_value_data['value'].values,index=special_value_data['variable']).to_dict()
		for i in special_value.keys():
				if type(special_value[i])== str:
					a = special_value[i].split('|')
					a = [int(v) for v in a]
					special_value[i]=a
					del a
		all_special_value.update(special_value)
	
for file_nm in range(len(raw_csv_file_name)):
		raw_file 	= pd.read_csv(input_folder+'/'+raw_csv_file_name[file_nm]+'.csv',names=['variable','#records','IV','monotonicity','#bins','bins','selction_level','cluster','rsquareratio','rsquaregm'],skiprows=2)
		raw_file 	= raw_file[raw_file['selection_level']==2]
		var_name 	= list(set(raw_file['variable']))
		var_name.append(key_var)
		data_file 	= pd.read_csv(subset_folder+'/'+raw_csv_file_name[file_nm]+'.csv')
		data_file 	= datafile[var_name]
		data_file.to_csv(output_folder+'/'+raw_csv_file_name[file_nm]+'.csv',index=False)
		raw_file	= raw_file['variable','IV']
		summary 	= pd.concat([summary, raw_file])
		if file_nm	==1:
				all_data	= data_file[:]
		else:
				all_data 	= pd.merge(all_data,data_file,on=key_var,how = 'outer')


all_data.to_csv(output_folder+'/appended_data.csv',index=False)

#Remove special values
for v in all_special_value.keys():
		all_data[v]=all_data[v].replace(all_special_value[v], np.nan)

#drop key
all_data=all_data.drop(key_var,1)
correlation_all_var = all_data.corr()
#get correlation in csv
correlation_all_var.to_csv(output_folder+'/correlation_all.csv')

def _init_(self,g,label):
		self.G=G
		Self.label=label
		def _cmp_(self,other):
				if (self.G.degree (self.label) gt other.G.degree (other.label)) or (self.G.degree (self.label) == other.G.degree (other.label)) and  (self.G.node (self.label)('weight') lt other.G.node (other.label)('weight'))) :
						return -1
		
def deCorrelate(corrMatrix , weightsSeries , corrThreshold=0.5) #corrMatrix must be a dataFrame
		s=corrMatrix.abs().unstack()
		G=nx.Graph()
		G.add_edges_from( list(s[(s > corrThreshold) & (s<1)].index.values))
		for x in G.nodes():
				G.node[x]['weight'] = weightsSeries[x]
		while any(v > 0 for k, v in G.degree().iteritems()):
				 nodesPriorityDict = {k : G.degree() - G.nodes[k]['weight']/10 for k in G.nodes() }
				 print(nodesPriorityDict)
				 nodeToBeRemoved=max(nodesPriorityDict.iteritemd(), key=operator.itemgetter(1))[0]
				 print('removing %s ....'%nodeToBeRemoved)
				 G.remove_node(nodeToBeRemoved)
		return G.nodes()
	 
deCorrelate(correlation_all_var, summary, corrThreshold=0.5)

#Add data type in row#2
new_record = pd.DataFrame([list(input_data.dtypes)],columns=list(input_data.columns))
new_record=new_record.replace({'object':'str', 'float64':'float', 'int64':'int'})
old_data_frame = pd.concat([new_record,input_data])

#Remove keys from dict.
#http://stackoverflow.com/questions/8995611/removing-multiple-keys-from-a-dictionary-safely
listC 		= 	[item for item in listB if item not in listA]
dict_new	=	{key: dict[key] for key in dict if key not in b}	#Override the original dict. Remove some key & value. Change dict and b
data2 		= 	t.set_index('two')
#########################################################################################################################

special_value_data=pd.read_csv('blackboxamount.csv',names=['variable','type','value'])
special_value_data_1 = special_value_data.drop('value',1)
Transpose=special_value_data_1.set_index('variable').T

'''
#os.chdir(special_values_folder)
#special_value_data=pd.read_csv('blackboxamount.csv',names=['variable','type','value'])
#special_value_data.dropna()
#replace special value with missing 	#b['NOP_before_purchase']=b['NOP_before_purchase'].replace([3,2], np.nan)
special_values_folder = 'C:\\Users\\amit.kumar\\Google Drive\\Study\\Python\\Kalu\\Python code\\Variable_reduction\\special_values'
data_folder = 'C:\\Users\\amit.kumar\\Google Drive\\Study\\Python\\Kalu\\Python code\\Variable_reduction\\original_data'
Correlation = 'C:\\Users\\amit.kumar\\Google Drive\\Study\\Python\\Kalu\\Python code\\Variable_reduction\\Correlation'
special_value_file_name = list(set(get_file_names(special_values_folder)))

special_values_folder = 'C:\\Users\\amit.kumar\\Google Drive\\Study\\Python\\Kalu\\Python code\\Variable_reduction\\special_values'
all_special_value = {}
### Dict for special value
for file_nm in range(len(special_value_file_name)):
	special_value = dict()
	special_value_data=pd.read_csv(special_values_folder+'/'+special_value_file_name[file_nm]+'.csv',names=['variable','type','value']).dropna()
	special_value=pd.Series(special_value_data['value'].values,index=special_value_data['variable']).to_dict()
	for i in special_value.keys():
		a = special_value[i].split('|')
		a = [int(v) for v in a]
		special_value[i]=a
		del a
	all_special_value.update(special_value)
### Dict for special value and remove special value from data
for file_nm in range(len(special_value_file_name)):
	special_value = dict()
	special_value_data=pd.read_csv(special_values_folder+'/'+special_value_file_name[file_nm]+'.csv',names=['variable','type','value']).dropna()
	special_value=pd.Series(special_value_data['value'].values,index=special_value_data['variable']).to_dict()
	for i in special_value.keys():
		a = special_value[i].split('|')
		a = [int(v) for v in a]
		special_value[i]=a
		del a
	raw_file = pd.read_csv(data_folder+'/'+special_value_file_name[file_nm]+'.csv',names=list(pd.read_csv(data_folder+'/'+special_value_file_name[file_nm]+'.csv',nrows=0).columns),skiprows=2)
	for v in special_value.keys():
		raw_file[v]=raw_file[v].replace(special_value[v], np.nan)
	raw_file.to_csv(Correlation+'/'+special_value_file_name[file_nm]+'.csv',index=False)
	del raw_file, special_value_data

### Dict for special value and remove special value from data
for file_nm in range(len(raw_csv_file_name)):
	special_value = dict()
	special_value_data=pd.read_csv(special_values_folder+'/'+raw_csv_file_name[file_nm]+'.csv',names=['variable','type','value']).dropna()
	special_value=pd.Series(special_value_data['value'].values,index=special_value_data['variable']).to_dict()
	for i in special_value.keys():
			a = special_value[i].split('|')
			a = [int(v) for v in a]
			special_value[i]=a
	raw_file = pd.read_csv(data_folder+'/'+raw_csv_file_name[file_nm]+'.csv',names=list(pd.read_csv(data_folder+'/'+raw_csv_file_name[file_nm]+'.csv',nrows=0).columns),skiprows=2)
	for v in special_value.keys():
			raw_file[v]=raw_file[v].replace(special_value[v], np.nan)
### Dict for special value and remove special value from data	
special_value_data=pd.read_csv('blackboxamount.csv',names=['variable','type','value']).dropna()
special_value=pd.Series(special_value_data['value'].values,index=special_value_data['variable']).to_dict()
for i in special_value.keys():
		a = special_value[i].split('|')
		a = [int(v) for v in a]
		special_value[i]=a

#dict two columns
#special_value = {k: g["Val"].tolist() for k,g in z.groupby("Var")}
#special_value = pd.Series(z['Val'].values,index=z['Var']).to_dict()
special_value=pd.Series(z['Val'].values,index=z['Var']).to_dict()
for i in special_value.keys():
		a=list(special_value[i])
		while '|' in a:
			a.remove('|')
		a = [int(v) for v in a]
		special_value[i]=a
	
	
#special_value = {'reservation':[0],'NOP_before_purchase':[3,2]}
var_name = list(b.columns)
for i in var_name:
		b[i]=b[i].replace(special_value[i], np.nan)

###This code will give 3 datasets ###
##1.summary with appended for all datasets and will have 2 columns variable and IV##
##2.raw_csv_file_name . csv's with filtered variables in output folder####
##3.correlation_all csv which has correlation output in output folder#####

#http://stackoverflow.com/questions/27060098/replacing-few-values-in-a-pandas-dataframe-column-with-another-value
#http://stackoverflow.com/questions/17142304/replace-string-value-in-entire-dataframe
#http://stackoverflow.com/questions/36072626/pandas-replace-multiple-values-at-once
'''