
#######################################################################################################################################
####################################### CART ##########################################################################################
#######################################################################################################################################

#######################################################################################################################################
###### call library ###################################################################################################################
#######################################################################################################################################

from __future__ import print_function
import subprocess
import os
import sys
#import pydotplus
from IPython.display import Image
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from time import time
from operator import itemgetter
from scipy.stats import randint
from collections import defaultdict
from sklearn import tree, metrics
from sklearn.tree import DecisionTreeClassifier, export_graphviz, _tree
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score, r2_score, roc_curve, auc
from sklearn.externals.six import StringIO
#from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.cross_validation import cross_val_score
from sklearn.datasets import load_iris
import pickle
print ('-------------------------------------------------------------------------------------------------')

#######################################################################################################################################
###### Call Functions##################################################################################################################
#######################################################################################################################################

def prepare_data(input_data):
		string_var_list 	=[]
		numeric_var_list 	=[]
		for i in list(input_data.columns):
				if input_data[i].dtype == 'object':
						string_var_list.append(i)
				else:
						numeric_var_list.append(i)
		#Seperate numeric and string data
		if len(string_var_list)  > 0 : 
				df_string 	= input_data[string_var_list]
				print ("string_var_list", string_var_list)
				print (df_string.head())
				print ('\n')
				LabelEncoder_mapping = defaultdict(LabelEncoder)
				df_string_numeric = df_string.apply(lambda x: LabelEncoder_mapping[x.name].fit_transform(x))
		if len(numeric_var_list) > 0 : 
				df_numeric 	= input_data[numeric_var_list]
				print ("numeric_var_list", numeric_var_list)
				print (df_numeric.head())
		if len(string_var_list)  > 0  and len(numeric_var_list) > 0	: 
				df 						= pd.concat([df_numeric, df_string_numeric], axis=1, join_axes=[df_numeric.index])
		elif len(string_var_list)  > 0:
				df 						= df_string_numeric[:]
				df_numeric 				= None
		elif len(numeric_var_list) > 0 :
				df 						= df_numeric[:]
				df_string 				= None
				LabelEncoder_mapping 	= None
		else	:
				df 						= None
				LabelEncoder_mapping 	= None
				
		print ('\n')
		print ('Final data')
		print (df.head())
		
		return LabelEncoder_mapping, string_var_list, numeric_var_list, df_string, df_numeric, df
		
def get_tree_in_dot_n_image_file(dot_file_name, decision_tree, independent_variable_name, dependent_variable_value, pdf_file_name):
		with open(dot_file_name+".dot", 'w') as f: f = tree.export_graphviz(decision_tree, out_file=f)
		dot_data = StringIO()
		tree.export_graphviz(decision_tree, feature_names=independent_variable_name, class_names=dependent_variable_value, filled=True, rounded=True, node_ids= True, out_file=dot_data)
		graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
		graph.write_pdf(pdf_file_name+".pdf")

def visualize_tree(tree, feature_names):
		with open("dt.dot", 'w') as f:export_graphviz(tree, out_file=f,feature_names=feature_names)
		command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
		try: subprocess.check_call(command)
		except: exit("Could not run dot, ie graphviz, to " "produce visualization")
		
def Performance_of_Decision_Tree(independent_variable_name, decision_tree, dependent_variable, independent_variable, predicted_class, in_csv,  output_csv_file_name ):
		print ("------------------------- Performance --------------------------------------------")
		c_matrix 		= metrics.confusion_matrix(dependent_variable, 		predicted_class)
		TN, FN, TP, FP	= c_matrix[0][0], c_matrix[1][0], c_matrix[1][1], c_matrix[0][1]
		TPR, FPR		= (float(TP)/float(TP+FN)), (float(FP)/float(FP+TN))
		#Var importance: Gini. [dict(zip(ind_var, clf.feature_importances_)) / for key, value in dict(zip(ind_var, clf.feature_importances_)).iteritems():key, value]
		print('-----------------------------------------------------------------------------')		
		feature_importances = pd.DataFrame(dict(zip(independent_variable_name, decision_tree.feature_importances_)).items(), columns=['Variable', 'Feature_Importances']).sort(['Feature_Importances'],ascending=False)
		if in_csv is True:
			feature_importances.to_csv(output_csv_file_name+'.csv',index=False)
		print("Variable Importance:")
		print(feature_importances)
		#print(pd.DataFrame(dict(zip(independent_variable_name, decision_tree.feature_importances_)).items(), columns=['Variable', 'Feature_Importances']).sort(['Feature_Importances'],ascending=False))
		#Returns the coefficient of determination R^2 of the prediction
		print('-----------------------------------------------------------------------------')
		print('\n')
		print("R Square of the prediction:")
		print(decision_tree.score(independent_variable, dependent_variable))
		#cross-validation: The cross-validation score can be directly calculated using the cross_val_score helper. 
		#Given an estimator, the cross-validation object and the input dataset, the cross_val_score splits the data repeatedly into a training and a testing set, 
		#trains the estimator using the training set and computes the scores based on the testing set for each iteration of cross-validation.
		print('-----------------------------------------------------------------------------')
		print('\n')
		print("Cross-Validation:")
		print(cross_val_score(decision_tree, independent_variable, dependent_variable, cv=10))
		print('-----------------------------------------------------------------------------')
		print('\n')
		print("Classification Report:")
		print(metrics.classification_report(dependent_variable, predicted_class))
		print('-----------------------------------------------------------------------------')
		print('\n')
		print("Confusion Matrix:")
		print(c_matrix)
		print('-----------------------------------------------------------------------------')		
		print('\n')
		print ("True Negative-", TN, '\n', "False Negative-", FN, '\n', "True Positive-", TP, '\n', "False Positive-",FP)
		print('-----------------------------------------------------------------------------')
		print ("Area Under the Curve:-", roc_auc_score(dependent_variable,predicted_class))
		print('-----------------------------------------------------------------------------')

def Plot_ROC(dependent_variable, predicted_class ):
		#Receiver Operating Characteristic
		false_positive_rate, true_positive_rate, thresholds = roc_curve(dependent_variable, predicted_class)
		roc_auc = auc(false_positive_rate, true_positive_rate)
		plt.figure(figsize=(10,10))
		plt.title('Receiver Operating Characteristic')
		plt.plot(false_positive_rate,true_positive_rate, color='black',label = 'AUC = %0.2f' % roc_auc)
		plt.legend(loc = 'lower right')
		plt.plot([0, 1], [0, 1],linestyle='--')
		plt.axis('tight')
		plt.ylabel('True Positive Rate')
		plt.xlabel('False Positive Rate')
		return plt
		
def get_predicted_class_n_probability_in_original_data(input_data, decision_tree, independent_variable, df_string, in_csv, output_csv_file_name ):
		#Merge predicted class with original data
		input_data_with_predicted_class = pd.DataFrame()
		input_data_with_predicted_class = pd.concat([input_data, pd.DataFrame(decision_tree.predict(independent_variable),columns=['predicted_class'])], axis=1, join_axes=[input_data.index])	
		#Probability of each class
		probability_class=decision_tree.predict_proba(independent_variable)
		#add probability in original data
		input_data_with_predicted_class = pd.concat([input_data_with_predicted_class, pd.DataFrame(probability_class,columns=['N_probability_class','Y_probability_class'])], axis=1, join_axes=[input_data_with_predicted_class.index])
		#Export original data with predicted class & probability
		if df_string is not None:
				df_string.columns 				= [str(col) + '_string' for col in df_string.columns]
				input_data_with_predicted_class = pd.concat([input_data_with_predicted_class, df_string], axis=1, join_axes=[input_data_with_predicted_class.index])
		#Export data in CSV
		if in_csv is True:
			input_data_with_predicted_class.to_csv(output_csv_file_name+'.csv',index=False)
			
		print (input_data_with_predicted_class.head())
		return input_data_with_predicted_class
		
def get_code(tree, feature_names, target_value_names, spacer_base="    "):
		from sklearn.tree import _tree
		left      	= tree.tree_.children_left
		right     	= tree.tree_.children_right
		threshold 	= tree.tree_.threshold
		features  	= [feature_names[i] for i in tree.tree_.feature]
		value 		= tree.tree_.value
		def recurse(left, right, threshold, features, node, depth):
				spacer = spacer_base * depth
				if (threshold[node] != -2):
					print(spacer + "if ( " + features[node] + " <= " + str(threshold[node]) + " ) {")
					if left[node] != -1:
							recurse(left, right, threshold, features,left[node], depth+1)
					print(spacer + "}\n" + spacer +"else {")
					if right[node] != -1:
							recurse(left, right, threshold, features,right[node], depth+1)
					print(spacer + "}")
				else:
					target = value[node]
					for i, v in zip(np.nonzero(target)[1],target[np.nonzero(target)]):
						target_name = target_value_names[i]
						target_count = int(v)
						print(spacer + "return " + str(target_name) + " ( " + str(target_count) + " examples )")
		
		recurse(left, right, threshold, features, 0, 0)

def get_code_in_txt(tree, feature_names, target_value_names, text_file_name):
		#get the funaction parameter in string format. [inspect.getargspec(<fn_name>).args[2]]
		orig_stdout = 	sys.stdout
		f 			= 	file(text_file_name+'.txt', 'w')
		sys.stdout 	= 	f
		get_code(tree, feature_names, target_value_names)
		sys.stdout 	= 	orig_stdout
		f.close()
		
def cross_freq(data, feature_names, freq_output_file_name):
		orig_stdout = 	sys.stdout
		f 			= 	file(freq_output_file_name+'.txt', 'w')
		sys.stdout 	= 	f
		for i in feature_names:
				print ('----Start--------------------------------')
				print (pd.crosstab(data[i],data[i+'_string']))
				print ('-----------', i , '-----------')
				print (pd.DataFrame(dict(zip(list(pd.crosstab(data[i],data[i+'_string']).axes[0]), list(pd.crosstab(data[i],data[i+'_string']).axes[1]))).items(), columns=['New', 'Old']))
				print ('----End--------------------------------')
				print ('\n')
		sys.stdout 	= 	orig_stdout
		f.close()
		
def run_gridsearch(X, y, clf, param_grid, cv=5):
		grid_search = GridSearchCV(clf,param_grid=param_grid,cv=cv)
		start 		= time()
		grid_search.fit(X, y)
		print(("\nGridSearchCV took {:.2f} ""seconds for {:d} candidate ""parameter settings.").format(time() - start,len(grid_search.grid_scores_)))
		top_params = report(grid_search.grid_scores_, 3)
		return  top_params
		
def run_randomsearch(X, y, clf, para_dist, cv=5, n_iter_search=20):
		random_search 	= RandomizedSearchCV(clf,param_distributions=param_dist,n_iter=n_iter_search)
		start 			= time()
		random_search.fit(X, y)
		print(("\nRandomizedSearchCV took {:.2f} seconds ""for {:d} candidates parameter ""settings.").format((time() - start),n_iter_search))
		top_params 		= report(random_search.grid_scores_, 3)
		return  top_params

def report(grid_scores, n_top=3):
		top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
		for i, score in enumerate(top_scores):
				print("Model with rank: {0}".format(i + 1))
				print(("Mean validation score: ""{0:.3f} (std: {1:.3f})").format(score.mean_validation_score,np.std(score.cv_validation_scores)))
				print("Parameters: {0}".format(score.parameters))
				print("")
		return top_scores[0].parameters

def generate_lift_table(score_data, no_bin, dep, non_dep,in_csv,output_csv_file_name):
		#dep : dependent_variable_name
		#non_dep : Non dependent_variable_name
		## Generate lift table, ks table
		scr_var_list 								= ['Y_probability_class',dependent_variable_name]
		data 										= score_data[scr_var_list]
		data.loc[:, 'score'] 						= data['Y_probability_class']*1000
		desred_decimals = 0
		#data['score'] 								= data['score'].apply(lambda x: round(x,desred_decimals))
		data.loc[:,'score'] 						= data['score'].apply(lambda x: round(x,desred_decimals))
		data										= data.drop('Y_probability_class',1)
		#data.rename(columns={'reservation': 'response'}, inplace=True)
		data.rename(columns={dependent_variable_name: 'response'}, inplace=True)
		data['non_response'] 						= 1 - data.response
		
		unique_val=len(list(data['score'].unique()))
		if unique_val < no_bin:
			if unique_val % 2 !=0:
				no_bin = (unique_val - 1)/2
			else :
				no_bin = (unique_val)/2
		print ("Unique score points", unique_val)
		
		try:
			#DEFINE 10 BUCKETS WITH EQUAL SIZE
			data['bucket'] 								= pd.qcut(data.score, no_bin)
			#GROUP THE DATA FRAME BY BUCKETS
			grouped 									= data.groupby('bucket', as_index = False)
			#CREATE A SUMMARY DATA FRAME
			agg1										= pd.DataFrame()
			agg1['min_scr'] 							= grouped.min().score
			agg1['max_scr'] 							= grouped.max().score
			agg1['total'] 								= agg1['total'] = grouped.sum().response + grouped.sum().non_response
			agg1['pct_total'] 							= (agg1.total/agg1.total.sum()).apply('{0:.2%}'.format)	
			agg1['non_response'] 						= grouped.sum().non_response
			agg1['pct_non_response'] 					= (agg1.non_response/agg1.non_response.sum()).apply('{0:.2%}'.format)	
			agg1['response'] 							= grouped.sum().response
			agg1['pct_response'] 						= (agg1.response/agg1.response.sum()).apply('{0:.2%}'.format)
			agg1['bad_rate'] 							= (agg1.response  / agg1.total).apply('{0:.2%}'.format)
			agg1['odds'] 								= (agg1.non_response / agg1.response).apply('{0:.2f}'.format)
			#SORT THE DATA FRAME BY SCORE
			lift_table 									= (agg1.sort_index(by = 'min_scr')).reset_index(drop = True)
			lift_table['cum_response'] 					= lift_table.response.cumsum()
			lift_table['cum_non_response'] 				= lift_table.non_response.cumsum()
			lift_table['cum_pct_response'] 				= (lift_table.cum_response/lift_table.response.sum()).apply('{0:.2%}'.format)	
			lift_table['cum_pct_non_response']			= (lift_table.cum_non_response/lift_table.non_response.sum()).apply('{0:.2%}'.format)
			#CALCULATE KS STATISTIC
			lift_table['ks'] 							= np.round(((lift_table.cum_non_response/lift_table.non_response.sum()) - (lift_table.cum_response/lift_table.response.sum()))*100,2)
			#DEFINE A FUNCTION TO FLAG MAX KS
			flag 										= lambda x: '<----' if x == lift_table.ks.max() else ''
			#FLAG OUT MAX KS
			lift_table['max_ks'] 						= lift_table.ks.apply(flag)
			lift_table.rename(columns={'non_response': non_dep, 'pct_non_response':'pct_'+non_dep, 'response':dep, 'pct_response':'pct_'+dep, 'bad_rate':dep+'_rate','cum_response':'cum_'+dep, 'cum_non_response':'cum_'+non_dep, 'cum_pct_response':'cum_pct_'+dep, 'cum_pct_non_response':'cum_pct_'+non_dep}, inplace=True)
			print (lift_table)
			del agg1 #remove a column [a.pop('cum_response')] [a=a.drop(['cum_non_response', 'cum_pct_response', 'cum_pct_non_response'], axis=1)]
			if in_csv is True:
				lift_table.to_csv(output_csv_file_name+'.csv',index=False)
			return lift_table
		except: 
			print(no_bin, "bins not possible in current score. Please check number of bins")
		
		
		
#######################################################################################################################################################
###### data preparation(Numeric & String data)#########################################################################################################
#######################################################################################################################################################

#Assign working folder
os.chdir(r'C:\Users\1567478\Documents\Projects\00.Info\Work\Python\Machine Learning\05.Decision Tree')

#Read csv data
input_data					= pd.read_csv('Dev1_Hilton_Model_Data.csv')

#Prepare data. Conver string to number
LabelEncoder_mapping, string_var_list, numeric_var_list, df_string, df_numeric, df = prepare_data(input_data)

#Name of dependent variable
dependent_variable_name 	= 'reservation'

#level names for dependent variable. dependent_variable_value=list(df[dependent_variable_name].unique())
dependent_variable_value 	= ['non-reservation','reservation']

independent_variable_name 	= list(df.columns)
independent_variable_name.remove(dependent_variable_name)
independent_variable 		= df[independent_variable_name].values
dependent_variable 			= df[dependent_variable_name].values

#### Validation #########
input_val_data				= pd.read_csv('Val1_Hilton_Data.csv')

if df_string is not None:
		df_string_val = input_val_data[string_var_list]
else :
		df_string_val = None

if LabelEncoder_mapping is not None: 
		df_val	= input_val_data.apply(lambda x: LabelEncoder_mapping[x.name].transform(x))
else:
		df_val	= input_val_data[:]
		
independent_variable_val 	= df_val[independent_variable_name].values
dependent_variable_val 		= df_val[dependent_variable_name].values

#############################################################################################################################################################
###### Build - Decision Tree Classifier #####################################################################################################################
#############################################################################################################################################################

clf = tree.DecisionTreeClassifier(max_leaf_nodes=5)
clf = clf.fit(independent_variable, dependent_variable)
#Get tree in PDF and Dot file
#get_tree_in_dot_n_image_file("Decision_Tree_Rules_For_Hilton", clf, independent_variable_name, dependent_variable_value, "Decision_Tree_Rules_For_Hilton")

visualize_tree(clf, independent_variable_name)

#############################################################################################################################################################
###### Performance of Decision Tree Classifier ##############################################################################################################
#############################################################################################################################################################

####### Performance - Development
#Predict the class
predicted_class = clf.predict(independent_variable)

#Print Performance in text file
os.remove('Performance_of_Decision_Tree_Development.txt') if os.path.exists('Performance_of_Decision_Tree_Development.txt') else None
orig_stdout = sys.stdout
f = file('Performance_of_Decision_Tree_Development.txt', 'w')
sys.stdout = f
#Print Performance
print ("------------------------- Performance - Development --------------------------------------------")
Performance_of_Decision_Tree(independent_variable_name, clf, dependent_variable, independent_variable, predicted_class, True, 'feature_importances' )
sys.stdout = 	orig_stdout
f.close()

Plot_ROC(dependent_variable, predicted_class )
#plt.savefig('Receiver Operating Characteristic.png')
plt.show()
#Export data in CSV. It will have predicted class, string variable mapped  with numeric
input_data_with_predicted_class=get_predicted_class_n_probability_in_original_data(df,clf,independent_variable,df_string,False,'Dev1_Hilton_Model_Data_output')
lift_table = generate_lift_table(input_data_with_predicted_class, 3, 'bad', 'good',False, 'dev-lift_table')
#Cross frequency [pd.crosstab(result.os_cat,result.os_cat_string)]
#if len(string_var_list)  > 0 : cross_freq(input_data_with_predicted_class, string_var_list, 'cross_freq')
# Get the rules.#get_code(clf, independent_variable_name, dependent_variable_value)
get_code_in_txt(clf, independent_variable_name, dependent_variable_value, 'Decision_Tree_Rules_For_Hilton')
print ("------------------------- END of Performance - Development --------------------------------------------")

########## Performance- Validation
predicted_class_val	=	clf.predict(independent_variable_val)
#propability_val	=	clf.predict_proba(independent_variable_val)

#Print Performance in text file
os.remove('Performance_of_Decision_Tree_Validation.txt') if os.path.exists('Performance_of_Decision_Tree_Validation.txt') else None
orig_stdout = sys.stdout
f = file('Performance_of_Decision_Tree_Validation.txt', 'w')
sys.stdout = f
print ("------------------------- Performance - Validation --------------------------------------------")
Performance_of_Decision_Tree(independent_variable_name, clf, dependent_variable, independent_variable, predicted_class, True, 'feature_importances' )
sys.stdout = 	orig_stdout
f.close()

Plot_ROC(dependent_variable_val, predicted_class_val )
#plt.savefig('Receiver Operating Characteristic.png')
plt.show()
#Plot_ROC(dependent_variable_val, predicted_class_val ) #plt.savefig('Receiver Operating Characteristic.png') #plt.show()
input_data_with_predicted_class = get_predicted_class_n_probability_in_original_data(df_val, clf, independent_variable_val, df_string_val, False,'Val1_Hilton_Dataoutput' )
lift_table = generate_lift_table(input_data_with_predicted_class, 3, 'bad', 'good',False, 'val-lift_table')
print ("------------------------- END of Performance - Validation --------------------------------------------")

#############################################################################################################################################################
###### Optimize Decision Tree by random search ##############################################################################################################
#############################################################################################################################################################

param_dist 	= 	{	"criterion"			: ["gini", "entropy"],
					"min_samples_split"	: randint(1, 20),
					"max_depth"			: randint(1, 20),
					"min_samples_leaf"	: randint(1, 20),
					"max_leaf_nodes"	: randint(2, 20)
				}
				
clf 		= DecisionTreeClassifier()
ts_rs 		= run_randomsearch(independent_variable, dependent_variable, clf, param_dist, cv=10, n_iter_search=288)
#print:Best Parameters
for k, v in ts_rs.items(): print("parameters: {:<20s} setting: {}".format(k, v))
#cross validation 
clf 		= DecisionTreeClassifier(**ts_rs)
scores 		= cross_val_score(clf, independent_variable, dependent_variable, cv=10)
print("mean: {:.3f} (std: {:.3f})".format(scores.mean(), scores.std()), end="\n\n" )
#Fit the tree
clf.fit(independent_variable, dependent_variable)

#Get tree in PDF and Dot file
#get_tree_in_dot_n_image_file("Random_search_Decision_Tree_Rules_For_Hilton", clf, independent_variable_name, dependent_variable_value, "Random_search_Decision_Tree_Rules_For_Hilton")
visualize_tree(clf, independent_variable_name)

#Predict the class
predicted_class = clf.predict(independent_variable)

#Print Performance in text file
os.remove('Random_Performance_of_Decision_Tree_Validation.txt') if os.path.exists('Random_Performance_of_Decision_Tree_Validation.txt') else None
orig_stdout = sys.stdout
f = file('Random_Performance_of_Decision_Tree_Validation.txt', 'w')
sys.stdout = f
#Print Performance
Performance_of_Decision_Tree(independent_variable_name, clf, dependent_variable, independent_variable, predicted_class, True, 'feature_importances' )
sys.stdout = 	orig_stdout
f.close()

#Export data in CSV. It will have predicted class, string variable mapped  with numeric
input_data_with_predicted_class = get_predicted_class_n_probability_in_original_data(input_data, clf, independent_variable, df_string, False, 'Random_search_Hilton_Model_Dev_Val_Data1_output' )
lift_table = generate_lift_table(input_data_with_predicted_class, 3, 'bad', 'good',False, 'Random_search-lift_table')
#Cross frequency [pd.crosstab(result.os_cat,result.os_cat_string)]
#if len(string_var_list)  > 0: cross_freq(input_data_with_predicted_class, string_var_list, 'Random_search_cross_freq')
# Get the rules.#get_code(clf, independent_variable_name, dependent_variable_value)
get_code_in_txt(clf, independent_variable_name, dependent_variable_value, 'Random_search_Decision_Tree_Rules_For_Hilton')


#############################################################################################################################################################
###### Optimize Decision Tree by Grid search ##############################################################################################################
#############################################################################################################################################################

param_grid 	= 	{	"criterion"			: ["gini", "entropy"],
					"min_samples_split"	: [2, 10, 20], 
					"max_depth"			: [None, 2, 5, 10],
					"min_samples_leaf"	: [1, 5, 10],
					"max_leaf_nodes"	: [None, 5, 10, 20],
				}
					
clf 			= DecisionTreeClassifier()
ts_gs 			= run_gridsearch(independent_variable, dependent_variable, clf, param_grid, cv=10)
for k, v in ts_gs.items(): print("parameters: {:<20s} setting: {}".format(k, v))
clf 			= DecisionTreeClassifier(**ts_gs)
scores 			= cross_val_score(clf, independent_variable, dependent_variable, cv=10)
print("mean: {:.3f} (std: {:.3f})".format(scores.mean(), scores.std()), end="\n\n" )
clf.fit(independent_variable, dependent_variable)
#get_code(clf, independent_variable_name, dependent_variable_value)
visualize_tree(clf, independent_variable_name)
predicted_class = clf.predict(independent_variable)

os.remove('Grid_Performance_of_Decision_Tree_Validation.txt') if os.path.exists('Grid_Performance_of_Decision_Tree_Validation.txt') else None
orig_stdout 	= sys.stdout
f = file('Grid_Performance_of_Decision_Tree_Validation.txt', 'w')
sys.stdout 		= f
#Print Performance
Performance_of_Decision_Tree(independent_variable_name, clf, dependent_variable, independent_variable, predicted_class, True, 'feature_importances' )
sys.stdout 		= 	orig_stdout
f.close()

#Export data in CSV. It will have predicted class, string variable mapped  with numeric
input_data_with_predicted_class = get_predicted_class_n_probability_in_original_data(input_data, clf, independent_variable, df_string, False, 'Grid_search_Hilton_Model_Dev_Val_Data1_output' )
lift_table = generate_lift_table(input_data_with_predicted_class, 3, 'bad', 'good',False, 'Grid_search-lift_table')
get_code_in_txt(clf, independent_variable_name, dependent_variable_value, 'Grid_search_Decision_Tree_Rules_For_Hilton')

#############################################################################################################################################################
##### END ###################################################################################################################################################
#############################################################################################################################################################