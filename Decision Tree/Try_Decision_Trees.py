####################################### CART ############################################################################################
# python decision tree how to get probability value
# Is decision tree output a prediction or class probabilities?
#######################################################################################################################################
# http://codereview.stackexchange.com/questions/109089/id3-decision-tree-in-python
# http://stats.stackexchange.com/questions/193424/is-decision-tree-output-a-prediction-or-class-probabilities
# http://hamelg.blogspot.in/2015/11/python-for-data-analysis-part-29.html
#######################################################################################################################################
# https://www.coursera.org/learn/ml-classification/home/welcome
# http://scikit-learn.org/stable/modules/tree.html
# http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
# https://www.quora.com/What-is-an-intuitive-explanation-of-F-score

# http://napitupulu-jon.appspot.com/posts/decision-tree-ud.html
#######################################################################################################################################

#######################################################################################################################################################
###### call library ##################################################################################################################################
#######################################################################################################################################################

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

#http://chrisstrelioff.ws/sandbox/2015/06/08/decision_trees_in_python_with_scikit_learn_and_pandas.html
'''
True Positives 	(TP) = Correctly Identified
True Negatives 	(TN) = Correctly Rejected
False Positives	(FP) = Incorrectly Identified 	= Type I Error
False Negatives	(FN) = Incorrectly Rejected		= Type II Error
Recall (Sensitivity) = Ability of the classifier to find positive samples from all positive samples. 
Precision			 = Ability of the classifier not to label as positive a sample that is negative. (positive predictive value)
Specificity 		 = Measures the proportion of actual negatives that are correctly identified. (true negative rate)


True Positive Rate			/ Sensitivity / Recall 	: TP 	/ (TP + FN)			= TP / Actual Positives
True Negative Rate			/ Specificity			: TN 	/ (TN + FP)			= TN / Actual Negatives
False Positive Rate 		/ Type I Error			: FP	/ (FP + TN) 		= FP / Actual Negatives		= 1 - Specificity
False Negative Rate 		/ Type II Error			: FN	/ (FN + TP) 		= FN / Actual Positives		= 1 - True Positive Rate
Positive Predictive Value 	/ Precision 			: TP 	/ (TP + FP)
Negative Predictive Value							: TN 	/ (TN + FN)
False Discovery Rate								: FP	/ (FP + TP)			= 1 - Positive Predictive Value
F1-Score 											: 2*TP	/ (2TP + FP + FN) 	= [2 * (Precision * Recall) / (Precision + Recall)]
Accuracy 											: (TP + TN)/ (TP  + TN + FP + FN)


F1 score (also F-score or F-measure) is a measure of a test's accuracy. The F1-score gives you the harmonic mean of precision and recall. 
The scores corresponding to every class will tell you the accuracy of the classifier in classifying the data points in that particular class compared to all other classes.
The F1 score is the harmonic average of the precision and recall, where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0.
It considers both the precision and the recall of the test to compute the score: 
	-	precision is the number of correct positive results divided by the number of all positive results returned by the classifier, 
	-	recall is the number of correct positive results divided by the number of all relevant samples (all samples that should have been identified as positive).
	
'''
#######################################################################################################################################################
###### Call Functions##################################################################################################################################
#######################################################################################################################################################

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
				#Inverse the encoded. [fit=fit.apply(lambda x: LabelEncoder_mapping[x.name].inverse_transform(x))]
				#Using the dictionary to label future data. [fit=df_string.apply(lambda x: LabelEncoder_mapping[x.name].transform(x))]				
		if len(numeric_var_list) > 0 : 
				df_numeric 	= input_data[numeric_var_list]
				print ("numeric_var_list", numeric_var_list)
				print (df_numeric.head())
		###LabelEncoder. Encode labels with value between 0 and n_classes-1. #http://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn
		# Encoding the variable
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
		
def Performance_of_Decision_Tree(independent_variable_name, decision_tree, dependent_variable, independent_variable, predicted_class ):
		print ("------------------------- Performance --------------------------------------------")
		c_matrix 		= metrics.confusion_matrix(dependent_variable, 		predicted_class)
		TN, FN, TP, FP	= c_matrix[0][0], c_matrix[1][0], c_matrix[1][1], c_matrix[0][1]
		TPR, FPR		= (float(TP)/float(TP+FN)), (float(FP)/float(FP+TN))
		#Var importance: Gini. [dict(zip(ind_var, clf.feature_importances_)) / for key, value in dict(zip(ind_var, clf.feature_importances_)).iteritems():key, value]
		print('-----------------------------------------------------------------------------')
		print("Variable Importance:")
		print(pd.DataFrame(dict(zip(independent_variable_name, decision_tree.feature_importances_)).items(), columns=['Variable', 'Feature_Importances']).sort(['Feature_Importances'],ascending=False))
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
		#https://www.kaggle.com/nirajvermafcb/d/dalpozz/creditcardfraud/random-forest
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
		
def get_predicted_class_n_probability_in_original_data(input_data, decision_tree, independent_variable, df_string, output_csv_file_name ):
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
		input_data_with_predicted_class.to_csv(output_csv_file_name+'.csv',index=False)
		print (input_data_with_predicted_class.head())
		return input_data_with_predicted_class
		
def get_code(tree, feature_names, target_value_names, spacer_base="    "):
		from sklearn.tree import _tree
		#http://chrisstrelioff.ws/sandbox/2015/06/08/decision_trees_in_python_with_scikit_learn_and_pandas.html
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

###############################################################################################################################################################
###### Build - Decision Tree Classifier #######################################################################################################################
###############################################################################################################################################################		

clf = tree.DecisionTreeClassifier(max_leaf_nodes=5)
clf = clf.fit(independent_variable, dependent_variable)
#Get tree in PDF and Dot file
#get_tree_in_dot_n_image_file("Decision_Tree_Rules_For_Hilton", clf, independent_variable_name, dependent_variable_value, "Decision_Tree_Rules_For_Hilton")

visualize_tree(clf, independent_variable_name)

##################################################################################################################################################################
###### Performance of Decision Tree Classifier ###################################################################################################################
##################################################################################################################################################################

####### Performance - Development
#Predict the class
predicted_class = clf.predict(independent_variable)
#Print Performance
print ("------------------------- Performance - Development --------------------------------------------")
Performance_of_Decision_Tree(independent_variable_name, clf, dependent_variable, independent_variable, predicted_class )
Plot_ROC(dependent_variable, predicted_class )
#plt.savefig('Receiver Operating Characteristic.png')
plt.show()
#Export data in CSV. It will have predicted class, string variable mapped  with numeric
input_data_with_predicted_class = get_predicted_class_n_probability_in_original_data(df, clf, independent_variable, df_string, 'Dev1_Hilton_Model_Data_output' )
#Cross frequency [pd.crosstab(result.os_cat,result.os_cat_string)]
if len(string_var_list)  > 0 : cross_freq(input_data_with_predicted_class, string_var_list, 'cross_freq')
# Get the rules.#get_code(clf, independent_variable_name, dependent_variable_value)
get_code_in_txt(clf, independent_variable_name, dependent_variable_value, 'Decision_Tree_Rules_For_Hilton')
print ("------------------------- END of Performance - Development --------------------------------------------")

########## Performance- Validation
predicted_class_val	=	clf.predict(independent_variable_val)
#propability_val	=	clf.predict_proba(independent_variable_val)
print ("------------------------- Performance - Validation --------------------------------------------")
Performance_of_Decision_Tree(independent_variable_name, clf, dependent_variable_val, independent_variable_val, predicted_class_val )
Plot_ROC(dependent_variable_val, predicted_class_val )
#plt.savefig('Receiver Operating Characteristic.png')
plt.show()
#Plot_ROC(dependent_variable_val, predicted_class_val ) #plt.savefig('Receiver Operating Characteristic.png') #plt.show()
input_data_with_predicted_class = get_predicted_class_n_probability_in_original_data(df_val, clf, independent_variable_val, df_string_val, 'Val1_Hilton_Dataoutput' )
print ("------------------------- END of Performance - Validation --------------------------------------------")

##############################################################################################
###### Save tree in pickle to use in future ##################
#save decision tree
pickle.dump(clf, open("Decision_Tree_Classifier_Hilton", 'wb'))
#Load decision tree
#clf = pickle.load(open("Decision_Tree_Classifier_Hilton",'r'))
#Call decision tree. below is for testing 
#print(cross_val_score(clf, independent_variable, dependent_variable, cv=10))


##################################################################################################################################################################
###### Optimize Decision Tree by random search ###################################################################################################################
##################################################################################################################################################################

param_dist 	= {"criterion": ["gini", "entropy"], "min_samples_split": randint(1, 20), "max_depth": randint(1, 20), "min_samples_leaf": randint(1, 20), "max_leaf_nodes": randint(2, 20)}
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
#Print Performance
Performance_of_Decision_Tree(independent_variable_name, clf, dependent_variable, independent_variable, predicted_class )
#Export data in CSV. It will have predicted class, string variable mapped  with numeric
input_data_with_predicted_class = get_predicted_class_n_probability_in_original_data(clf, independent_variable, df_string, 'Random_search_Hilton_Model_Dev_Val_Data1_output' )
#Cross frequency [pd.crosstab(result.os_cat,result.os_cat_string)]
if len(string_var_list)  > 0: cross_freq(input_data_with_predicted_class, string_var_list, 'Random_search_cross_freq')
# Get the rules.#get_code(clf, independent_variable_name, dependent_variable_value)
get_code_in_txt(clf, independent_variable_name, dependent_variable_value, 'Random_search_Decision_Tree_Rules_For_Hilton')


#save decision tree
pickle.dump(clf, open("Random_Search_Decision_Tree_Classifier_Hilton", 'wb'))
#Load decision tree
#clf = pickle.load(open("Random_Search_Decision_Tree_Classifier_Hilton",'r'))
#Call decision tree. below is for testing 
#print(cross_val_score(clf, independent_variable, dependent_variable, cv=10))

########################################
#How to output RandomForest Classifier from python. http://stackoverflow.com/questions/23000693/how-to-output-randomforest-classifier-from-python
#from sklearn.externals import joblib
#joblib.dump(clf, 'filename.pkl') 
#then your colleagues can load it
#clf = joblib.load('filename.pk1')
########################################
		
########################################################################################################################################
##### END ##############################################################################################################################
########################################################################################################################################


## input only numeric data
os.chdir('C:\\Users\\amit.kumar\\Google Drive\\Study\\Other\\Decision Tree')
#np.genfromtxt('Hilton_Model_Dev_Val_Data1.csv', delimiter=',', skip_header=1)
df 							= pd.read_csv('Hilton_Model_Dev_Val_Data1.csv')
dependent_variable_name 	= 'reservation'
#level names for dependent variable. dependent_variable_value=list(df[dependent_variable_name].unique())
dependent_variable_value 	= ['non-reservation','reservation']
#list of independent variables ['reservation','os_cat','dd','browser_cat']
independent_variable_name 	= list(df.columns)
independent_variable_name.remove(dependent_variable_name)
#df[:,1:]
independent_variable 		= df[independent_variable_name].values
#df[:,0]
dependent_variable 			= df[dependent_variable_name].values

clf = tree.DecisionTreeClassifier(max_leaf_nodes=5)
clf = clf.fit(independent_variable, dependent_variable)

with open("Decision_Tree_Rules_For_Hilton.dot", 'w') as f: f = tree.export_graphviz(clf, out_file=f)
dot_data = StringIO()
tree.export_graphviz(clf, feature_names=independent_variable_name, class_names=dependent_variable_value, filled=True, rounded=True, node_ids= True, out_file=dot_data)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("Decision_Tree_Rules_For_Hilton.pdf")

#Var importance: Gini 
#dict(zip(ind_var, clf.feature_importances_)) / for key, value in dict(zip(ind_var, clf.feature_importances_)).iteritems():key, value
pd.DataFrame(dict(zip(independent_variable_name, clf.feature_importances_)).items(), columns=['Variable', 'Feature_Importances']).sort(['Feature_Importances'],ascending=[False])
#Returns the coefficient of determination R^2 of the prediction
clf.score(independent_variable, dependent_variable)
#cross-validation: The cross-validation score can be directly calculated using the cross_val_score helper. 
#Given an estimator, the cross-validation object and the input dataset, the cross_val_score splits the data repeatedly into a training and a testing set, 
#trains the estimator using the training set and computes the scores based on the testing set for each iteration of cross-validation.
cross_val_score(clf, independent_variable, dependent_variable, cv=10)
#Predict the class
predicted_class = clf.predict(independent_variable)
#Merge predict the class with original data
result = pd.concat([df, pd.DataFrame(clf.predict(independent_variable),columns=['predicted_class'])], axis=1, join_axes=[df.index])	
#Probability of each class
probability_class=clf.predict_proba(independent_variable)
#add probability in original data
result = pd.concat([result, pd.DataFrame(probability_class,columns=['N_probability_class','Y_probability_class'])], axis=1, join_axes=[result.index])
#Export original data with predicted class & probability
df_string.columns = [str(col) + '_string' for col in df_string.columns]
result = pd.concat([result, df_string], axis=1, join_axes=[result.index])
#Cross frequency [pd.crosstab(result.os_cat,result.os_cat_string)]
def cross_freq(data, feature_names, freq_output_file_name):
		orig_stdout 	= 	sys.stdout
		f 		= 	file(freq_output_file_name+'.txt', 'w')
		sys.stdout 	= 	f
		for i in feature_names:
			print ('----Start--------------------------------')
			print (pd.crosstab(data[i],data[i+'_string']))
			print ('-----------', i , '-----------')
			print (pd.DataFrame(dict(zip(list(pd.crosstab(data[i],data[i+'_string']).axes[0]), list(pd.crosstab(data[i],data[i+'_string']).axes[1]))).items(), columns=['New', 'Old']))
			print ('----End--------------------------------')
		sys.stdout 	= 	orig_stdout
		f.close()
		
cross_freq(result, independent_variable_name, 'cross_freq')

result.to_csv('C:\\Users\\amit.kumar\\Google Drive\\Study\\Other\\Decision Tree\\Hilton_Model_Dev_Val_Data1_output.csv',index=False)

#Confusion Matrix
print(metrics.classification_report(dependent_variable, predicted_class))
print(metrics.confusion_matrix(dependent_variable, 		predicted_class))

# Get the rules
def get_code(tree, feature_names, target_value_names, spacer_base="    "):
		from sklearn.tree import _tree
		#http://chrisstrelioff.ws/sandbox/2015/06/08/decision_trees_in_python_with_scikit_learn_and_pandas.html
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

## Call function
get_code(clf, independent_variable_name, dependent_variable_value)


def get_code_in_txt(tree, feature_names, target_value_names, text_file_name):
		#get the funaction parameter in string format. [inspect.getargspec(<fn_name>).args[2]]
		orig_stdout = 	sys.stdout
		f 			= 	file(text_file_name+'.txt', 'w')
		sys.stdout 	= 	f
		get_code(tree, feature_names, target_value_names)
		sys.stdout 	= 	orig_stdout
		f.close()

get_code_in_txt(clf, independent_variable_name, dependent_variable_value, 'Decision_Tree_Rules_For_Hilton')

########################################################################

'''
#Convert categorical variable into dummy/indicator variables 
pd.get_dummies(df_string)
###LabelEncoder. Encode labels with value between 0 and n_classes-1. #http://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn
d = defaultdict(LabelEncoder)
# Encoding the variable
df_string_numeric = df_string.apply(lambda x: d[x.name].fit_transform(x))
#Inverse the encoded
fit=fit.apply(lambda x: d[x.name].inverse_transform(x))
#Using the dictionary to label future data
fit=df_string.apply(lambda x: d[x.name].transform(x))
#df=fit[:]
----------
df_numeric = df[:]
le=LabelEncoder()
for col in df_numeric.columns.values:
		if df_numeric[col].dtypes=='object':
			data=df_numeric[col]
			le.fit(data.values)
			df_numeric[col]=le.transform(df_numeric[col])
#----------------------------------------			
####################################################################		
#sys.stdout=open("Decision_Tree_Rules_For_Hilton.txt","w")
#print (get_code(clf, independent_variable_name, 'reservation'))
#sys.stdout.close()
	
#http://nbviewer.jupyter.org/gist/aflaxman/d20c723f75d336865940
# tree.predict(iris.data)
# http://stackoverflow.com/questions/32506951/how-to-explore-a-decision-tree-built-using-scikit-learn
# clf = clf.fit(X_train, Y_train)
# tree.tree_.__getstate__()['nodes']
'''
 
####### END ################################################################################################################################
############################################################################################################################################

##### Optimize decision tree################################################################################################################
### Optimize decision tree by Develop tree with random search/grid search
### http://chrisstrelioff.ws/sandbox/2015/06/25/decision_trees_in_python_again_cross_validation.html

from __future__ import print_function
import os
import subprocess
from time import time
from operator import itemgetter
from scipy.stats import randint
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.cross_validation import cross_val_score

########################################################################

##### call all function from below
 
#### 0. get the data and define variable name	
os.chdir('C:\\Users\\amit.kumar\\Google Drive\\Study\\Other\\Decision Tree')
#print("\n-- get data:")
df = get_iris_data()
#print("")
#variable name
independent_variable_name 		= ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]
#Actual dependent variable name. It's in char format.
dependent_variable_name_string	= 'Name'
#New dependent variable name. It's in numeric format.
dependent_variable_name			= 'Target'
#unique value of dependent variable. Maintain the order of name as its populated in data set. [dependent_variable_value=list(df[dependent_variable_name_string].unique())]
dependent_variable_value 		= ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
#convert target variable text to number.[df,dependent_variable_value=encode_target(df, dependent_variable_name_string, dependent_variable_name)]
df								= encode_target(df, dependent_variable_name_string, dependent_variable_name)
#data
dependent_variable  			= df[dependent_variable_name]
independent_variable			= df[independent_variable_name]

########################################################################
#### 1. first cross-validation
#print("-- 10-fold cross-validation ""[using setup from previous post]")
dt_old = DecisionTreeClassifier(min_samples_split=20, random_state=99)
dt_old.fit(independent_variable, dependent_variable)
scores = cross_val_score(dt_old, independent_variable, dependent_variable, cv=10)
print("mean: {:.3f} (std: {:.3f})".format(scores.mean(),scores.std()), end="\n\n" )

########################################################################	
### 2.a. application of random search	
#print("-- Random Parameter Search via 10-fold CV")
#dict of parameter list/distributions to sample
param_dist 	= {"criterion": ["gini", "entropy"], "min_samples_split": randint(1, 20), "max_depth": randint(1, 20), "min_samples_leaf": randint(1, 20), "max_leaf_nodes": randint(2, 20)}
dt 			= DecisionTreeClassifier()
ts_rs 		= run_randomsearch(independent_variable, dependent_variable, dt, param_dist, cv=10, n_iter_search=288)
						 
#print("\n-- Best Parameters:")
for k, v in ts_rs.items():
		print("parameters: {:<20s} setting: {}".format(k, v))
	
#print("\n\n-- Testing best parameters [Random]...")
dt_ts_rs 	= DecisionTreeClassifier(**ts_rs)
scores 		= cross_val_score(dt_ts_rs, independent_variable, dependent_variable, cv=10)
print("mean: {:.3f} (std: {:.3f})".format(scores.mean(), scores.std()), end="\n\n" )
	
#print("\n-- get_code for best parameters [Random]:")
dt_ts_rs.fit(independent_variable, dependent_variable)
get_code(dt_ts_rs, independent_variable_name, dependent_variable_value)
visualize_tree(dt_ts_rs, independent_variable_name, dependent_variable_value, fn="rand_best")
####################	
### 2.b. application of grid search. Its time consuming method.
#print("-- Grid Parameter Search via 10-fold CV")
#set of parameters to test

param_grid 	= 	{	"criterion"			: ["gini", "entropy"],
					"min_samples_split"	: [2, 10, 20], 
					"max_depth"			: [None, 2, 5, 10],
					"min_samples_leaf"	: [1, 5, 10],
					"max_leaf_nodes"	: [None, 5, 10, 20],
				}
				
dt 			= DecisionTreeClassifier()
ts_gs 		= run_gridsearch(independent_variable, dependent_variable, dt, param_grid, cv=10)

#print("\n-- Best Parameters:")
for k, v in ts_gs.items():
		print("parameter: {:<20s} setting: {}".format(k, v))
		
#print("\n\n-- Testing best parameters [Grid]...")
dt_ts_gs 	= DecisionTreeClassifier(**ts_gs)
scores 		= cross_val_score(dt_ts_gs, independent_variable, dependent_variable, cv=10)
print("mean: {:.3f} (std: {:.3f})".format(scores.mean(), scores.std()), end="\n\n" )

#print("\n-- get_code for best parameters [Grid]:", end="\n\n")
dt_ts_gs.fit(independent_variable, dependent_variable)
get_code(dt_ts_gs, independent_variable_name, dependent_variable_value)
visualize_tree(dt_ts_gs, independent_variable_name, dependent_variable_value, fn="grid_best")

####################################################
##### Define all functions ########################

def get_iris_data():
		"""Get the iris data, from local csv or pandas repo."""
		if os.path.exists("iris.csv"):
				print("-- iris.csv found locally")
				df = pd.read_csv("iris.csv", index_col=0)
		else:
				print("-- trying to download from github")
				fn = ("https://raw.githubusercontent.com/pydata/""pandas/master/pandas/tests/data/iris.csv")
				try:
					df = pd.read_csv(fn)
				except:
					exit("-- Unable to download iris.csv")

				with open("iris.csv", 'w') as f:
					print("-- writing to local iris.csv file")
					df.to_csv(f)
		return df

def encode_target(df, target_column, target_column_new_name):
		"""Add column to df with integers for the target.
		Args
		----
		df 				-- pandas Data Frame.
		target_column 	-- column to map to int, producing new Target column.
		Returns
		-------
		df 				-- modified Data Frame.
		targets 		-- list of target names.
		"""
		
		df_mod 							= df.copy()
		targets 						= list(df_mod[target_column].unique())
		map_to_int 						= {name: n for n, name in enumerate(targets)}
		df_mod[target_column_new_name] 	= df_mod[target_column].replace(map_to_int)
		#old: return (df_mod, targets)
		return df_mod

def run_gridsearch(X, y, clf, param_grid, cv=5):
		"""Run a grid search for best Decision Tree parameters.
		Args
		----
		X 			-- features
		y 			-- targets (classes)
		cf			-- scikit-learn Decision Tree
		param_grid 	-- [dict] parameter settings to test
		cv 			-- fold of cross-validation, default 5

		Returns
		-------
		top_params -- [dict] from report()
		"""
		
		grid_search = GridSearchCV(clf,param_grid=param_grid,cv=cv)
		start 		= time()
		grid_search.fit(X, y)
		print(("\nGridSearchCV took {:.2f} ""seconds for {:d} candidate ""parameter settings.").format(time() - start,len(grid_search.grid_scores_)))
		top_params = report(grid_search.grid_scores_, 3)
		return  top_params
	
	
def run_randomsearch(X, y, clf, para_dist, cv=5, n_iter_search=20):
		"""Run a random search for best Decision Tree parameters.
		Args
		----
		X 				-- features
		y 				-- targets (classes)
		cf 				-- scikit-learn Decision Tree
		param_dist 		-- [dict] list, distributions of parameters to sample
		cv 				-- fold of cross-validation, default 5
		n_iter_search 	-- number of random parameter sets to try, default 20.
		Returns
		-------
		top_params 		-- [dict] from report()
		"""
		
		random_search 	= RandomizedSearchCV(clf,param_distributions=param_dist,n_iter=n_iter_search)
		start 			= time()
		random_search.fit(X, y)
		print(("\nRandomizedSearchCV took {:.2f} seconds ""for {:d} candidates parameter ""settings.").format((time() - start),n_iter_search))
		top_params 		= report(random_search.grid_scores_, 3)
		return  top_params

def report(grid_scores, n_top=3):
		"""Report top n_top parameters settings, default n_top=3.
		Args
		----
		grid_scores -- output from grid or random search
		n_top 		-- how many to report, of top models

		Returns
		-------
		top_params -- [dict] top parameter settings found in search
		"""
		
		top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
		for i, score in enumerate(top_scores):
				print("Model with rank: {0}".format(i + 1))
				print(("Mean validation score: ""{0:.3f} (std: {1:.3f})").format(score.mean_validation_score,np.std(score.cv_validation_scores)))
				print("Parameters: {0}".format(score.parameters))
				print("")
		return top_scores[0].parameters
		
def get_code(tree, feature_names, target_value_names, spacer_base="    "):
    
		"""Produce pseudo-code for decision tree.
		Args
		----
		tree 				-- scikit-leant Decision Tree.
		feature_names 		-- list of feature names.
		target_value_names 	-- list of target value names.
		spacer_base 		-- used for spacing code (default: "    ").
		Notes
		-----
		based on http://stackoverflow.com/a/30104792.
		"""
		
		left      	= tree.tree_.children_left
		right     	= tree.tree_.children_right
		threshold 	= tree.tree_.threshold
		features  	= [feature_names[i] for i in tree.tree_.feature]
		value 		= tree.tree_.value

		def recurse(left, right, threshold, features, node, depth):
				spacer = spacer_base * depth
				if (threshold[node] != -2):
						print(spacer + "if ( " + features[node] + " <= " + \
							  str(threshold[node]) + " ) {")
						if left[node] != -1:
								recurse (left, right, threshold, features, left[node], depth+1)
						print(spacer + "}\n" + spacer +"else {")
						if right[node] != -1:
								recurse (left, right, threshold, features, right[node], depth+1)
						print(spacer + "}")
				else:
						target = value[node]
						for i, v in zip(np.nonzero(target)[1], target[np.nonzero(target)]):
							target_name 	= target_value_names[i]
							target_count 	= int(v)
							print(spacer + "return " + str(target_name) + \
								  " ( " + str(target_count) + " examples )")
		recurse(left, right, threshold, features, 0, 0)
	
def visualize_tree(tree, feature_names, target_name, fn="dt"):
		"""Create tree png using graphviz.
		Args
		----
		tree 			-- scikit-learn Decision Tree.
		feature_names 	-- list of feature names.
		fn 				-- [string], root of filename, default `dt`.
		"""
		
		dotfile = fn + ".dot"
		pngfile = fn + ".png"

		#tree.export_graphviz(clf, feature_names=ind_var, class_names=dep_var, filled=True, rounded=True, node_ids= True, out_file=dot_data)
		with open(dotfile, 'w') as f:
				export_graphviz(tree, out_file=f, feature_names=feature_names,class_names=target_name, filled=True, rounded=True, node_ids= True)
		command = ["dot", "-Tpng", dotfile, "-o", pngfile]
		try:
				subprocess.check_call(command)
		except:
				exit("Could not run dot, ie graphviz, ""to produce visualization")
				
########### End of all function #####################################################################################
#####################################################################################################################

#####################################################################################################################
## http://chrisstrelioff.ws/sandbox/2015/06/08/decision_trees_in_python_with_scikit_learn_and_pandas.html
#####################################################################################################################

def get_iris_data():
		#"""Get the iris data, from local csv or pandas repo."""
		if os.path.exists("iris.csv"):
			print("-- iris.csv found locally")
			df = pd.read_csv("iris.csv", index_col=0)
		else:
			print("-- trying to download from github")
			fn = "https://raw.githubusercontent.com/pydata/pandas/" + "master/pandas/tests/data/iris.csv"
			try:
				df = pd.read_csv(fn)
			except:
				exit("-- Unable to download iris.csv")

			with open("iris.csv", 'w') as f:
				print("-- writing to local iris.csv file")
				df.to_csv(f)

		return df

df = get_iris_data()

print("* df.head()", df.head(), sep="\n", end="\n\n")
print("* df.tail()", df.tail(), sep="\n", end="\n\n")
print("* iris types:", df["Name"].unique(), sep="\n")

def encode_target(df, target_column):
		"""Add column to df with integers for the target.
		Args
		----
		df 				-- pandas DataFrame.
		target_column 	-- column to map to int, producing new Target column.
		Returns
		-------
		df_mod 			-- modified DataFrame.
		targets 		-- list of target names.
		"""
		
		df_mod = df.copy()
		targets = df_mod[target_column].unique()
		map_to_int = {name: n for n, name in enumerate(targets)}
		df_mod["Target"] = df_mod[target_column].replace(map_to_int)

		return (df_mod, targets)
		
df2, targets = encode_target(df, "Name")

print("* df2.head()", df2[["Target", "Name"]].head(), sep="\n", end="\n\n")
print("* df2.tail()", df2[["Target", "Name"]].tail(), sep="\n", end="\n\n")
print("* targets", targets, sep="\n", end="\n\n")

features = list(df2.columns[:4])
print("* features:", features, sep="\n")

y = df2["Target"]
X = df2[features]
dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
dt.fit(X, y)

def visualize_tree(tree, feature_names):
    #"""Create tree png using graphviz.Args ----    tree -- scikit-learn DecsisionTree.    feature_names -- list of feature names.  """
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,feature_names=feature_names)
    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
       subprocess.check_call(command)	#Not working
    except:
        exit("Could not run dot, ie graphviz, to ""produce visualization")
		
visualize_tree(dt, features)

def get_code(tree, feature_names, target_names, spacer_base="    "):
    """Produce psuedo-code for decision tree.

    Args
    ----
    tree -- scikit-leant DescisionTree.
    feature_names -- list of feature names.
    target_names -- list of target (class) names.
    spacer_base -- used for spacing code (default: "    ").

    Notes
    -----
    based on http://stackoverflow.com/a/30104792.
    """
    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features  = [feature_names[i] for i in tree.tree_.feature]
    value = tree.tree_.value

    def recurse(left, right, threshold, features, node, depth):
        spacer = spacer_base * depth
        if (threshold[node] != -2):
            print(spacer + "if ( " + features[node] + " <= " + \
                  str(threshold[node]) + " ) {")
            if left[node] != -1:
                    recurse(left, right, threshold, features,
                            left[node], depth+1)
            print(spacer + "}\n" + spacer +"else {")
            if right[node] != -1:
                    recurse(left, right, threshold, features,
                            right[node], depth+1)
            print(spacer + "}")
        else:
            target = value[node]
            for i, v in zip(np.nonzero(target)[1],
                            target[np.nonzero(target)]):
                target_name = target_names[i]
                target_count = int(v)
                print(spacer + "return " + str(target_name) + \
                      " ( " + str(target_count) + " examples )")

    recurse(left, right, threshold, features, 0, 0)
	
get_code(dt, features, targets)


#######################################################################################################################################
'''
## http://scikit-learn.org/stable/modules/tree.html
#----------------------------------------------
iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

with open("iris.dot", 'w') as f:
		f = tree.export_graphviz(clf, out_file=f)
		
#-------- Method-1 --------------------------------------
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=iris.feature_names,  class_names=iris.target_names,  filled=True, rounded=True,  special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("iris.pdf")
#Image(graph.create_png())

#--------- Method-2 -------------------------------------		
os.unlink('iris.dot')
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = pydotplus.graph_from_dot_data(dot_data) 
graph.write_pdf("iris.pdf")

#######################################################################################
## http://stackoverflow.com/questions/38907220/graphviz-executables-not-found

clf = tree.DecisionTreeClassifier()
iris = load_iris()
clf = clf.fit(iris.data, iris.target)
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")
	
#### Not Tested ##########################################################################################

## https://statcompute.wordpress.com/2012/12/05/decision-tree-with-python/
from sklearn import tree
from pandas import *

# Provide Data 
data = read_table('/home/liuwensui/Documents/data/credit_count.txt', sep = ',')
Y = data[data.CARDHLDR == 1].BAD
X = data[data.CARDHLDR == 1][['AGE', 'ADEPCNT', 'MAJORDRG', 'MINORDRG', 'INCOME', 'OWNRENT']]
clf = tree.DecisionTreeClassifier(min_samples_leaf = 500)
clf = clf.fit(X, Y)
from StringIO import StringIO
out = StringIO()
out = tree.export_graphviz(clf, out_file = out)
# OUTPUT DOT LANGUAGE SCRIPTS
print out.getvalue()
'''
############################################################################################################
#### http://www.patricklamle.com/Tutorials/Decision%20tree%20python/tuto_decision%20tree.html
#### https://github.com/arthur-e/Programming-Collective-Intelligence/blob/master/chapter7/treepredict.py
########################################################################################################

from __future__ import print_function

my_data=[['slashdot','USA','yes',18,'None'],
        ['google','France','yes',23,'Premium'],
        ['digg','USA','yes',24,'Basic'],
        ['kiwitobes','France','yes',23,'Basic'],
        ['google','UK','no',21,'Premium'],
        ['(direct)','New Zealand','no',12,'None'],
        ['(direct)','UK','no',21,'Basic'],
        ['google','USA','no',24,'Premium'],
        ['slashdot','France','yes',19,'None'],
        ['digg','USA','no',18,'None'],
        ['google','UK','no',18,'None'],
        ['kiwitobes','UK','no',19,'None'],
        ['digg','New Zealand','yes',12,'Basic'],
        ['slashdot','UK','no',21,'None'],
        ['google','UK','yes',18,'Basic'],
        ['kiwitobes','France','yes',19,'Basic']]

# Divides a set on a specific column. Can handle numeric or nominal values
def divideset(rows,column,value):
	   # Make a function that tells us if a row is in the first group (true) or the second group (false)
	   split_function=None
	   if isinstance(value,int) or isinstance(value,float): # check if the value is a number i.e int or float
		  split_function=lambda row:row[column]>=value
	   else:
		  split_function=lambda row:row[column]==value
	   
	   # Divide the rows into two sets and return them
	   set1=[row for row in rows if split_function(row)]
	   set2=[row for row in rows if not split_function(row)]
	   return (set1,set2)

divideset(my_data,2,'yes')
divideset(my_data,3,20)

# Create counts of possible results (the last column of each row is the result)
def uniquecounts(rows):
	   results={}
	   for row in rows:
		  # The result is the last column
		  r=row[len(row)-1]
		  if r not in results: results[r]=0
		  results[r]+=1
	   return results


print(uniquecounts(my_data))
print(divideset(my_data,3,20)[0])
print(uniquecounts(divideset(my_data,3,20)[0]))
print("")
print(divideset(my_data,3,20)[1])
print(uniquecounts(divideset(my_data,3,20)[1]))

# Entropy is the sum of p(x)log(p(x)) across all the different possible results
def entropy(rows):
	   from math import log
	   log2=lambda x:log(x)/log(2)  
	   results=uniquecounts(rows)
	   # Now calculate the entropy
	   ent=0.0
	   for r in results.keys():
		  p=float(results[r])/len(rows)
		  ent=ent-p*log2(p)
	   return ent
	   
set1,set2=divideset(my_data,3,20)
entropy(set1), entropy(set2)
entropy(my_data)	   


class decisionnode:
		def __init__(self,col=-1,value=None,results=None,tb=None,fb=None):
				self.col=col
				self.value=value
				self.results=results
				self.tb=tb
				self.fb=fb
				
def buildtree(rows,scoref=entropy): #rows is the set, either whole dataset or part of it in the recursive call, scoref is the method to measure heterogeneity. By default it's entropy.
				if len(rows)==0: return decisionnode() #len(rows) is the number of units in a set
				current_score=scoref(rows)
				 # Set up some variables to track the best criteria
				best_gain=0.0
				best_criteria=None
				best_sets=None				 
				column_count=len(rows[0])-1   #count the # of attributes/columns. It's -1 because the last one is the target attribute and it does not count.
				for col in range(0,column_count):
						# Generate the list of all possible different values in the considered column
						global column_values        #Added for debugging
						column_values={}            
						for row in rows:
								column_values[row[col]]=1   
						# Now try dividing the rows up for each value in this column
						for value in column_values.keys(): #the 'values' here are the keys of the dictionnary
								(set1,set2)=divideset(rows,col,value) #define set1 and set2 as the 2 children set of a division
							  
							  # Information gain
								p=float(len(set1))/len(rows) #p is the size of a child set relative to its parent
								gain=current_score-p*scoref(set1)-(1-p)*scoref(set2) #cf. formula information gain
								if gain>best_gain and len(set1)>0 and len(set2)>0: #set must not be empty
										best_gain=gain
										best_criteria=(col,value)
										best_sets=(set1,set2)						
				 # Create the sub branches   
				if best_gain>0:
						trueBranch=buildtree(best_sets[0])
						falseBranch=buildtree(best_sets[1])
						return decisionnode(col=best_criteria[0],value=best_criteria[1],tb=trueBranch,fb=falseBranch)
				else:
						return decisionnode(results=uniquecounts(rows))

tree=buildtree(my_data)

print(tree.col)
print(tree.value)
print(tree.results)
print("")
print(tree.tb.col)
print(tree.tb.value)
print(tree.tb.results)
print("")
print(tree.tb.tb.col)
print(tree.tb.tb.value)
print(tree.tb.tb.results)
print("")
print(tree.tb.fb.col)
print(tree.tb.fb.value)
print(tree.tb.fb.results)
						

def printtree(tree,indent=''):
	   # Is this a leaf node?
		if tree.results!=None:
				print(str(tree.results))
		else:
				print(str(tree.col)+':'+str(tree.value)+'? ')
				# Print the branches
				print(indent+'T->', end=" ")
				printtree(tree.tb,indent+'  ')
				print(indent+'F->', end=" ")
				printtree(tree.fb,indent+'  ')
	  

printtree(tree)

def getwidth(tree):
	  if tree.tb==None and tree.fb==None: return 1
	  return getwidth(tree.tb)+getwidth(tree.fb)

def getdepth(tree):
	  if tree.tb==None and tree.fb==None: return 0
	  return max(getdepth(tree.tb),getdepth(tree.fb))+1


from PIL import Image,ImageDraw

def drawtree(tree,jpeg='tree.jpg'):
	  w=getwidth(tree)*100
	  h=getdepth(tree)*100+120

	  img=Image.new('RGB',(w,h),(255,255,255))
	  draw=ImageDraw.Draw(img)

	  drawnode(draw,tree,w/2,20)
	  img.save(jpeg,'JPEG')
  
def drawnode(draw,tree,x,y):
	  if tree.results==None:
		# Get the width of each branch
		w1=getwidth(tree.fb)*100
		w2=getwidth(tree.tb)*100

		# Determine the total space required by this node
		left=x-(w1+w2)/2
		right=x+(w1+w2)/2

		# Draw the condition string
		draw.text((x-20,y-10),str(tree.col)+':'+str(tree.value),(0,0,0))

		# Draw links to the branches
		draw.line((x,y,left+w1/2,y+100),fill=(255,0,0))
		draw.line((x,y,right-w2/2,y+100),fill=(255,0,0))
		
		# Draw the branch nodes
		drawnode(draw,tree.fb,left+w1/2,y+100)
		drawnode(draw,tree.tb,right-w2/2,y+100)
	  else:
		txt=' \n'.join(['%s:%d'%v for v in tree.results.items()])
		draw.text((x-20,y),txt,(0,0,0))
				
drawtree(tree,jpeg='treeview.jpg')
				
def classify(observation,tree):
	  if tree.results!=None:
			return tree.results
	  else:
			v=observation[tree.col]
			branch=None
			if isinstance(v,int) or isinstance(v,float):
				  if v>=tree.value: branch=tree.tb
				  else: branch=tree.fb
			else:
				  if v==tree.value: branch=tree.tb
				  else: branch=tree.fb
			return classify(observation,branch)				

classify(['(direct)','USA','yes',5],tree)
classify(['(direct)','USA','no',23],tree)

###########################################
def get_predicted_class_n_probability_in_original_data(decision_tree, independent_variable, df_string, output_csv_file_name ):
		#Merge predicted class with original data
		input_data_with_predicted_class = pd.DataFrame()
		input_data_with_predicted_class = pd.concat([df, pd.DataFrame(decision_tree.predict(independent_variable),columns=['predicted_class'])], axis=1, join_axes=[df.index])	
		#Probability of each class
		probability_class=decision_tree.predict_proba(independent_variable)
		#add probability in original data
		input_data_with_predicted_class = pd.concat([input_data_with_predicted_class, pd.DataFrame(probability_class,columns=['N_probability_class','Y_probability_class'])], axis=1, join_axes=[input_data_with_predicted_class.index])
		#Export original data with predicted class & probability
		if df_string is not None:
			df_string.columns = [str(col) + '_string' for col in df_string.columns]
			input_data_with_predicted_class = pd.concat([input_data_with_predicted_class, df_string], axis=1, join_axes=[input_data_with_predicted_class.index])
		#Export data in CSV
		input_data_with_predicted_class.to_csv(output_csv_file_name+'.csv',index=False)
		print (input_data_with_predicted_class.head())
		return input_data_with_predicted_class