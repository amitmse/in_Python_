###################################################################################################################################################################
###### Random FOrest ##############################################################################################################################################
###################################################################################################################################################################
## http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
## https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/ 
## http://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting
## https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
###################################################################################################################################################################

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
from sklearn import tree, metrics, datasets, model_selection, cross_validation
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score, r2_score, roc_curve, auc, mean_squared_error
from sklearn.tree import DecisionTreeClassifier, export_graphviz, _tree, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.externals.six import StringIO
from sklearn.preprocessing import LabelEncoder
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_iris, make_blobs
import pickle
print ('-------------------------------------------------------------------------------------------------')

#from treeinterpreter import treeinterpreter as ti
############################################################################################################

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

def iterate_tree_in_Gradient_Boosting_Classifier(no_iteration,independent_variable,dependent_variable):
		#Running a different number of trees and see the effect of that on the accuracy of the prediction
		trees=range(no_iteration)
		accuracy=np.zeros(no_iteration)
		for idx in range(len(trees)):
			   classifier			= GradientBoostingClassifier(n_estimators=idx + 1)
			   classifier			= classifier.fit(independent_variable, dependent_variable)
			   predicted_class		= classifier.predict(independent_variable)
			   accuracy[idx]		= accuracy_score(dependent_variable, predicted_class)
		plt.cla()
		plt.title("Number of Trees vs Accuracy")
		plt.xlabel("Number of Trees")
		plt.ylabel("Accuracy")
		plt.plot(trees, accuracy)
		#plt.show()
		return plt
		
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
		#Generate a simple plot of the test and training learning curve
		##http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
		plt.figure()
		plt.title(title)
		if ylim is not None:
				plt.ylim(*ylim)
		plt.xlabel("Training examples")
		plt.ylabel("Score")
		train_sizes, train_scores, test_scores = model_selection.learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
		train_scores_mean 	= np.mean(train_scores, axis=1)
		train_scores_std 	= np.std(train_scores,  axis=1)
		test_scores_mean 	= np.mean(test_scores,  axis=1)
		test_scores_std 	= np.std(test_scores,   axis=1)
		plt.grid()
		plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
		plt.fill_between(train_sizes, test_scores_mean  - test_scores_std , test_scores_mean  + test_scores_std , alpha=0.1, color="g")
		plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
		plt.plot(train_sizes, test_scores_mean , 'o-', color="g", label="Cross-validation score")
		plt.legend(loc = "best")
		return plt

def selecting_good_features(independent_variable, dependent_variable, independent_variable_name, no_iteration, validation_sample_size):
		#Selecting good features by Mean decrease impurity. 
		#http://blog.datadive.net/selecting-good-features-part-iii-random-forests/
		names = independent_variable_name
		scores = defaultdict(list)
		#no_iteration: generate no of samples, validation_sample_size: sperate validation data in proportation 
		for train_idx, test_idx in cross_validation.ShuffleSplit(len(independent_variable), no_iteration, validation_sample_size):
				X_train, X_test = independent_variable[train_idx], 	independent_variable[test_idx]
				Y_train, Y_test = dependent_variable[train_idx], 	dependent_variable[test_idx]
				r 				= model.fit(X_train, Y_train)
				#coefficient_of_determination
				coeff_of_deter 	= r2_score(Y_test, model.predict(X_test))
				for i in range(independent_variable.shape[1]):
						X_t = X_test.copy()
						np.random.shuffle(X_t[:, i])
						shuff_coeff_of_deter = r2_score(Y_test, model.predict(X_t))
						scores[names[i]].append((coeff_of_deter - shuff_coeff_of_deter)/coeff_of_deter)
		return scores
		
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
		
def Performance_of_Gradient_Boosting_Classifier(independent_variable_name, decision_tree, dependent_variable, independent_variable, predicted_class ):
		scores 			= model_selection.cross_val_score(decision_tree, independent_variable, dependent_variable, cv=10)
		c_matrix 		= confusion_matrix(dependent_variable,predicted_class)
		TN, FN, TP, FP	= c_matrix[0][0], c_matrix[1][0], c_matrix[1][1], c_matrix[0][1]
		TPR, FPR		= (float(TP)/float(TP+FN)), (float(FP)/float(FP+TN))
		print('-----------------------------------------------------------------------------')
		print("Variable importance")
		print(pd.DataFrame(dict(zip(independent_variable_name, decision_tree.feature_importances_)).items(), columns=['Variable', 'Feature_Importances']).sort(['Feature_Importances'],ascending=False))
		print('-----------------------------------------------------------------------------')
		print('\n')
		print("coefficient of determination OR R^2 of the prediction")
		print ("Mean Accuracy - ", decision_tree.score(independent_variable, dependent_variable), '\n', "Accuracy classification score - ", accuracy_score(dependent_variable, predicted_class))
		print('-----------------------------------------------------------------------------')
		print('\n')		
		print("Cross-Validation")
		print ("RandomForestClassifier : ---- ", "\n", "Mean: ", scores.mean(), "\n", "STD: ", scores.std())
		print('-----------------------------------------------------------------------------')
		print('\n')
		print("Classification Report")
		print ("Precision, Recall ,F1-Score : ", '\n', classification_report(dependent_variable, predicted_class, target_names=dependent_variable_value, digits=4))
		print('-----------------------------------------------------------------------------')
		print('\n')
		print("Confusion Matrix")
		print (c_matrix)
		print('-----------------------------------------------------------------------------')
		print('\n')
		print ("True Negative-", TN, '\n', "False Negative-", FN, '\n', "True Positive-", TP, '\n', "False Positive-",FP)
		print('-----------------------------------------------------------------------------')
		print ("Area Under the Curve - ", roc_auc_score(dependent_variable,predicted_class))
		print('-----------------------------------------------------------------------------')

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

		
############################################################################################################
os.chdir("C:\\Users\\amit.kumar\\Google Drive\\Study\\Other\\07.Boosting")
#Read csv data
input_data					= pd.read_csv('Dev1_Hilton_Model_Data.csv')
#Prepare data. Conver string to number
#LabelEncoder_mapping, string_var_list, numeric_var_list, df_string, df_numeric, df = prepare_data(input_data)
df							= input_data[:]
#string_var_list, numeric_var_list, df_string, df_numeric, df = prepare_data(input_data)
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

df_val	= input_val_data[:]

if df_string is not None:
		df_string_val = input_val_data[string_var_list]
else :
		df_string_val = None

#you may ger an error if value is not available in dev dataset. Fix this later.
if LabelEncoder_mapping is not None: 
		df_val	= input_val_data.apply(lambda x: LabelEncoder_mapping[x.name].transform(x))
else:
		df_val	= input_val_data[:]
	
independent_variable_val 	= df_val[independent_variable_name].values
dependent_variable_val 		= df_val[dependent_variable_name].values

############################################################################################################
### Build Random Forest - Development #######################

### Default [add later #oob_score=True, ]

#clf = 	GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(independent_variable, dependent_variable)
clf = 	GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
clf = 	clf.fit(independent_variable, dependent_variable)


#Selecting good features by Mean decrease impurity. 
model 	= GradientBoostingClassifier()
scores 	= selecting_good_features(independent_variable, dependent_variable, independent_variable_name, 100, 0.3)
print sorted([(round(np.mean(score), 4), feat) for feat, score in scores.items()], reverse=True)

#Running a different number of trees and see the effect of that on the accuracy of the prediction
iterate_tree_in_Gradient_Boosting_Classifier(25, independent_variable, dependent_variable)
plt.savefig('Number of Trees vs Accuracy.png')
plt.show()

#Generate a simple plot of training learning curve
title 		= "Learning Curves - Gradient Boosting Classifier"
cv 			= model_selection.ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)	#n_splits:30 (Try)
estimator 	= GradientBoostingClassifier()
plot_learning_curve(estimator, title, independent_variable, dependent_variable, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
plt.savefig(title+'.png')
plt.show()

#Random Search
#param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
#gsearch5 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=5,max_depth=9,min_samples_split=10, min_samples_leaf=10, subsample=0.8, random_state=10,max_features=2),param_grid = param_test5, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
#gsearch5.fit(independent_variable,dependent_variable)
#gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_

param_grid = 	{ 	"learning_rate"		: [0.001, 0.4],
					"n_estimators"  	: [1, 60],					
					"max_depth"         : [1, 30], 
					"min_samples_split" : [2, 40],
					"min_samples_leaf"  : [1, 20] ,
					"subsample"			: [0.3, 0.9],
					"max_features"      : [1, 5]
				}
				
clf 	= 	GradientBoostingClassifier()
ts_gs 	= 	run_gridsearch(independent_variable, dependent_variable, clf, param_grid)
for k, v in ts_gs.items(): print("parameters: {:<20s} setting: {}".format(k, v))

# n_estimators 		: 	The number of trees in the forest.
# criterion 		: 	The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.
# max_features 		: 	The number of features to consider when looking for the best split.
# max_depth 		: 	The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
# min_samples_split	: 	The minimum number of samples required to split an internal node.
# bootstrap			:	Whether bootstrap samples are used when building trees.

clf 	= GradientBoostingClassifier(**ts_gs)
clf.fit(independent_variable, dependent_variable)


### Performance ######################################################################################################
#Development
predicted_class	=	clf.predict(independent_variable)
print ("------------------------- Performance - Development --------------------------------------------")
Performance_of_Gradient_Boosting_Classifier(independent_variable_name, clf, dependent_variable, independent_variable, predicted_class )
Plot_ROC(dependent_variable, predicted_class )
#plt.savefig('Receiver Operating Characteristic.png')
plt.show()
input_data_with_predicted_class = get_predicted_class_n_probability_in_original_data(df, clf, independent_variable, df_string, "RF_Output_Dev2_Hilton_Model_Data" )
print ("------------------------- END of Performance - Development --------------------------------------------")

#Validation
predicted_class_val	=	clf.predict(independent_variable_val)
print ("------------------------- Performance - Validation --------------------------------------------")
Performance_of_Gradient_Boosting_Classifier(independent_variable_name, clf, dependent_variable_val, independent_variable_val, predicted_class_val )
Plot_ROC(dependent_variable_val, predicted_class_val )
#plt.savefig('Receiver Operating Characteristic.png')
plt.show()
input_data_with_predicted_class = get_predicted_class_n_probability_in_original_data(df_val, clf, independent_variable_val, df_string_val, "RF_Output_Val2_Hilton_Data" )
print ("------------------------- END of Performance - Validation --------------------------------------------")

##############################################################################
###### Save tree in pickle to use in future #######################
pickle.dump(clf, open("Gradient_Boosting_Classifier_Hilton", 'wb'))
#Load decision tree
#clf = pickle.load(open("Decision_Tree_Classifier_Hilton",'r'))
#Call decision tree. below is for testing 
#print(cross_val_score(clf, independent_variable, dependent_variable, cv=10))

############################################################################################################################
##### END ##################################################################################################################
############################################################################################################################

#https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/
#Choose all predictors except target & IDcols
predictors = [x for x in train.columns if x not in [target, IDcol]]
gbm0 = GradientBoostingClassifier(random_state=10)
modelfit(gbm0, train, predictors)
