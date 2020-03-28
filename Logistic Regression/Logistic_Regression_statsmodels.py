###################################################################################################################################################################
###### Logistic Regression ##############################################################################################################################################
###################################################################################################################################################################

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
from sklearn import tree, metrics, datasets, model_selection#, cross_validation
from sklearn.model_selection import train_test_split #cross_validation
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score, r2_score, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier, export_graphviz, _tree, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import statsmodels.api as sm
from sklearn.externals.six import StringIO
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_iris, make_blobs
import pickle
print ('-------------------------------------------------------------------------------------------------')

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


def iterate_tree_in_random_forest(no_iteration,independent_variable,dependent_variable):
		#Running a different number of trees and see the effect of that on the accuracy of the prediction
		trees=range(no_iteration)
		accuracy=np.zeros(no_iteration)
		for idx in range(len(trees)):
			   classifier			= RandomForestClassifier(n_estimators=idx + 1)
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

def Performance_of_Random_Forest_Classifier(independent_variable_name, decision_tree, dependent_variable, independent_variable, predicted_class ):
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
os.chdir("C:\\Users\\AMIT\\Google Drive\\Study\ML\\06.Random_Forest")

#Read csv data
input_data					= pd.read_csv('Dev1_Hilton_Model_Data.csv')

#Prepare data. Conver string to number
LabelEncoder_mapping, string_var_list, numeric_var_list, df_string, df_numeric, df = prepare_data(input_data)

#df							= input_data[:]
#df['Intercept']			=1

#string_var_list, numeric_var_list, df_string, df_numeric, df = prepare_data(input_data)
#Name of dependent variable
dependent_variable_name 	= 'reservation'
#level names for dependent variable. dependent_variable_value=list(df[dependent_variable_name].unique())
dependent_variable_value 	= ['non-reservation','reservation']
independent_variable_name 	= list(df.columns)
independent_variable_name.remove(dependent_variable_name)

#independent_variable 		= df[independent_variable_name].values
#dependent_variable 		= df[dependent_variable_name].values
independent_variable 		= df[independent_variable_name]
dependent_variable 			= df[dependent_variable_name]

#### Validation #########
input_val_data				= pd.read_csv('Val2_Hilton_Data.csv')
if df_string is not None:
		df_string_val = input_val_data[string_var_list]
else :
		df_string_val = None
		
#you may ger an error if value is not available in dev dataset. Fix this later.
if LabelEncoder_mapping is not None: 
		df_val	= input_val_data.apply(lambda x: LabelEncoder_mapping[x.name].transform(x))
else:
		df_val	= input_val_data[:]
		
#independent_variable_val 	= df_val[independent_variable_name].values
#dependent_variable_val 	= df_val[dependent_variable_name].values

independent_variable_val 	= df_val[independent_variable_name]
dependent_variable_val 		= df_val[dependent_variable_name]

############################################################################################################
#P value unavailable in skleran
''' 
clf = LogisticRegression(penalty='none', fit_intercept=True, random_state=None).fit(independent_variable, dependent_variable)
clf.score(independent_variable, dependent_variable)
clf.score(independent_variable_val, dependent_variable_val)
clf.coef_, independent_variable_name
predictions = clf.predict(independent_variable)
probs = clf.predict_proba(independent_variable)
print(classification_report(dependent_variable, predictions))
print(confusion_matrix(dependent_variable, predictions))
print(accuracy_score(dependent_variable, predictions))
print (roc_auc_score(dependent_variable, probs[:, 1]))
pd.DataFrame(zip(independent_variable_name, np.transpose(clf.coef_))) #clf.intercept_
log_loss(dependent_variable, predictions)
log_loss(dependent_variable, predictions)
'''

model = sm.Logit(dependent_variable, independent_variable)
result = model.fit(method='newton')
result.summary() #Method
result.summary2() #AIC, BIC
print (np.exp(result.params)) # odds ratios only
result.params #coefficients 


######## Check and update ################################################################################################################################
#Selecting good features by Mean decrease impurity. 
model 	= RandomForestClassifier()
scores 	= selecting_good_features(independent_variable, dependent_variable, independent_variable_name, 100, 0.3)
print sorted([(round(np.mean(score), 4), feat) for feat, score in scores.items()], reverse=True)

#Running a different number of trees and see the effect of that on the accuracy of the prediction
iterate_tree_in_random_forest(25, independent_variable, dependent_variable)
plt.savefig('Number of Trees vs Accuracy.png')
plt.show()

#Generate a simple plot of training learning curve
title 		= "Learning Curves - Random Forest"
cv 			= model_selection.ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)	#n_splits:30 (Try)
estimator 	= RandomForestClassifier()
plot_learning_curve(estimator, title, independent_variable, dependent_variable, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
plt.savefig(title+'.png')
plt.show()

#Random Search
param_grid = 	{ 	"n_estimators"  	: [1, 5],
					"criterion"         : ["gini", "entropy"],
					"max_features"      : [3, 5],
					"max_depth"         : [10, 20],
					"min_samples_split" : [2, 4] ,
					"bootstrap"			: [True, False]
				}
				
clf 	= 	RandomForestClassifier()
ts_gs 	= 	run_gridsearch(independent_variable, dependent_variable, clf, param_grid)		
for k, v in ts_gs.items(): print("parameters: {:<20s} setting: {}".format(k, v))

# n_estimators 		: 	The number of trees in the forest.
# criterion 		: 	The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.
# max_features 		: 	The number of features to consider when looking for the best split.
# max_depth 		: 	The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
# min_samples_split	: 	The minimum number of samples required to split an internal node.
# bootstrap			:	Whether bootstrap samples are used when building trees.

clf 	= RandomForestClassifier(**ts_gs)
clf.fit(independent_variable, dependent_variable)


### Performance ######################################################################################################
#Development
predicted_class	=	clf.predict(independent_variable)
print ("------------------------- Performance - Development --------------------------------------------")
Performance_of_Random_Forest_Classifier(independent_variable_name, clf, dependent_variable, independent_variable, predicted_class )
Plot_ROC(dependent_variable, predicted_class )
#plt.savefig('Receiver Operating Characteristic.png')
plt.show()
input_data_with_predicted_class = get_predicted_class_n_probability_in_original_data(df, clf, independent_variable, df_string, "RF_Output_Dev2_Hilton_Model_Data" )
print ("------------------------- END of Performance - Development --------------------------------------------")

#Validation
predicted_class_val	=	clf.predict(independent_variable_val)
print ("------------------------- Performance - Validation --------------------------------------------")
Performance_of_Random_Forest_Classifier(independent_variable_name, clf, dependent_variable_val, independent_variable_val, predicted_class_val )
Plot_ROC(dependent_variable_val, predicted_class_val )
#plt.savefig('Receiver Operating Characteristic.png')
plt.show()
input_data_with_predicted_class = get_predicted_class_n_probability_in_original_data(df_val, clf, independent_variable_val, df_string_val, "RF_Output_Val2_Hilton_Data" )
print ("------------------------- END of Performance - Validation --------------------------------------------")

##############################################################################
###### Save tree in pickle to use in future #######################
pickle.dump(clf, open("Random_Forest_Classifier_Hilton", 'wb'))
#Load decision tree
#clf = pickle.load(open("Decision_Tree_Classifier_Hilton",'r'))
#Call decision tree. below is for testing 
#print(cross_val_score(clf, independent_variable, dependent_variable, cv=10))

############################################################################################################################
##### END ##################################################################################################################
############################################################################################################################


























############################################################################################################################
############################################################################################################################
#http://stackoverflow.com/questions/35164310/random-forest-hyperparameter-tuning-scikit-learn-using-gridsearchcv
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestRegressor
digits = load_boston()
X, y = dataset.data, dataset.target
model = RandomForestRegressor(random_state=30)

param_grid = { "n_estimators"  		: [250, 300],
				"criterion"         : ["gini", "entropy"],
				"max_features"      : [3, 5],
				"max_depth"         : [10, 20],
				"min_samples_split" : [2, 4] ,
				"bootstrap"			: [True, False]}
				
grid_search = GridSearchCV(clf, param_grid, n_jobs=-1, cv=2)
grid_search.fit(X, y)
print grid_search.best_params_
###################################################################################################################################
#http://scikit-learn.org/stable/modules/ensemble.html
X, y 	= make_blobs(n_samples=10000, n_features=10, centers=100, random_state=0)

#DecisionTreeClassifier
clf 	= DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
scores 	= cross_val_score(clf, X, y)
print 	"DecisionTreeClassifier : ---- ", "\n", "Mean: ", scores.mean(), "\n", "STD: ", scores.std()

#ExtraTreesClassifier
clf 	= ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
scores 	= cross_val_score(clf, X, y)
print 	"ExtraTreesClassifier : ---- ", "\n", "Mean: ", scores.mean(), "\n", "STD: ", scores.std()

#RandomForestClassifier
clf 		= 	RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
scores 		= 	cross_val_score(clf, X, y)
print 			"RandomForestClassifier : ---- ", "\n", "Mean: ", scores.mean(), "\n", "STD: ", scores.std()
clf			=	clf.fit(X,y)
predictions	=	clf.predict(X)
propability	=	clf.predict_proba(X)
confusion_matrix(y,predictions)
accuracy_score(y, predictions)
print(clf.feature_importances_)
##########################################################################################################
## os.chdir("C:\TREES")
#Load the dataset
AH_data = pd.read_csv("tree_addhealth.csv")
#data_clean = AH_data.dropna()
sum([True for idx,row in AH_data.iterrows() if any(row.isnull())])

data_clean.dtypes
data_clean.describe()

#Split into training and testing sets
predictors 	= data_clean[['BIO_SEX','HISPANIC','WHITE','BLACK','NAMERICAN','ASIAN','age','ALCEVR1','ALCPROBS1','marever1','cocever1','inhever1', 'cigavail','DEP1','ESTEEM1','VIOL1','PASSIST','DEVIANT1','SCHCONN1','GPA1','EXPEL1','FAMCONCT','PARACTV','PARPRES']]
targets 	= data_clean.TREG1
pred_train, pred_test, tar_train, tar_test  = cross_validation.train_test_split(predictors, targets, test_size=.4)
pred_train.shape
pred_test.shape
tar_train.shape
tar_test.shape

#Build model on training data
classifier=RandomForestClassifier(n_estimators=25)
classifier=classifier.fit(pred_train,tar_train)

predictions=classifier.predict(pred_test)

sklearn.metrics.confusion_matrix(tar_test,predictions)
sklearn.metrics.accuracy_score(tar_test, predictions)


# fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(pred_train,tar_train)
# display the relative importance of each attribute
print(model.feature_importances_)


#Running a different number of trees and see the effect of that on the accuracy of the prediction
trees=range(25)
accuracy=np.zeros(25)
for idx in range(len(trees)):
   classifier=RandomForestClassifier(n_estimators=idx + 1)
   classifier=classifier.fit(pred_train,tar_train)
   predictions=classifier.predict(pred_test)
   accuracy[idx]=sklearn.metrics.accuracy_score(tar_test, predictions)
plt.cla()
plt.plot(trees, accuracy)
#########################################################################
# Random Forest Regressor 
#########################################################################

## http://blog.datadive.net/random-forest-interpretation-with-scikit-learn/

boston = load_boston()
print(boston.data.shape)
##boston.data, boston.target

rf = RandomForestRegressor()
rf.fit(boston.data[:300], boston.target[:300])

instances = boston.data[[300, 309]]
print "Instance 0 prediction:", rf.predict(instances[0])
print "Instance 1 prediction:", rf.predict(instances[1])

prediction, bias, contributions = ti.predict(rf, instances)

for i in range(len(instances)):
    print "Instance", i
    print "Bias (trainset mean)", bias[i]
    print "Feature contributions:"
    for c, feature in sorted(zip(contributions[i], boston.feature_names), key=lambda x: -abs(x[0])):
        print feature, round(c, 2)
    print "-"*20 
	
print prediction
print bias + np.sum(contributions, axis=1)


ds1 = boston.data[300:400]
ds2 = boston.data[400:]
 
print np.mean(rf.predict(ds1))
print np.mean(rf.predict(ds2))

prediction1, bias1, contributions1 = ti.predict(rf, ds1)
prediction2, bias2, contributions2 = ti.predict(rf, ds2)

totalc1 = np.mean(contributions1, axis=0) 
totalc2 = np.mean(contributions2, axis=0) 

print np.sum(totalc1 - totalc2)
print np.mean(prediction1) - np.mean(prediction2)

for c, feature in sorted(zip(totalc1 - totalc2, boston.feature_names), reverse=True):
    print feature, round(c, 2)	
	
#####################################
### Classification trees and forests
#####################################


iris = load_iris()
 
rf = RandomForestClassifier(max_depth = 4)
idx = range(len(iris.target))
np.random.shuffle(idx)
 
rf.fit(iris.data[idx][:100], iris.target[idx][:100])

instance = iris.data[idx][100:101]
print rf.predict_proba(instance)

prediction, bias, contributions = ti.predict(rf, instance)
print "Prediction", prediction
print "Bias (trainset prior)", bias
print "Feature contributions:"
for c, feature in zip(contributions[0], iris.feature_names):
    print feature, c
################################################################################
## https://www.analyticsvidhya.com/blog/2015/06/tuning-random-forest-model/
## 2.a. n_jobs : This parameter tells the engine how many processors is it allowed to use. 
## A value of “-1” means there is no restriction whereas a value of “1” means it can only use one processor. 
## Here is a simple experiment you can do with Python to check this metric :

# Only in Ipython
## %timeit
model = RandomForestRegressor(n_estimators = 100, oob_score = True,n_jobs = 1,random_state =1)
model.fit(X,y)
## Output  ———-  1 loop best of 3 : 1.7 sec per loop
## %timeit
model = RandomForestRegressor(n_estimators = 100,oob_score = True,n_jobs = -1,random_state =1)
model.fit(X,y)
## Output  ———-  1 loop best of 3 : 1.1 sec per loop

#########################################################
## Titanic data	  
os.chdir("C:\\Users\\amit.kumar\\Google Drive\\Study\\Other\\Boosting\\Random_Forest\\Kaggle - Titanic Machine Learning from Disaster")

## too many categories. it will not work for RF
x = pd.read_csv("train.csv")
y = x.pop("Survived")
## need to clean the data
model =  RandomForestRegressor(n_estimators = 100, oob_score=True, random_state = 42)
model.fit(x(numeric_variable,y)
print "AUC - ROC : ", roc_auc_score(y,model.oob_prediction)
## AUC – ROC : 0.7386
## Try runing the following code and find the optimal leaf size in the comment box.
sample_leaf_options = [1,5,10,50,100,200,500]
for leaf_size in sample_leaf_options :
	model = RandomForestRegressor(n_estimators = 200, oob_score = True, n_jobs = -1,random_state =50,max_features = "auto", min_samples_leaf = leaf_size)
	model.fit(x(numeric_variable,y)
	print "AUC - ROC : ", roc_auc_score(y,model.oob_prediction)
############################################################################################################################################################
##########################################################################################################
#http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

		"""Generate a simple plot of the test and training learning curve.
			Parameters
			----------
			estimator : object type that implements the "fit" and "predict" methods An object of that type which is cloned for each validation.
			title : string
				Title for the chart.
			X : array-like, shape (n_samples, n_features)
				Training vector, where n_samples is the number of samples and n_features is the number of features.
			y : array-like, shape (n_samples) or (n_samples, n_features), optional Target relative to X for classification or regression;
				None for unsupervised learning.
			ylim : tuple, shape (ymin, ymax), optional
				Defines minimum and maximum yvalues plotted.
			cv : int, cross-validation generator or an iterable, optional
				Determines the cross-validation splitting strategy.
				Possible inputs for cv are:
				  - None, to use the default 3-fold cross-validation,
				  - integer, to specify the number of folds.
				  - An object to be used as a cross-validation generator.
				  - An iterable yielding train/test splits.
				For integer/None inputs, if ``y`` is binary or multiclass, :class:`StratifiedKFold` used. If the estimator is not a classifier or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.
				Refer :ref:`User Guide <cross_validation>` for the various cross-validators that can be used here.

			n_jobs : integer, optional
				Number of jobs to run in parallel (default 1).
		"""
		
		plt.figure()
		plt.title(title)
		if ylim is not None:
				plt.ylim(*ylim)
		plt.xlabel("Training examples")
		plt.ylabel("Score")
		train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
		train_scores_mean = np.mean(train_scores, axis=1)
		train_scores_std = np.std(train_scores, axis=1)
		test_scores_mean = np.mean(test_scores, axis=1)
		test_scores_std = np.std(test_scores, axis=1)
		plt.grid()
		plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
		plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
		plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
		plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
		plt.legend(loc="best")
		return plt


digits = load_digits()
X, y = digits.data, digits.target

title = "Learning Curves (Random Forest)"
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = RandomForestClassifier()
plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
plt.show()


title = "Learning Curves (Naive Bayes)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
estimator = GaussianNB()
plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = SVC(gamma=0.001)
plot_learning_curve(estimator, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4)

plt.show()

############################################################################################################################
#http://blog.datadive.net/selecting-good-features-part-iii-random-forests/
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import r2_score
from collections import defaultdict
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor

boston = load_boston()
X = boston["data"]
Y = boston["target"]
names = boston["feature_names"]
rf = RandomForestRegressor()
scores = defaultdict(list)

for train_idx, test_idx in ShuffleSplit(len(X), 100, .3):
		X_train, X_test = X[train_idx], X[test_idx]
		Y_train, Y_test = Y[train_idx], Y[test_idx]
		r = rf.fit(X_train, Y_train)
		acc = r2_score(Y_test, rf.predict(X_test))
		for i in range(X.shape[1]):
			X_t = X_test.copy()
			np.random.shuffle(X_t[:, i])
			shuff_acc = r2_score(Y_test, rf.predict(X_t))
			scores[names[i]].append((acc-shuff_acc)/acc)

print sorted([(round(np.mean(score), 4), feat) for feat, score in scores.items()], reverse=True)

##################################################################################################################################
#Performance - Development
scores 				= 	model_selection.cross_val_score(clf, independent_variable, dependent_variable)
predicted_class		=	clf.predict(independent_variable)
propability			=	clf.predict_proba(independent_variable)
c_matrix			=	confusion_matrix(dependent_variable,predicted_class)	#confusion_matrix
TN, FN, TP, FP		= 	c_matrix[0][0], c_matrix[1][0], c_matrix[1][1], c_matrix[0][1]
TPR, FPR			=	(float(TP)/float(TP+FN)), (float(FP)/float(FP+TN))
Plot_ROC(dependent_variable, predicted_class )
#plt.savefig('Receiver Operating Characteristic.png')
plt.show()
print "RandomForestClassifier : ---- ", "\n", "Mean: ", scores.mean(), "\n", "STD: ", scores.std()
print "True Negative-", TN, '\n', "False Negative-", FN, '\n', "True Positive-", TP, '\n', "False Positive-",FP
print "Mean Accuracy - ", clf.score(independent_variable, dependent_variable), '\n', "Accuracy classification score - ", accuracy_score(dependent_variable, predicted_class)
print "Area Under the Curve - ", roc_auc_score(dependent_variable,predicted_class)
print "Precision, Recall ,F1-Score : ", '\n', classification_report(dependent_variable, predicted_class, target_names=dependent_variable_value, digits=4)
print(pd.DataFrame(dict(zip(independent_variable_name, clf.feature_importances_)).items(), columns=['Variable', 'Feature_Importances']).sort(['Feature_Importances'],ascending=False))

#clf.oob_score_

#Performance- Validation
scores_val 					= model_selection.cross_val_score(clf, independent_variable_val, dependent_variable_val)
predictions_class_val		= clf.predict(independent_variable_val)
propability					= clf.predict_proba(independent_variable_val)
confusion_matrix(dependent_variable_val,predictions_class_val)
print "Mean Accuracy - ", clf.score(independent_variable_val, dependent_variable_val), '\n', "Accuracy classification score - ", accuracy_score(dependent_variable_val, predictions_class_val)
print "RandomForestClassifier : ---- ", "\n", "Mean: ", scores_val.mean(), "\n", "STD: ", scores_val.std()
print classification_report(dependent_variable_val, predictions_class_val, target_names=dependent_variable_value, digits=4)
################################################################################################################################################
