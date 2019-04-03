##################################################################################################################################################
														# Neural Network
##################################################################################################################################################

from   	__future__ import print_function
import 	subprocess
import 	os
import 	sys
import 	pydotplus
from   	IPython.display import Image
import 	matplotlib.pylab as plt
import 	numpy as np
import 	pandas as pd
from   	time import time
from   	operator import itemgetter
from 	scipy.stats import randint
from 	collections import defaultdict
from 	sklearn.neural_network import MLPClassifier
from 	sklearn import tree, metrics
from 	sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score, r2_score, roc_curve, auc
from 	sklearn.externals.six import StringIO
from 	sklearn.model_selection import cross_val_score, GridSearchCV
from 	sklearn.preprocessing import LabelEncoder
import 	pickle
#from 	sklearn.tree import DecisionTreeClassifier, export_graphviz, _tree
#from 	sklearn.grid_search import GridSearchCV, RandomizedSearchCV
#from 	sklearn.model_selection import GridSearchCV
#from 	sklearn.cross_validation import cross_val_score
#from 	sklearn.datasets import load_iris
print ('-------------------------------------------------------------------------------------------------')
####################################################################################################################

##############
def Model_Performance(independent_variable_name, model, dependent_variable, independent_variable, predicted_class ):
		scores 			= cross_val_score(model, independent_variable, dependent_variable, cv=10)
		#confusion_matrix
		#cm = pd.concat([pd.concat([pd.Series(confusion_matrix(dependent_variable,predicted_class)[1][1], index=['Actual-1'], name='Predicted-1'), pd.Series(confusion_matrix(dependent_variable,predicted_class)[1][0], index=['Actual-1'], name='Predicted-0')], axis=1), pd.concat([pd.Series(confusion_matrix(dependent_variable,predicted_class)[0][1], index=['Actual-0'], name='Predicted-1'), pd.Series(confusion_matrix(dependent_variable,predicted_class)[0][0], index=['Actual-0'], name='Predicted-0')], axis=1)])
		c_matrix 		= confusion_matrix(dependent_variable,predicted_class)
		#cm = pd.concat([pd.concat([pd.Series(c_matrix[1][1], index=['Actual-1'], name='Predicted-1'), pd.Series(c_matrix[1][0], index=['Actual-1'], name='Predicted-0')], axis=1), pd.concat([pd.Series(c_matrix[0][1], index=['Actual-0'], name='Predicted-1'), pd.Series(c_matrix[0][0], index=['Actual-0'], name='Predicted-0')], axis=1)])
		TN, FN, TP, FP	= c_matrix[0][0], c_matrix[1][0], c_matrix[1][1], c_matrix[0][1]
		#cm = pd.concat([pd.concat([pd.Series(TP, index=['Actual-1'], name='Predicted-1'), pd.Series(FN, index=['Actual-1'], name='Predicted-0')], axis=1), pd.concat([pd.Series(FP, index=['Actual-0'], name='Predicted-1'), pd.Series(TN, index=['Actual-0'], name='Predicted-0')], axis=1)])
		TPR, FPR		= (float(TP)/float(TP+FN)), (float(FP)/float(FP+TN))
		print('-----------------------------------------------------------------------------')
		print('\n')
		print("coefficient of determination OR R^2 of the prediction")
		print ("Mean Accuracy - ", model.score(independent_variable, dependent_variable), '\n', "Accuracy classification score - ", accuracy_score(dependent_variable, predicted_class))
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
		#print (c_matrix)
		print('-----------------------------------------------------------------------------')
		print('\n')
		print ("True Negative-", TN, '\n', "False Negative-", FN, '\n', "True Positive-", TP, '\n', "False Positive-",FP)
		print('-----------------------------------------------------------------------------')
		print ("Area Under the Curve - ", roc_auc_score(dependent_variable,predicted_class))
		print('-----------------------------------------------------------------------------')

################
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

######
def run_gridsearch(X, y, clf, param_grid, cv=5):
		grid_search = GridSearchCV(clf,param_grid=param_grid,cv=cv)
		start 		= time()
		grid_search.fit(X, y)
		print(("\nGridSearchCV took {:.2f} ""seconds for {:d} candidate ""parameter settings.").format(time() - start,len(grid_search.grid_scores_)))
		top_params = report(grid_search.grid_scores_, 3)
		return  top_params

def report(grid_scores, n_top=3):
		top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
		for i, score in enumerate(top_scores):
				print("Model with rank: {0}".format(i + 1))
				print(("Mean validation score: ""{0:.3f} (std: {1:.3f})").format(score.mean_validation_score,np.std(score.cv_validation_scores)))
				print("Parameters: {0}".format(score.parameters))
				print("")
		return top_scores[0].parameters
		
####################################################################################################################

os.chdir('C:\\Users\\amit.kumar\\Google Drive\\Study\\Other\\08.NN')
#Dev data
input_data					= pd.read_csv('Dev_Data.csv')
dependent_variable_name 	= 'reservation'
dependent_variable_value 	= ['non-reservation','reservation']
df							= input_data[:]
independent_variable_name 	= list(df.columns)
independent_variable_name.remove(dependent_variable_name)
independent_variable 		= df[independent_variable_name].values
dependent_variable 			= df[dependent_variable_name].values

#Val data
input_data_val				= pd.read_csv('Val_Data.csv')
df_val						= input_data_val[:]
independent_variable_val 	= df_val[independent_variable_name].values
dependent_variable_val 		= df_val[dependent_variable_name].values

################################################

#clf = MLPClassifier(random_state=1)
clf = MLPClassifier()
clf = clf.fit(independent_variable, dependent_variable)

[coef.shape for coef in clf.coefs_]

########################
param_grid = 	[	{'hidden_layer_sizes'	: [(nb,) for nb in range(20,500,10)			]}	,
					{'activation'			: ['logistic', 	'relu'						]}	,
					{'solver'				: ['lbfgs', 	'sgd',			'adam'		]}	,
					{'alpha'				: [a/100 for a  in range(0,100,5)			]}	,
					{'learning_rate'		: ['constant', 'invscaling', 	'adaptive'	]}
				]
				
nn = GridSearchCV(MLPClassifier(random_state=1), param_grid)
nn.fit(independent_variable, dependent_variable)
nn.best_estimator_

#clf 	= MLPClassifier(random_state=1)
#ts_gs 	= 	run_gridsearch(independent_variable, dependent_variable, clf, param_grid)				
#for k, v in ts_gs.items(): print("parameters: {:<20s} setting: {}".format(k, v))
clf 	= MLPClassifier(**ts_gs)
clf.fit(independent_variable, dependent_variable)


################################################
#Performance - Development
predicted_class = clf.predict(independent_variable)
Model_Performance(independent_variable_name, clf, dependent_variable, independent_variable, predicted_class )
Plot_ROC(dependent_variable, predicted_class )

#Performance - Validation
predicted_class_val = clf.predict(independent_variable_val)
Model_Performance(independent_variable_name, clf, dependent_variable_val, independent_variable_val, predicted_class_val )
Plot_ROC(dependent_variable_val, predicted_class_val )
plt.show()


####################################################################################################################
####################################################################################################################
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

# http://www.xavierdupre.fr/app/ensae_teaching_cs/helpsphinx/notebooks/solution_2016_credit_clement.html

# optimisation - choix du nombre de couches
param_grid = [{'hidden_layer_sizes': [(nb,) for nb in range(20,500,10)]}, {'alpha': [a/100 for a in range(0,100,5)]}]
neural2 = GridSearchCV(MLPClassifier(), param_grid)
neural2.fit(X_train, Y_train)
neural2.best_estimator_
###
neural = MLPClassifier(	activation='relu', 	alpha=0.0001, batch_size='auto', beta_1=0.9, beta_2=0.999, 		early_stopping=False, epsilon=1e-08, hidden_layer_sizes=(170,), learning_rate='constant',
						learning_rate_init=0.001, max_iter=200, momentum=0.9, nesterovs_momentum=True, power_t=0.5, random_state=None, shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1, 
						verbose=False, warm_start=False)
###
neural.fit(X_train, Y_train)
predicted = neural.predict(X_cross)
print(metrics.confusion_matrix(expected, predicted))
neural.predict_proba(X_cross[:50])


###################################################

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from pandas import DataFrame

X, y = make_classification(random_state=42)
gs = GridSearchCV(MLPClassifier(random_state=42), param_grid=[{'learning_rate': ['constant', 'invscaling', 'adaptive'], 'solver': ['sgd',]}, {'solver': ['adam',]}])
DataFrame(gs.fit(X, y).cv_results_)

####################################################

from sklearn.neural_network import *
mlp = MLPClassifier(solver='lbfgs',  hidden_layer_sizes=(100, ), random_state=1)
#paramgrid = {'estimator__alpha':logspace(-3,2,20),}	# not correct
mlpcv = grid_search.GridSearchCV(mlp, paramgrid, cv = 5)
mlpcv.fit(trainXtf, trainY)
print mlpcv.best_params_

####################################################
# http://www.gequest.com/am00634/d/primaryobjects/voicegender/anas1/run/419353

##Neural Network - Classifier. ifferent solver and number of hidden layers
print("Neural Network - Classifier")
clf = MLPClassifier(activation = 'logistic',solver='lbfgs', alpha=0.0001, random_state=1)
clf.fit(X_train,y_train)
predictedlabel = clf.predict(X_test)
precision 	= metrics.precision_score(y_test, predictedlabel, average='weighted', sample_weight=None)
accuracy 	= metrics.accuracy_score(y_test, predictedlabel, normalize=True, sample_weight=None)
print("Accuracy is: " + str(accuracy))
print("Precision is: " + str(precision))

print("Neural Network - Classifier with different hidden layers and solver")
clf = MLPClassifier(activation = 'logistic',solver='adam', hidden_layer_sizes=(200,25), alpha=0.001, random_state=1,max_iter=300)
clf.fit(X_train,y_train)
predictedlabel = clf.predict(X_test)
precision 	= metrics.precision_score(y_test, predictedlabel, average='weighted', sample_weight=None)
accuracy 	= metrics.accuracy_score(y_test, predictedlabel, normalize=True, sample_weight=None)
print("Accuracy is: " + str(accuracy))
print("Precision is: " + str(precision))

##Neural Netowkr - Regressor
print("Neural Network - Regression")
clf = MLPRegressor(activation = 'logistic',solver='lbfgs', alpha=0.0001, random_state=1)
clf.fit(X_train,y_train)
predictedlabel = clf.predict(X_test)
##Changing less than 0.5 to 0 and greater than equal to 0.5 to 1
predictedlabel [predictedlabel<0.5]  = 0
predictedlabel [predictedlabel>=0.5] = 1
precision = metrics.precision_score(y_test, predictedlabel, average='weighted', sample_weight=None)
accuracy = metrics.accuracy_score(y_test, predictedlabel, normalize=True, sample_weight=None)
print("Accuracy is: " + str(accuracy))
print("Precision is: " + str(precision))

print("Neural Network Regressor with different number of hidden layers")
clf = MLPRegressor(activation = 'logistic',solver='adam', hidden_layer_sizes=(200,25), alpha=0.001, random_state=1,max_iter=300)
clf.fit(X_train,y_train)
predictedlabel = clf.predict(X_test)
predictedlabel [predictedlabel<0.5] = 0
predictedlabel [predictedlabel>=0.5] = 1
precision = metrics.precision_score(y_test, predictedlabel, average='weighted', sample_weight=None)
accuracy = metrics.accuracy_score(y_test, predictedlabel, normalize=True, sample_weight=None)
print("Accuracy is: " + str(accuracy))
print("Precision is: " + str(precision))

####################################################
# https://anaconda.org/hhllcks/monsters-nn/notebook

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import neural_network
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

parameters 			= {'solver': ['lbfgs'], 'max_iter': [500,1000,1500], 'alpha': 10.0 ** -np.arange(1, 7), 'hidden_layer_sizes':np.arange(5, 12), 'random_state':[0,1,2,3,4,5,6,7,8,9]}
clf_grid 			= GridSearchCV(neural_network.MLPClassifier(), parameters, n_jobs=-1)
clf_grid_hair_soul 	= GridSearchCV(neural_network.MLPClassifier(), parameters, n_jobs=-1)
clf_grid.fit(x,y.values.ravel())
clf_grid_hair_soul.fit(x_hair_soul,y.values.ravel())
print("-----------------Original Features--------------------")
print("Best score: %0.4f" % clf_grid.best_score_)
print("Using the following parameters:")
print(clf_grid.best_params_)
print("------------------------------------------------------")
clf = neural_network.MLPClassifier(alpha=0.001, hidden_layer_sizes=(6), max_iter=500, random_state=3, solver='lbfgs')
clf.fit(x, y.values.ravel())
preds = clf.predict(x_test)
sub = pd.DataFrame(preds)
pd.concat([testset["id"],sub], axis=1).rename(columns = {0: 'type'}).to_csv("submission_neural_net.csv", index=False)
####################################################
