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
input_data			= pd.read_csv('Dev1_Hilton_Model_Data.csv')
df				= input_data[:]
df['Intercept']			=1

#Name of dependent variable
dependent_variable_name 	= 'reservation'
dependent_variable_value 	= ['non-reservation','reservation']
independent_variable_name 	= list(df.columns)
independent_variable_name.remove(dependent_variable_name)

independent_variable 		= df[independent_variable_name]
dependent_variable 		= df[dependent_variable_name]

#### Validation #########
input_val_data			= pd.read_csv('Val2_Hilton_Data.csv')
df_val				= input_val_data[:]
independent_variable_val 	= df_val[independent_variable_name]
dependent_variable_val 		= df_val[dependent_variable_name]

############################################################################################################
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
pd.read_html(result.summary().tables[1].as_html(), header=0, index_col=0)[0]

nan_value = float("NaN")
summary1 = {}
for item in result.summary().tables[0].data:
    summary1[item[0].strip()] = item[1].strip()
    summary1[item[2].strip()] = item[3].strip()
summary1 = pd.DataFrame(summary1.items(), columns=['Metrics', 'Value'])
summary1.replace("", nan_value, inplace=True)
summary1.dropna(subset = ['Metrics'], inplace=True)
summary1 = summary1[~summary1.Metrics.isin(['Dep. Variable:','Date:','Time:'])]

summary2_1=result.summary2().tables[0].loc[:, [0, 1]]
summary2_1.rename(columns = {0:'Metrics',1:'Value'}, inplace = True)
summary2_1.replace("", nan_value, inplace=True)
summary2_1.dropna(subset = ['Metrics'], inplace=True)
summary2_1 = summary2_1[~summary2_1.Metrics.isin(['Model:','No. Observations:','Df Model:','Df Residuals:','Converged:'])]

summary2_2=result.summary2().tables[0].loc[:, [2, 3]]
summary2_2.rename(columns = {2:'Metrics',3:'Value'}, inplace = True)
summary2_2.replace("", nan_value, inplace=True)
summary2_2.dropna(subset = ['Metrics'], inplace=True)
summary2_2 = summary2_2[~summary2_2.Metrics.isin(['Pseudo R-squared:','Log-Likelihood:','LL-Null:','LLR p-value:'])]

summary_all = None
summary_all=pd.concat([summary1,summary2_1,summary2_2])
summary_all.reset_index(drop=True, inplace=True)

result.summary2().tables[1] #more decimal places

######## Check and update ################################################################################################################################
'''
def probability_calculation(coefficient):
		#calculate Yhat or probability or predicted value
		def logit_function(independent_variable_and_coefficient):
		#Logit function/Sigmoid Function
			return 1 / (1 + np.exp(-independent_variable_and_coefficient))

		return logit_function(np.dot(independent_variable, coefficient))


#actual_score = np.column_stack((dependent_variable, np.around(1000*probability_calculation(output_estimate_logistic_model[0]))))
'''

#predictions = result.predict(independent_variable)
df['odds']=-3.670706 + (0.022926*df['NOP_before_purchase']) - (0.061525*df['nop_last_visit']) - (0.913963*df['no_of_visits_last_7_days']) + (3.176300*df['no_of_purchases_last_7_days']) + (0.195037*df['Hilton_Honors_Status_Ever_flag'])
df['prob']= 1 / (1 + np.exp(-df['odds']))

def calculate_ROC(input=None, target=None,score=None ):
        actual_predicted 		= 	np.column_stack((input[target], input[score]))
        actual_predicted.dtype 	= 	{'names':['target', 'predicted_value'], 'formats':[np.float64, np.float64]}
        concordant 				= 	0
        discordant 				= 	0
        tied					=	0
        for i in actual_predicted[actual_predicted['target']==1]['predicted_value']:
            for j in actual_predicted[actual_predicted['target']==0]['predicted_value']:
                if i > j:
                    concordant = concordant + 1
                elif j > i:
                    discordant = discordant + 1
                else:
                    tied  = tied + 1		

        total_pairs 			= 	concordant + discordant + tied
        percent_concordant 		= 	round((float(concordant) / float(total_pairs))*100,1)
        percent_discordant 		= 	round((float(discordant) / float(total_pairs))*100,1)
        percent_tied 			= 	round((float(tied) / float(total_pairs))*100,1)
        roc 					= 	((float(concordant) / float(total_pairs)) + 0.5 * (float(tied) / float(total_pairs)))
        gini 					= 	(2*((float(concordant) / float(total_pairs)) + 0.5 * (float(tied) / float(total_pairs))) - 1)
        
        metrics_type 			= np.dtype([('Col1', 'S100'), ('Col2', 'float64')])
        ln_1 					= np.asarray([("Percent Concordant:", 	percent_concordant)], dtype=metrics_type)
        ln_2 					= np.asarray([("Percent Discordant:", 	percent_discordant)], dtype=metrics_type)
        ln_3 					= np.asarray([("Percent Tied:",			percent_tied)], dtype=metrics_type)	
        ln_4 					= np.asarray([("Total Pairs:", total_pairs		)], dtype=metrics_type)
        ln_5 					= np.asarray([("ROC:", roc						)], dtype=metrics_type)
        ln_6 					= np.asarray([("Gini:",gini 					)], dtype=metrics_type)	
        
        metrics = np.array([], dtype=metrics_type)
        for i in [ln_1, ln_2, ln_3, ln_4, ln_5, ln_6]:
            metrics= np.hstack((metrics, i))
            
        del actual_predicted, ln_1, ln_2, ln_3, ln_4, ln_5, ln_6
            
        return metrics
    
    
metrics = pd.DataFrame(calculate_ROC(input=df, target='reservation',score='prob' ))
metrics.rename(columns = {'Col1':'Metrics','Col2':'Value'}, inplace = True)
metrics['Metrics'] = metrics['Metrics'].str.decode('utf-8')
metrics



def generate_lift_table(input_data=None, dependent_variable=None, score_variable=None, high_score_for_bad=False):
    ## Generate lift table, ks table
    temp = pd.DataFrame(input_data, columns=[dependent_variable, score_variable])
    temp.rename(columns = {dependent_variable:'response', score_variable:'score'}, inplace = True) 
    temp['non_response'] = 1 - temp['response'] #temp.response
    #DEFINE 10 BUCKETS WITH EQUAL SIZE
    try:
        temp['bucket'] = pd.qcut(temp.score, 10)
    except:
        temp['Rank'] = temp["score"].rank(method='first')
        temp['bucket'] = pd.qcut(temp.Rank, 10)
        #temp=temp.drop('Rank',1)
        #temp['bucket'] = pd.qcut(temp.score,len(temp.score.dropna()),duplicates='drop')
        
    #GROUP THE DATA FRAME BY BUCKETS
    grouped = temp.groupby('bucket', as_index = False)
    
    ####################################################################
    #CREATE A SUMMARY DATA FRAME
    agg1= pd.DataFrame()
    agg1['min_scr'] = grouped.min().score
    agg1['max_scr'] = grouped.max().score
    agg1['total'] = agg1['total'] = grouped.sum().response + grouped.sum().non_response
    agg1['pct_total'] = (agg1.total/agg1.total.sum())
    agg1['non_response'] = grouped.sum().non_response
    agg1['pct_non_response']= (agg1.non_response/agg1.non_response.sum())
    agg1['response'] = grouped.sum().response
    agg1['pct_response'] = (agg1.response/agg1.response.sum()).apply('{0:.2%}'.format)
    agg1['bad_rate'] = (agg1.response / agg1.total).apply('{0:.2%}'.format)
    agg1['odds']= (agg1.non_response / agg1.response).apply('{0:.2f}'.format)
    ##################################################################
    
    #SORT THE DATA FRAME BY SCORE
    if high_score_for_bad == True:
        lift_table = (agg1.sort_values(by = 'min_scr', ascending=False)).reset_index(drop = True)
    else:
        lift_table = (agg1.sort_values(by = 'min_scr', ascending=True)).reset_index(drop = True)
    
    lift_table['cum_response'] = lift_table.response.cumsum()
    lift_table['cum_non_response'] = lift_table.non_response.cumsum()
    lift_table['cum_pct_response'] = (lift_table.cum_response/lift_table.response.sum())
    lift_table['cum_pct_non_response']= (lift_table.cum_non_response/lift_table.non_response.sum()).apply('{0:.2%}'.format)
    #CALCULATE KS STATISTIC
    lift_table['ks'] = np.round(((lift_table.cum_non_response/lift_table.non_response.sum()) - (lift_table.cum_response/lift_table.response.sum()))*100,2).abs()
    #DEFINE A FUNCTION TO FLAG MAX KS
    flag = lambda x: '<----' if x == lift_table.ks.max() else ''
    #FLAG OUT MAX KS
    lift_table['max_ks'] = lift_table.ks.apply(flag)
    #accuracy ratio (AR) using ROC method 
    lift_table['AUC_by_ROC'] = ((lift_table['cum_pct_response'] + lift_table.cum_pct_response.shift().fillna(0))*lift_table['pct_non_response'])/2
    #accuracy ratio (AR) using CAP method
    lift_table['AUC_by_CAP'] = ((lift_table['cum_pct_response'] + lift_table.cum_pct_response.shift().fillna(0))*lift_table['pct_total'])/2
    # AR is same from ROC and CAP method. Using only ROC for final AR 
    lift_table['AR']= ((2*lift_table.AUC_by_ROC.sum())-1)
    lift_table['AR']= (lift_table['AR']).apply('{0:.2%}'.format)
    lift_table['AUC_by_ROC']= (lift_table['AUC_by_ROC']).apply('{0:.2%}'.format)
    lift_table['AUC_by_CAP']= (lift_table['AUC_by_CAP']).apply('{0:.2%}'.format)
    lift_table['cum_pct_response'] = (lift_table['cum_pct_response']).apply('{0:.2%}'.format)
    lift_table['pct_non_response']= (lift_table['pct_non_response']).apply('{0:.2%}'.format)
    lift_table['pct_total'] = (lift_table['pct_total']).apply('{0:.2%}'.format)
    return lift_table
    
temp=generate_lift_table(input_data=df, dependent_variable='reservation', score_variable='prob',high_score_for_bad=True)
temp
