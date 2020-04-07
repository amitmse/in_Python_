###################################################################################################################################################################
###### Logistic Regression ##############################################################################################################################################
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

def LR(target=None, model_variable=None):
    model = sm.Logit(target, model_variable)
    result = model.fit(method='newton', disp=False)
    #result.summary() #Method
    #result.summary2() #AIC, BIC
    #pd.read_html(result.summary().tables[1].as_html(), header=0, index_col=0)[0]
    #result.params #coefficients
    #print (np.exp(result.params)) # odds ratios only

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

    LR_summary = None
    LR_summary=pd.concat([summary1,summary2_1,summary2_2])
    LR_summary.reset_index(drop=True, inplace=True)
    
    LR_Coefficients=result.summary2().tables[1].drop(['z','[0.025','0.975]'],1)
    LR_Coefficients.rename(columns = {'Coef.':'Coefficients','Std.Err.':'Standard Error','P>|z|':'P-Value'}, inplace = True)
    LR_Coefficients['Wald Chi-Square'] = (LR_Coefficients['Coefficients']**2/LR_Coefficients['Standard Error']**2).round(decimals=2)
    LR_Coefficients['P-Value']=LR_Coefficients['P-Value'].round(decimals=2)
    LR_Coefficients['Standard Error']=LR_Coefficients['Standard Error'].round(decimals=2)
    LR_Coefficients['Coefficients']=LR_Coefficients['Coefficients'].round(decimals=15)
    LR_Coefficients=LR_Coefficients[['Coefficients','Standard Error','Wald Chi-Square','P-Value']]

    print (LR_summary.to_string(index=False ))
    print("\n")
    print (LR_Coefficients)
    return LR_summary, LR_Coefficients
    
LR_summary, LR_Coefficients =LR(target=dependent_variable, model_variable=independent_variable)
#LR_summary
#LR_Coefficients

##############################################################################################
#predictions = result.predict(independent_variable)
df['odds']  = -3.670706 + (0.022926*df['NOP_before_purchase']) - (0.061525*df['nop_last_visit']) - (0.913963*df['no_of_visits_last_7_days']) + (3.176300*df['no_of_purchases_last_7_days']) + (0.195037*df['Hilton_Honors_Status_Ever_flag'])
df['prob']  = 1 / (1 + np.exp(-df['odds']))
df['score'] =  (- ((df['odds'] + np.log(100))*30 / np.log(2) ) + 500)
#curr_score = round(-( (curr_log_odds + log(100)) *30/log(2)) + 500,1);
###################################################################################################
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
metrics.round(decimals=2)

##################################################################################################################

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
    lift_table['min_scr'] = lift_table['min_scr'].round(decimals=4)
    lift_table['max_scr'] = lift_table['max_scr'].round(decimals=4)
    return lift_table
    
temp=generate_lift_table(input_data=df, dependent_variable='reservation', score_variable='prob',high_score_for_bad=True)
temp

##############################################################################################################
'''
######## Check and update
def probability_calculation(coefficient):
		#calculate Yhat or probability or predicted value
		def logit_function(independent_variable_and_coefficient):
		#Logit function/Sigmoid Function
			return 1 / (1 + np.exp(-independent_variable_and_coefficient))

		return logit_function(np.dot(independent_variable, coefficient))


#actual_score = np.column_stack((dependent_variable, np.around(1000*probability_calculation(output_estimate_logistic_model[0]))))
'''

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

# Created a function : LR
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
'''
