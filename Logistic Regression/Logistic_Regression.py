
###################################################################################
### Implementation of Logistic Regression in Python
###################################################################################
#Import python libraries
import os
import sys
from math import sqrt
from scipy.optimize import fmin_ncg
from scipy.stats import norm
import numpy as np
import pandas as pd
import datetime
import time
import numpy.lib.recfunctions as nlr
import matplotlib.mlab as mlab

from io import StringIO

#####################################################################################

def change_independent_variable_name(input_independent_variable, input_independent_variable_name=False, header=False):
		##change independent variable name  into list format
		if 	input_independent_variable_name is not False:
				if 		type(input_independent_variable_name)  is list:				
						independent_variable_name = input_independent_variable_name[:]
				elif 	type(input_independent_variable_name).__module__ == np.__name__:
						independent_variable_name = input_independent_variable_name.tolist()
				elif 	type(input_independent_variable_name) is tuple:
						independent_variable_name = list(input_independent_variable_name)
				else :
						independent_variable_name = "False"								
		elif type(input_independent_variable).__module__ == np.__name__ and input_independent_variable.dtype.names is not None:
				independent_variable_name = list(input_independent_variable.dtype.names)						
		elif header == True and type(input_independent_variable) is list:
				independent_variable_name = input_independent_variable[0]
		else:
				independent_variable_name = "False"
		##
		if	independent_variable_name == "False":
				if 		type(input_independent_variable) is list:
						independent_variable_name = ["Column" + str(i+1) for i in range(len(input_independent_variable[0]))]
				elif 	type(input_independent_variable).__module__ == np.__name__ :
						if len(input_independent_variable.shape) > 1 :
								independent_variable_name = ["Column" + str(i+1) for i in range(input_independent_variable.shape[1])]
						else :
								independent_variable_name = ["Column" + str(i+1) for i in range(len(input_independent_variable[0]))]
				else :
						independent_variable_name = "False"

		return independent_variable_name


def change_dependent_variable_name(input_dependent_variable, input_dependent_variable_name=False, header=False):
		##change dependent variable name in a list format
		if 	input_dependent_variable_name is not False:
				if 		type(input_dependent_variable_name)  is list:				
						dependent_variable_name = input_dependent_variable_name[:]
				elif 	type(input_dependent_variable_name).__module__ == np.__name__:
						dependent_variable_name = input_dependent_variable_name.tolist()
				elif 	type(input_dependent_variable_name) is str:
						dependent_variable_name = []
						dependent_variable_name.append(input_dependent_variable_name)
				else :
						dependent_variable_name = input_dependent_variable_name[:]
		elif type(input_dependent_variable).__module__ == np.__name__ and input_dependent_variable.dtype.names is not None:
				dependent_variable_name = list(input_dependent_variable.dtype.names)						
		elif header == True and type(input_dependent_variable) is list:
				dependent_variable_name = input_dependent_variable[0]
		else:
				dependent_variable_name = "False"
		##
		if	dependent_variable_name == "False":
				dependent_variable_name = ["Response_Variable"]
		##
		return dependent_variable_name


def change_independent_variable(input_independent_variable, header=False):	
		##convert independent variable into NUMPY format
		if 		type(input_independent_variable)  is list:	
				if header == True:
					independent_variable = np.array(input_independent_variable[1:])
				else	:
					independent_variable = np.array(input_independent_variable)
		elif 	type(input_independent_variable).__module__ == np.__name__ :
				if input_independent_variable.dtype.names is None:
					independent_variable = input_independent_variable[:] 
				else:
					independent_variable = np.array(input_independent_variable.tolist())
		else:
				independent_variable = "False"

		return independent_variable


def change_dependent_variable(input_dependent_variable, header = False):	
		##convert dependent variable into NUMPY format
		if 		type(input_dependent_variable)  is list:	
				if header == True:
					dependent_variable = np.array(input_dependent_variable[1:])
				else:
					dependent_variable = np.array(input_dependent_variable)
		elif 	type(input_dependent_variable).__module__ == np.__name__ :
				if input_dependent_variable.dtype.names is None:
					dependent_variable = input_dependent_variable[:]
				else:
					dependent_variable = np.array(input_dependent_variable.tolist())
		else:
				dependent_variable = "False"

		return dependent_variable


def data_prep(input_independent_variable,input_dependent_variable,input_independent_variable_name=False,input_dependent_variable_name=False,header=False):

		error_information = []

		try	:
			independent_variable_name 	= change_independent_variable_name(input_independent_variable, input_independent_variable_name, header)
		except :	
			independent_variable_name	= "False"
		error_information.append("Issue in Name of Independent variable") if independent_variable_name == "False" else None

		try	:
			dependent_variable_name 	= change_dependent_variable_name(input_dependent_variable,input_dependent_variable_name, header)
		except :	
			dependent_variable_name		= "False"
		error_information.append("Issue in Name of Dependent variable") if dependent_variable_name == "False" else None

		try	:
			independent_variable 		= change_independent_variable(input_independent_variable, header)
		except :	
			independent_variable		= "False"
		error_information.append("Issue in Independent variable") if independent_variable == "False" else None

		try	:
			dependent_variable			= change_dependent_variable(input_dependent_variable, header)
		except :	
			dependent_variable			= "False"
		error_information.append("Issue in Dependent variable") if dependent_variable == "False" else None

		return (error_information, independent_variable_name,dependent_variable_name, independent_variable, dependent_variable)


def check_data_type():

		try	:
			independent_variable_test = sum(independent_variable[0].tolist())
		except :	
			error_information.append("Independent variable has string")

		try	:
			dependent_variable_test = np.sum(dependent_variable)
		except :	
			error_information.append("Dependent variable has string")

		return error_information

def check_value_of_dependent_variable():

		value_of_dependent_variable = list(np.unique(dependent_variable,return_counts = True))[0].tolist()

		if 0 in value_of_dependent_variable and 1 in value_of_dependent_variable and len(value_of_dependent_variable)==2:
			pass
		else:
			error_information.append("Dependent variable is not binary")

		return error_information

#############################################################################		

def negative_log_likelihood(coefficient, independent_variable, dependent_variable):
		#Negative of the log likelihood function.	(Ref: Gujarati, Porter: Basic Econometrics, 5e, page 590: Formula #8)
		#y*xb - log(1+exp(xb)). [y=dependent_variable, x=independent_variable, b=coefficient]
		return -(np.sum((dependent_variable*(np.multiply(independent_variable,coefficient)).sum(axis=1)) - (np.log(1 + np.exp((np.multiply(independent_variable,coefficient)).sum(axis=1))))))

def gradient_log_likelihood(coefficient, independent_variable,dependent_variable):
		#Gradient of log likelihood function. 
		#First differentiation of negative_log_likelihood
		#(y*x - ((x*exb)/(1.0 + exb))) [y=dependent_variable, x=independent_variable, b=coefficient, exb=exp(xb)]
		return -(np.dot(dependent_variable,independent_variable) - np.dot(np.exp((np.multiply(independent_variable,coefficient)).sum(axis=1)) / (1 + np.exp((np.multiply(independent_variable,coefficient)).sum(axis=1))),independent_variable))

def hessian_logistic_model(coefficient, independent_variable):
		#Hessian matrix of logistic model. 
		#First differentiation of gradient_log_likelihood with respect to coefficient.
		#x*transpose(x)*(exb/exb/(1.0 + exb)^2)
		return np.dot(np.multiply((np.exp((np.multiply(independent_variable,coefficient)).sum(axis=1))/((1 + np.exp((np.multiply(independent_variable,coefficient)).sum(axis=1)))**2)),np.transpose(independent_variable) ),independent_variable)

def estimate_coefficient_logistic_model(initial_coefficient, independent_variable, dependent_variable):
		#estimates of the unknown coefficient of the logistic equation.
		#Negative of the log likelihood function
		def n_l_l(coefficient):
			return negative_log_likelihood(coefficient, independent_variable, dependent_variable)
		#gradient of the log likelihood function
		def g_l_l(coefficient):
			return gradient_log_likelihood(coefficient, independent_variable, dependent_variable)
		#Hessian matrix for logistic model
		def h_l_m(coefficient):
			return hessian_logistic_model(coefficient, independent_variable)

		output_estimate_logistic_model = fmin_ncg(n_l_l, initial_coefficient, fprime = g_l_l, fhess=h_l_m, full_output = True,	disp=1)

		return output_estimate_logistic_model

###########################################################################

def probability_calculation(coefficient):
		#calculate Yhat or probability or predicted value

		def logit_function(independent_variable_and_coefficient):
		#Logit function/Sigmoid Function
			return 1 / (1 + np.exp(-independent_variable_and_coefficient))

		return logit_function(np.dot(independent_variable, coefficient))

def generate_lift_table(output_estimate_logistic_model):
		##Generate lift table, ks table
		actual_score = np.column_stack((dependent_variable, np.around(1000*probability_calculation(output_estimate_logistic_model[0]))))
		data 										= pd.DataFrame(actual_score, columns=['response', 'score'])
		data['non_response'] 						= 1 - data.response
		#DEFINE 10 BUCKETS WITH EQUAL SIZE
		data['bucket'] 								= pd.qcut(data.score, 10)
		#GROUP THE DATA FRAME BY BUCKETS
		grouped 									= data.groupby('bucket', as_index = False)
		#CREATE A SUMMARY DATA FRAME
		agg1										= pd.DataFrame()
		#agg1 = pd.DataFrame(grouped.min().score, columns = ['min_scr'])
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

		del actual_score, agg1

		return lift_table


def converged_Number_of_iterations(display_msg_fmin_ncg):
		##Get the value of converged and Number of iterations from fmin_ncg display message
		converged = False
		Number_of_iterations = None
		zfile = open(display_msg_fmin_ncg, 'r')
		for zline in zfile:
			zline = zline.strip()
			if "Optimization terminated successfully." in zline:
				converged = True
			if zline.startswith('Iterations:'):
				value = (zline.replace('Iterations:', '')).strip()
				try:
					Number_of_iterations = int(value)
				except:
					Number_of_iterations = 0
		converged_Number_of_iterations_value = [converged, Number_of_iterations]
		zfile.close()
		os.remove('fmin_ncg_msg.txt') if os.path.exists('fmin_ncg_msg.txt') else None

		return converged_Number_of_iterations_value


def collect_model_information(Degrees_of_Freedom, output_estimate_logistic_model,converged_Number_of_iterations_value, AIC, BIC):
		#combine high level information for model
		model_information_type = np.dtype([('Col1', 'S1000'), ('Col2', 'S100'),( 'Col3', 'S100'), ('Col4', 'float64')])
		ln_1 = np.asarray([("Dependent variable:", 		dependent_variable_name, 					"Total Observations", 	independent_variable.shape[0]		)], dtype=model_information_type)
		ln_2 = np.asarray([("Model:", 					"Binary Logit", 							"Degrees of Freedom", 	Degrees_of_Freedom 					)], dtype=model_information_type)
		ln_3 = np.asarray([("Optimization Technique",	"Newton CG", 								"Log Likelihood", 		-output_estimate_logistic_model[1]	)], dtype=model_information_type)
		#Newton CG: Newton Conjugate Gradient
		ln_4 = np.asarray([("Converged", 				converged_Number_of_iterations_value[0], 	"AIC",					AIC									)], dtype=model_information_type)
		ln_5 = np.asarray([("Warning",					output_estimate_logistic_model[5],			"BIC", 					BIC 								)], dtype=model_information_type)
		ln_6 = np.asarray([("Time", 					str(" "+datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')), "Iterations", converged_Number_of_iterations_value[1])],dtype=model_information_type)

		model_information 	= np.array([], dtype=model_information_type)
		for i in [ln_1, ln_2, ln_3, ln_4, ln_5,ln_6]:
			model_information 	= np.hstack((model_information, i))

		del ln_1, ln_2, ln_3, ln_4, ln_5, ln_6

		return model_information


def vif_calculation():
		### calculate VIF from linear regression
		### http://www.real-statistics.com/multiple-regression/ridge-and-lasso-regression/ridge-regression-basic-concepts/
		independent_without_intercept 	= independent_variable[:,1:] if 'Intercept' in independent_variable_name else independent_variable[:]
		vif								= []

		if 'Intercept' in independent_variable_name:
			vif.insert(0,0.0)

		for i in range(independent_without_intercept.shape[1]):
			yy 			= independent_without_intercept[:,i]
			xx 			= np.delete(independent_without_intercept,i,1)

			try:
				rss 		= np.linalg.lstsq(xx, yy)[1]
				tss			= sum((yy - np.mean(yy))**2)
				r_squared	= 1.0 - (rss/tss)
				vif_cal 	= 1.0 / (1.0 - r_squared)
			except:
				vif_cal		= 0

			if 'Intercept' in independent_variable_name:
				vif.insert(i+1, vif_cal[0])
			else:
				vif.insert(i, vif_cal[0])

		del independent_without_intercept, yy, xx, rss, tss, r_squared, vif_cal

		return vif


def collect_model_summry(coefficient, standard_error, vif, Wald_Chi_Square, z_value, p_value):
		###Combine model estimates metrics
		combine_model_summry = zip(*[independent_variable_name, coefficient, standard_error, vif, Wald_Chi_Square, z_value, p_value])
		#Data type for Columns
		model_summary_type 	= 	np.dtype([('Variable', 'S100'), ('Coefficient', 'float64'),( 'Standard Error', 'float64'), ( 'VIF', 'float64'), ('Wald Chi-Square', 'float64'),('Z Value', 'float64'), ('P Value(>|z|)', 'float64')])
		#Create empty array for "Logistic model summary"
		model_summary 		= 	np.array([], dtype=model_summary_type)
		for i in range(len(combine_model_summry)):
				model_summary 	= np.hstack((model_summary, np.asarray(combine_model_summry[i],dtype=model_summary_type)))

		del combine_model_summry

		return model_summary


def calculate_metrics(output_estimate_logistic_model, lift_table):
		##calculate overall metrics 
		#Final Probability value with dependent variable
		actual_predicted 		= 	np.column_stack((dependent_variable, probability_calculation(output_estimate_logistic_model[0])))
		actual_predicted.dtype 	= 	{'names':[dependent_variable_name, 'predicted_value'], 'formats':[np.float64, np.float64]}
		concordant 	= 	0
		discordant 	= 	0
		tied		=	0
		for i in actual_predicted[actual_predicted[dependent_variable_name]==1]['predicted_value']:
			for j in actual_predicted[actual_predicted[dependent_variable_name]==0]['predicted_value']:
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
		divergence  			= 	((float(np.mean(actual_predicted[actual_predicted[dependent_variable_name]==0]['predicted_value'])) - float(np.mean(actual_predicted[actual_predicted[dependent_variable_name]==1]['predicted_value'])))**2)/(0.5*(float(np.var(actual_predicted[actual_predicted[dependent_variable_name]==0]['predicted_value'],ddof=1)) + float(np.var(actual_predicted[actual_predicted[dependent_variable_name]==1]['predicted_value'],ddof=1))))

		metrics_type 			= np.dtype([('Col1', 'S100'), ('Col2', 'float64'),( 'Col3', 'S100'), ('Col4', 'float64')])

		ln_1 					= np.asarray([("Percent Concordant:", 	percent_concordant, "ROC:", 		roc					)], dtype=metrics_type)
		ln_2 					= np.asarray([("Percent Discordant:", 	percent_discordant, "Gini:", 		gini 				)], dtype=metrics_type)
		ln_3 					= np.asarray([("Percent Tied:",			percent_tied, 		"Divergence:",	divergence			)], dtype=metrics_type)		
		ln_4 					= np.asarray([("Total Pairs:", 			total_pairs,		"KS:",			lift_table.ks.max() )], dtype=metrics_type)

		metrics = np.array([], dtype=metrics_type)
		for i in [ln_1, ln_2, ln_3, ln_4]:	
			metrics 	= np.hstack((metrics, i))		

		del actual_predicted, ln_1, ln_2, ln_3		#, ln_4 

		return metrics


def calculate_correlation():
	#Pearson Correlation
		variable_name_type 	= np.dtype([('Variable', 'S100')])
		variable_name 		= np.array([],dtype=variable_name_type)
		variable_name 		= np.hstack((variable_name, np.array(str(dependent_variable_name), dtype=variable_name_type)))

		for i in range(1, len(independent_variable_name)):
			variable_name 	= np.hstack((variable_name, np.array(str(independent_variable_name[i]), dtype=variable_name_type)))

		for i in range(1, len(independent_variable_name)):
			correlation_type 	= np.dtype([(str(independent_variable_name[i]), 'float64')])
			corr 		= np.array([], dtype=correlation_type)
			corr 		= np.hstack((corr, np.array(np.corrcoef(independent_variable[:,i],dependent_variable)[0,1], dtype=correlation_type)))
			for j in range(1, len(independent_variable_name)):
				corr 	= np.hstack((corr, np.array(np.corrcoef(independent_variable[:,i],independent_variable[:,j])[0,1], dtype=correlation_type)))
			if i == 1:
				correlation = corr
			else :
				correlation = nlr.merge_arrays([correlation, corr], flatten=True)

		correlation = nlr.merge_arrays([variable_name,correlation], flatten=True)

		del variable_name_type, variable_name, correlation_type,corr

		return correlation


def print_model_summary(model_information, model_summary, metrics, lift_table, correlation):
		## Print model output
		print "====================================================================================================================================================="
		print "                                           Model Information                                                                                         "
		print "-----------------------------------------------------------------------------------------------------------------------------------------------------"
		for line in model_information:
			print ('{:>30} {:>25} {:>50} {:>25}'.format(*line))
		print "====================================================================================================================================================="
		print "\n"
		#print "\n"

		print "====================================================================================================================================================="
		print "                                           Estimates                                                                                                 "
		print "-----------------------------------------------------------------------------------------------------------------------------------------------------"		
		print ('{:>20} {:>20} {:>20} {:>20} {:>20} {:>20} {:>20}'.format(*model_summary.dtype.names))
		print ('{:>20} {:>20} {:>20} {:>20} {:>20} {:>20} {:>20}'.format(*('-------------------', '-------------------',  '-------------------', '-------------------', '-------------------', '-------------------', '-------------------' )))
		for line in model_summary:
			print ('{:>20} {:>20} {:>20} {:>20} {:>20} {:>20} {:>20}'.format(*line))
		print "====================================================================================================================================================="
		print "\n"
		#print "\n"

		print "====================================================================================================================================================="
		print "                                           Metrics                                                                                                 "
		print "-----------------------------------------------------------------------------------------------------------------------------------------------------"
		for line in metrics:
			print ('{:>30} {:>25} {:>50} {:>25}'.format(*line))		
		print "====================================================================================================================================================="
		print "\n"
		#print "\n"

		print "====================================================================================================================================================="
		print "                                           Lift Table                                                                                                 "
		print "-----------------------------------------------------------------------------------------------------------------------------------------------------"
		summary_lift_table = lift_table[:]
		print summary_lift_table.drop(['pct_total', 'pct_non_response', 'pct_response','odds', 'cum_response', 'cum_non_response', 'cum_pct_response', 'cum_pct_non_response'], axis=1)
		print "====================================================================================================================================================="
		print "\n"		

		"""
		#Could not format the correlation table hence not printing it on console.
		print "\n"
		print "====================================================================================================================================================="
		print "                                           Pearson Correlation Table                                                                                                 "
		print "-----------------------------------------------------------------------------------------------------------------------------------------------------"
		
		print "====================================================================================================================================================="
		print correlation.dtype.names
		for line in correlation:
			print	line
		"""

def export(model_information, model_summary, metrics, lift_table, correlation):
		## Export model output in csv
		blank_type 	= np.dtype([('Col1', 'S100'), ('Col2', 'S100'),( 'Col3', 'S100'), ('Col4', 'S100'), ('Col5', 'S100'), ('Col6', 'S100'), ('Col7', 'S100')])
		blank_line 	= np.asarray([("  ", "  ", " ", " ", " ", " ", " ")], dtype=blank_type)
		blank_line 	= np.hstack((blank_line, blank_line))
		dot_line 	= np.asarray([("=======================", "=======================",  "=======================",  "=======================", "=======================", "=======================", "=======================")], dtype=blank_type)
		title_1 	= np.asarray([("  ", "  ", "Model Information", " ",  " ",  " ", " "	)], dtype=blank_type)
		title_2 	= np.asarray([("  ", "  ", "Model Summary", " ",  " ",  " ", " "		)], dtype=blank_type)
		title_3 	= np.asarray([("  ", "  ", "Metrics", " ",  " ",  " ", " "				)], dtype=blank_type)
		title_4 	= np.asarray([("  ", "  ", "Lift Table", " ",  " ",  " ", " "			)], dtype=blank_type)
		title_5 	= np.asarray([("  ", "  ", "Correlation Table", " ",  " ",  " ", " "	)], dtype=blank_type)

		file_name 	= str('Logistic Regression Output '+datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')+'.csv').replace(":", "-")

		with open(file_name, 'wb') as outfile:
			mlab.rec2csv(blank_line, 			outfile, withheader=False)
			mlab.rec2csv(title_1, 				outfile, withheader=False)
			mlab.rec2csv(dot_line, 				outfile, withheader=False)
			mlab.rec2csv(model_information, 	outfile, withheader=False)
			mlab.rec2csv(dot_line, 				outfile, withheader=False)

			mlab.rec2csv(blank_line, 			outfile, withheader=False)
			mlab.rec2csv(title_2, 				outfile, withheader=False)
			mlab.rec2csv(dot_line, 				outfile, withheader=False)
			mlab.rec2csv(model_summary, 		outfile, withheader=True)
			mlab.rec2csv(dot_line, 				outfile, withheader=False)

			mlab.rec2csv(blank_line, 			outfile, withheader=False)
			mlab.rec2csv(title_3, 				outfile, withheader=False)
			mlab.rec2csv(dot_line, 				outfile, withheader=False)
			mlab.rec2csv(metrics, 				outfile, withheader=False)
			mlab.rec2csv(dot_line, 				outfile, withheader=False)

			mlab.rec2csv(blank_line, 			outfile, withheader=False)
			mlab.rec2csv(title_4, 				outfile, withheader=False)
			mlab.rec2csv(dot_line, 				outfile, withheader=False)
			mlab.rec2csv(lift_table.to_records(),outfile, withheader=True)
			mlab.rec2csv(dot_line, 				outfile, withheader=False)

			mlab.rec2csv(blank_line, 			outfile, withheader=False)
			mlab.rec2csv(title_5, 				outfile, withheader=False)
			mlab.rec2csv(dot_line, 				outfile, withheader=False)
			mlab.rec2csv(correlation, 			outfile, withheader=True)
			mlab.rec2csv(dot_line, 				outfile, withheader=False)

			outfile.close()

def combine_all_functions_together_for_fitting_logistic_model(input_independent_variable,input_dependent_variable, input_independent_variable_name = False, input_dependent_variable_name=False, intercept=True, header=False ):

		###call all function
		global error_information, independent_variable_name,dependent_variable_name, independent_variable, dependent_variable

		##Prepare data
		error_information, independent_variable_name,dependent_variable_name, independent_variable, dependent_variable = data_prep(input_independent_variable,input_dependent_variable,input_independent_variable_name,input_dependent_variable_name,header)

		## check data type of independent & dependent variable
		error_information = check_data_type()

		## check binary (0,1) condition of dependent variable
		error_information = check_value_of_dependent_variable()
		#########################################################
		## add intercept
		if intercept == True and independent_variable != "False":
				independent_variable = np.append( np.ones((independent_variable.shape[0], 1)), independent_variable, axis=1)
				independent_variable_name.insert(0,"Intercept")

		## run model if there is no error
		if not error_information:
				try:
						#global initial_coefficient
						initial_coefficient 						= 	np.zeros(independent_variable.shape[1])

						#################################################
						##store the display message from fmin_ncg in text file
						os.remove('fmin_ncg_msg.txt') if os.path.exists('fmin_ncg_msg.txt') else None
						orig_stdout 								= 	sys.stdout
						f 											= 	file('fmin_ncg_msg.txt', 'w')
						sys.stdout 									= 	f
						#estimates of the unknown coefficient
						output_estimate_logistic_model 				= 	estimate_coefficient_logistic_model(initial_coefficient, independent_variable, dependent_variable)
						#title of above output in order ["coefficient", "Negative Log Likelihood", "Number of function calls", "Number of gradient calls", "Number of Hessian  calls", "Warning flag", 	"Result at each iteration"]
						sys.stdout 									= 	orig_stdout
						f.close()
						##get the display message
						converged_Number_of_iterations_value 		= 	converged_Number_of_iterations('fmin_ncg_msg.txt')	
						################################################		
						#Inverse of Hessian matrix for logistic model. It is called variance-covariance estimates.					
						inverse_hessian_matrix 						= 	np.linalg.inv(hessian_logistic_model(output_estimate_logistic_model[0], independent_variable))

						Degrees_of_Freedom = len(independent_variable_name) - 1 if 'Intercept' in independent_variable_name else len(independent_variable_name)

						#add all results in one hash
						AIC											=	2*independent_variable.shape[1] - 2*(-output_estimate_logistic_model[1])
						BIC											=	(independent_variable.shape[1]*np.log(independent_variable.shape[0])) - (2*(-output_estimate_logistic_model[1]))

						coefficient 								= 	[output_estimate_logistic_model[0][j] for j in range(len( output_estimate_logistic_model[0]))]

						#SQRT of inverse of H is the estimate of the standard error of coefficient.
						standard_error								= 	[sqrt(inverse_hessian_matrix[i][i]) for i in range(len(inverse_hessian_matrix))]

						#z = coefficient divided by standard error.
						z_value 									= 	[beta/std_err for beta, std_err in zip(coefficient, standard_error)]

						#Wald Chi-Square = Square of (Coefficient Estimate / Standard Error)
						Wald_Chi_Square 							= 	[(beta/std_err)**2 for beta, std_err in zip(coefficient, standard_error)]

						#cumulative density function for standard normal distribution / cumulative normal distribution	
						p_value										= 	[2 *(1.0 - norm.cdf(abs(z))) for z in z_value]
						#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html

						#################################################
						#lift table
						try:		
							del lift_table
							lift_table 		= generate_lift_table(output_estimate_logistic_model)
						except:
							lift_table 		= generate_lift_table(output_estimate_logistic_model)

						#summarize all information
						model_information 	= 	collect_model_information(Degrees_of_Freedom, output_estimate_logistic_model,converged_Number_of_iterations_value, AIC, BIC)
						vif 				= 	vif_calculation()
						model_summary 		= 	collect_model_summry(coefficient, standard_error, vif, Wald_Chi_Square, z_value, p_value)
						metrics 			= 	calculate_metrics(output_estimate_logistic_model, lift_table)
						correlation 		= 	calculate_correlation()
						#################################################################################################################
						##Print on console
						print_model_summary(model_information, model_summary, metrics, lift_table, correlation)		

						#export in csv
						export(model_information, model_summary, metrics, lift_table, correlation)

				except:
						error_information.append("Issue in Estimating Parameters")
						print error_information
		else:
				print error_information

		del error_information, independent_variable_name,dependent_variable_name, independent_variable, dependent_variable


############################################################
### End of code ####
############################################################	

### Call above logistic model
if __name__=="__main__":
	##call logistc model
	#Provide below information and do not change the below name
		#independent_variable
		#dependent_variable
		#dependent_variable_name
		#independent_variable_name
		sample_data = u"""
		pn spending extra
		0  32.1007  0
		1  34.3706  1
		0   4.8749  0
		0   8.1263  0
		0  12.9783  0
		0  16.0471  0
		0  20.6648  0
		1  42.0483  1
		0  42.2264  1
		1  37.9900  1
		1  53.6063  1
		0  38.7936  0
		0  27.9999  0
		1  42.1694  0
		1  56.1997  1
		0  23.7609  0
		0  35.0388  1
		1  49.7388  1
		0  24.7372  0
		1  26.1315  1
		0  31.3220  1
		1  40.1967  1
		0  35.3899  0
		0  30.2280  0
		1  50.3778  0
		0  52.7713  0
		0  27.3728  0
		1  59.2146  1
		1  50.0686  1
		1  35.4234  1
		"""

		read_data 					= np.genfromtxt(StringIO(sample_data), skip_header=2, usecols=(0, 1, -1))
		dep_variable 				= np.genfromtxt(StringIO(sample_data), skip_header=2, usecols=(0, 1, -1))[:,0]
		dep_variable_name 			= sample_data.split("\n")[1].split()[0]
		indep_variable				= np.genfromtxt(StringIO(sample_data), skip_header=2, usecols=(0, 1, -1))[:,1:]
		indep_variable_name 		= sample_data.split("\n")[1].split()[1:]
		###change location
		os.chdir('D:\\Training\\Python\\Logistic_Model_Development')		
		#Model Summary
		combine_all_functions_together_for_fitting_logistic_model(
							input_independent_variable			=	indep_variable, 
							input_dependent_variable			=	dep_variable, 
							input_independent_variable_name 	= 	indep_variable_name, 
							input_dependent_variable_name		=	dep_variable_name, 
							intercept							=	True, 
							header								=	False
		)

###############################################################
###############################################################

#Specify column name and type
np_test_indp_type 	= np.dtype([('Intercept', 'float64'), ('spending', 'float64'), ('extra', 'float64')])
np_test_dep_type 	= np.dtype([('pn', 'float64')])
#Read data
np_test_indp 		= np.genfromtxt('np_test_indp.csv',delimiter=',', names=True, missing_values='nan', dtype = np_test_indp_type)
np_test_dep 		= np.genfromtxt('np_test_dep.csv',delimiter=',', names=True, missing_values='nan', dtype = np_test_dep_type)

### call LR module
if __name__=="__main__":
		dependent_variable 			= np_test_dep
		dependent_variable_name 	= list(np_test_dep.dtype.names)
		independent_variable		= np_test_indp
		independent_variable_name 	= list(np_test_indp.dtype.names)
		os.chdir('D:\\Training\\Python\\Logistic_Model_Development')
		#Model Summary
		combine_all_functions_together_for_fitting_logistic_model(independent_variable, dependent_variable)			

###############################################################
#Compare the above output with standard library:
import statsmodels.api as sm
logit 	= sm.Logit(dependent_variable, independent_variable)
result 	= logit.fit()
results = sm.OLS(y, X).fit()
print result.summary()
print results.summary()
################################################################
################# END OF CODE ##################################
################################################################
