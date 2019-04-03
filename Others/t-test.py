
####################################################################################################################################################################################
# t-test 
# t-test for independent samples
# t-test for dependent samples
# https://machinelearningmastery.com/how-to-code-the-students-t-test-from-scratch-in-python/
####################################################################################################################################################################################
# t-test for independent samples
from math import sqrt
from numpy.random import seed
from numpy.random import randn
from numpy import mean
from scipy.stats import sem
from scipy.stats import t

# function for calculating the t-test for two independent samples
def independent_ttest(data1, data2, alpha):
	# calculate means
	mean1, mean2 = mean(data1), mean(data2)
	# calculate standard errors
	''' # calculate sample standard deviations: # std1, std2 = std(data1, ddof=1), std(data2, ddof=1)
		# calculate standard errors : 			# n1, n2 	 = len(data1), len(data2) 				# se1, se2 = std1/sqrt(n1), std2/sqrt(n2)
	'''	
	se1, se2 = sem(data1), sem(data2)
	# standard error on the difference between the samples
	sed = sqrt(se1**2.0 + se2**2.0)
	# calculate the t statistic
	t_stat = (mean1 - mean2) / sed
	# degrees of freedom
	df = len(data1) + len(data2) - 2
	# calculate the critical value
	cv = t.ppf(1.0 - alpha, df)
	# calculate the p-value
	p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
	# return everything
	return t_stat, df, cv, p

# seed the random number generator
seed(1)
# generate two independent samples
data1 = 5 * randn(100) + 50
data2 = 5 * randn(100) + 51
# calculate the t test
alpha = 0.05
t_stat, df, cv, p = independent_ttest(data1, data2, alpha)
print('Calculate t statistic = %.3f, Degrees of Freedom = %d, Critical Value = %.3f, P-value = %.3f' % (t_stat, df, cv, p))
# interpret via critical value
if abs(t_stat) <= cv:	print('Accept null hypothesis that the means are equal. 	Calculate t statistic is LE Critical Value : ',	'Calculate t statistic:', abs(t_stat), 'Critical Value:', cv	)
else:					print('Reject the null hypothesis that the means are equal. Calculate t statistic is GT Critical Value : ', 'Calculate t statistic:', abs(t_stat), 'Critical Value:', cv	)
# interpret via p-value
if p > alpha:	print('Accept null hypothesis that the means are equal. 	P-value is LT Significance Level : ', 'P-value:', p, 'Significance Level :', alpha	)
else:			print('Reject the null hypothesis that the means are equal. P-value is GE Significance Level : ', 'P-value:', p, 'Significance Level :', alpha	)

####################################################################################################################################################################################
################# END: t-test for independent samples ##############################################################################################################################
####################################################################################################################################################################################

####################################################################################################################################################################################
# t-test for dependent samples
####################################################################################################################################################################################
from math import sqrt
from numpy.random import seed
from numpy.random import randn
from numpy import mean
from scipy.stats import t

# function for calculating the t-test for two dependent samples
def dependent_ttest(data1, data2, alpha):
	# calculate means
	mean1, mean2 = mean(data1), mean(data2)
	# number of paired samples
	n = len(data1)
	# sum squared difference between observations
	d1 = sum([(data1[i]-data2[i])**2 for i in range(n)])
	# sum difference between observations
	d2 = sum([data1[i]-data2[i] for i in range(n)])
	# standard deviation of the difference between means
	sd = sqrt((d1 - (d2**2 / n)) / (n - 1))
	# standard error of the difference between the means
	sed = sd / sqrt(n)
	# calculate the t statistic
	t_stat = (mean1 - mean2) / sed
	# degrees of freedom
	df = n - 1
	# calculate the critical value
	cv = t.ppf(1.0 - alpha, df)
	# calculate the p-value
	p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
	# return everything
	return t_stat, df, cv, p

# seed the random number generator
seed(1)
# generate two independent samples (pretend they are dependent)
data1 = 5 * randn(100) + 50
data2 = 5 * randn(100) + 51
# calculate the t test
alpha = 0.05
t_stat, df, cv, p = dependent_ttest(data1, data2, alpha)
print('Calculate t statistic = %.3f, Degrees of Freedom = %d, Critical Value = %.3f, P-value = %.3f' % (t_stat, df, cv, p))

# interpret via critical value
if abs(t_stat) <= cv:	print('Accept null hypothesis that the means are equal. 	Calculate t statistic is LE Critical Value : ',	'Calculate t statistic:', abs(t_stat), 'Critical Value:', cv	)
else:					print('Reject the null hypothesis that the means are equal. Calculate t statistic is GT Critical Value : ', 'Calculate t statistic:', abs(t_stat), 'Critical Value:', cv	)
# interpret via p-value
if p > alpha:	print('Accept null hypothesis that the means are equal. 	P-value is LT Significance Level : ', 'P-value:', p, 'Significance Level :', alpha	)
else:			print('Reject the null hypothesis that the means are equal. P-value is GE Significance Level : ', 'P-value:', p, 'Significance Level :', alpha	)	
	
####################################################################################################################################################################################
################# END:t-test for dependent samples #################################################################################################################################
####################################################################################################################################################################################	