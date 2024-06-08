
#######################################################################################################################################################################
###	Monotonic Binning of a variable 	
#######################################################################################################################################################################
	# import packages
	import pandas as pd
	import numpy as np
	import scipy.stats.stats as stats
	 
	# import data
	data = pd.read_csv("/home/liuwensui/Documents/data/accepts.csv", sep = ",", header = 0)
	 
	# define a binning function
	def mono_bin(Y, X, n = 20):
	  # fill missings with median
	  X2 = X.fillna(np.median(X))
	  r = 0
	  while np.abs(r) < 1:
		d1 = pd.DataFrame({"X": X2, "Y": Y, "Bucket": pd.qcut(X2, n)})
		d2 = d1.groupby('Bucket', as_index = True)
		r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
		n = n - 1
	  d3 = pd.DataFrame(d2.min().X, columns = ['min_' + X.name])
	  d3['max_' + X.name] = d2.max().X
	  d3[Y.name] = d2.sum().Y
	  d3['total'] = d2.count().Y
	  d3[Y.name + '_rate'] = d2.mean().Y
	  d4 = (d3.sort_index(by = 'min_' + X.name)).reset_index(drop = True)
	  print "=" * 60
	  print d4
	 
	mono_bin(data.bad, data.ltv)
	mono_bin(data.bad, data.bureau_score)
	mono_bin(data.bad, data.age_oldest_tr)
	mono_bin(data.bad, data.tot_tr)
	mono_bin(data.bad, data.tot_income)

#######################################################################################################################################################################
#####Statistical correlation: Pearson or Spearman?			http://stackoverflow.com/questions/6731540/statistical-correlation-pearson-or-spearman
#######################################################################################################################################################################
	import numpy as np
	import scipy.stats
	x = np.random.randn(1000)
	y = np.random.randn(1000)
	print scipy.stats.spearmanr(x, y)
	#(-0.013847401847401847, 0.66184551507218536)
	#The first number (-0.01) is the rank correlation coefficient; the second number (0.66) is the associated p-value.
#######################################################################################################################################################################
##Calculating K-S Statistic with Python. 	https://statcompute.wordpress.com/2012/11/18/calculating-k-s-statistic-with-python/
#######################################################################################################################################################################
	# IMPORT PACKAGES
	import pandas as pd
	import numpy as np
	# LOAD DATA FROM CSV FILE
	data = pd.read_csv('c:\\projects\\data.csv')
	data.describe()
	data['good'] = 1 - data.bad
	# DEFINE 10 BUCKETS WITH EQUAL SIZE
	data['bucket'] = pd.qcut(data.score, 10)
	# GROUP THE DATA FRAME BY BUCKETS
	grouped = data.groupby('bucket', as_index = False)
	# CREATE A SUMMARY DATA FRAME
	#agg1 = grouped.min().score
	#del agg1 delete entire datframe
	agg1 = pd.DataFrame(grouped.min().score, columns = ['min_scr'])
	agg1['min_scr'] = grouped.min().score
	agg1['max_scr'] = grouped.max().score
	agg1['bads'] = grouped.sum().bad
	agg1['goods'] = grouped.sum().good
	agg1['total'] = agg1.bads + agg1.goods
	# SORT THE DATA FRAME BY SCORE
	agg2 = (agg1.sort_index(by = 'min_scr')).reset_index(drop = True)
	agg2['odds'] = (agg2.goods / agg2.bads).apply('{0:.2f}'.format)
	agg2['bad_rate'] = (agg2.bads / agg2.total).apply('{0:.2%}'.format)
	# CALCULATE KS STATISTIC
	agg2['ks'] = np.round(((agg2.bads / data.bad.sum()).cumsum() - (agg2.goods / data.good.sum()).cumsum()), 4) * 100
	# DEFINE A FUNCTION TO FLAG MAX KS
	flag = lambda x: '<----' if x == agg2.ks.max() else ''
	# FLAG OUT MAX KS
	agg2['max_ks'] = agg2.ks.apply(flag)
######################################################################################################################################################################
##Fitting A Logistic Regression with Python.		https://statcompute.wordpress.com/2012/11/08/fitting-a-logistic-regression-with-python/
#######################################################################################################################################################################

	from pandas import *
	import statsmodels.api as sm
	# LOAD EXTERNAL DATA
	data = read_table('C:\\data\\credit_count.txt', sep = ',')
	data
	# DEFINE RESPONSE
	Y = data[data.CARDHLDR == 1].DEFAULT
	# SUMMARIZE RESPONSE VARIABLE
	Y.describe()
	 # DEFINE PREDICTORS
	 X = sm.add_constant(data[data.CARDHLDR == 1][['AGE', 'ADEPCNT', 'MAJORDRG', 'MINORDRG', 'INCOME', 'OWNRENT']]
	 # SUMMARIZE PREDICTORS
	 X.describe()
	 # DEFINE A MODEL
	 model = sm.GLM(Y, X, family = sm.families.Binomial())
	 # FIT A MODEL
	 result = model.fit()
	 # PRINT RESULTS
	 print result.summary()

#######################################################################################################################################################################
## .		
#######################################################################################################################################################################


