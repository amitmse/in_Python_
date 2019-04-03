
XG Boost: 
			https://www.quora.com/What-is-the-difference-between-the-R-gbm-gradient-boosting-machine-and-xgboost-extreme-gradient-boosting
			https://www.quora.com/When-would-one-use-Random-Forests-over-Gradient-Boosted-Machines-GBMs/answer/Tianqi-Chen-1
			https://www.quora.com/What-makes-xgboost-run-much-faster-than-many-other-implementations-of-gradient-boosting/answer/Tianqi-Chen-1
			http://xgboost.readthedocs.io/en/latest/model.html						
			
		- I am the author of xgboost. Both xgboost and gbm follows the principle of gradient boosting.  There are however, the difference in modeling details. 
				Specifically,  xgboost used a more regularized model formalization to control over-fitting, which gives it better performance.

############################################
## https://jessesw.com/XG-Boost/
## https://gist.github.com/tonicebrian/4018084
## https://gist.github.com/tristanwietsma/5486024
############################################

import numpy as np
import pandas as pd

train_set = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', header = None)
# Make sure to skip a row for the test set
test_set = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test', skiprows = 1, header = None) 

train_set.head()
test_set.head()

col_labels = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country','wage_class']

train_set.columns = col_labels
test_set.columns = col_labels

## check to see if pandas has identified any of these missing values.
train_set.info()
test_set.info()

train_set.replace(' ?', np.nan).dropna().shape 
test_set.replace(' ?', np.nan).dropna().shape

train_nomissing = train_set.replace(' ?', np.nan).dropna()
test_nomissing = test_set.replace(' ?', np.nan).dropna()

test_nomissing['wage_class'] = test_nomissing.wage_class.replace({' <=50K.': ' <=50K', ' >50K.':' >50K'})

test_nomissing.wage_class.unique()
train_nomissing.wage_class.unique()

combined_set = pd.concat([train_nomissing, test_nomissing], axis = 0) # Stacks them vertically
combined_set.info()

for feature in combined_set.columns: # Loop through all columns in the dataframe
    if combined_set[feature].dtype == 'object': # Only apply for columns with categorical strings
        combined_set[feature] = pd.Categorical(combined_set[feature]).codes # Replace strings with an integer

combined_set.info()

final_train = combined_set[:train_nomissing.shape[0]] # Up to the last initial training set row
final_test = combined_set[train_nomissing.shape[0]:] # Past the last initial training set row

y_train = final_train.pop('wage_class')
y_test = final_test.pop('wage_class')



		




