
#### XG Boost ############
#########################

pip install shap


import shap
import pandas as pd
import numpy as np
shap.initjs()

customer = pd.read_csv("data/customer_churn.csv")
customer.head()

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

X = customer.drop("Churn", axis=1) # Independent variables
y = customer.Churn # Dependent variable

# Split into train and test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Train a machine learning model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Make prediction on the testing data
y_pred = clf.predict(X_test)

# Classification Report
print(classification_report(y_pred, y_test))

xplainer = shap.Explainer(clf)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)

shap.summary_plot(shap_values[0], X_test)

shap.dependence_plot("Subscription Length", shap_values[0], X_test,interaction_index="Age")

shap.plots.force(explainer.expected_value[0], shap_values[0][0,:], X_test.iloc[0, :], matplotlib = True)

shap.plots.force(explainer.expected_value[1], shap_values[1][6, :], X_test.iloc[6, :],matplotlib = True)

shap.decision_plot(explainer.expected_value[1], shap_values[1], X_test.columns)

shap.decision_plot(explainer.expected_value[0], shap_values[0], X_test.columns)
#### https://www.datacamp.com/tutorial/introduction-to-shap-values-machine-learning-interpretability

# -----------------------------------------------------------------------------

# prediction's explanation
shap.plots.waterfall(shap_values[0])
#### https://github.com/shap/shap

# -----------------------------------------------------------------------------

pip install lime shap


import numpy as np
import sklearn
import shap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer

X, y = shap.datasets.imdb()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7, stratify=y)

vectorizer = TfidfVectorizer(min_df=10)
X_train_vec = vectorizer.fit_transform(X_train).toarray()
X_test_vec = vectorizer.transform(X_test).toarray()

print(len(y_train), len([t for t in y_train if t])) # 20000 10000
print(len(y_test), len([t for t in y_test if t])) # 5000 2500

model = RandomForestClassifier()
model.fit(X_train_vec, y_train)

pipeline = make_pipeline(vectorizer, model)

class_names = ['negative', 'positive']
explainer = LimeTextExplainer(class_names=class_names)

ind = 6
text = X_test[ind]
print(text)
# This amazing documentary gives us a glimpse into the lives of the
# brave women in Cameroun's judicial system-- policewomen, lawyers and
# judges. Despite tremendous difficulties-- lack of means, the desperate
# poverty of the people, multiple languages and multiple legal precedents
# depending on the region of the country and the religious/ethnic
# background of the plaintiffs and defendants-- these brave, [...]

exp = explainer.explain_instance(text, pipeline.predict_proba, num_features=6)
exp.show_in_notebook(text=True)



https://medium.com/nlplanet/two-minutes-nlp-explain-predictions-with-lime-aec46c7c25a2



























































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



		




