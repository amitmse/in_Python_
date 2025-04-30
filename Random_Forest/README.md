# Random Forest

	A group of decision trees.

## Advantages:
	- Same as decision Tree.
	- Each decision tree in the forest is independent.
	- Enhanced accuracy over a decision tree.
    
## Limitations:
	- Computationally expensive.
	- Black-box model; harder to interpret than single decision trees.

## Eaxmple:
	To choose the best strategy for an upcoming product launch, a project manager seeks advice 
 	from multiple teams and tailors the strategy based on their responses.

----------------------------------------------------------------------------------------------------------------------------
## Random Forest in Python

https://github.com/amitmse/in_Python_/blob/master/Random_Forest/Random_Forest_Try.py

----------------------------------------------------------------------------------------------------------------------------
## Key Element

	Decision trees use the entire data, while random forests utilize an ensemble method to reduce variance.

### Ensemble methods:
- Ensemble learning is a machine learning paradigm where multiple models (often called "weak learners") are trained to solve the same problem and combined to get better results. 

### Bagging (Bootstrap aggregating): Decrease Variance
- Fit a weak learner (several independent models) on each of bootstarp samples and finally aggregate the outputs (average model predictions) in order to obtain a model with a lower variance. It builds model parallelly.
- Bootstrap samples: Draw repeated samples from the population, a large number of times. Samples are approximatively independent and identically distributed (i.i.d.).
- Bagging ensembles methods applied on Random Forest.
 
### Boosting: Decrease Bias
- Similar to bagging but it fits weak learner sequentially (a model depends on the previous ones) in a very adaptative way. Each model in the sequence is fitted giving more importance to the observations which are not classified correctly (high error). Mainly focus on reducing bias.
	
- Bagging mainly focus at getting an ensemble model with less variance than its components whereas boosting and stacking will mainly try to produce strong models less biased than their components (even if variance can also be reduced).

Following techniques are based on Boosting:
	- AdaBoost (Adaptive Boosting)
	- Gradient Tree Boosting (GBM)
	- XGBoost

![image](https://github.com/user-attachments/assets/b3adb086-6c00-479b-b170-76e761e9a5f4)


### Stacking: Improve Predictions
- Stacking mainly differ from bagging and boosting on two points. First stacking often considers heterogeneous weak learners (different learning algorithms are combined) whereas bagging and boosting consider mainly homogeneous weak learners. Second, stacking learns to combine the base models using a meta-model whereas bagging and boosting combine weak learners following deterministic algorithms.
	
### Bias - Variance
- Bias: 
	- Bias is the difference between the prediction of model and actual value. 
	- It always leads to high error on training and test data.
	- It creates underfitting problem.
	- If model is too simple and has very few parameters then it may have high bias and low variance.
	
- Variance: 
	- Model performs very well on development data but poor performance on on OOT validation.
	- It creates overfitting problem.
	- If model has large number of parameters then it’s going to have high variance and low bias
	
	- For high variance, one common solution is to reduce parameter/features. This very frequently increases bias, so there’s a tradeoff to take into consideration.
	
----------------------------------------------------------------------------------------------------------------------

## Algorithm (for both classification and regression)

1. Select a subset of data and variables: Draw ntree bootstrap samples from the original data
    
2. Develop decision trees on the selected data: For each of the bootstrap samples, grow an unpruned classification or regression tree, with the following modification: at each node, rather than choosing the best split among all predictors, randomly sample mtry of the predictors and choose the best split from among those variables. (Bagging can be thought of as the special case of random forests obtained when mtry = p, the number of predictors)
        
3. Final tree based on averaging:
- Predict new data by aggregating the predictions of the ntree trees (i.e., majority votes for classification, average for regression).		
- An estimate of the error rate can be obtained, based on the training data, by the following:
	1. At each bootstrap iteration, predict the data not in the bootstrap sample (what Breiman calls “out-of-bag”, or OOB, data) using the tree grown with the bootstrap sample.        
	2. Aggregate the OOB predictions. (On the average, each data point would be out-of-bag around 36% of the times, so aggregate these predictions.)  Calcuate the error rate, and call it the OOB estimate of error rate.
	
- Our experience has been that the OOB estimate of error rate is quite accurate, given that enough trees have been grown (otherwise the OOB estimate can bias upward; see Bylander (2002))
http://www.bios.unc.edu/~dzeng/BIOS740/randomforest.pdf

![Function](https://github.com/amitmse/in_Python_/blob/master/Random_Forest/RF.PNG)

-----------------------------------------------------------------------------------------------
## Implementation 

1. Sample (N) cases at random with replacement to create a subset of the data. The subset should be about 66% of the total set.
2. At each node:
	- For some number m (see below), m predictor variables are selected at random from all the predictor variables.
	- The predictor variable that provides the best split, according to some objective function, is used to do a binary split on that node.
	- At the next node, choose another m variables at random from all predictor variables and do the same.
		
3. Depending upon the value of m, there are three slightly different systems:
	- Random splitter selection: m =1
	- Breiman’s bagger: m = total number of predictor variables
	- Random forest: m << number of predictor variables Brieman suggests three possible values for m: 1/2(sqrt(vm)), sqrt(vm), and sqrt(2vm)
			
4. Running a Random Forest.
	- When a new input is entered into the system, it is run down all of the trees. The result may either be an average or weighted average of all of the terminal nodes that are reached, or, in the case of categorical variables, a voting majority.
	- Note that:
		- With a large number of predictors, the eligible predictor set will be quite different from node to node
		- The greater the inter-tree correlation, the greater the random forest error rate, so one pressure on the model is to have the trees as uncorrelated as possible
		- As m goes down, both inter-tree correlation and the strength of individual trees go down. So some optimal value of m must be discovered
	
---------------------------------------------------------------------------------------------------------------------------

  
#### Random Forest hyperparameters

- Number of trees:
	- The number of decision trees in the forest. Generally, a larger number of trees can improve accuracy but also increase training time.
	- n_estimators is not really worth optimizing. The more estimators you give it, the better it will do.
	- 500 or 1000 is usually sufficient. ususally bigger the forest the better, there is small chance of overfitting here. a larger number of trees can improve accuracy but also increase training time.

- Maximum depth of each tree: 
	- A deeper tree can capture more complex relationships in the data but may also lead to overfitting.
	- max depth of each tree (default none, leading to full tree) - reduction of the maximum depth helps fighting with overfitting.
	- This will reduce the complexity of the learned models, lowering over fitting risk. Try starting small, say 5-10, and increasing you get the best result. 
	- A deeper tree can capture more complex relationships in the data but may also lead to overfitting.  	

- Minimum number of samples required to split a node:  
	- Helps prevent overfitting by ensuring that nodes aren't split on very small subsets of the data.
	- Minimum number of samples required to split a node:  Helps prevent overfitting by ensuring that nodes aren't split on very small subsets of the data.

- Minimum number of samples per leaf node: 
	- The minimum number of samples required to be at a leaf node. It helps to prevent overfitting by ensuring that leaf nodes have a sufficient number of samples.
	- - Minimum number of samples per leaf node: The minimum number of samples required to be at a leaf node. It helps to prevent overfitting by ensuring that leaf nodes have a sufficient number of samples.

- Number of features to consider when making a split: 
	- It controls the diversity of the trees in the forest, with more features leading to potentially more diverse trees.
	- Number of features to consider when making a split: IT controls the diversity of the trees in the forest, with more features leading to potentially more diverse trees.
	- It may have a large impact on the behavior of the RF because it decides how many features each tree in the RF considers at each split. 
	- Try reducing this number (try 30-50% of the number of features). 
	- This determines how many features each tree is randomly assigned. The smaller, the less likely to overfit, but too small will start to introduce under fitting.

- Bootstrap: 
	- Determines whether or not to use bootstrap sampling when building the trees. Bootstrap sampling involves drawing samples with replacement, which can help increase diversity. 

- Criterion: 
	- The function used to measure the quality of a split. Common choices include "gini" for Gini impurity and "entropy" for information gain. 
	- criterion may have a small impact, but usually the default is fine. If you have the time, try it out.

- Class weight: 
	- To adjust the weights of classes in imbalanced datasets, which can be useful when one class is significantly more prevalent than others.

- Default setting for a first run is: 1000 trees, 1/2 features per node, out of bag performance weighting, Gini Index for node evaluation.
- Make sure to use sklearn's GridSearch (preferably GridSearchCV, but your data set size is too small) when trying out these parameters. Details are in below link.

##### Grid Search, Randomized Search or Bayesian Optimization

https://github.com/amitmse/in_Python_/blob/master/Others/README.md#grid-search-randomized-search-or-bayesian-optimization

-----------------------------------------------------------------------------------------  

# Feature importance

- Feature Importance: Analyze the importance of each input feature in the model's predictions. Techniques like tree-based models or methods that calculate the importance of each feature based on its contribution to the model's predictions. Tree-Based Algorithms feature importance scores based on how much each feature reduces impurity (e.g., Gini index or information gain) in the decision tree nodes. 
- Tree based: Decision Trees, Random Forests, XGBoost, LightGBM.
  
- Variable importance is calculated by the sum of the decrease in error when split by a variable. Then, the relative importance is the variable importance divided by the highest variable importance value so that values are bounded between 0 and 1.

- Below both are similar to Feature importance:
	- LIME: Local Interpretable Model-Agnostic Explanations.  
	- SHAP: SHapley Additive exPlanations.

Details are in https://github.com/amitmse/in_Python_/blob/master/Others/README.md#feature-importance

## Mean Decrease in Accuracy (MDA) / Accuracy-based importance / Permutation Importance:

- The values of the variable in the out-of-bag-sample are randomly shuffled, keeping all other variables the same. Finally, the decrease in prediction accuracy on the shuffled data is measured. 
- The mean decrease in accuracy across all trees is reported. 
- For example, age is important for predicting that a person earns over $50,000, but not important for predicting a person earns less. Intuitively, the random shuffling means that, on average, the shuffled variable has no predictive power. This importance is a measure of by how much removing a variable decreases accuracy, and vice versa — by how much including a variable increases accuracy.
- Note that if a variable has very little predictive power, shuffling may lead to a slight increase in accuracy due to random noise. This in turn can give rise to small negative importance scores, which can be essentially regarded as equivalent to zero importance.	
- This is most interesting measure, because it is based on experiments on out-of-bag(OOB) samples, via destroying the predictive power of a feature without changing its marginal distribution.
- Percentage increase in mean square error is analogous to accuracy-based importance, and is calculated by shuffling the values of the out-of-bag samples.

- ** Scikit-learn doesn’t implement this measure, Python user may not even know it exists.**

			from sklearn.datasets import load_boston
			from sklearn.ensemble import RandomForestRegressor
			import numpy as np
			#Load boston housing dataset as an example
			boston = load_boston()
			X = boston["data"]
			Y = boston["target"]
			names = boston["feature_names"]
			rf = RandomForestRegressor()
			rf.fit(X, Y)
			print "Features sorted by their score:"
			print sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), reverse=True)

## Gini Importance / Mean Decrease in Impurity (MDI) :

- Gini Index and Gini Impurity refer to the same.
- Gini Impurity is the probability of incorrectly classifying a randomly chosen element in the dataset if it were randomly labeled according to the class distribution in the dataset. It’s calculated as 
- Gini impurity index (G) = P * (1 - P)
- Importance = G (parent node) - G (child node 1) - G (child node 2)
- The initial gini index before split  Overall = 1 − P(success)^2 − P(Failure)^2
- Node level :
	- impurity in Left node  =1 − P(Success in left node)^2  − P(Failure in left node)^2
	- impurity in Right node =1 − P(Success in right node)^2 − P(Failure in right node)^2
- Now the final formula for GiniGain would be = Overall − impurity in  Left node − impurity in Right node
	- Lets assume we have 3 classes and 80 objects. 19 objects are in class 1, 21 objects in class 2, and 40 objects in class 3 (denoted as (19,21,40) ). 
	- The Gini index would be: 	= 1 - [ (19/80)^2 + (21/80)^2 + (40/80)^2] = 0.6247      
		- costbefore Gini(19,21,40) = 0.6247

	- In order to decide where to split, we test all possible splits. For example splitting at 2.0623, 
		- which results in a split (16,9,0) and (3,12,40).
		- After testing x1 < 2.0623:
			- costL Gini(16,9,0)  = 0.4608
			- costR Gini(3,12,40) = 0.4205
	- Then we weight branch impurity by empirical branch probabilities: costx1<2.0623 = 25/80 costL + 55/80 costR = 0.4331
	- We do that for every possible split, for example x1 < 1:
		- costx1<1 = FractionL Gini(8,4,0) + FractionR Gini(11,17,40) = 12/80 * 0.4444 + 68/80 * 0.5653 = 0.5417
	- After that, we chose the split with the lowest cost. This is the split x1 < 2.0623 with a cost of 0.4331.

------------------------------------------------------      
****** Implemented in scikit-learn ********************************************************************
		
|outlook|temp|humidity|windy|play|
|---|---|---|---|---|
|sunny|hot|high|Weak|no|
|sunny|hot|high|Strong|no|
|overcast|hot|high|Weak|yes|
|rainy|mild|high|Weak|yes|
|rainy|cool|normal|Weak|yes|
|rainy|cool|normal|Strong|no|
|overcast|cool|normal|Strong|yes|
|sunny|mild|high|Weak|no|
|sunny|cool|normal|Weak|yes|
|rainy|mild|normal|Weak|yes|
|sunny|mild|normal|Strong|yes|
|overcast|mild|high|Strong|yes|
|overcast|hot|normal|Weak|yes|
|rainy|mild|high|Strong|no|


- Calculate the gini index of dep variable	
	- Gini(S) = 1 - [(9/14)² + (5/14)²] = 0.459
		
- calculate gini gain. For that first, we will find the average weighted gini impurity of Outlook, Temperature, Humidity and Windy 
	- Gini(S, outlook)
		- (5/14)*gini(3,2) + (4/14)*gini(4,0) + (5/14)*gini(2,3)			
		- [5/14]*[1 - (3/5)² - (2/5)²] + (4/14)*[0] + (5/14)*[1 - (2/5)² - (3/5)²]
		- 0.171 + 0 + 0.171
		- 0.342
			
	- Gini gain (S, outlook) 	= 0.459 - 0.342 	= 0.117
	- Gini gain(S, Temperature) 	= 0.459 - 0.4405 	= 0.0185		
	- Gini gain(S, Humidity) 	= 0.459 - 0.3674 	= 0.0916		
	- Gini gain(S, windy) 		= 0.459 - 0.4286 	= 0.0304

	- Choose one that having higher gini gain. Gini gain is higher for outlook.So we can choose it as our root node.

---------------------------------------------------------------------------------------------------------------  
