# Random Forest

## Algorithm (for both classification and regression)

	1. Draw ntree bootstrap samples from the original data
    
	2. For each of the bootstrap samples, grow an unpruned classification or regression tree, with the following
		modification: at each node, rather than choosing the best split among all predictors, randomly sample
		mtry of the predictors and choose the best split from among those variables. (Bagging can be thought 
		of as the special case of random forests obtained when mtry = p, the number of predictors)
        
	3. Predict new data by aggregating the predictions of the ntree trees 
        	(i.e., majority votes for classification, average for regression).
		
	An estimate of the error rate can be obtained, based on the training data, by the following:
		1. At each bootstrap iteration, predict the data not in the bootstrap sample 
        		(what Breiman calls “out-of-bag”, or OOB, data) using the tree grown with the bootstrap sample.
        
		2. Aggregate the OOB predictions. (On the average, each data point would be out-of-bag around 36% of 
			the times, so aggregate these predictions.)  
			Calcuate the error rate, and call it the OOB estimate of error rate.
	
  Our experience has been that the OOB estimate of error rate is quite accurate, given that enough trees have 
  been grown (otherwise the OOB estimate can bias upward; see Bylander (2002))
  
http://www.bios.unc.edu/~dzeng/BIOS740/randomforest.pdf

## Implementation 

	1. Sample (N) cases at random with replacement to create a subset of the data.
		The subset should be about 66% of the total set.
		
	2. At each node:
		- For some number m (see below), m predictor variables are selected at random from 
			all the predictor variables.
		
		- The predictor variable that provides the best split, according to some objective function, 
			is used to do a binary split on that node.
			
		- At the next node, choose another m variables at random from all predictor variables and do the same.
		
	3. Depending upon the value of m, there are three slightly different systems:
		- Random splitter selection: m =1
		
		- Breiman’s bagger: m = total number of predictor variables
		
		- Random forest: m << number of predictor variables
			Brieman suggests three possible values for m: 1/2(sqrt(vm)), sqrt(vm), and sqrt(2vm)
			
	4. Running a Random Forest. When a new input is entered into the system, 
		it is run down all of the trees. The result may either be an average or weighted average of 
		all of the terminal nodes that are reached, or, in the case of categorical variables, 
		a voting majority.
		
		Note that:
			- With a large number of predictors, the eligible predictor set will be quite 
				different from node to node
				
			- The greater the inter-tree correlation, the greater the random forest error rate, 
				so one pressure on the model is to have the trees as uncorrelated as possible
				
			- As m goes down, both inter-tree correlation and the strength of individual trees go down. 
				So some optimal value of m must be discovered
				
	5. To understand how we test the classifier, we must explain several concepts:
		- cross-validation 
		- thresholds 
		- mean precision
		- precision above chance


# Bias - Variance
	For high variance, one common solution is to add more features from which to learn. 
	This very frequently increases bias, so there’s a tradeoff to take into consideration.

# Tune
	- n_estimators is not really worth optimizing. The more estimators you give it, the better it will do. 
		500 or 1000 is usually sufficient. ususally bigger the forest the better, 
		there is small chance of overfitting here
		
	- max_features is worth exploring for many different values. It may have a large impact on the behavior 
		of the RF because it decides how many features each tree in the RF considers at each split. 
		Try reducing this number (try 30-50% of the number of features). This determines how many features 
		each tree is randomly assigned. The smaller, the less likely to overfit, but too small will start 
		to introduce under fitting.
		
	- max depth of each tree (default none, leading to full tree) - reduction of the maximum depth helps 
		fighting with overfitting. This will reduce the complexity of the learned models, lowering over 
		fitting risk. Try starting small, say 5-10, and increasing you get the best result.
		
	- criterion may have a small impact, but usually the default is fine. If you have the time, try it out.
	
	- Make sure to use sklearn's GridSearch (preferably GridSearchCV, but your data set size is too small) 
		when trying out these parameters.
		
	- my default setting for a first run is: 1000 trees, 1/2 features per node, out of bag performance weighting, 
		Gini Index for node evaluation.

plot_learning_curve: http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

https://medium.com/@chris_bour/6-tricks-i-learned-from-the-otto-kaggle-challenge-a9299378cd61#.62v2igttw

http://stackoverflow.com/questions/36107820/how-to-tune-parameters-in-random-forest-using-scikit-learn

http://scikit-learn.org/stable/auto_examples/model_selection/randomized_search.html#sphx-glr-auto-examples-model-selection-randomized-search-py



# Feature importance

https://medium.com/@srnghn/the-mathematics-of-decision-trees-random-forest-and-feature-importance-in-scikit-learn-and-spark-f2861df67e3

https://github.com/scikit-learn/scikit-learn/blob/18cdaa69c14a5c84ab03fce4fb5dc6cd77619e35/sklearn/tree/_tree.pyx#L1056

Accuracy-based importance:

https://www.displayr.com/how-is-variable-importance-calculated-for-a-random-forest/

	Each tree has its own out-of-bag sample of data that was not used during construction. 
	This sample is used to calculate importance of a specific variable. First, the prediction 
	accuracy on the out-of-bag sample is measured. Then, the values of the variable in the 
	out-of-bag-sample are randomly shuffled, keeping all other variables the same. Finally, 
	the decrease in prediction accuracy on the shuffled data is measured.
	
	
	The mean decrease in accuracy across all trees is reported. This importance measure is also broken 
	down by outcome class. For example, age is important for predicting that a person earns over $50,000,
	but not important for predicting a person earns less.

	Intuitively, the random shuffling means that, on average, the shuffled variable has no predictive power. 
	This importance is a measure of by how much removing a variable decreases accuracy, and vice versa — b
	y how much including a variable increases accuracy.

	Note that if a variable has very little predictive power, shuffling may lead to a slight increase in 
	accuracy due to random noise. This in turn can give rise to small negative importance scores, 
	which can be essentially regarded as equivalent to zero importance.	


Gini-based importance

	When a tree is built, the decision about which variable to split at each node uses a 
	calculation of the Gini impurity.

	For each variable, the sum of the Gini decrease across every tree of the forest is accumulated 
	every time that variable is chosen to split a node. The sum is divided by the number of trees in 
	the forest to give an average. The scale is irrelevant: only the relative values matter. 
	In the example above, occupation is over five times more important than country.

	The importances are roughly aligned between the two measures, with numeric variables age and 
	hrs_per_week being lower on the Gini scale. This may indicate a bias towards using numeric variables 
	to split nodes because there are potentially many split points.	

Importance for numeric outcomes

	The previous example used a categorical outcome. For a numeric outcome (as show below) 
	there are two similar measures:

	Percentage increase in mean square error is analogous to accuracy-based importance, 
	and is calculated by shuffling the values of the out-of-bag samples.
	Increase in node purity is analogous to Gini-based importance, and is calculated based on the reduction 
	in sum of squared errors whenever a variable is chosen to split.

	One advantage of the Gini-based importance is that the Gini calculations are already performed during training, 
	so minimal extra computation is required. A disadvantage is that splits are biased towards variables 
	with many classes, which also biases the importance measure. Both methods may overstate 
	the importance of correlated predictors.

## Gini Importance / Mean Decrease in Impurity (MDI) :

https://medium.com/the-artificial-impostor/feature-importance-measures-for-tree-models-part-i-47f187c1a2c3

		MDI counts the times a feature is used to split a node, weighted by the number of samples it splits:
		Gini Importance or Mean Decrease in Impurity (MDI) calculates each feature importance as 
		the sum over the number of splits (across all tress) that include the feature, 
		proportionally to the number of samples it splits.
		
		However, Gilles Louppe gave a different version in [4]. Instead of counting splits, 
		the actual decrease in node impurity is summed and averaged across all trees. 
		(weighted by the number of samples it splits).
		
		In scikit-learn, we implement the importance as described in
		(often cited, but unfortunately rarely read…). It is sometimes called “gini importance” 
		or “mean decrease impurity” and is defined as the total decrease in node impurity 
		(weighted by the probability of reaching that node 
		(which is approximated by the proportion of samples reaching that node)) 
		averaged over all trees of the ensemble.
		
		At each split in each tree, the improvement in the split-criterion is the importance 
		measure attributed to the splitting variable, and is accumulated over all the trees in 
		the forest separately for each variable.
		
## Permutation Importance or Mean Decrease in Accuracy (MDA)
		
		This is IMO most interesting measure, because it is based on experiments on out-of-bag(OOB) 
		samples, via destroying the predictive power of a feature without changing its 
		marginal distribution. Because scikit-learn doesn’t implement this measure, 
		people who only use Python may not even know it exists.
		
		Random forests also use the OOB samples to construct a different variable-importance measure, 
		apparently to measure the prediction strength of each variable. When the bth tree is grown, 
		the OOB samples are passed down the tree, and the prediction accuracy is recorded. Then the values 
		for the jth variable are randomly permuted in the OOB samples, and the accuracy is again computed. 
		The decrease in accuracy as a result of this permuting is averaged over all trees, and is used as a 
		measure of the importance of variable j in the random forest. … The randomization effectively voids 
		the effect of a variable, much like setting a coefficient to zero in a linear model (Exercise 15.7). 
		This does not measure the effect on prediction were this variable not available, because if the model
		was refitted without the variable, other variables could be used as surrogates.

## Feature Importance Measure in Gradient Boosting Models

		LightGBM: importance_type (string, optional (default=”split”)) — How the importance is calculated. 
		If “split”, result contains numbers of times the feature is used in a model. If “gain”, result contains 
		total gains of splits which use the feature.
		
		XGBoost: ‘weight’ — the number of times a feature is used to split the data across all trees. 
		‘gain’ — the average gain of the feature when it is used in trees ‘cover’ — the average coverage of 
		the feature when it is used in trees, where coverage is defined as the number of 
		samples affected by the split
		
		 It’s basically the same as the Gini Importance implemented in R packages and in scikit-learn 
		 with Gini impurity replaced by the objective used by the gradient boosting model.

How do we calculate variable importance for a regression tree in random forests?

https://www.quora.com/How-do-we-calculate-variable-importance-for-a-regression-tree-in-random-forests

		1. %IncMSE - It is computed from permuting test data: For each tree, the prediction error on test 
		is recorded (Mean Squared Error - MSE ). Then the same is done after permuting each predictor variable. 
		The difference between the two are then averaged over all trees, and normalized by the standard 
		deviation of the differences. If the standard deviation of the differences is equal to 0 for a 
		variable, the division is not done (but the average is almost always equal to 0 in that case).
		Higher the difference is, more important the variable. MSE = mean((actual_y - predicted_y)^2)

		2. IncNodePurity -  Total decrease in node impurities from splitting on the variable, averaged 
		over all trees. Impurity is measured by residual sum of squares. Impurity is calculated only 
		at node at which that variable is used for that split. Impurity before that node, 
		and impurity after the split has occurred.

https://blog.datadive.net/selecting-good-features-part-iii-random-forests/

		Random forests are among the most popular machine learning methods thanks to their relatively 
		good accuracy, robustness and ease of use. They also provide two straightforward methods for
		feature selection: mean decrease impurity and mean decrease accuracy.

		Mean decrease impurity:
			Random forest consists of a number of decision trees. Every node in the decision trees 
			is a condition on a single feature, designed to split the dataset into two so that similar 
			response values end up in the same set. The measure based on which the (locally) optimal 
			condition is chosen is called impurity. For classification, it is typically either Gini 
			impurity or information gain/entropy and for regression trees it is variance. Thus when 
			training a tree, it can be computed how much each feature decreases the weighted impurity 
			in a tree. For a forest, the impurity decrease from each feature can be averaged and 
			the features are ranked according to this measure.

			This is the feature importance measure exposed in sklearn’s Random Forest implementations 
			(random forest classifier and random forest regressor).


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
			
		Mean decrease accuracy
			Another popular feature selection method is to directly measure the impact of each feature 
			on accuracy of the model. The general idea is to permute the values of each feature and 
			measure how much the permutation decreases the accuracy of the model. Clearly, 
			for unimportant variables, the permutation should have little to no effect on model 
			accuracy, while permuting important variables should significantly decrease it.

			This method is not directly exposed in sklearn, but it is straightforward to implement it. 
			Continuing from the previous example of ranking the features in the Boston housing dataset:
		
				from sklearn.cross_validation import ShuffleSplit
				from sklearn.metrics import r2_score
				from collections import defaultdict

				X = boston["data"]
				Y = boston["target"]

				rf = RandomForestRegressor()
				scores = defaultdict(list)

				#crossvalidate the scores on a number of different random splits of the data
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
				print "Features sorted by their score:"
				print sorted([(round(np.mean(score), 4), feat) for feat, score in scores.items()], reverse=True)

https://stackoverflow.com/questions/15810339/how-are-feature-importances-in-randomforestclassifier-determined

		In scikit-learn, we implement the importance as described in. It is sometimes called 
		"gini importance" or "mean decrease impurity" and is defined as the total decrease in node 
		impurity (weighted by the probability of reaching that node averaged over all trees of the ensemble.

		In the literature or in some other packages, you can also find feature importances implemented as 
		the "mean decrease accuracy". Basically, the idea is to measure the decrease in accuracy on 
		OOB data when you randomly permute the values for that feature. If the decrease is low, 
		then the feature is not important, and vice-versa.

