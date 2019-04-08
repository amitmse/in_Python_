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



## feature’s importance

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

## Gini Importance / Mean Decrease in Impurity (MDI)

