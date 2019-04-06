
----------------------------------------------------------------------------------------------------------------

| Ensembling	| Bagging	| RF	| Handle Overfitting	| Reduce Variance		| Independent Calssifiers |
|----|----|----|----|----|----|
| Ensembling	| Boosting	| GBM	| Can Overfit		| Reduce Bias and Variance	| Sequential Classifier   |
|----|----|----|----|----|----|

## ADA Boost

	--------------------------------------------------------------------------------------------------
	1. Assign same weight to all obs ( 1/No. of obs )
  
	2. Develop a tree and pick node (decision stump) with lowest weighted training error
  
	3. Compute coefficient		(1/2 ln[{1 - weighted error(ft)}/weighted error(ft)])
  
	4. Recompute weights		( ai e^(-Wt))
  
	5. Normalize weights
	
  	-------------------------------------------------------------------------------------

	y^ = sign(sum(Wt ft(X))) 	(Wt=Coefficient)
	
	first compute Wt (Coefficient)
		Wt = 1/2 ln[{1 - weighted error(ft)}/weighted error(ft)] 
			(weighted error	= total weight of mistake/total weight of all data point)
			
	second compute ai(weight)
		for first iteration ai = 1/N (N=no of observation)
		ai = "ai e^(-Wt)" if ft(Xi)=Yi OR "ai e^(Wt)" if ft(Xi) not equal to Yi	(ai=Weight)
		
	Normalize weight ai
		ai = ai/(sum of all a's)
		
	Function	: (1/sum (Wi)) sum(Wi exp(-2Yi -1)f(Xi))
	
	Initial Value	: 1/2 log [(sum (Yi Wi e^-oi))/sum ((1-Yi)Wi e^oi)]
	
	Gradient	: Zi = -(2Yi - 1) exp(-(2Yi-1)fXi))
  
  	-------------------------------------------------------------------------------------
  
 ## GBM
 
 	--------------------------------------
	1. Learn a regression predictor
	2. compute the error of residual
	3. learn to predict the residual
	4. identify good weights
	---------------------------------------	
 
	1. Y 	  = M(x) + error (Logistic Regression if dep is binay). get the weight as well
  
	2. Error  = G(x) + error2 (Regress error with other ind var. Apply linear regression as error is continuous. 
				   get the weight as well
                  
	3. Error2 = H(x) + error3 (continue the #2 until you get low error)
  
	4. Y 	  = M(x) + G(x) + H(x) + error3	(combine all model together)
  
	5. Y 	  = alpha * M(x) + beta * G(x) + gamma * H(x) + error4 	(alpha, beta, gamma are weight of each model)
  
	--------------------------------------
  	Pseudo-code of the GBM algorithm
	
		1. Initialize the outcome
		
		2. Iterate from 1 to total number of trees
	
		  	2.1 Update the weights for targets based on previous run (higher for the ones mis-classified) 
			[weight = 0.5log(1-error/error)]	
			It would indicate higher weights to trees with lower error rate.
			
		  	2.2 Fit the model on selected subsample of data
		  
		  	2.3 Make predictions on the full set of observations
			
		  	2.4 Update the output with current results taking into account the learning rate
			
		3. Return the final output.
		
Gradient boosting involves three elements: 

	1. A loss function to be optimized: 
		- The loss function used depends on the type of problem being solved.
		- It must be differentiable, but many standard loss functions are supported and you can define your own. 
			For example, regression may use a squared error and classification may use logarithmic loss.
		- A benefit of the gradient boosting framework is that a new boosting algorithm does not have to 
			be derived for each loss function that may want to be used, instead, it is a generic enough 
			framework that any differentiable loss function can be used.
	
	2. A weak learner to make predictions: 
		- Decision trees are used as the weak learner in gradient boosting.
		- Specifically regression trees are used that output real values for splits and whose output 
			can be added together, allowing subsequent models outputs to be added and “correct” 
			the residuals in the predictions.
		- Trees are constructed in a greedy manner, choosing the best split points based on purity 
			scores like Gini or to minimize the loss.
		- Initially, such as in the case of AdaBoost, very short decision trees were used that only had 
			a single split, called a decision stump. Larger trees can be used generally with 4-to-8 levels.
		- It is common to constrain the weak learners in specific ways, such as a maximum number of layers, 
			nodes, splits or leaf nodes.
		- This is to ensure that the learners remain weak, but can still be constructed in a greedy manner.

	3. An additive model to add weak learners to minimize the loss function:
		- Trees are added one at a time, and existing trees in the model are not changed.
		- A gradient descent procedure is used to minimize the loss when adding trees.
		- Traditionally, gradient descent is used to minimize a set of parameters, such as the coefficients 
			in a regression equation or weights in a neural network. After calculating error or loss, 
			the weights are updated to minimize that error.
		- Instead of parameters, we have weak learner sub-models or more specifically decision trees. After 
			calculating the loss, to perform the gradient descent procedure, we must add a tree to 
			the model that reduces the loss. We do this by parameterizing the tree, then modify the 
			parameters of the tree and move in the right direction by reducing the residual loss.
		- Generally this approach is called functional gradient descent or gradient descent with functions.

Improvements to Basic Gradient Boosting

	- Gradient boosting is a greedy algorithm and can overfit a training dataset quickly.
	- It can benefit from regularization methods that penalize various parts of the algorithm and generally 
		improve the performance of the algorithm by reducing overfitting.
	- In this this section we will look at 4 enhancements to basic gradient boosting:
		1. Tree Constraints:
			- It is important that the weak learners have skill but remain weak.
			- There are a number of ways that the trees can be constrained.
			- A good general heuristic is that the more constrained tree creation is, the more trees 
				you will need in the model, and the reverse, where less constrained individual trees, 
				the fewer trees that will be required.
			- Below are some constraints that can be imposed on the construction of decision trees:
				# Number of trees: generally adding more trees to the model can be very slow to overfit. 
					The advice is to keep adding trees until no further improvement is observed.
				# Tree depth: deeper trees are more complex trees and shorter trees are preferred. 
					Generally, better results are seen with 4-8 levels.
				# Number of nodes or number of leaves: like depth, this can constrain the size of tree, 
					but is not constrained to a symmetrical structure if other constraints are used.
				# Number of observations per split: imposes a minimum constraint on the amount of 
					training data at a training node before a split can be considered
				# Minimum improvement to loss:
					is a constraint on the improvement of any split added to a tree
		2. Shrinkage:
		3. Random sampling:				
		4. Penalized Learning:



Adaboost vs Gradient Boosting: 

	- Adaboost		:"shortcomings” are identified by high-weight data points.
	- Gradient Boosting	: “shortcomings” are identified by gradients. 

Boosting vs Bagging

	- Boosting: It is similar, however the selection of sample is made more intelligently. We subsequently 
			give more and more weight to hard to classify observations.
	- Bagging: It is an approach where you take random samples of data, build learning algorithms and take 
			simple means to find bagging probabilities.
