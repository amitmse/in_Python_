
----------------------------------------------------------------------------------------------------------------
# Boosting: 

	It decreases Bias.
	Similar to bagging but it fits weak learner sequentially (a model depends on the previous ones) in a very 
	adaptative way. Each model in the sequence is fitted giving more importance to the observations which are not 
	classified correctly (high error). Mainly focus on reducing bias.
	
	Bagging mainly focus at getting an ensemble model with less variance than its components whereas 
	boosting and stacking will mainly try to produce strong models less biased than their components 
	(even if variance can also be reduced).

	Following techniques are based on Boosting:
		- AdaBoost (Adaptive Boosting)
		- Gradient Tree Boosting (GBM)
		- eXtreme Gradient Boosting (XGBoost)
		- Light Gradient Boosting Machine (LightGBM)
		- Categorical Boosting (CatBoost)

![Function](https://github.com/amitmse/in_Python_/blob/master/Boosting/Compare.PNG)

----------------------------------------------------------------------------------------------------------------

## Pros:
	1. Step-by-Step Error Fixing:
	2. Flexible Error Measures
	3. High Accuracy
    
## Cons:
	1. Overfitting: No of trees or models    
	2. Computationally Expensive: Cost of #1, Gradient Boosting, Optimal learning rate, tree depth, and number of trees
	3. Slow Training Process: sequential learning

## Loss Functions:
	1. Mean Squared Error (MSE): ( 1/2 ) * ( Actual - Predicted )^2
	2. Cross-entropy: Difference between two probability distributions

----------------------------------------------------------------------------------------------------------------
## Boosting in Python

https://github.com/amitmse/in_Python_/blob/master/Boosting/Boosting_Try.py

----------------------------------------------------------------------------------------------------------------

| 	| 	| 	| 	| 		|  |
|----|----|----|----|----|----|
| Ensembling	| Bagging	| RF	| Handle Overfitting	| Reduce Variance		| Independent Calssifiers |
| Ensembling	| Boosting	| GBM	| Can Overfit		| Reduce Bias and Variance	| Sequential Classifier   |


![Function](https://github.com/amitmse/in_Python_/blob/master/Boosting/Boosting.PNG)

## Adaptive (ADA) Boosting: Adapts from mistakes

https://github.com/amitmse/in_Python_/blob/master/Boosting/Example.xlsx

	--------------------------------------------------------------------------------------------------
	1. Initialize Weights: Assign same weight to all obs ( 1/No. of obs ). The sum of weights is 1.
 		If there are 10 total obs then weight is 0.1 (1/10).
		Actual weight for target "Yes" = +0.1
		Actual weight for target "No" = -0.1 
  
  
	2. Iterative Learning: In each iteration, a model is trained.
 		- Each tree learns from previous ones. Misclassified observations gain more weight in the next iteration.
 		- Correctly classified observations retain their weights. All weights are adjusted to sum to 1.
   
		- For each iteration compute accuracy: (1/2) * ln [ (1 - Total Error) / Total Error ] 
			3 obs are misclassified out of 10 in the first iteration, the total error is 3/10.
   			accuracy: 0.43
   
		- Recompute weights: Adjust weight for misclassified observation: 
			correctly classified = Previous Weight * e^(-accuracy) = 0.07
			wrongly classified   = Previous Weight * e^(+accuracy) = 0.15
  
		- Normalize weights: All weights are adjusted to sum to 1. Divide each weight by the sum of all weights.
   			7 correctly classified and 3 wrongly classified, sum of weight = (0.07 X 7) + (0.15 X 3) =  0.92
      			Normalize weights for correctly classified = 0.071
			Normalize weights for wrongly   classified = 0.167
  
	3. Final Tree: Combine all models using weights.

-------------------------------------------------------------------------------------
  
 ## Gradient Boosting Machine (GBM)
 
	--------------------------------------
	1. Initialize with a base model
	2. Calculate residuals
	3. Predict the residuals with learning rate (0 to 1) or weight of model
	4. Combine previous models and check accuracy
	5. Repeat #2,3,4 until target achieved
	6. Final: Sum up all models
 
 
	1. Y 	  = M(x) + error (Logistic Regression if dep is binay). get the weight as well
  
	2. Error  = G(x) + error2 (Regress error with other ind var. Apply linear regression as error is continuous. 
				   Apply learning rate (0 to 1) or weight of the model.
                  
	3. Error2 = H(x) + error3 (continue the #2 until you get low error)
  
	4. Y 	  = M(x) + G(x) + H(x) + error3	(combine all model together)
  
	5. Y 	  = alpha * M(x) + beta * G(x) + gamma * H(x) + error4 	
 			(alpha, beta, gamma are weight / learning rates of each model)
  
	--------------------------------------
  	Pseudo-code of the GBM algorithm
	
		1. Initialize the outcome
		
		2. Iterate from 1 to total number of trees
	
		  	2.1 Update the weights for targets based on previous run (higher for the ones mis-classified) 
			[weight = 0.5*log[(1-error)/error]
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

------------------------------------------------------------------------------------------------------------

### Feature Importance

	- Analyze the importance of each input feature in the model's predictions. 
	- Techniques like tree-based models or methods that calculate the importance of each feature based 
		on its contribution to the model's predictions.
	- Tree-Based Algorithms feature importance scores based on how much each feature reduces impurity 
		(e.g., Gini index or information gain) in the decision tree nodes.
		Tree based: Decision Trees, Random Forests, XGBoost, LightGBM.
  	- Below are same as Feature Importance
		LIME: Local Interpretable Model-Agnostic Explanations. 
 		SHAP: SHapley Additive exPlanations.
  		Details are below in link
    
------------------------------------------------------------------------------------------------------------

## hyperparameters
	- Use techniques like Grid Search, Randomized Search or Bayesian Optimization to explore the parameter 
 		space and find the optimal combination. Details are below in link
   
https://github.com/amitmse/in_Python_/blob/master/Others/README.md
 
### AdaBoost hyperparameters:
	- Number of Estimators: This determines how many weak learners (e.g., decision trees) are 
		combined in the ensemble. More estimators can improve accuracy but also increase training time. 
	- Learning Rate: This controls the contribution of each weak learner to the final prediction. 
		A smaller learning rate means each weak learner has less influence, potentially requiring 
		more estimators to achieve the same performance. 
	- Base Estimator Hyperparameters: If the base estimator (e.g., decision trees) has its own hyperparameters 
		(like max_depth for decision trees), tuning these can also impact the AdaBoost model's performance. 
	- Loss Function (loss): AdaBoost supports different loss functions for classification, like exponential, 
		linear, and square, which affect how weights are assigned to misclassified samples. 
	- Random Seed: Setting a random seed ensures reproducibility, but experimenting with different random seeds 
		during hyperparameter tuning can improve the robustness of the model.

------------------------------------------------------------------------------------------------------------
### Gradient Boosting hyperparameters:
	- Learning Rate: This controls the contribution of each tree to the final prediction. 
		A smaller learning rate leads to more stable and robust models, but requires more trees 
		to achieve optimal performance. 
	- Number of Estimators (Trees): This parameter dictates how many trees are used in the ensemble. 
		A larger number of trees can improve performance, but also increases computational cost 
		and risk of overfitting. 
	- Max Depth: This limits the complexity of individual trees, preventing overfitting by restricting 
		how deep they can grow. 
	- Subsampling: This involves randomly selecting a subset of the training data for each tree. 
		Subsampling helps to prevent overfitting and can improve the generalizability of the model, 
		explains a guide on Hands-On Machine Learning with R. 
   
------------------------------------------------------------------------------------------------------------
### XGBoost hyperparameters: 
	- Maximum depth of each tree: A deeper tree can capture more complex relationships in the data 
 		but may also lead to overfitting.
	- minimum sum of instance weights (Hessian) needed in a child. It helps prevent overfitting 
		by controlling the creation of new nodes in the tree. 
	- subsample: This determines the fraction of training instances used for each tree, reducing the risk of overfitting. 
	- colsample bytree: This parameter specifies the fraction of features used for each tree. 
		Similar to subsample, it helps prevent overfitting by reducing the model's reliance on specific features. 
	- learning rate (eta): This parameter controls the step size of the gradient descent algorithm. 
		A smaller learning rate can lead to more stable training but may require more iterations to converge. 
	- gamma: This parameter specifies the minimum loss reduction required to make a split. 
		It can be useful for pruning the tree and preventing overfitting. 
	- L2 regularization: This parameter adds a penalty proportional to the squared magnitude 
		of the coefficients, helping to prevent overfitting. 
	- L1 regularization: This parameter adds a penalty proportional to the absolute value of 
		the coefficients, promoting sparsity in the model. 

------------------------------------------------------------------------------------------------------------  

https://s3.amazonaws.com/thinkific-import-development/118220/TreeBasedAlgorithms_ACompleteBookfromScratchinRPython-200403-111115.pdf
