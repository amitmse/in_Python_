
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


![Function](https://github.com/amitmse/in_Python_/blob/master/Random_Forest/Bagging%20vs%20Boosting.PNG)

------------------------------------------------------------------------------------------------------------

 ## eXtreme Gradient Boosting (XGBoost)

	- XGBoost builds a series of trees to make predictions, and each tree corrects errors made by the previous ones. 
	- Offers speed and performance both.
	- Minimizes a loss function: mean squared error for regression and the log loss for classification.
	- Regularization: Controls model complexity to prevent overfitting.
		L1 (Lasso) and L2 (Ridge) regularization terms are added to the objective function. 
		This penalizes overly complex trees to avoid overfitting by discouraging the overly deep or detailed trees.

	- Shrinkage (Learning Rate): Each tree's contribution is modulated, reducing the impact of outliers.
		This mechanism is designed to improve the balance between model complexity and learning speed. 
		
	- Cross-Validation: XGBoost internally performs cross-validation tasks to fine-tune hyperparameters, 
		such as the number of trees, boosting round, etc.

	- The algorithm minimizes a predefined loss function by following the steepest descent in the model's parameter space.
 
	- Split: Histogram based approach to find best split and reduces complexity.
		Computational efficiency via split finding algorithms using approximate tree boosting.
		Employs the exact or approximate greedy algorithm for split discovery.
  		Splits up to the specified max_depth and then starts pruning back the tree, 
    			removing splits beyond which there is no positive gain.

		Gini index or information gain are used to evaluate to split.
  
		XGBoost employs a greedy search, evaluating each possible split and choosing the one that 
  			results in the greatest gain in the split quality metric.  
    
	- Early stopping
 
	- It allows to specify whether the model should have a positive or negative relationship with each feature, 
		implementing business logic into the model.

	- Feature Importance: Mechanisms to rank and select features, empowering better decision-making.
		Provides a way to calculate feature importance scores based on the number of times a feature is used 
		in the model and how much it contributes to reducing the objective function.

		Different ways to get feature importance
			- use built-in feature importance.
			- use permutation based importance.
			- use shap based importance.

	- Feature Selection: Refer below Similarity Score.
		XGBoost uses a few criteria for feature selection.		
		Feature importance scores: gain, cover, weight.
			- Gain: Average loss reduction when using a feature for splitting.
				- XGBoost considers the impurity (e.g., Gini impurity, entropy) of the parent node 
					(before the split) and the impurity of the child nodes (after the split). 
				- Gain Calculation: Difference between the impurity of the parent node and the weighted 
					sum of the impurity of the child nodes. 
					This represents the reduction in impurity achieved by the split. 
				- Feature Importance: The feature with the highest gain value is considered the most important 
					feature for that particular split and, cumulatively, for the entire tree.
					https://xgboost.readthedocs.io/en/latest/tutorials/model.html
			- Cover: Indicates how many times a feature is used to split data across all trees, 
				weighted by the number of data points that go through those splits. 
			- Weight: Represents the total number of times a feature is used to split data across all trees.
   			Access above using the get_feature_importance() method after training your XGBoost model.
  
		Thresholding above scores: Recursive Feature Elimination (RFE), SHAP values. 
			You can set a threshold on the feature importance scores and select features that exceed that threshold. 
			The `SelectFromModel` class in Scikit-learn can be used to apply this threshold-based selection.

		Recursive Feature Elimination (RFE): It iteratively removes features based on their importance 
				and evaluates the model's performance on the remaining features. 
			This can help identify the most relevant features while improving model performance and 
				reducing training time. 
    
		SHAP Values (SHapley Additive exPlanations): values provide a way to understand how each feature contributes 
				to the model's predictions. Similar to the beta of linear regression.
			They can help in identifying the most important features and understanding their impact on the model.
			
 
		mRMR (Minimum Redundancy, Maximum Relevance): It's a feature selection algorithm that identifies 
			the most relevant features for predicting the target variable while minimizing redundancy 
			between selected features. This process improves model performance by focusing on 
			the most important information and reducing overfitting.
   
			- Relevance: mRMR aims to select features that have a strong relationship with the target variable 
   					(high correlation).
				The F-statistic, which is derived from ANOVA if the target is discrete or correlation 
					if the target is categorical.
				The F-statistic determines the degree of linear association between the features and the target. 
				If the target is categorical, the F-statistic is calculated using Scikit-learn’s f_classif function. 
				If the target is continuous, the F-statistic is determined using f_regression.
				Mutual information: Quantifies how much we know about one variable, by examining the values of a 
					second variable. In other words, it measures the non-linear association between features. 
					Higher values indicate stronger associations.

			- Redundancy: It also strives to minimize the correlation between the selected features themselves, 
				meaning they should not be highly correlated with each other.
			- Feature Ranking: mRMR ranks features based on a score that considers both relevance and redundancy. 
			- Selection Process: The algorithm iteratively selects features, starting with the highest-ranking feature, 
				and adds features that maximize the score. 
			- SULOV Method: Some implementations, like those in Featurewiz, 
				use the SULOV (Searching for Uncorrelated List of Variables) method to ensure low redundancy 
				and high relevance in the selection process. 
			- Benefits: Improved Model Performance, Reduced Complexity, Faster Training.
			- Limitation: May not capture feature interactions.
 
 		Featurewiz: It's feature selection is powered by recursive XGBoost ranking.  
		- Start with Everything. Feed the entire dataset into the selection process.
		- XGBoost Feature Ranking. Train an XGBoost model to assess feature importance.
		- Select Key Features. Extract the most significant features based on importance scores.
		- Prune and Repeat. Keep only the top-ranked features and rerun the process on a refined subset.
		- Iterate Until Optimal. Continue the cycle until a stopping criterion 
			(like stability or diminishing returns) is met.
		- Finalize the Feature Set. Merge selected features from all cycles, eliminating duplicates 
			to form the final optimized set.

		Split-Driven Recursive: This improved method introduces validation-based feature selection, 
  			leveraging Polars for speed and efficiency.
		- Split Data for Validation: The dataset is divided into training and validation sets.
		- Feature Ranking (with Validation): Features are ranked based on importance in the training set 
			while evaluating performance on the validation set.
		- Select Key Features (with Validation): Features are chosen based on both importance scores 
			and how well they generalize.
		- Repeat with New Splits: The process is rerun multiple times with different train/validation splits.
		- Stabilized Feature Set: Selected features from all runs are merged, removing duplicates, 
			leading to a more robust and reliable selection.

	- Handling Missing Data: It can manage missing data in both the training and evaluation phases.
 
	- Parallel Processing: Parallel and distributed computing, deliver high efficiency.
		While it can parallelize certain aspects of training, it fundamentally operates by sequentially 
		building decision trees, with each tree learning from the errors of its predecessors. 

		XGBoost employs parallel computing primarily to speed up the training process, 
		not to build trees concurrently. It can parallelize tasks like:
		- Data loading and preprocessing: The initial steps of preparing the data for training 
			can be done in parallel. 
		- Tree construction: Building the individual decision trees can also be parallelized, 
			allowing for faster computation. 
		- Gradient calculation and update: Calculating the gradient and updating model parameters during 
			each iteration can be parallelized.
   
		It leverages Column Block for Parallel Learning, Weighted Quantile Sketch and Cache-aware Access.
		- Column Block for Parallel Learning: XGBoost organizes the data into blocks of columns, 
			where each block corresponds to a subset of features. By independently processing column blocks, 
			multiple CPU cores of computing nodes can simultaneously work on the tree-building process.
		- Weighted Quantile Sketch: It’s a data structure that approximates the distribution of feature values, 
			which helps quickly identifying potential split points for decision tress without having 
			to sort the entire dataset.
		- Cache-aware Access: XGBoost organizes its data structures and computations to align with the CPU cache 
			architecture. By doing so, frequently accessed elements are more likely to be present in the cache.
    
	- Despite the parallelization capabilities, XGBoost's core learning mechanism remains sequential. 
		It starts with a base model (usually a simple tree) and then iteratively adds new trees. 
		Each new tree focuses on correcting the errors made by the previous trees. This is achieved by: 
		- Calculating Residuals: The algorithm calculates the difference between the actual target values 
			and the predictions made by the ensemble of existing trees (the "residual errors"). 
		- Training a New Tree: A new decision tree is trained to predict these residuals. 
		- Updating Predictions: The predictions from the new tree are added to the predictions of 
			the previous trees, effectively reducing the residual errors. 
		- Iterating: This process of calculating residuals, training a new tree, and updating predictions is 
			repeated iteratively until the desired level of accuracy is achieved. 
    
	- Quality score or Similarity score for the Residuals:
 		It's used to split. Info gain 
		For regressor = (sum of residuals squared) / (number of residuals + λ)
					λ  is a regularisation parameter
		For classifier = (sum of residuals squared) / [ pr(1-pr) + λ ]
					pr  is probability

		Feature importance scores:
    			- Gain: Average loss reduction gained when using a feature for splitting.
    			- Cover: The number of times a feature is used to split data across trees weighted 
       				by training data points.
    			- Weight: Total number of times a feature is used to split data across all trees.
     
	- Loss Functions
		- Logistic Loss: Commonly employed in binary classification problems. It calculates the likelihood of 
			the predicted class, converting it to a probability with a sigmoid function.
 		- Softmax Loss: Generally used for multi-class classification tasks. It calculates a 
			probability distribution for each class and maximizes the likelihood across all classes.
		- Adaptive Log Loss (ALogLoss): Introduced in XGBoost, this loss function provides a balance between speed 
			and accuracy. It's derived by approximating the Poisson likelihood.

		Regression 	= (1/2)*(actual - predicted)^2
		Classification 	= -[y*log(p) - (1-y)log(1-p)]
		
		Total loss will be the sum of this function
 
### XGBoost algorithm:
	- The first step in XGBoost is to make the first guess, that is, to determine the base score. 
		The base score is usually set at 0.5. So the initial prediction values are 0.5. 
		Then the residuals are obtained by subtracting 0.5 from the actual y values. 
		A tree model is established with these residues obtained as in GBM.
		Initial Prediction is 0.5 for both regression and classification.

	- Residuals are collected at the initial node and the similarity score of this node is calculated. 
		Then, trees are created by dividing each independent variable by threshold values. 
		The similarity score and gain value in each tree are calculated.
		In this way, all possible trees are created and the tree with the highest gain value is continued. 
		These operations are done with greedy algorithm logic.
		Similarity score is the evaluation metric for nodes. 
		Gain score is an evaluation criterion for trees.

	- Similarity Score:
		The smaller the similarity, the less they are similar.  
		To get all the residuals into one leaf and calculate the similarity score.
   
		For regressor = (Sum of Residuals)^2 / (number of Residuals + λ )
			λ (lambda) is the regularization parameter, which helps prevent overfitting.    
    
		For classifier = (Sum of Residuals)^2 / [pr*(1-pr) + λ]
			pr  is previous probability
  
	- Gain:
		How great is it the leaves classify similar residuals compared to the root.
   
		Gain = Left Similarity + Right Similarity - Root Similarity
			Root Similarity: The Similarity Score of the Previous Tree is the Similarity Score.
    
			Gain — Gamma > 0 Keep the tree.
			Gain — Gamma < 0 Prune the tree.


	- Lambda (λ): regularisation parameter
		As the lambda increases, the similarity score will decrease and therefore this will also decrease 
		the gain score. This allows for more pruning, only branches with a high gain score are preserved 
		and overfitting can be prevented.
   
		The fewer instances in the branch, the lower the similarity score and the higher the probability 
		of these branches being pruned. It prevents overfitting and having less instances in leaf nodes.

		Lambda value is in the denominator in the output formula, as the lambda increases, the output value 
		will decrease. The correct prediction will be reached with more iterations, 
		that is, the number of trees.

	- Calculating these similarity and gain scores would take a long time on large datasets, 
		xgboost divides the data into quantiles instead of examining each value in the data.
		The default number of quantile is 33. As the number of quantiles increases, 
		xgboost will look at smaller ranges and make better predictions, but at the same time 
		the training time will be longer.An algorithm called "Sketches" is used to overcome 
		this training time problem. The “Sketches” algorithm converges to find the quantiles.

	- After the tree with the highest gain score is determined, the pruning process is started.
		During the pruning process, the "gamma" hyperparameter is used as a metric. 
		if Gain Score < Gamma,the branch is pruned. As gamma increases, the most valuable branches 
		remain on the tree, and this pruning helps prevent overfitting. Pruning is done from 
		the bottom to the top. If the bottom branch is not pruned, the upper branches are not examined.

	- Prediction is made after the pruning process is completed. The tree prediction is multiplied by 
		the learning rate and added to the prediction value of the first tree, 
		and a new prediction value is formed. 
		These operations continue until the specified number of iterations, namely n_estimators 
		(number of boosting trees).
   
		Predicted value (regressor) = First Prediction + (Learnin Rate)* (Second Prediction)
       
		Predicted Value (classifier) =  
  			log of odds of Initial prediction + eta(learning rate) * output from the leaves(mean value)
			
			Convert above value to probability with logistic function
			Probability = Exp^log(odds) / [1 + Exp^log(odds)]

	- Example: 
  
https://github.com/amitmse/in_Python_/blob/master/Boosting/Example.xlsx  

------------------------------------------------------------------------------------------------------------
### XGBoost hyperparameters: 
	- Maximum depth of each tree: A deeper tree can capture more complex relationships in the data 
 		but may also lead to overfitting. max_depth: [ 3, 4, 5, 6, 8, 10, 12, 15]
		Default is 6
  
	- minimum sum of instance weights (Hessian) needed in a child. It helps prevent overfitting 
		by controlling the creation of new nodes in the tree.
		Default is 1  
		min_child_weight:[ 1, 3, 5, 7 ]
  
	- subsample: This determines the fraction of training instances used for each tree, reducing the risk of overfitting. 
		Default is 1
  
	- colsample bytree: This parameter specifies the fraction of features used for each tree. 
		Similar to subsample, it helps prevent overfitting by reducing the model's reliance on specific features.
		Default is 1  
		colsample_bytree:[ 0.3, 0.4, 0.5 , 0.7 ]
  
	- learning rate (eta): This parameter controls the step size of the gradient descent algorithm. 
		A smaller learning rate can lead to more stable training but may require more iterations to converge.
		Default is 0.3
  		learning_rate: [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ]
    
	- gamma: This parameter specifies the minimum loss reduction required to make a split. 
		It can be useful for pruning the tree and preventing overfitting.
		Default is 0
		gamma:[ 0.0, 0.1, 0.2 , 0.3, 0.4 ]
  
	- L2 regularization: This parameter adds a penalty proportional to the squared magnitude 
		of the coefficients, helping to prevent overfitting.
		Default is 1
  
	- L1 regularization: This parameter adds a penalty proportional to the absolute value of 
		the coefficients, promoting sparsity in the model.
		Default is 0  
  
	------------------------------------------------------------------------------------
	- Hyper Parameter Optimization: RandomizedSearchCV
		params={
		 	"learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
		 	"max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
		 	"min_child_weight" : [ 1, 3, 5, 7 ],
		 	"gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
		 	"colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
			}
		https://github.com/krishnaik06/Hyperparameter-Optimization
  
------------------------------------------------------------------------------------------------------------

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

	- GBM is ensemble methods that build models in a forward stepwise manner, using decision trees as base learners. 
		The algorithm can be computationally intensive, making it a somewhat challenging learning model. 

	- Pre-sorting approach to find best split which is computationally expensive.
	- GBM stops splitting a node when it encounters a negative loss in the split. 
		Therefore it is more of a greedy algorithm.

 
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
  
	---------------------------------------------------------------------
 
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

	---------------------------------------------------------------------
 
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

------------------------------------------------------------------------------------------------------------



      
------------------------------------------------------------------------------------------------------------

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

### SHapley Additive exPlanations (SHAP) 

- SHAP does not go on and retrain the model for each subset. Instead, for the removed or left out feature, 
		it just replaces it with the average value of the feature and generates the predictions.

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

https://s3.amazonaws.com/thinkific-import-development/118220/TreeBasedAlgorithms_ACompleteBookfromScratchinRPython-200403-111115.pdf
