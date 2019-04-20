# Regularization


## Bias–variance trade-off

	- The bias (under-fitting) is an error from erroneous assumptions in the learning algorithm. 
		High bias can cause an algorithm to miss the relevant relations between features and target outputs.
		
		Bias is the difference between the average prediction of our model and the correct value 
		which we are trying to predict. Model with high bias pays very little attention to 
		the training data and oversimplifies the model. It always leads to high error on training and test data.
		
https://towardsdatascience.com/understanding-the-bias-variance-tradeoff-165e6942b229
		
	- The variance (over-fitting) is an error from sensitivity to small fluctuations in the training set. 
		High variance can cause an algorithm to model the random noise in the training data, 
		rather than the intended outputs.
		
		Model with high variance pays a lot of attention to training data and does not 
		generalize on the data which it hasn’t seen before. As a result, such models perform very well on 
		training data but has high error rates on test data.
		

## Ridge Regression: L2 norm (sum of square of coefficients)
	- Objective = RSS + α * (sum of square of coefficients) [RSS : Residual Sum of Squares]
	- Shrink the estimated association of each variable with the response, except the intercept β0. 
		Intercept is a measure of the mean value of the response
	- The coefficients that are produced by the standard least squares method are scale equi-variant 
		i.e. if we multiply each input by c then the corresponding coefficients are scaled by a factor of 1/c. 
		Therefore, regardless of how the predictor is scaled, the multiplication of predictor and 
		coefficient(Xjβj) remains the same. However, this is not the case with ridge regression, 
		and therefore, we need to standardize the predictors or bring the predictors to the same scale 
		before performing ridge regression.
	- The ridge coefficients are a reduced factor of the simple linear regression coefficients and thus 
		never attain zero values but very small values

## Lasso (Least Absolute Shrinkage and Selection Operator): L1 norm (sum of absolute value of coefficients)
	- Objective = RSS + α * (sum of absolute value of coefficients)
	- Differs from ridge regression only in penalizing the high coefficients
		
## Disadvantage :
	- For the same values of alpha, the coefficients of lasso regression are much smaller as compared 
		to ridge regression. For the same alpha, lasso has higher RSS (poorer fit) as compared 
		to ridge regression. Many of the coefficients are zero even for very small values of alpha 
		(LASSO) and this phenomenon of most of the coefficients being zero is called "sparsity".
	- Ridge: It will shrink the coefficients for least important predictors, very close to zero. 
		But it will never make them exactly zero. 
	- Lasso: L1 penalty has the effect of forcing some of the coefficient estimates to be exactly 
		equal to zero when the tuning parameter λ is sufficiently large. Therefore, the lasso method 
		also performs variable selection and is said to yield sparse models.
	- Generally, regularizing the intercept is not a good idea and it should be left out of regularization.
	
Use Cases:

	- Ridge: It is majorly used to prevent over-fitting. Since it includes all the features, it is not very 
		useful in case of exorbitantly high #features, say in millions, 
		as it will pose computational challenges.
	
	- Lasso: Since it provides sparse solutions, it is generally the model of choice 
		(or some variant of this concept) for modelling cases where the #features are in millions or more. 
		In such a case, getting a sparse solution is of great computational advantage as the features 
		with zero coefficients can simply be ignored.
	

	Regularization, significantly reduces the variance of the model, without substantial increase in its bias. 
	So the tuning parameter λ, controls the impact on bias and variance. As the value of λ rises, 
	it reduces the value of coefficients and thus reducing the variance. Till a point, this increase in λ is 
	beneficial as it is only reducing the variance(hence avoiding over-fitting), without loosing any important 
	properties in the data. But after certain value, the model starts loosing important properties, 
	giving rise to bias in the model and thus under-fitting. Therefore, the value of λ should be carefully selected.


Both L1 and L2 work differently in the way that they penalize the size of a weight. 

	L2 will force the weights into a pattern similar to a Gaussian distribution
	
	L1 will force the weights into a pattern similar to a Laplace distribution
	

## ElasticNet Regression:
	- Objective = RSS + α * [{(1-b)/2}*(sum of square of coefficients) + {b}*(sum of absolute value of coefficients)]
				ridge : b=0
				lasso : b=1

## Dropout Regularization

Most neural network frameworks implement dropout as a separate layer. Dropout layers function as a regular, densely connected neural network layer. The only difference is that the dropout layers will periodically drop some of their neurons during training. You can use dropout layers on regular feedforward neural networks.

https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
