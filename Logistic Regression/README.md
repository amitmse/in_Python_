# Logistic Regression

## Logistic Regression is a classification technique which predicts a binary outcome.

https://www.linkedin.com/pulse/logistic-regression-algorithm-step-amit-kumar/

-----------------------------------------------------------------------------------------------------------------------
## Logistic regression assumptions: 		

	1. Dependent variable should be binary
	2. Linearity between independent & log odds 
	     (non-linear relationship between the dependent and independent variables)
	3. Independence of errors
	4. No perfect multicollinearity
    	  
-----------------------------------------------------------------------------------------------------------------------
## Logistic Regression Algorithm Coded in Python:

https://github.com/amitmse/in_Python_/blob/master/Logistic%20Regression/Logistic_Regression.py

----------------------------------------------------------------------------------------------------------------------- 
## Derivation of Logistic Regression:
 	Model Equation:
		
		Y = a + bX (Y = dependent_variable, a=Intercept, b = coefficient, X = independent_variable)
	
-----------------------------------------------------------------------------------------------------------------------

	Logit function/Sigmoid Function:
		Y 		= Exp(a + bX)/{1 + Exp(a + bX)}   = [1/{1 + Exp -(a + bX)}] = 1/(1+exp^-y)
		1 - Y 		= Exp-(a + bX)/{1 + Exp-(a + bX)} =  1/{1 + Exp(a + bX)}    = 1/(1+exp^y)
		Y/(1-Y)		= Exp(a + bX)/{1 + Exp(a + bX)}]/ [1/{1 + Exp(a + bX)}]     = Exp(a + bX) = exp^y
		Log{Y/(1-Y)}	= a + bX (Apply log to convert non-linear relationship into linear relationship)
	
-----------------------------------------------------------------------------------------------------------------------

	Maximum Likelihood: 
		finds parameter values that maximize the likelihood of making the observations given the parameters
		Pi = {Pr(Yi = 1/Xi) if Yi = 1}	= Pr^Yi         --> (P, Yi is a Bernoulli random variable)
		{1 - Pr(Yi = 1/Xi)  if Yi = 0}	= (1-Pr)^(1-Yi)	--> (1-P)
			
	Likelihood function/Joint probability density function: (Yi is success and failure)
		= Product[(Pr^Yi){(1-Pr)^(1-Yi)}]
		
-----------------------------------------------------------------------------------------------------------------------

	Log Likelihood Function:
		(Applying Logs on likelihood equation and  product will become sum. Refer to property of LOG)
		= Sum[{Yi*Log(Pr)} + {(1-Yi)*Log(1-Pr)}] (Apply log in above eq. and simplify it. cost function/log loss)
		= Sum[Yi*Log(Pr) - Yi*Log(1-Pr) + Log(1-Pr)]
		= Sum[Yi*Log{Pr/(1-Pr)}] + Sum[Log(1-Pr)] 
			[Substitute [Log{Pr/(1-Pr) = a + bX] and [1-Pr = 1 / {1 + Exp(a + bX)}]]
		= Sum[Yi*(a + bX)] + Sum[Log{1/1 + Exp(a + bX)}]
		= Sum[Yi*(a + bX)] + Sum[ Log(1) - Log{1 + Exp(a + bX)}] (Log(1) = 0)
		= Sum[Yi*(a + bX)] - Sum[Log{1 + Exp(a + bX)}]				
		= -[Sum[Yi*(a + bX)] - Sum[Log{1 + Exp(a + bX)}]] 
		(Apply negative to minimize the Log Likelihood Function)
		
-----------------------------------------------------------------------------------------------------------------------

	Gradient of Log Likelihood Function : 
		First Differentiation (with respect to beta) of Log Likelihood Function
		= [Yi*X] - [X*Exp(a + bX) / {1 + Exp(a + bX)}]
		= -[[Yi*X] - [X*Exp(a + bX) / {1 + Exp(a + bX)}]]	
		(Negative is part of negative log likelihood function. Refer to gradient_log_likelihood)
		
-----------------------------------------------------------------------------------------------------------------------

	Hessian Matrix :
		Second Differentiation (with respect to beta) of Log Likelihood Function
		First Differentiation of Gradient of Log Likelihood Function
		= 0 - [{(X*Exp(a + bX)*X)/(1 + Exp(a + bX))} + {(X*Exp(a + bX))/((1+Exp(a + bX))^2)*(Exp(a + bX)*X)}]	
			(Differentiation of [Yi*X] will be 0 due to no beta.
		= [(X*X*Exp(a + bX))/((1+Exp(a + bX))^2)*{Exp(a + bX) - (1+Exp(a + bX))}]
		= -[(X*X*Exp(a + bX))/((1+Exp(a + bX))^2)]
		= (X*X*Exp(a + bX))/((1+Exp(a + bX))^2) 
			(minus will be cancel out due to minus sign in Gradient of Log Likelihood Function)

-----------------------------------------------------------------------------------------------------------------------
## Maximum likelihood estimation (MLE): 
	- Finds parameter values that maximize the likelihood of making the observations given the parameters
	- MLE allows more flexibility in the data and analysis because it has fewer restrictions
	    
-----------------------------------------------------------------------------------------------------------------------
### Cost function :
	It tried to quantify the error factor of logistic regression. It calculates how well the logistic 
	regression is performing based on the actual vs predicted value. Error factor = Predicted – Actual.
		
-----------------------------------------------------------------------------------------------------------------------
## Types of Gradient Descent:

1. Batch Gradient Descent: 
	It uses a complete dataset available to compute the gradient of the cost function hence and it's 
	very slow. 
	- Cost function is calculated after the initialization of parameters.
	- It reads all the records into memory from the disk.
	- After calculating sigma for one iteration, we move one step further, and repeat the process.

2. Mini-batch Gradient Descent:
	It is a widely used algorithm that makes faster and accurate results. The dataset, here, 
	is clustered into small groups of ‘n’ training datasets hence it's faster. In every iteration, 
	it uses a batch of ‘n’ training datasets to compute the gradient of the cost function.
	It reduces the variance of the parameter updates, which can lead to more stable convergence. 
	It can also make use of a highly optimized matrix that makes computing of the gradient very efficient.

3. Stochastic Gradient Descent:
	Stochastic gradient descent used for faster computation. First, it randomizes the complete dataset, 
	and then uses only one training example in every iteration to calculate the gradient. Its benifical 
	for huge datasets.

## Gradient Descent vs Newton's method

### Gradient Descent:
	- Simple
	- Need learning Rate
	- 2nd derivate not required
	- More number of iterations
	- Each iteration is cheap (no 2nd derivative )
	- If number of observation is large then its cheaper
	
### Newton's method:
	- Complex
	- No learning Rate required
	- 2nd derivate is required
	- Less number of iteration
	- Each iteration is expesive (2nd derivative )
	- If less number of observation (may be 1000) then its cheaper
	

### Solve a equation (identify beta):  

#### 1. Calculus: 
	It will faster if equation is simple. But in real life equations are very complex and messy and 
	its difficult to solve.   
			f(x) 	= X^2 - 2X + 2   
			df/dx 	= 2X - 2  
			2X	= 2  
			X	= 1   
	
#### 2. Gradient Descent:   
			Xi+1 = Xi - a f'(Xi)  	
			[Xi = initial guess, a = learning rate or step length or jump, Xi+1 = next guess]  
			f(x) 	= X^2 - 2X + 2  
			df/dx 	= 2X - 2  
			apply "Xi+1 = Xi - a f'(Xi)" on above equation. start with zero "0"  
			X1 = X0 - 0.2f'(3)			X0= 3 (initial guess), a=0.2 (guess)  
			X1 = 3  - 0.2(4)			[put 3 in "2X - 2": 2*3 - 2 = 6-2 =4]  
			X1 = 3 - 0.8  
			X1 = 2.2  
			X2 = X1 - 0.2f'(X1)  
			X2 = 2.2 - 0.2(2.4)			[put 2.2 in "2X - 2": 2*2.2 - 2 = 4.4-2 =2.4]  
			X2 = 1.72  
			X3 = X2 - 0.2f'(X2)  
			X3 = 1.72 - 0.2(1.44)		[put 1.72 in "2X - 2": 2*1.72 - 2 = 3.44-2 =1.44]  
			X3 = 1.72 - 0.288  
			X3 = 1.432  
			
	continue doing this untill we are close to 1 which is the exact solution. As we approach to local minimum, 
	Gradient Descent will automatically take smaller steps. So no need to decrease "a" over time. 
	optimization gradient descent:
			cX + d 		= Y [equation of line and solve this for c & d]  
			(cX + d) -  Y 	= 0 ( "cX + d" is predected Y^, Y^-Y is error and it should be zero)   
			min by(a,b) 	= sum ([cX + d]-Yi)^2		[c = cofficient, d=intercept]  
		
	First make initial guess for c & d then do the derivative by c & d seperately to get the optimium value of c & d. 
	Above process will apply on Gradient Descent "Xi+1 = Xi - a f'(Xi)". Gradient descent is based on 1st derivatives 
	only and it use all data at one time. Gradient descent generally requires more iterations. If data size is big then 
	it will take long time to compute.
			
	Stochastic Gradient descent: It takes portion of data at one time and do the computation and continue in same way. 
	cofficients are not exactly equals to Gradient descent but its close. 
	For BIG data its only option to apply Gradient descent in faster way.
			
#### 3. Newton Raphson:   
	Newton's method generally requires fewer iterations, but each iteration is slow as we need to 
	compute 2nd dervatives too. There is no guarantee that the Hessian is nonsingular. Additionally, we must 
	supply the second partial derivatives to the computer (and they can sometimes be very difficult to calculate).
	(http://www.stat.missouri.edu/~spinkac/stat8320/Nonlinear.pdf)
	
			Xn+1 	= Xn - f(X)/f'(X)  
			f(X) 	= X^2 - 8  
			f'(X)	= 2X  
			X1	= 3 (guess)  
			X2	= X1 - f(X)/f'(X)  
				= 3  - [(3^2-8)/2*3]  
				= 3  - (1/6)  
				= 18-1/6  
				= 17/6  
			X3	= X2 - f(X2)/f'(X2)  
				= 17/6 - [(17/6)^2 - 8]/[2(17/6)]  
				= 2.828  
	
-----------------------------------------------------------------------------------------------------------------------
		
	Jacobian is similar to first order derivative and Hessian is similar to second order derivative. 
	The determinant of a matrix is also sometimes referred to as the Hessian. The Hessian matrix can 
	be considered related to the Jacobian matrix. Hessian matrices are used in large-scale optimization 
	problems within Newton-type methods because they are the coefficient of the quadratic term of a local 
	Taylor expansion of a function. A bordered Hessian (Lagrange function) is used for the second-derivative 
	test in certain constrained optimization problems. The Hessian matrix of a convex function is positive 
	semi-definite. And this property allows us to test if a critical point x is a local maximum, local minimum, 
	or a saddle point, as follows:
	 - If the Hessian is positive definite at x, then f attains an isolated local minimum at x (concave up)
	 
	 - If the Hessian is negative definite at x, then f attains an isolated local maximum at x (concave down)
	 
	 - If the Hessian has both positive and negative eigenvalues then x is a saddle point for f. 
	   Otherwise the test is inconclusive. Graph is concave up in one direction and concave down in the other.
	   
	- This implies that, at a local minimum (respectively, a local maximum), the Hessian is positive-semi-definite
	  (respectively, negative semi-definite).
	  
	If the gradient (the vector of the partial derivatives) of a function f is zero at some point x, then f has a
	critical point (or stationary point) at x. The determinant of the Hessian at x is then called the discriminant.
	If this determinant is zero then x is called a degenerate critical point of f. Otherwise it is non-degenerate.
	  
	Jacobian matrix is the matrix of first-order partial derivatives of a vector-valued function. When the matrix 
	is a square matrix, both the matrix and its determinant are referred to as the Jacobian determinant. 
	The Jacobian of the gradient of a scalar function of several variables has a special name: the Hessian matrix, 
	which in a sense is the "second derivative" of the function.
				
---------------------------------------------------------------------------------------------------------------------------------------

	beta(x) 	= covariance(x,y) / variance(x)
	correlation(x,y)= covariance(x,y) / [variance(x)*variance(y)]
	TSS 		= SUM[y-mean(y)]^2
	RSS 		= SUM[y-predicted(y)]^2
	R Squared	= 1.0 - (RSS/TSS)
	AIC		= (No of variable*2)               - (2*-Log Likelihood)
	BIC		= {No of variable*log(No of obs)}  - (2*-Log Likelihood)
	VIF 		= 1.0 / (1.0 - R Squared)
	Gini/Somer’s D 	= [2AUC-1] OR [(Concordant - Disconcordant) / Total  pairs]
	Divergence 	= [(meanG – meanB)^2] / [0.5(varG + varB)]	
			     [meanG = mean of score only for good, varB= variance of score only for bad ]
	Area under curve /C statistics = Percent Concordant + 0.5 * Percent Tied
			
		The ROC curve is a graphical plot that illustrates the performance of any binary classifier 
		system as its discrimination threshold is varied. True positive rate (Sensitivity : Y axis ) 
		is plotted in function of the false positive rate (100-Specificity : X axis) for different 
		cut-off points. Each point on the ROC curve represents a sensitivity/specificity pair 
		corresponding to a particular decision threshold.
	
	Standard Error Coef: 
		Linear regression standard error of Coef : SE  = sqrt [ S(yi - yi)2 / (n - 2) ] / sqrt [ S(xi - x)2 ]
	
		The standard error of the coefficient estimates the variability between coefficient estimates 
		that you would obtain if you took samples from the same population again and again. 
		The calculation assumes that the sample size and the coefficients to estimate would remain 
		the same if you sampled again and again. Use the standard error of the coefficient to measure 
		the precision of the estimate of the coefficient. 
		The smaller the standard error, the more precise the estimate.
		
		
	Recall and Precision:
		Recall is the fraction of instances that have been classified as true. On the contrary, 
		precision is a measure of weighing instances that are actually true. 
		While recall is an approximation, precision is a true value that represents factual knowledge.

	ROC curve:
		Receiver Operating Characteristic is a measurement of the True Positive Rate (TPR) against False 
		Positive Rate (FPR). We calculate True Positive (TP) as TPR = TP/ (TP + FN). On the contrary, 
		false positive rate is determined as FPR = FP/FP+TN where where TP = true positive, TN = true negative, 
		FP = false positive, FN = false negative.

	AUC vs ROC:
		AUC curve is a measurement of precision against the recall. Precision = TP/(TP + FP) and TP/(TP + FN).
		This is in contrast with ROC that measures and plots True Positive against False positive rate.
-----------------------------------------------------------------------------------------------------------------------

### Logistic regression uses MLE rather than OLS, it avoids many of the typical assumptions (listed below) 
tested in statistical analysis.

	Does not assume: 
		- normality of variables (both DV and IVs)
		
		- linearity between DV and IVs
		
		- homoscedasticity
		
		- normal errors
		
## Ordinary Least Squares (OLS): 
	Finds parameter values that minimizing the error. Assumptions of Linear regression: 

For Model:

	1. Linear in parameters : 
		Issue	: Beta not multiplied or divided by any other parameter. 
                          Incorrect and unreliable model which leads to error in result.
		Solution: Transformations of independent variables

For Variable: 

	2. No perfect multicollinearity :
		Issue: Issue: Regression coefficient variance will increase
		Test: VIF
		
For Error Tearm: 

	3. Normality of residuals :
		Issue: OLS estimators won’t have the desirable BLUE property
		
	4. Mean of residuals is zero :
		Issue: Error terms has zero mean and doesn’t depend on the independent variables. 
			Thus, there must be no relationship between the independent variable and the error term
	
	5. Homoscedasticity of residuals /equal variance of residuals
		Example	: Family income to predict luxury spending. Residuals are very small for low values of 
			  family income (less spend on luxury) while there is great variation in the size of 
			  the residuals for wealthier families. Standard errors are biased and it leads to 
			  incorrect conclusions about the significance of the regression coefficients
		Test	: Breush-Pagan test
		Solution: Weighted least squares regression.
			  Transform the dependent variable using one of the variance stabilizing transformations
	
	6. No autocorrelation of residuals :
		Issue: correlation with own lag (stock price today linked with yesterday's price). if above fails 
		 	then OLS estimators are no longer the Best Linear Unbiased Estimators. While it does not 
			bias the OLS coefficient estimates, the standard errors tend to be underestimated 
			(t-scores overestimated) when the autocorrelations of the errors at low lags are positive.
		Test :  Durbin–Watson
	
	7. X variables and residuals are uncorrelated 
	
	
	Number of observations must be greater than number of Xs

	Linear model should have residuals mean zero, have a constant variance, and not correlated with themselves 
	or other variables. If these assumptions hold true, the OLS procedure creates the best possible estimates.
		

---------------------------------------------------------------------------------------------------------------------------------------
