# Logistic Regression
		https://www.linkedin.com/pulse/logistic-regression-algorithm-step-amit-kumar/

## Logistic regression assumptions: 		
http://www.soc.iastate.edu/sapp/soc512LogisticNotes.pdf

	1. Dependent variable should be binary
	2. Linearity between independent & log odds 
	     (non-linear relationship between the dependent and independent variables)
	3. Independence of errors
	4. No perfect multicollinearity
    
Logistic regression uses MLE rather than OLS, it avoids many of the typical assumptions (listed below) tested in statistical analysis.

	Does not assume: 
		- normality of variables (both DV and IVs)
		
		- linearity between DV and IVs
		
		- homoscedasticity
		
		- normal errors
		
MLE allows more flexibility in the data and analysis because it has fewer restrictions
  
Maximum likelihood estimation (MLE): 
	- Finds parameter values that maximize the likelihood of making the observations given the parameters

Ordinary Least Squares (OLS): Finds parameter values that minimizing the error
Linear regression assumptions: (http://r-statistics.co/Assumptions-of-Linear-Regression.html)
	1. Linear in parameters
	2. Mean of residuals is zero
	3. Homoscedasticity of residuals /	equal variance of residuals
	4. No autocorrelation of residuals
	5. Normality of residuals 
	6. X variables and residuals are uncorrelated 
	7. No perfect multicollinearity
	8. number of observations must be greater than number of Xs

## Derivation of Logistic Regression:
 	Model Equation:
		
		Y = a + bX (Y = dependent_variable, a=Intercept, b = coefficient, X = independent_variable)
	
	Logit function/Sigmoid Function:
		Y 		= xp(a + bX)/{1 + Exp(a + bX)}   = [1/{1 + Exp -(a + bX)}]  = 1/(1+exp^-y)
		1 - Y 		= xp-(a + bX)/{1 + Exp-(a + bX)} =  1/{1 + Exp(a + bX)}	    = 1/(1+exp^y)
		Y/(1-Y)		= Exp(a + bX)/{1 + Exp(a + bX)}]/ [1/{1 + Exp(a + bX)}]     = Exp(a + bX) = exp^y
		Log{Y/(1-Y)}	= a + bX (Apply log to convert non-linear relationship into linear relationship)
	
	Maximum Likelihood: 
		finds parameter values that maximize the likelihood of making the observations given the parameters
		(https://onlinecourses.science.psu.edu/stat414/node/191)
		Pi = {Pr(Yi = 1/Xi) if Yi = 1}	= Pr^Yi (Yi is a Bernoulli random variable) P
		{1 - Pr(Yi = 1/Xi)  if Yi = 0}	= (1-Pr)^(1-Yi)	1-P
			
	Likelihood function/Joint probability density function: (Yi is success and failure)
	(https://stats.stackexchange.com/questions/211848/likelihood-why-multiply)
		= Product[(Pr^Yi){(1-Pr)^(1-Yi)}]
		
	Log Likelihood Function:
		(Applying Logs on likelihood equation and  product will become sum. Refer to property of LOG)
		= Sum[{Yi*Log(Pr)} + {(1-Yi)*Log(1-Pr)}] (Apply log in above eq. and simplify it. cost function)
		= Sum[Yi*Log(Pr) - Yi*Log(1-Pr) + Log(1-Pr)]
		= Sum[Yi*Log{Pr/(1-Pr)}] + Sum[Log(1-Pr)] 
			[Substitute [Log{Pr/(1-Pr) = a + bX] and [1-Pr = 1 / {1 + Exp(a + bX)}]]
		= Sum[Yi*(a + bX)] + Sum[Log{1/1 + Exp(a + bX)}]
		= Sum[Yi*(a + bX)] + Sum[ Log(1) - Log{1 + Exp(a + bX)}] (Log(1) = 0)
		= Sum[Yi*(a + bX)] - Sum[Log{1 + Exp(a + bX)}]				
		= -[Sum[Yi*(a + bX)] - Sum[Log{1 + Exp(a + bX)}]] 
		(Apply negative to minimize the Log Likelihood Function)
		
	Gradient of Log Likelihood Function : 
		First Differentiation (with respect to beta) of Log Likelihood Function
		= [Yi*X] - [X*Exp(a + bX) / {1 + Exp(a + bX)}]
		= -[[Yi*X] - [X*Exp(a + bX) / {1 + Exp(a + bX)}]]	
		(Negative is part of negative log likelihood function. Refer to gradient_log_likelihood)
		
	Hessian Matrix :
		Second Differentiation (with respect to beta) of Log Likelihood Function
		First Differentiation of Gradient of Log Likelihood Function
		= 0 - [{(X*Exp(a + bX)*X)/(1 + Exp(a + bX))} + {(X*Exp(a + bX))/((1+Exp(a + bX))^2)*(Exp(a + bX)*X)}]	
			(Differentiation of [Yi*X] will be 0 due to no beta.
		= [(X*X*Exp(a + bX))/((1+Exp(a + bX))^2)*{Exp(a + bX) - (1+Exp(a + bX))}]
		= -[(X*X*Exp(a + bX))/((1+Exp(a + bX))^2)]
		= (X*X*Exp(a + bX))/((1+Exp(a + bX))^2) 
			(minus will be cancel out due to minus sign in Gradient of Log Likelihood Function)
			
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
			
The ROC curve is a graphical plot that illustrates the performance of any binary classifier system as 
its discrimination 	threshold is varied. True positive rate (Sensitivity : Y axis ) is plotted in 
function of the false positive rate (100-Specificity : X axis) for different cut-off points. 
Each point on the ROC curve represents a sensitivity/specificity pair corresponding to a particular decision threshold.
	
Standard Error Coef: 
	Linear regression standard error of Coef : SE  = sqrt [ S(yi - yi)2 / (n - 2) ] / sqrt [ S(xi - x)2 ]
	http://support.minitab.com/en-us/minitab-express/1/help-and-how-to/modeling-statistics/regression/how-to/binary-logistic-regression/interpret-the-results/all-statistics-and-graphs/coefficients/
	The standard error of the coefficient estimates the variability between coefficient estimates that you would 
	obtain if you took samples from the same population again and again. The calculation assumes that the sample 
	size and the coefficients to estimate would remain the same if you sampled again and again.
	Interpretation : Use the standard error of the coefficient to measure the precision of the estimate of the coefficient. 
	The smaller the standard error, the more precise the estimate. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3976195/

Jacobian is similar to first order derivative and Hessian is similar to second order derivative.
	The determinant of a matrix is also sometimes referred to as the Hessian. 
	The Hessian matrix can be considered related to the Jacobian matrix. 
	Hessian matrices are used in large-scale optimization problems within Newton-type methods because they are 
	the coefficient of the quadratic term of a local Taylor expansion of a function.
	A bordered Hessian (Lagrange function) is used for the second-derivative test in certain constrained 
	optimization problems.
	The Hessian matrix of a convex function is positive semi-definite. And this property allows us to test 
	if a critical point x is a local maximum, local minimum, or a saddle point, as follows:
	 - If the Hessian is positive definite at x, then f attains an isolated local minimum at x. This is like “concave up”. 
	 - If the Hessian is negative definite at x, then f attains an isolated local maximum at x. This is like “concave down”.
	 - If the Hessian has both positive and negative eigenvalues then x is a saddle point for f. 
	   Otherwise the test is inconclusive. Graph is concave up in one direction and concave down in the other.
	- This implies that, at a local minimum (respectively, a local maximum), the Hessian is positive-semi-definite
	  (respectively, negative semi-definite).

If the gradient (the vector of the partial derivatives) of a function f is zero at some point x, then f has a 
critical point (or stationary point) at x. The determinant of the Hessian at x is then called the discriminant. 
If this determinant is zero then x is called a degenerate critical point of f. Otherwise it is non-degenerate.

Jacobian matrix is the matrix of first-order partial derivatives of a vector-valued function. When the matrix is a 
square matrix, both the matrix and its determinant are referred to as the Jacobian determinant.
The Jacobian of the gradient of a scalar function of several variables has a special name: the Hessian matrix, 
which in a sense is the "second derivative" of the function.

http://www.bioinfo.org.cn/~wangchao/maa/Numerical_Optimization.pdf
