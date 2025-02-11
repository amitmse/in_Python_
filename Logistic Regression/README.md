# Logistic Regression

	Logistic Regression is a classification technique which predicts a binary outcome.

https://www.linkedin.com/pulse/logistic-regression-algorithm-step-amit-kumar/

-----------------------------------------------------------------------------------------------------------------------
## Logistic Regression Assumptions: 		

	1. Dependent variable should be binary
	2. Linearity between independent & log odds 
	     (non-linear relationship between the dependent and independent variables)
	3. Independence of errors
	4. No perfect multicollinearity
    	  
-----------------------------------------------------------------------------------------------------------------------
## Logistic Regression Algorithm Coded in Python:

https://github.com/amitmse/in_Python_/blob/master/Logistic%20Regression/Logistic_Regression.py

----------------------------------------------------------------------------------------------------------------------- 
## Statistical Derivation of Logistic Regression:
### Model Equation:
		
		Y = a + bX (Y = dependent_variable, a=Intercept, b = coefficient, X = independent_variable)
	
-----------------------------------------------------------------------------------------------------------------------

### Logit function / Sigmoid Function:
		Y 		= Exp(a + bX)/{1 + Exp(a + bX)}   = [1/{1 + Exp -(a + bX)}] = 1/(1+exp^-y)
		1 - Y 		= Exp-(a + bX)/{1 + Exp-(a + bX)} =  1/{1 + Exp(a + bX)}    = 1/(1+exp^y)
		Y/(1-Y)		= Exp(a + bX)/{1 + Exp(a + bX)}]/ [1/{1 + Exp(a + bX)}]     = Exp(a + bX) = exp^y
		Log{Y/(1-Y)}	= a + bX (Apply log to convert non-linear relationship into linear relationship)
		  
	- The sigmoid function is a mathematical function used to map the predicted values to probabilities 
 	  which has a characteristic of S-shaped or sigmoid curve. 
    	- Logistic / logit function has the same property of a sigmoid function.	
	- The sigmoid function takes any real number as input and output probabilities (a value between 0 to 1), 
 	  which forms a S-shaped curve.
	- Due to Sigmoid function, Logistic Regression is not a Linear Regression model.
 
-----------------------------------------------------------------------------------------------------------------------

### Maximum Likelihood: 
	Finds parameter values that maximize the likelihood of making the observations given the parameters
		Pi = {Pr(Yi = 1/Xi) if Yi = 1}	= Pr^Yi         --> (P, Yi is a Bernoulli random variable)
		{1 - Pr(Yi = 1/Xi)  if Yi = 0}	= (1-Pr)^(1-Yi)	--> (1-P)
			
	Likelihood function/Joint probability density function: (Yi is success and failure)
		= Product[(Pr^Yi){(1-Pr)^(1-Yi)}]

	Maximum likelihood estimation (MLE):
		- Logistic Regression uses Maximum likelihood estimation finds parameter values that maximize 
  		  the likelihood of making the observations given the parameters. Linear regression uses 
      		  Ordinary Least Squares (OLS) which finds parameter values that minimizing the error.
		- MLE allows more flexibility because it has fewer restrictions.  
    
-----------------------------------------------------------------------------------------------------------------------

### Log Likelihood Function:
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

	Cost function :
		- Log Loss is cost function of logistic regression.
    		- It quantifies the error of a logistic regression by assessing how effectively it separates actual 
      		  from predicted. Error = Predicted – Actual.
    
-----------------------------------------------------------------------------------------------------------------------

### Gradient of Log Likelihood Function : 
	First Differentiation (with respect to beta) of Log Likelihood Function
		= [Yi*X] - [X*Exp(a + bX) / {1 + Exp(a + bX)}]
		= -[[Yi*X] - [X*Exp(a + bX) / {1 + Exp(a + bX)}]]	
		(Negative is part of negative log likelihood function. Refer to gradient_log_likelihood)  		
  
-----------------------------------------------------------------------------------------------------------------------

#### Hessian Matrix :
	Second Differentiation (with respect to beta) of Log Likelihood Function
	First Differentiation of Gradient of Log Likelihood Function
		= 0 - [{(X*Exp(a + bX)*X)/(1 + Exp(a + bX))} + {(X*Exp(a + bX))/((1+Exp(a + bX))^2)*(Exp(a + bX)*X)}]	
			(Differentiation of [Yi*X] will be 0 due to no beta.
		= [(X*X*Exp(a + bX))/((1+Exp(a + bX))^2)*{Exp(a + bX) - (1+Exp(a + bX))}]
		= -[(X*X*Exp(a + bX))/((1+Exp(a + bX))^2)]
		= (X*X*Exp(a + bX))/((1+Exp(a + bX))^2) 
			(minus will be cancel out due to minus sign in Gradient of Log Likelihood Function)

	Jacobian is similar to first order derivative.
 	Hessian is similar to second order derivative. 
	  
-----------------------------------------------------------------------------------------------------------------------

# Gradient Descent

 	Gradient Descent is an optimization algorithm which finds global or local minima of a 
  	differentiable function (error or cost function).

## Types of Gradient Descent:
	1. Batch Gradient Descent: It uses a complete dataset available to compute the gradient of the cost 
  	   function hence and it's very slow.
		- Cost function is calculated after the initialization of parameters.
		- It reads all the records into memory from the disk.
		- After calculating sigma for one iteration, we move one step further, and repeat the process.

	2. Mini-batch Gradient Descent: It is a widely used algorithm that makes faster and accurate results. 
  	   The dataset, here, is clustered into small groups of ‘n’ training datasets hence it's faster. 
       	   In every iteration, it uses a batch of ‘n’ training datasets to compute the gradient of the cost function. 
	   It reduces the variance of the parameter updates, which can lead to more stable convergence. 
	   It can also make use of a highly optimized matrix that makes computing of the gradient very efficient.

	3. Stochastic Gradient Descent: Stochastic gradient descent used for faster computation. First, it randomizes
  	   the complete dataset, and then uses only one training example in every iteration to calculate the gradient.
       	   Its benifical for huge datasets.

### Gradient Descent vs Newton's method
#### Gradient Descent:
	- Simple
	- Need learning Rate
	- 2nd derivate not required
	- More number of iterations
	- Each iteration is cheap (no 2nd derivative )
	- If number of observation is large then its cheaper

#### Newton's method:
	- Complex
	- No learning Rate required
	- 2nd derivate is required
	- Less number of iteration
	- Each iteration is expesive (2nd derivative )
	- If less number of observation (may be 1000) then its cheaper

### Solve a equation (identify beta):  

##### 1. Calculus: 
	It will faster if equation is simple. But in real life equations are very complex and messy and 
	its difficult to solve.   
					f(x) 	= X^2 - 2X + 2   
					df/dx 	= 2X - 2  
					2X	= 2  
					X	= 1   
	
##### 2. Gradient Descent:   
					Xi+1 = Xi - a f'(Xi)  	
					[Xi = initial guess, a = learning rate or step length or jump, Xi+1 = next guess]  
					f(x) 	= X^2 - 2X + 2  
					df/dx 	= 2X - 2  
					apply "Xi+1 = Xi - a f'(Xi)" on above equation. start with zero "0"  
					X1 = X0 - 0.2f'(3)	X0= 3 (initial guess), a=0.2 (guess)  
					X1 = 3  - 0.2(4)	[put 3 in "2X - 2": 2*3 - 2 = 6-2 =4]  
					X1 = 3 - 0.8  
					X1 = 2.2  
					X2 = X1 - 0.2f'(X1)  
					X2 = 2.2 - 0.2(2.4)	[put 2.2 in "2X - 2": 2*2.2 - 2 = 4.4-2 =2.4]  
					X2 = 1.72  
					X3 = X2 - 0.2f'(X2)  
					X3 = 1.72 - 0.2(1.44)	[put 1.72 in "2X - 2": 2*1.72 - 2 = 3.44-2 =1.44]  
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
			
##### 3. Newton Raphson:   
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

### Metrics:

	- beta(x): covariance(x,y) / variance(x)
		Standardized beta: Beta * [ standard deviation of a model variable / {Pi / SQRT(3)}] 	standard deviation of a model variable get it from proc means 'Std Dev'
			Pi / SQRT(3): standard deviation of standard logistic regression =  1.81379936423422
			Value of Pi: 3.14159265358979 	SQRT of 3: 1.73205080756888 
   
	- Standard Error of Beta: Square root of variance of beta coefficient. 
 		Square root of the diagonal elements of the covariance matrix. 		[Covariance Matrix = Inverse(Hessian matrix)]
                SQRT(Diagonal(Covariance Matrix))
 
	- Correlation(x,y): covariance(x,y) / [variance(x)*variance(y)]
 
	- AIC: (No of variable*2) - (2*-Log Likelihood)
 		"2*- Log Likelihood" is deviance of LR and its similar to residual sum of squares(RSS) of a linear regression. 
  		Ordinary least squares minimizes RSS and LR minimizes deviance.
 
	- BIC: {No of variable*log(No of obs)} - (2*-Log Likelihood)

	- Area under curve / C statistics: Percent Concordant + 0.5 * Percent Tied
		(The ROC curve is a graphical plot that illustrates the performance of any binary classifier system as its 
 		discrimination threshold is varied.) 
		True positive rate (Sensitivity : Y axis ) is plotted in function of the false positive rate (100-Specificity : X axis) 
 		for different cut-off points. Each point on the ROC curve represents a sensitivity/specificity pair corresponding to a 
  		particular decision threshold. 

	- Somer’s D (Gini): [2AUC-1] OR [(Concordant - Disconcordant) / Total  pairs]
 
 	- Divergence: [(meanG – meanB)^2] / [0.5(varG + varB)]
  			[meanG = mean of score only for good, varB= variance of score only for bad ]

	- TSS: SUM[y-mean(y)]^2
	- RSS: SUM[y-predicted(y)]^2
	- R Squared: 1.0 - (RSS/TSS)
	- VIF: 1.0 / (1.0 - R Squared)

-----------------------------------------------------------------------------------------------------------------------
    
