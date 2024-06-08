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
		

