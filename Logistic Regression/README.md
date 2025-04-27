# Logistic Regression

	- Logistic Regression is a classification technique which predicts a binary outcome.

https://www.linkedin.com/pulse/logistic-regression-algorithm-step-amit-kumar/

-----------------------------------------------------------------------------------------------------------------------
## Logistic Regression Assumptions:

1. Dependent variable should be binary
 
2. Linearity between independent & log odds.
   	- Log converts odds into linear form. log(p/q) = [log(p) - log(q)]
   	  
		(non-linear relationship between the dependent and independent variables)

	- Test: Box-Tidwell test	
  
3. Independence of errors
	- Each observation in the dataset is unrelated to any other observation, If observations are not independent, the error terms (residuals) will be correlated.
	- This can lead to biased coefficient estimates and inflated standard errors.
	- Example: Data collected from the same individuals over time. Observations collected at consecutive time points.
	- Test: Durbin-Watson test, Residual plots
  
4. No perfect multicollinearity

-------------------------------------------------------------

- Logistic regression relaxes several key assumptions required by linear regression (linearity between the dependent and independent variables, normality of errors, homoscedasticity)
- Logistic regression the target variable follows Bernoulli / binomial distribution, not normal distribution. The errors in logistic regression are not normally distributed, as the outcome is a probability (0 to 1).
- Logistic regression does not require homoscedasticity as the variance of the errors can vary depending on the predicted probability, as it's a binomial random variable.
       
-----------------------------------------------------------------------------------------------------------------------
## Logistic Regression Algorithm Coded in Python:

https://github.com/amitmse/in_Python_/blob/master/Logistic%20Regression/Logistic_Regression.py

-----------------------------------------------------------------------------------------------------------------------
  
#### Why use odds and log-odds
- Probability output ranges from 0 to 1  
- Odds Ratio = P/(1-P)		[Odds output range from 0 to ∞ ]
  
	odds = 0 when p = 0     [ 0 / (1-0) = 0] 
	odds = ∞ when p = 1	[ when denominator is very small number]

Odds of an event occurring in one group compared to another, provides a measure of the strength of association between the predictor and the outcome.
  
- Log of Odds: log (p/(1-P))  	[Log output ranges from −∞ to ∞]
  
	- Log of Odds is also called logit function.
	- Logit established a linear relationship between Predictors and Target.
	- The logit function takes a probability (0 to 1) and converts it back into a linear combination of predictors.
	- Converting a sigmoid function to logit for an easier interpretation of the output results in the logistic model equation.
	- Converting the probability to the logit (log odds), it transforms the nonlinear relationship into a linear one, making it easier to interpret. 
	- The coefficients in the logit model tell us how a one-unit change in a predictor affects the log odds (i.e., logit) of the outcome.
	- One unit increase in logit means exactly is still challenging. Thus, convert regression coefficients to something easier for interpretation, like odds ratios. This can be done easily by exponentiating the coefficient.
	- Log odds with a negative value indicating the odds of failure and a positive value showing higher chances of success.

- Sigmoid function: 1/(1+exp^-y)
  
	- The inverse of the logit function.
	- The sigmoid function maps arbitrary real values back to the range [0, 1].
	- Generalised form of logit function. For probability p, sigmoid(logit(p)) = p. 
		 
- Cost Function: [{Yi*Log(Pr)} + {(1-Yi)*Log(1-Pr)}]

- Logistic regression estimates an unknown probability for any given linear combination (log odds #2Assumptions) of the independent variables.
  
	- logit(p) => Log(Odds) => log[p/(1-P)] => [log(p) - log(1-P)] => logit(p)
	- log[p/(1-P)] = a + bX (Logistic Model)
	  Anti Log is exponential function which converts logit to sigmoid for probability.
	- inverse of logit(p)= 1/[1+exp^(a+bX)] (output is probability between 0 to 1)
  
----------------------------------------------------------------------------------------------------------------------- 
## Statistical Derivation of Logistic Regression:
### Model Equation:
		
		Y = a + bX (Y = dependent_variable, a=Intercept, b = coefficient, X = independent_variable)
	
-----------------------------------------------------------------------------------------------------------------------

### Logit function / Sigmoid Function:

https://github.com/amitmse/in_Python_/blob/master/Others/README.md#logistic-distribution

- Below shows probability to logit
  
		Y 		= Exp(a + bX)/{1 + Exp(a + bX)}   = [1/{1 + Exp -(a + bX)}] = 1/(1+exp^-y)
		1 - Y 		= Exp-(a + bX)/{1 + Exp-(a + bX)} =  1/{1 + Exp(a + bX)}    = 1/(1+exp^y)
		Y/(1-Y)		= Exp(a + bX)/{1 + Exp(a + bX)}]/ [1/{1 + Exp(a + bX)}]     = Exp(a + bX) = exp^y
		Log{Y/(1-Y)}	= a + bX (Apply log to convert non-linear relationship into linear relationship)
		If above not clear then read from bottom to top (above 4 lines) to understand logit to probability.
		  
- The sigmoid function is a mathematical function used to map the predicted values to probabilities which has a characteristic of S-shaped or sigmoid curve. 
- Logistic / logit function has the same property of a sigmoid function. 
- The sigmoid function takes any real number as input and output probabilities (a value between 0 to 1), which forms a S-shaped curve.
- Due to Sigmoid function, Logistic Regression is not a Linear Regression model (Sigmoid introduces non-linearity).

-----------------------------------------------------------------------------------------------------------------------

### Maximum Likelihood: 
- Finds parameter values that maximize the likelihood of making the observations given the parameters
  
		Pi = {Pr(Yi = 1/Xi) if Yi = 1}	= Pr^Yi         --> (P, Yi is a Bernoulli random variable)
		{1 - Pr(Yi = 1/Xi)  if Yi = 0}	= (1-Pr)^(1-Yi)	--> (1-P)
			
- Likelihood function/Joint probability density function: (Yi is success and failure)
  
		= Product[(Pr^Yi){(1-Pr)^(1-Yi)}]

Maximum likelihood estimation (MLE):
- Logistic Regression uses Maximum likelihood estimation finds parameter values that maximize the likelihood of making the observations given the parameters. Linear regression uses Ordinary Least Squares (OLS) which finds parameter values that minimizing the error.
- MLE allows more flexibility because it has fewer restrictions.  
    
-----------------------------------------------------------------------------------------------------------------------

### Log Likelihood Function:
- Apply Logs on likelihood equation and  product will become sum. Refer to property of LOG
  
		= Sum[{Yi*Log(Pr)} + {(1-Yi)*Log(1-Pr)}] (Apply log in above eq. and simplify it. cost function/log loss)
		= Sum[Yi*Log(Pr) - Yi*Log(1-Pr) + Log(1-Pr)]
		= Sum[Yi*Log{Pr/(1-Pr)}] + Sum[Log(1-Pr)] 
			[Substitute [Log{Pr/(1-Pr) = a + bX] and [1-Pr = 1 / {1 + Exp(a + bX)}]]
		= Sum[Yi*(a + bX)] + Sum[Log{1/1 + Exp(a + bX)}]
		= Sum[Yi*(a + bX)] + Sum[ Log(1) - Log{1 + Exp(a + bX)}] (Log(1) = 0)
		= Sum[Yi*(a + bX)] - Sum[Log{1 + Exp(a + bX)}]				
		= -[Sum[Yi*(a + bX)] - Sum[Log{1 + Exp(a + bX)}]] 
		(Apply negative to minimize the Log Likelihood Function)
	
- Cost function :
	- Log Loss is cost function of logistic regression.
	- It quantifies the error of a logistic regression by assessing how effectively it separates actual from predicted. Error = Predicted – Actual.

-----------------------------------------------------------------------------------------------------------------------

### Gradient of Log Likelihood Function : 
- First Differentiation (with respect to beta) of Log Likelihood Function
  
		= [Yi*X] - [X*Exp(a + bX) / {1 + Exp(a + bX)}]
		= -[[Yi*X] - [X*Exp(a + bX) / {1 + Exp(a + bX)}]]
  
- Negative is refers to negative log likelihood function. Refer to gradient_log_likelihood
  
-----------------------------------------------------------------------------------------------------------------------

#### Hessian Matrix :
- Second Differentiation (with respect to beta) of Log Likelihood Function. First Differentiation of Gradient of Log Likelihood Function
  
		= 0 - [{(X*Exp(a + bX)*X)/(1 + Exp(a + bX))} + {(X*Exp(a + bX))/((1+Exp(a + bX))^2)*(Exp(a + bX)*X)}]	
			(Differentiation of [Yi*X] will be 0 due to no beta.
		= [(X*X*Exp(a + bX))/((1+Exp(a + bX))^2)*{Exp(a + bX) - (1+Exp(a + bX))}]
		= -[(X*X*Exp(a + bX))/((1+Exp(a + bX))^2)]
		= (X*X*Exp(a + bX))/((1+Exp(a + bX))^2) 
			(minus will be cancel out due to minus sign in Gradient of Log Likelihood Function)

- Jacobian is similar to first order derivative.
- Hessian is similar to second order derivative. 
	  
-----------------------------------------------------------------------------------------------------------------------

# Gradient Descent

- Gradient Descent is an optimization algorithm which finds global or local minima of a cost/loss function. (Cost Function quantifies the error between predicted values and expected values). A gradient (slope) is nothing but a derivative (first-order) of cost function.  
- The gradient of the loss function is a vector that indicates the direction and magnitude of the steepest increase in the loss. It tells which way to change the model's parameters to increase the loss.
- Gradient Descent Process:
	- Initialization: Start with an initial set of model parameters (e.g., weights and biases).
	- Iteration: Calculate the gradient of the loss function with respect to the current parameters.
	- Learning Rate: A parameter that determines how large of a step to take during each iteration. A smaller learning rate may lead to slower convergence but avoid overshooting the minimum, while a larger learning rate can converge faster but potentially overshoot and oscillate around the minimum.
- By iteratively adjusting the model's parameters in the direction of the negative gradient, gradient descent aims to find the set of parameters that minimizes the loss function.
		New Value = Old Value — Step Size
![image](https://github.com/user-attachments/assets/197d05e3-6c47-4cd8-b064-9d34a1767d6a)
	- Derivative calculates slope and helps to find direction to reach minima.
	- Constant (Learning rate) also referred to as step size or the alpha, is the size of the steps that are taken to reach the minimum. Learning rate must be chosen wisely as:
		- if it is too small, then the model will take some time to learn.
		- if it is too large, model will converge as our pointer will shoot and we’ll not be able to get to minima.

![image](https://github.com/user-attachments/assets/ff5e905d-df2d-4e38-8d9e-503498cfdd62)

- Regression: Loss function is mean squared loss. only positive values (squared loss) are picked to obtain positive, and squaring is done to obtain the model’s real performance. When positive and negative numbers are added together, the result could be 0. This will inform the model that, although the net error is zero and it is operating well, it is still operating poorly. Larger errors are likewise given more weight when squaring. Squaring the error will penalize the model more and help it approach the minimal value faster when the cost function is far from its minimal value.

![image](https://github.com/user-attachments/assets/ea0c26a4-caa1-4173-8b66-e14ec8ddc15c)
  
- Mean of Absolute of error (MAE) is difference between the actual and the predicted prediction by the model. The absolute of residuals is done to convert negative values to positive values. Mean is taken to make the loss function independent of number of datapoints in the training set. MAE is generally less preferred over MSE as it is harder to calculate the derivative of the absolute function because absolute function is not differentiable at the minima.

![image](https://github.com/user-attachments/assets/8e1e15ef-30bf-4892-8030-09d08a5638b2)

- Classification: loss function is cross entropy loss. Cross-entropy, also known as logarithmic loss or log loss or Negative Log Likelihood.

![image](https://github.com/user-attachments/assets/ea2009ca-5cac-4334-af54-bfd6226fb7af)

---------------------------------------------------------------------------------------------------

### Challenges with Gradient Descent

- Local minima and saddle points: For convex problems, gradient descent can find the global minimum with ease, but as nonconvex problems emerge, gradient descent can struggle to find the global minimum, where the model achieves the best results.

![image](https://github.com/user-attachments/assets/ecba556f-d6e5-4dd2-a710-d0c3245d8c82)

- Vanishing and Exploding Gradients mostly happes in deeper neural networks, particular recurrent neural networks. Activation functions, like the logistic function (sigmoid), have a huge difference between the variance of their inputs and the outputs. In simpler words, they shrink and transform a larger input space into a smaller output space between the range of [0,1]. Cen be fixed with proper weight initialization, activation functions like ReLU, gradient clipping, and batch normalization. 
	- Vanishing gradients: This occurs when the gradient is too small. As move backwards during backpropagation, the gradient continues to become smaller, causing the earlier layers in the network to learn more slowly than later layers. When this happens, the weight parameters update until they become insignificant—i.e. 0—resulting in an algorithm that is no longer learning. The parameters of the higher layers change significantly whereas the parameters of lower layers would not change much (or not at all). The model weights may become 0 during training. The model learns very slowly and perhaps the training stagnates at a very early stage just after a few iterations. 

	- Exploding gradients: This happens when the gradient is too large, creating an unstable model. In this case, the model weights will grow too large, and they will eventually be represented as NaN. One solution to this issue is to leverage a dimensionality reduction technique, which can help to minimize complexity within the model. There is an exponential growth in the model parameters. The model weights may become NaN during training. The model experiences avalanche learning.
   
---------------------------------------------------------------------------------------------------

## Types of Gradient Descent:
1. Batch Gradient Descent: It uses a complete dataset available to compute the gradient of the cost function hence and it's very slow.
	- Cost function is calculated after the initialization of parameters.
	- It reads all the records into memory from the disk.
	- After calculating sigma for one iteration, we move one step further, and repeat the process.

2. Mini-batch Gradient Descent: It is a widely used algorithm that makes faster and accurate results. The dataset, here, is clustered into small groups of ‘n’ training datasets hence it's faster. In every iteration, it uses a batch of ‘n’ training datasets to compute the gradient of the cost function. It reduces the variance of the parameter updates, which can lead to more stable convergence. It can also make use of a highly optimized matrix that makes computing of the gradient very efficient.

3. Stochastic Gradient Descent: Stochastic gradient descent used for faster computation. First, it randomizes the complete dataset, and then uses only one training example in every iteration to calculate the gradient. Its benifical for huge datasets.

![image](https://github.com/user-attachments/assets/4c300ff5-7a12-4625-84b4-1c7bf4aafa7d)

---------------------------------------------------------------------------------------------------

### Gradient Descent vs Newton's method

---------------------------------------------------------------------------------------------------

#### Gradient Descent:
	- Simple
	- Need learning Rate
	- 2nd derivate not required
	- More number of iterations
	- Each iteration is cheap (no 2nd derivative )
	- If number of observation is large then its cheaper
 
---------------------------------------------------------------------------------------------------

#### Newton's method:
	- Complex
	- No learning Rate required
	- 2nd derivate is required
	- Less number of iteration
	- Each iteration is expesive (2nd derivative )
	- If less number of observation (may be 1000) then its cheaper

---------------------------------------------------------------------------------------------------

### Solve a equation (identify beta):  

---------------------------------------------------------------------------------------------------

##### 1. Calculus: 
- It will faster if equation is simple. But in real life equations are very complex and messy and its difficult to solve.
  
		f(x) 	= X^2 - 2X + 2   
		df/dx 	= 2X - 2  
		2X	= 2  
		X	= 1
  
---------------------------------------------------------------------------------------------------
	
##### 2. Gradient Descent:

		f(x) 	= X^2 - 2X + 2  
		df/dx 	= 2X - 2
		Gradient Descent: Xi+1 = Xi - a f'(Xi)
			(Xi+1 is next guess, Xi = initial guess, a = learning rate, f'(Xi) = df/dx)  
		apply "Xi+1 on above function. 
		X1 = X0 - 0.2f'(3)
		Initial guess X0=3, a=0.2
		X1 = 3  - 0.2*f'(Xi) 				[f'(Xi) = df/dx = 2X - 2 = 2*3 -2 = 4]
		X1 = 3  - 0.2(4) = 3 - 0.8 = 2.2
		X1 = 2.2
		X2 = X1 - 0.2f'(X1) = 2.2 - 0.2(2.4) 		[f'(Xi) = 2*2.2 - 2 = 4.4-2 =2.4 ]
		X2 = 1.72  
		X3 = X2 - 0.2f'(X2)  = 1.72 - 0.2(1.44)		[f'(Xi) 2*1.72 - 2 = 3.44-2 =1.44]  
		X3 = 1.72 - 0.288  
		X3 = 1.432  
			
- continue doing this untill we are close to 1 which is the exact solution. As we approach to local minimum, Gradient Descent will automatically take smaller steps. So no need to decrease "a" over time. 
- optimization gradient descent:
  
		cX + d 	        = Y 			[equation of line and solve this for c & d]  
		(cX + d) -  Y 	= 0 			("cX + d" is predected Y^, Y^-Y is error and it should be zero)   
		min by(a,b) = sum ([cX + d]-Yi)^2	[c = cofficient, d=intercept]  
		
- First make initial guess for c & d then do the derivative by c & d seperately to get the optimium value of c & d. Above process will apply on Gradient Descent "Xi+1 = Xi - a f'(Xi)". Gradient descent is based on 1st derivatives only and it use all data at one time. Gradient descent generally requires more iterations. If data size is big then it will take long time to compute.
			
- Stochastic Gradient descent: It takes portion of data at one time and do the computation and continue in same way. cofficients are not exactly equals to Gradient descent but its close. For BIG data its only option to apply Gradient descent in faster way.

---------------------------------------------------------------------------------------------------
   
##### 3. Newton Raphson:   
- Newton's method generally requires fewer iterations, but each iteration is slow as we need to compute 2nd dervatives too. There is no guarantee that the Hessian is nonsingular. Additionally, we must  	supply the second partial derivatives to the computer (and they can sometimes be very difficult to calculate). http://www.stat.missouri.edu/~spinkac/stat8320/Nonlinear.pdf
   
- The Newton-Raphson method requires the second derivative because it uses a quadratic approximation of the function, and the second derivative provides information about the curvature of the function, allowing the method to converge faster than methods that use only the first derivative.

- Second derivative tells about the concavity of the function's graph:
	- Positive second derivative means the graph is concave up (like a "U" shape), indicating a minimum.
	- A negative second derivative means the graph is concave down (like a hump), indicating a maximum

- The second derivative is used to determine whether a critical point of a function is a local minimum, maximum, or neither. A positive second derivative at a critical point indicates a local minimum, while a negative second derivative indicates a local maximum.
- Find Critical Points: Start by finding the points where the first derivative of the function is equal to zero. These points are potential candidates for local minima or maxima.
-  Second Derivative Test: second derivative of the function at each of the critical points
	- minimum: second derivative > 0 at a critical point
	- maximum: second derivative < 0 at a critical point
	- inconclusive: second derivative = 0 at a critical point

	f(x) = x² - 4x + 5
	first derivative: f'(x) = 2x - 4
	Set f'(x) = 0 and solve for x: 2x - 4 = 0 => x = 2. This is a critical point.
	second derivative: f''(x) = 2
    	Evaluate f''(x) at x = 2: f''(2) = 2. Since f''(2) > 0, the function has a local minimum at x = 2.

- Newton Raphson without second derivative
  
	Xn+1 	= Xn - f(X)/f'(X)
		(Xn is initail guess, f'(X) first derivative)
	f(X) 	= X^2 - 8  
	f'(X)	= 2X  
	X1	= 3 (initail guess)  
	X2	= X1 - f(X)/f'(X)  
		= 3  - [( 3^2 - 8 )/ 2*3 ]   = 3  - (1/6)  = 18-1/6  
	X2	= 17/6
	X3	= X2 - f(X2)/f'(X2)  = 17/6 - [(17/6)^2 - 8] / [2(17/6)] 
	X3	= 2.828
    
-----------------------------------------------------------------------------------------------------------------------

### Metrics:

https://github.com/amitmse/in_Python_/blob/master/Others/README.md#model-metrics

- beta(x): covariance(x,y) / variance(x)
  
	- Standardized beta: Beta * [ standard deviation of a model variable / {Pi / SQRT(3)}]
	- standard deviation of a model variable get it from proc means 'Std Dev'
	- Pi / SQRT(3): standard deviation of standard logistic regression =  1.81379936423422
	- Value of Pi: 3.14159265358979 	SQRT of 3: 1.73205080756888 
   
- Standard Error of Beta: Square root of variance of beta coefficient.
	- Square root of the diagonal elements of the covariance matrix. 	[Covariance Matrix = Inverse(Hessian matrix)]
	- SQRT(Diagonal(Covariance Matrix))
 
- Correlation(x,y): covariance(x,y) / [variance(x)*variance(y)]
 
- AIC: (No of variable*2) - (2*-Log Likelihood)
  
"2*- Log Likelihood" is deviance of LR and its similar to residual sum of squares(RSS) of a linear regression. Ordinary least squares minimizes RSS and LR minimizes deviance.
 
- BIC: {No of variable*log(No of obs)} - (2*-Log Likelihood)

- Area under curve / C statistics: Percent Concordant + 0.5 * Percent Tied
  
(The ROC curve is a graphical plot that illustrates the performance of any binary classifier system as its discrimination threshold is varied.) True positive rate (Sensitivity : Y axis ) is plotted in function of the false positive rate (100-Specificity : X axis) for different cut-off points. Each point on the ROC curve represents a sensitivity/specificity pair corresponding to a particular decision threshold. 

- Somer’s D (Gini): [2AUC-1] OR [(Concordant - Disconcordant) / Total  pairs]
 
- Divergence: [(meanG – meanB)^2] / [0.5(varG + varB)]
  
	[meanG = mean of score only for good, varB= variance of score only for bad ]

- TSS: SUM[y-mean(y)]^2
  
- RSS: SUM[y-predicted(y)]^2
  
- R Squared: 1.0 - (RSS/TSS)
  
- VIF: 1.0 / (1.0 - R Squared)

-----------------------------------------------------------------------------------------------------------------------
