# Type of data:
## Nominal / categorical/qualitative /non-parametric:  
		Example	: 	colour,gender. 
		check	: 	frequency each category. 
		Test	: 	Comparison/Difference: 
					- Test for proportion (for one categorical variable)
					- Difference of two proportions
					- Chi-Square test for independence (for two categorical variables)
		Relationship:	Chi-Square test for independence
	
	
## Ordinal : Similar as Nominal
		Example	:	rank, satisfaction.  
		Check	: 	frequency or mean (special case). 
		Test	:	Similar as Nominal
		
		
## Interval / Ratio / quantitative/continuous  : 
		Example	:	number of customers, income,age. 
		Check	: 	Mean ..
		Test	:	Comparison/Difference:
					- Test for a mean / T test (for one continuous variable)
					- Difference of two means (independent samples)
					- Difference of two means(paired T test. Pre and Post scenario)
		Relationship:	Regression Analysis / Correlation (for two continuous variables/for relationship)
			
			
#### One categorical and one continuous: T test (Anova when more than 2 category)
	https://www.youtube.com/watch?v=tfiDu--7Gmg
	
	
---------------------------------------------------------------------------------------------

# Probability Distribution:
- Probability distributions describe what we think the probability of each outcome is.
- They come in many shapes, but in only one size: probabilities in a distribution always add up to 1.
- A probability distribution is a function that assigns to each event a number in [0,1] which is 
  the probability that this event occurs.
- A statistical model is a set of probability distributions. We assume that the observations are generated 
  from one of these distributions.
- Chart: Horizontal axis set of possible numeric outcomes. Vertical axis probability of outcomes.
- Example:
	- Flipping a fair coin has two outcomes: it lands heads or tails. 
	- Before the flip, we believe there’s a 0.5 probability, of heads and same for tails. 
	- That’s a probability distribution over the two outcomes of the flip (Bernoulli distribution).
	
![Function](https://github.com/amitmse/in_Python_/blob/master/Others/distribution.png)


## Uniform distribution:
- Many equally-likely outcomes (Bernoulli):the uniform distribution, characterized by its flat PDF. 
- It can be defined for any number of outcomes or even as a continuous distribution.
- Function
	- PDF		: 1/(b-a) 	       {-infinite <- (a,b) -> infinite}
	- Mean  	: (a+b)/2
	- Variance 	: (b-a)^2/12
Example: 
	- Imagine rolling a fair die. The outcomes 1 to 6 are equally likely.


## Bernoulli distribution:
- Bernoulli distribution has only two possible outcomes i.e. success and failure in a single trial
- The Bernoulli PDF has two lines of equal height, representing the two equally-probable outcomes of 0 and 1 at either end.
- Bernoulli Distribution is a special case of Binomial Distribution with a single trial
- Function
	- PDF		: P^x*(1-P)^(1-x)       {x in 0 or 1}
	- Mean  	: P
	- Variance 	: P(1-P)
- Example: 
	- Flipping a fair coin
	- it’s going to rain tomorrow or not


## Binomial distribution:
- The binomial distribution may be thought of as the sum of outcomes of things that follow a Bernoulli distribution.
- Function
	- PDF		: [n!/(n-x)!*x!] * [P^x*(Q)^(n-x)]	{! factorial}
	- Mean  	: nP
	- Variance 	: nPQ
- Example: 
	- Toss a fair coin 20 times; how many times does it come up heads? This count is an outcome that follows 
	  the binomial distribution. Each flip is a Bernoulli-distributed outcome. Converted to binomial 
	  distribution when counting the number of successes, where each flip is independent and has 
	  the same probability of success.
	
	- Imagine an urn with equal numbers of white and black balls. Draw a ball and note whether it is black, 
	  then put it back and Repeat this process. How many times black ball was drawn? 
	  This count also follows a binomial distribution.
	
	
## Hyper-Geometric distribution:
- Example: 
	- This is the distribution of that same count if the balls were drawn without replacement instead. 
	  Undeniably it’s a cousin to the binomial distribution, but not the same, because the probability 
	  of success changes as balls are removed. 
	- If the number of balls is large relative to the number of draws, the distributions are similar
	  because the chance of success changes less with each draw.
	
	
## Poisson distribution:
- Simialr to the binomial distribution, the Poisson distribution is the distribution of a 
  count - the count of times something happened. 
- The Poisson distribution is when trying to count events over a time given the continuous rate of events occurring
- Poisson Distribution is a limiting case of binomial distribution.
- Function								
	- PDF		: Expo(-Mean)*{(Mean^x)/x!} 		{x in o,1,2,3,4,5}
	- Mean  	: Mean
	- Variance 	: Mean
- Example:
	- Packets arrive at routers, or customers arrive at a store, or things wait in some kind of queue Count 
	  of customers calling a support hot-line each minute doesn't follow binomial/Bernoulli but Poisson.
	- The number of emergency calls recorded at a hospital in a day.
	- The number of thefts reported in an area on a day.
	- The number of customers arriving at a salon in an hour.
	- The number of suicides reported in a particular city.
	- The number of printing errors at each page of the book.
	- The number of incoming calls at a call center in a day.
	
	
## Geometric distribution:
- If the binomial distribution is “How many successes?” then the geometric distribution is
  “How many failures until a success?”
- Example:
	- From simple Bernoulli trials arises another distribution. How many times does a flipped coin 
	  come up tails before it first comes up heads? This count of tails follows a geometric distribution.


## Negative Binomial distribution:
- It's a simple generalization. It’s the number of failures until r successes have occurred,not just 1.
- Example: 
	
	
## Exponential distribution:
- The exponential distribution is one of the widely used continuous distributions. 
- It is often used to model the time elapsed between events.
- The exponential distribution should come to mind when thinking of "time until event", maybe "time until failure".
- Exponential distribution is widely used for survival analysis. From the expected life of a machine to 
  the expected life of a human, exponential distribution successfully delivers the result.
- There is a strong relationship between the Poisson distribution and the Exponential distribution.
- Function								
	- PDF		: λe^(-λx)    		{x ≥ 0}
	- Mean  	: 1/λ
	- Variance 	: (1/λ)²
- Example: 
	- let’s say a Poisson distribution models the number of births in a given time period. 
	  The time in between each birth can be modeled with an exponential distribution.
	- Duration of a telephone call
	- How long does it take to perform a service, fix something at a service point etc.
	- Duration between two phone calls
	- Half life of atoms (radioactive decay)
	- Expected lifetime of electronic (or other) parts, 
	  if wearing is not considered (this is called Mean Time Between Failures, MTBF)
	- Age of plants or animals
	- Very simple model used by insurance companies


## Weibull:
- Weibull distribution can model increasing (or decreasing) rates of failure over time. 
- The exponential is merely a special case.
- Commonly used to assess product reliability, analyze life data and model failure times
- Weibull isn’t an appropriate model for every situation i.e. chemical reactions and corrosion failures are 
  usually modeled with the lognormal distribution.


## Normal Distribution:
- The sum of Bernoulli trials follows a binomial distribution, and as the number of trials increases, 
  that binomial distribution becomes more like the normal distribution. 
- Its cousin the hyper-geometric distribution does too. 
- The Poisson distribution—an extreme form of binomial—also approaches the normal distribution as 
  the rate parameter increases.
- The mean, median and mode of the distribution coincide.
- The curve of the distribution is bell-shaped and symmetrical about the line x=μ.
- Normal distribution is another limiting form of binomial distribution.
- Its popular due to Central Limit Theorem.
- Function								
	- PDF		: [1/{SQRT(2Pai)*STD}]*Expo[(X-Mean)^2/-2Variance]   {-infinite <-x-> infinite}
	- Mean  	: Mean
	- Variance 	: Variance
- Example:
	- Heights of people, Measurement errors, Blood pressure, Points on a test, IQ scores, Salaries.


### Standard Normal distribution: 
- It is also known as the Z distribution and it follows normal distribution 
  with a mean of zero and a variance of one.
- Example: 
	- For example, if you get a score of 90 in Math and 95 in English, you might think that you are 
  	  better in English than in Math. However, in Math, your score is 2 standard deviations above 
	  the mean. In English, it’s only one standard deviation above the mean. It tells you that in Math, 
  	  your score is far higher than most of the students (your score falls into the tail).


### Z-test:
- The sample is assumed to be normally distributed. A z-score is calculated with population parameters 
  such as "population mean" and "population standard deviation" and is used to validate a hypothesis 
  that the sample drawn belongs to the same population. Sample mean is same as the population mean.
- Function								
	- PDF		: (x - mean)/standard deviation
	- Mean  	: 0
	- Variance 	: 1			  

## t /Student  Distribution:
- The t test tells how significant the differences between groups are. A t-test is used to compare the mean of 
  two given samples.
- A t-test is used when the population mean and population standard deviation are unknown.
- Independent samples t-test which compares mean for two groups. t test for equality of 
  population mean when variance is same.
- Before t test, F test is required for equality for variance.
- One sample t-test which tests the mean of a single group against a known mean.
- Test the significance of regression coefficient. 
- Example:
	- A very simple example: Let’s say you have a cold and you try a naturopathic remedy. 
	  Your cold lasts a couple of days. The next time you have a cold, you buy an over-the-counter 
	  pharmaceutical and the cold lasts a week. You survey your friends and they all tell you that 
	  their colds were of a shorter duration (an average of 3 days) when they took the homeopathic remedy. 
	  What you really want to know is, are these results repeatable? A t test can tell you by comparing
	  the means of the two groups and letting you know the probability of those results happening by chance.

	- Paired sample t-test which compares means from the same group at different times. 
	  Choose the paired t-test if you have two measurements on the same item, person or thing
		
		
## Chi-Squared Distribution :
- Tests for the strength of the association between two categorical variables. Chi Square lets you know whether 
  two groups have significantly different opinions, which makes it a very useful statistic for survey research.
- Population mean is known and test the variance of normal distributed. chi squared distribution is the square 
  of a normal distribution.
- Function
	- PDF		: (Observed - Mean)^2/Mean
	- Mean  	: mean
- Example:
	- The chi-squared distribution is used primarily in hypothesis testing
	- Goodness of fit test, which determines if a sample matches the population 
		(does a coin tossed 20 times turn up 10 heads and 10 tails?)
	- A chi-square fit test for two independent variables is used to compare two variables in a contingency table 
		to check if the data fits
	- Chi-squared test of independence in contingency tables (is there a relationship between gender and salary?)
	- Likelihood-ratio test for nested models
	- Log-rank test in survival analysis
	- Cochran–Mantel–Haenszel test for stratified contingency tables


## Likelihood-ratio :
- This test assesses the goodness of fit of two competing statistical models based on the ratio of their likelihoods


## F-test:
- F-test of equality of variances is a test for the null hypothesis that two normal populations have the same variance. 

- It is most often used when comparing statistical models that have been fitted to a data set, 
  in order to identify the model that best fits the population from which the data were sampled. 
  "F-tests" mainly arise when the models have been fitted to the data using least squares.


-----------------------------------------------------------------------------------------------------------------------------------

https://medium.com/@srowen/common-probability-distributions-347e6b945ce4

https://blog.cloudera.com/blog/2015/12/common-probability-distributions-the-data-scientists-crib-sheet/

https://www.johndcook.com/blog/distribution_chart/

https://www.statisticshowto.datasciencecentral.com/probability-distribution/

-----------------------------------------------------------------------------------------------------------------------------------
## Covariance:
- It refers to the measure of how two variables will change (directional relationship) when they are compared to each other
- It measures the Variance between two variables
- Covariance indicates the direction of the linear relationship between variables. Correlation on the other hand measures 
   both the strength and direction of the linear relationship between two variables. 

## Correlation vs Regression:
- Correlation 
	- It measures the degree of relationship between two variables. 
	- correlation doesn’t capture causality.
	- Correlation between x and y is the same as the one between y and x.
- Regression analysis 
	- It is about how one variable affects another or what changes it triggers in the other.
	- Regression is based on causality (cause and effect).
	- Regression of x and y, and y and x, yields completely different results.
	
## ANOVA:
- Known as analysis of variance, is used to compare multiple (three or more) samples with a single test 
	i.e. all sample means are equal
	

# Ordinary Least Squares (OLS): 
- Finds parameter values that minimizing the error. 
## Assumptions of Linear regression: 

### For Model:

	1. Linear in parameters : 
		Issue	: Beta not multiplied or divided by any other parameter. 
                          Incorrect and unreliable model which leads to error in result.
		Solution: Transformations of independent variables

### For Variable: 

	2. No perfect multicollinearity :
		Issue: Issue: Regression coefficient variance will increase
		Test: VIF
		
### For Error Tearm: 

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
	
	8. Number of observations must be greater than number of Xs

#### Linear model should have residuals mean zero, have a constant variance, and not correlated with themselves or other variables. If these assumptions hold true, the OLS procedure creates the best possible estimates.

	
# Logistic regression 
- It uses MLE rather than OLS, it avoids many of the typical assumptions (listed below) tested in statistical analysis.
### Assumptions:
	1. Dependent variable should be binary
	2. Linearity between independent & log odds. (non-linear relationship between the dependent and independent variables)
	3. Independence of errors
	4. No perfect multicollinearity
### Does not assume: 
	- normality of variables (both DV and IVs)
	- linearity between DV and IVs
	- homoscedasticity
	- normal errors
	
### Maximum likelihood estimation (MLE):
- Finds parameter values that maximize the likelihood of making the observations given the parameters
- MLE allows more flexibility in the data and analysis because it has fewer restrictions
	
# Model Metrics:
- beta(x) 	= covariance(x,y) / variance(x)
- correlation(x,y)= covariance(x,y) / [variance(x)*variance(y)]
- TSS 		= SUM[y-mean(y)]^2
- RSS 		= SUM[y-predicted(y)]^2
- R Squared	= 1.0 - (RSS/TSS)
- AIC		= (No of variable*2)               - (2*-Log Likelihood)
- BIC		= {No of variable*log(No of obs)}  - (2*-Log Likelihood)
- VIF 		= 1.0 / (1.0 - R Squared)
- Gini/Somer’s D = [2AUC-1] OR [(Concordant - Disconcordant) / Total  pairs]
- Divergence 	= [(meanG – meanB)^2] / [0.5(varG + varB)]       

			[meanG = mean of score only for good, varB= variance of score only for bad ]
			

- True Positives (TP) = Correctly Identified
- True Negatives (TN) = Correctly Rejected
- False Positives (FP) = Incorrectly Identified = Type I Error
- False Negatives (FN) = Incorrectly Rejected	= Type II Error
- Recall (Sensitivity) = Ability of the classifier to find positive samples from all positive samples
- Precision = Ability of the classifier not to label as positive a sample that is negative (positive predictive value)
- Specificity = Measures the proportion of actual negatives that are correctly identified (true negative rate)

![Function](https://github.com/amitmse/in_Python_/blob/master/Formula/Confusion%20Matrxi.jpg)

- True Positive Rate / Sensitivity / Recall : TP  / (TP + FN) = TP / Actual Positives
- True Negative Rate / Specificity : 	    TN  / (TN + FP) = TN / Actual Negatives
- False Positive Rate / Type I Error: 	    FP  / (FP + TN) = FP / Actual Negatives = 1 - Specificity
- False Negative Rate / Type II Error : 	    FN  / (FN + TP) = FN / Actual Positives = 1 - True Positive Rate
- Positive Predictive Value / Precision :     TP  / (TP + FP)
- Negative Predictive Value : 		    TN  / (TN + FN)
- False Discovery Rate: 			    FP  / (FP + TP) = 1 - Positive Predictive Value	
- Accuracy : 				   (TP + TN)/ (TP  + TN + FP + FN)
- F1-Score : 2*TP/ (2TP + FP + FN)   =   [2 * (Precision * Recall) / (Precision + Recall)]
	- F1 score (also F-score or F-measure) is a measure of a test's accuracy. 
	- The F1-score gives you the harmonic mean of precision and recall.
	- The scores corresponding to every class will tell you the accuracy of the classifier in classifying the data points in that particular class compared to all other classes.
	- The F1 score is the harmonic average of the precision and recall, where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0.
	- It considers both the precision and the recall of the test to compute the score: 
		- precision is the number of correct positive results divided by the number of all positive results returned by the classifier, 
		- recall is the number of correct positive results divided by the number of all relevant samples (all samples that should have been identified as positive).
	 
			F1-Score : 2*TP	/ (2TP + FP + FN) = [2 * (Precision * Recall) / (Precision + Recall)]
	
- Area under curve /C statistics = Percent Concordant + 0.5 * Percent Tied		
	The ROC curve is a graphical plot that illustrates the performance of any binary classifier 
	system as its discrimination threshold is varied. True positive rate (Sensitivity : Y axis ) 
	is plotted in function of the false positive rate (100-Specificity : X axis) for different 
	cut-off points. Each point on the ROC curve represents a sensitivity/specificity pair 
	corresponding to a particular decision threshold.
- Standard Error Coef: 
	Linear regression standard error of Coef : SE  = sqrt [ S(yi - yi)2 / (n - 2) ] / sqrt [ S(xi - x)2 ]
	The standard error of the coefficient estimates the variability between coefficient estimates 
	that you would obtain if you took samples from the same population again and again. 
	The calculation assumes that the sample size and the coefficients to estimate would remain 
	the same if you sampled again and again. Use the standard error of the coefficient to measure 
	the precision of the estimate of the coefficient. 
	The smaller the standard error, the more precise the estimate.
- Recall and Precision:
	Recall is the fraction of instances that have been classified as true. On the contrary, 
	precision is a measure of weighing instances that are actually true. 
	While recall is an approximation, precision is a true value that represents factual knowledge.
- ROC curve:
	Receiver Operating Characteristic is a measurement of the True Positive Rate (TPR) against False 
	Positive Rate (FPR). We calculate True Positive (TP) as TPR = TP/ (TP + FN). On the contrary, 
	false positive rate is determined as FPR = FP/FP+TN where where TP = true positive, TN = true negative, 
	FP = false positive, FN = false negative.
- AUC vs ROC:
	AUC curve is a measurement of precision against the recall. Precision = TP/(TP + FP) and TP/(TP + FN).
	This is in contrast with ROC that measures and plots True Positive against False positive rate.

- Feature importances: 
	It is also known as the Gini importance. The importance of a feature is computed as the (normalized) total reduction of the criterion brought by that feature.  	That reduction or weighted information gain is defined as. The weighted impurity decrease equation is the following: 

	N_t / N * (impurity - N_t_R / N_t * right_impurity - N_t_L / N_t * left_impurity)
			N 	: Total number of samples
			N_t 	: No. of samples at the current node
			N_t_L 	: No. of samples in the left child 
			N_t_R 	: No. of samples in the right child

# Types of Gradient Descent:
1. Batch Gradient Descent: It uses a complete dataset available to compute the gradient of the cost function hence and it's very slow.
	- Cost function is calculated after the initialization of parameters.
	- It reads all the records into memory from the disk.
	- After calculating sigma for one iteration, we move one step further, and repeat the process.

2. Mini-batch Gradient Descent: It is a widely used algorithm that makes faster and accurate results. The dataset, here, is clustered into small groups of ‘n’ training datasets hence it's faster. In every iteration, it uses a batch of ‘n’ training datasets to compute the gradient of the cost function. It reduces the variance of the parameter updates, which can lead to more stable convergence. It can also make use of a highly optimized matrix that makes computing of the gradient very efficient.

3. Stochastic Gradient Descent: Stochastic gradient descent used for faster computation. First, it randomizes the complete dataset, and then uses only one training example in every iteration to calculate the gradient. Its benifical for huge datasets.

### Gradient Descent vs Newton's method
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





-----------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------
https://machinelearningmastery.com/how-to-code-the-students-t-test-from-scratch-in-python/
