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
Finds parameter values that minimizing the error. 
## Assumptions of Linear regression:
1. Linear in parameters 
	Issue	: Incorrect and unreliable model which leads to error in result.
	Solution: Transformations of independent variables

2. Mean of residuals is zero 
	Issue	: Error terms has zero mean and doesn’t depend on the independent variables. 
	       	  Thus, there must be no relationship between the independent variable and the error term

3. Homoscedasticity of residuals /equal variance of residuals
	Example	: Family income to predict luxury spending. Residuals are very small for low values of 
  		  family income (less spend on luxury) while there is great variation in the size of 
  		  the residuals for wealthier families. Standard errors are biased and it leads to 
  		  incorrect conclusions about the significance of the regression coefficients
	Test	: Breush-Pagan test
	Solution: Weighted least squares regression.
  		  Transform the dependent variable using one of the variance stabilizing transformations

4. No autocorrelation of residuals
	Issue	: correlation with own lag (stock price today linked with yesterday's price). if above fails 
		  then OLS estimators are no longer the Best Linear Unbiased Estimators. While it does not 
		  bias the OLS coefficient estimates, the standard errors tend to be underestimated 
		  (t-scores overestimated) when the autocorrelations of the errors at low lags are positive.
	Test 	:  Durbin–Watson

5. Normality of residuals
	Issue	: OLS estimators won’t have the desirable Best Linear Unbiased Estimate (BLUE) property

6. X variables and residuals are uncorrelated 

7. No perfect multicollinearity
	Issue	: Issue: Regression coefficient variance will increase
			Test	: VIF

8. Number of observations must be greater than number of Xs
		
#### Linear model should have residuals mean zero, have a constant variance, and not correlated with themselves 
#### or other variables. If these assumptions hold true, the OLS procedure creates the best possible estimates.

## Maximum likelihood estimation (MLE):
- Finds parameter values that maximize the likelihood of making the observations given the parameters
- MLE allows more flexibility in the data and analysis because it has fewer restrictions
	
# Logistic regression assumptions:
1. Dependent variable should be binary
2. Linearity between independent & log odds 
   (non-linear relationship between the dependent and independent variables)
3. Independence of errors
4. No perfect multicollinearity


# Model Metrics:
- beta(x) 		= covariance(x,y) / variance(x)
- correlation(x,y)	= covariance(x,y) / [variance(x)*variance(y)]
- TSS 			= SUM[y-mean(y)]^2
- RSS 			= SUM[y-predicted(y)]^2
- R Squared		= 1.0 - (RSS/TSS)
- AIC			= (No of variable*2)               - (2*-Log Likelihood)
- BIC			= {No of variable*log(No of obs)}  - (2*-Log Likelihood)
- VIF 			= 1.0 / (1.0 - R Squared)
- Gini/Somer’s D 	= [2AUC-1] OR [(Concordant - Disconcordant) / Total  pairs]
- Divergence 		= [(meanG – meanB)^2] / [0.5(varG + varB)]	

	     		  [meanG = mean of score only for good, varB= variance of score only for bad ]
- Area under curve 	= Percent Concordant + 0.5 * Percent Tied 	(Also refered as C statistics)


# Types of Gradient Descent:
1. Batch Gradient Descent: It uses a complete dataset available to compute the gradient of the cost function hence and it's very slow.
	- Cost function is calculated after the initialization of parameters.
	- It reads all the records into memory from the disk.
	- After calculating sigma for one iteration, we move one step further, and repeat the process.

2. Mini-batch Gradient Descent: It is a widely used algorithm that makes faster and accurate results. The dataset, here, is clustered into small groups of ‘n’ training datasets hence it's faster. In every iteration, it uses a batch of ‘n’ training datasets to compute the gradient of the cost function. It reduces the variance of the parameter updates, which can lead to more stable convergence. It can also make use of a highly optimized matrix that makes computing of the gradient very efficient.

3. Stochastic Gradient Descent: Stochastic gradient descent used for faster computation. First, it randomizes the complete dataset, and then uses only one training example in every iteration to calculate the gradient. Its benifical for huge datasets.

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


-----------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------
https://machinelearningmastery.com/how-to-code-the-students-t-test-from-scratch-in-python/
