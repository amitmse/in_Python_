# Naïve Bayes

--------------------------------------------------------------------------------------------------
	It computes probability (conditional probability) based on prior probability / knowledge.
 
	- It's Naive as it assumes one feature doesn't affect another.
	- Bayes refers Bayes’ Theorem which is based on sequential events.
 
	P(A|B) = [P(B|A) * P(A)] / [P(B)]
 
	- P(A) and P(B): Prior probability refers to probability of a particular class occurring, 
 			without any condition.
	- P(B|A): Likelihood refers to the conditional probability.
	- P(A|B): Posterior probability which is combination of prior probability and likelihood.


--------------------------------------------------------------------------------------------------   

## Assumption

  1. Violation of Independence Assumption:
      
	- The Naïve Bayes assumption is that all the features are conditionally independent given the class label. 
	  	Even though this is usually false (since features are usually dependent).
              
	- Naive Bayesian classifiers assume that the effect of an attribute value on a given 
		class is independent of the values of the other attributes. This assumption is called class 
          	conditional independence. It is made to simplify the computations involved and, in this sense, 
          	is considered “naive.”

  2. Zero conditional probability Problem:
      
	- If a given class and feature value never occur together in the training set then the frequency-based 
          	probability estimate will be zero.
            
	- This is problematic since it will wipe out all information in the other probabilities 
	  	when they are multiplied. 
          
	- It is therefore often desirable to incorporate a small-sample correction in all probability estimates such 
          	that no probability is ever set to be exactly zero.

--------------------------------------------------------------------------------------------------

## Conclusions:
	- The naive Bayes model is tremendously appealing because of its simplicity, elegance, and robustness.
      
	- It is one of the oldest formal classification algorithms, and yet even in its simplest form 
          	it is often surprisingly effective.
      
	- It is widely used in areas such as text classification and spam filtering. 
      
	- A large number of modifications have been introduced, by the statistical, data mining, 
          	machine learning, and pattern recognition communities, in an attempt to make it more flexible.
        
	- but some one  has to recognize that such modifications are necessarily complications, 
          	which detract from its basic simplicity.

![Function](https://github.com/amitmse/in_Python_/blob/master/Naive%20Bayes/bayes%20theorem%20in%20one%20picture.png)

--------------------------------------------------------------------------------------------------

### Example: 
	Let's consider an example related to email spam detection:
	Prior Probability: This is the initial belief about the probability of an email being spam 
 		before analyzing its content. Suppose 30% of the emails are usually spam. 
   		Prior probability 𝑃(𝐴)=0.3

	Likelihood: This is the probability of observing a certain feature in the email (like a specific word) 
 		given that the email is spam. For instance, if the word "free" appears in spam emails 80% of the time.
		Likelihood 𝑃(𝐵∣𝐴)=0.8
		

	Posterior Probability: This is the updated probability that the email is spam after considering both 
 		the prior probability and the observed feature. If the email contains the word "free", 
   		the posterior probability would be higher than the prior probability, 
     		reflecting the increased likelihood of the email being spam given the presence of the word "free".
		Posterior Probability is higher than 30%, updated based on the presence of the word "free".
 		Prior probability 𝑃(B)=0.2  [probability of the word "free"]
		Posterior Probability P(A|B) = [P(B|A) * P(A)] / [P(B)] = (0.8 * 0.3) / 0.25 = 0.96
		This means that given the presence of the word "free," the updated belief that the email is spam is 96%.

--------------------------------------------------------------------------------------------------
