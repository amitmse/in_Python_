# Naïve Bayes

	It computes probability based on prior knowledge.

 
	It's Naive as it assumes one feature's presence doesn't affect another's
	Bayes refers Bayes’ Theorem which is based on sequential events.


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
