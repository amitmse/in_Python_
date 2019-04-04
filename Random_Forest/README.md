# Random Forest

## Algorithm (for both classification and regression)

	1. Draw ntree bootstrap samples from the original data
    
	2. For each of the bootstrap samples, grow an unpruned classification or regression tree, with the following
		modification: at each node, rather than choosing the best split among all predictors, randomly sample
		mtry of the predictors and choose the best split from among those variables. (Bagging can be thought 
		of as the special case of random forests obtained when mtry = p, the number of predictors)
        
	3. Predict new data by aggregating the predictions of the ntree trees 
        	(i.e., majority votes for classification, average for regression).
		
	An estimate of the error rate can be obtained, based on the training data, by the following:
		1. At each bootstrap iteration, predict the data not in the bootstrap sample 
        		(what Breiman calls “out-of-bag”, or OOB, data) using the tree grown with the bootstrap sample.
        
		2. Aggregate the OOB predictions. (On the average, each data point would be out-of-bag around 36% of 
			the times, so aggregate these predictions.)  
			Calcuate the error rate, and call it the OOB estimate of error rate.
	
  Our experience has been that the OOB estimate of error rate is quite accurate, given that enough trees have 
  been grown (otherwise the OOB estimate can bias upward; see Bylander (2002))
  
http://www.bios.unc.edu/~dzeng/BIOS740/randomforest.pdf

Here is how such a system is trained; for some number of trees T:

	1. Sample N cases at random with replacement to create a subset of the data 
		The subset should be about 66% of the total set.
		
	2. At each node:
		- For some number m (see below), m predictor variables are selected at random from 
			all the predictor variables.
		
		- The predictor variable that provides the best split, according to some objective function, 
			is used to do a binary split on that node.
			
		- At the next node, choose another m variables at random from all predictor variables and do the same.
		
	3. Depending upon the value of m, there are three slightly different systems:
		- Random splitter selection: m =1
		- Breiman’s bagger: m = total number of predictor variables
		- Random forest: m << number of predictor variables. 
			Brieman suggests three possible values for m: 1/2(sqrt(vm)), sqrt(vm), and sqrt(2vm)
			
	4. Running a Random Forest. When a new input is entered into the system, 
		it is run down all of the trees. The result may either be an average or weighted average of 
		all of the terminal nodes that are reached, or, in the case of categorical variables, 
		a voting majority.
		
		Note that:
			- With a large number of predictors, the eligible predictor set will be quite 
				different from node to node.
			- The greater the inter-tree correlation, the greater the random forest error rate, 
				so one pressure on the model is to have the trees as uncorrelated as possible.
			- As m goes down, both inter-tree correlation and the strength of individual trees go down. 
				So some optimal value of m must be discovered
				
	5. To understand how we test the classifier, we must explain several concepts:
		- cross-validation 
		- thresholds 
		- mean precision
		- precision above chance
