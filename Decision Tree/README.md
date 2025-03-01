# Decision Tree Algorithms

-----------------------------------------------------------------------------------------------------

### structure of tree: 
	Root Node: The starting point of the tree.
	Branches: Paths connecting nodes based on decisions.
	Internal Nodes: Intermediate nodes that facilitate further splitting.
	Leaf Nodes: Final or terminal nodes of the tree.


![Function](https://github.com/amitmse/in_Python_/blob/master/Decision%20Tree/structure.PNG)

-----------------------------------------------------------------------------------------------------
## Decision Trees in Python:

https://github.com/amitmse/in_Python_/blob/master/Decision%20Tree/Decision_Trees.py

-----------------------------------------------------------------------------------------------------

## Limitations of Decision Trees:

	- Low-Performance
	- Poor Resolution on Data With Complex Relationships Among the Variables
	- Practically Limited to Classification
	- Poor Resolution With Continuous Expectation Variables
  
## Gini Index:

	Gini index says, if we select two items from a population at random then they must be of same class 
      		and probability for this is 1 if population is pure.

	- It works with categorical target variable “Success” or “Failure”.
	- It performs only Binary splits
	- Higher the value of Gini higher the homogeneity.
	- CART (Classification and Regression Tree) uses Gini method to create binary splits.
											
	Steps to Calculate Gini for a split: 
			
	1. Calculate Gini for sub-nodes, using formula sum of square of probability for success and failure (p^2+q^2).
	
	2. Calculate Gini for split using weighted Gini score of each node of that split
				
		Total Student = 30 and only 15 play cricket(50%)
			Split on Gender:- Female, Male
				1.Female = 10 (2  play cricket, Play= 2/10 = 0.2,   No Play = 8/10 =0.8 )
				2.Male	 = 20 (13 play cricket, Play= 13/20= 0.65,  No Play = 7/20 =0.35)
				3.Gini for sub-node Female = (0.2  )*(0.2 )+(0.8 )*(0.8 )=0.68 (p^2+q^2)
				4.Gini for sub-node Male = (0.65 )*(0.65)+(0.35)*(0.35)=0.55 (p^2+q^2)
				5.weighted Gini for Split Gender  = (10/30)*0.68+(20/30)*0.55   =0.59

			Similar for Split on Class:- XII, X
				1.XII = 14  (6  play cricket, Play= 6/14 = 0.43,   No Play = 8/14 =0.57 )
				2.X   = 16  (9  play cricket, Play= 9/16 = 0.56,   No Play = 7/16 =0.44 )
				3.Gini for sub-node Class IX = (0.43 )*(0.43)+(0.57 )*(0.57)=0.51 (p^2+q^2)
				4.Gini for sub-node Class X = (0.56 )*(0.56)+(0.44 )*(0.44)=0.51 (p^2+q^2)
				5.Calculate weighted Gini for Split Class = (14/30)* 0.51 +(16/30)* 0.51 =0.51
						
			Above, we can see that Gini score for Split on Gender is higher (0.59> 0.51) 
			than Class so node will split on Gender
			
	------------------------------------------------------------------------------------
	Above is Gini and below is for Gini Index (1-gini)
	https://github.com/amitmse/in_Python_/tree/master/Random_Forest
	
	Gini Index:
	    for each branch in split:
		Calculate percent branch represents #Used for weighting
		for each class in branch:
		    Calculate probability of class in the given branch.
		    Square the class probability.
		Sum the squared class probabilities.
		Subtract the sum from 1. #This is the Ginin Index for branch
	    Weight each branch based on the baseline probability.
	    Sum the weighted gini index for each split.

## Chi-Square

	It is an algorithm to find out the statistical significance between the differences between sub-nodes and 
	parent node. We measures it by sum of squares of standardized differences between observed and expected 
	frequencies of target variable.

	- It works with categorical target variable “Success” or “Failure”.
	- It can performs two or more splits
	- Higher the value of Chi-Square higher the statistical significance of differences 
		between sub-node and Parent node.
	- Chi-Square of each node is calculated using formula,
	- Chi-square = SQRT((Actual – Expected)^2 / Expected)
	- It generates tree called CHAID (Chi-square Automatic Interaction Detector)
			
	Steps to Calculate Chi-square for a split:
		- Calculate Chi-square for individual node by calculating the deviation for 
			Success and Failure both
		- Calculated Chi-square of Split using Sum of all Chi-square of success and Failure 
			of each node of the split					
			
	Example: Let’s work with above example that we have used to calculate Gini.		
		Split on Gender:
			1. First we are populating for node Female, Populate the actual value for “Play Cricket” 
				and “Not Play Cricket”, here these are 2 and 8 respectively.
			2. Calculate expected value for “Play Cricket” and “Not Play Cricket”, here it would be 5 
				for both because parent node has probability of 50% and we have applied same 
				probability on Female count(10).
			3. Calculate deviations by using formula, Actual – Expected. 
				It is for “Play Cricket” (2 – 5 = -3) and for “Not play cricket” ( 8 – 5 = 3).
			4. Calculate Chi-square of node for “Play Cricket” and “Not Play Cricket” using formula 
				with formula, = ((Actual – Expected)^2 / Expected)^1/2. 				
			5. Follow similar steps for calculating Chi-square value for Male node.			
			6. Now add all Chi-square values to calculate Chi-square for split Gender.
		
			       Play Cricket	 			PC
			       Play not Cricket 			NPC 		
			       Expected 	Play Cricket 		EPC	[Total*%oftarget ]
			       Expected 	Play not Cricket	ENPC	[Total*%oftarget ]
			       Deviatation 	Play Cricket		DPC	[PC - EPC]
			       Deviatation 	NOT Play Cricket	DNPC	[NPC - ENPC]
			       Chi-Square  	Play Cricket		CPC 	[(DPC^2)/EPC]^1/2
			       Chi-Square  	Not Play Cricket	CNPC	[(DNPC^2)/ENPC]^1/2


			TOTAL CHI-SQUARE for Gender = 1.34 + 1.34 + 0.95 + 0.95 = 4.58
			TOTAL CHI-SQUARE for Class  = 0.38 + 0.38 + 0.35 + 0.35 = 1.46

## Information Gain
	We can say that less impure node requires less information to describe it and more impure node 
	requires more information. Information theory has a measure to define this degree of disorganization 
	in a system, which is called Entropy. Lower Entropy is better. If the sample is completely homogeneous, 
	then the entropy is zero and if the sample is an equally divided it has entropy of one. 
	Entropy can be calculated using formula:   - P*Log2(P) - Q*Log2(Q)
		Here P and Q is probability of success and failure respectively in that node. 
		Entropy is also used with categorical target variable. 
		It chooses the split which has lowest entropy compared to parent node and other splits.
				
	Steps to calculate entropy for a split:
		- Calculate entropy of parent node
		- Calculate entropy of each individual node of split and calculate weighted average 
			of all sub-nodes available in split.	
				
	Example: Let’s use this method to identify best split for student example. 	
			Total=30, PC=15, NPC=15, P=PC/Total, Q=NPC/Total			
		- Entropy for parent node = -(15/30)log2(15/30) – (15/30)log2(15/30) 	= 1. 		
			Here 1 shows that it is a impure node. 
		- Entropy for Female node = -(2/10)log2(2/10) – (8/10)log2(8/10) 	= 0.72 and 
			  for male node   = -(13/20)log2(13/20) – (7/20)log2(7/20) 	= 0.93.
		- Entropy for split Gender = Weighted entropy of sub-nodes [10 Female, 20 Male] 
			 (10/30)*0.72 + (20/30)*0.93 					= 0.86 
		- Entropy for Class IX node = -(6/14) log2 (6/14) – (8/14) log2 (8/14) 	= 0.99 and 
			  for Class X node  = -(9/16) log2 (9/16) – (7/16) log2 (7/16) 	= 0.99
		- Entropy for split Class =  (14/30)*0.99 + (16/30)*0.99 		= 0.99
		- Above you can see that entropy of split on Gender is lower compare to 
			 Class so we will again go with split Gender. 
		- We can derive information gain from entropy as 1- Entropy.



	Entropy:
	    for each branch in split:
		Calculate percent branch represents #Used for weighting
		for each class in branch:
		    Calculate probability of class in the given branch.
		    Multiply probability times log(Probability,base=2)
		    Multiply that product by -1
		Sum the calculated probabilities.
	    Weight each branch based on the baseline probability.
	    Sum the weighted entropy for each split.


## Reduction in Variance
	Till now, we have discussed the algorithms for categorical target variable. Reduction in Variance 
	is an algorithm for continuous target variable. This algorithm uses the same formula of variance to 
	choose the right split that we went through the descriptive statistics. 
	The split with lower variance is selected as the criteria to split the population:
			
	Steps to calculate Variance:
		- Calculate variance for each node.
		- Calculate Variance for each split as weighted average of each node variance
			
	Example:- Let’s assign numerical value 1 for play cricket and 0 for not playing cricket. 
			Now follow the steps to identify the right split:
		- Variance for Root node, here mean value is (15*1 + 15*0)/30 = 0.5 and we have 15 one and 15 zero. 
			Now variance would be ((1-0.5)^2+(1-0.5)^2+….15 times+(0-0.5)^2+(0-0.5)^2+…15 times) / 30,
			this can be written as (15*(1-0.5)^2+15*(0-0.5)^2) / 30 = 0.25
		- Mean of Female node =(2*1+8*0)/10=0.2 and Variance = (2*(1-0.2)^2+8*(0-0.2)^2) / 10 = 0.16
		- Mean of Male Node =(13*1+7*0)/20=0.65 and Variance = (13*(1-0.65)^2+7*(0-0.65)^2) / 20 = 0.23
		- Variance for Split Gender = Weighted Variance of Sub-nodes = (10/30)*0.16 + (20/30) *0.23 = 0.21
		- Mean of Class IX node =(6*1+8*0)/14=0.43 and Variance = (6*(1-0.43)^2+8*(0-0.43)^2) / 14 = 0.24
		- Mean of Class X node =(9*1+7*0)/16=0.56 and Variance = (9*(1-0.56)^2+7*(0-0.56)^2) / 16 = 0.25
		- Variance for Split Gender =Weighted Variance of Sub-nodes = (14/30)*0.24 + (16/30) *0.25 = 0.25
		Above, you can see that Gender split has lower variance compare to parent node so 
		the split would be on Gender only.
		
## Splitting / Pruning
	Above, we have have looked at various algorithms to split a node into sub nodes. Now to create a decision tree, 
	sub-nodes are further split into two or more sub-nodes and all input variables are considered for creating 
	the split again. Fields already involved in split also get considered for split. It is a recursive process 
	and it stops if the node ends up as a pure node or it reaches the maximum depth of the tree or number of 
	records in the node reaches the preset limit.
	
	In a extreme scenario, a decision tree can have number of nodes equals to total number of observation, 
	but that would be a very complex tree. If we are expanding decision tree towards more complexity based 
	on training data set, then it causes over fitting and losses the predictive power of the model because 
	it is not generalized. Over fitting can be removed by pruning the nodes.

## Type I / II Error

![Function](https://github.com/amitmse/in_Python_/blob/master/Formula/Confusion%20Matrxi.jpg)

	True Positives 	(TP) = Correctly Identified
	True Negatives 	(TN) = Correctly Rejected
	False Positives	(FP) = Incorrectly Identified 	= Type I Error
	False Negatives	(FN) = Incorrectly Rejected	= Type II Error
	Recall (Sensitivity) = Ability of the classifier to find positive samples from all positive samples
	Precision 	     = Ability of the classifier not to label as positive a sample that is negative
				(positive predictive value)
	Specificity 	     = Measures the proportion of actual negatives that are correctly identified
				(true negative rate)


	True Positive Rate / Sensitivity / Recall : TP  / (TP + FN) = TP / Actual Positives
	True Negative Rate / Specificity : 	    TN  / (TN + FP) = TN / Actual Negatives
	False Positive Rate / Type I Error: 	    FP  / (FP + TN) = FP / Actual Negatives = 1 - Specificity
	False Negative Rate / Type II Error : 	    FN  / (FN + TP) = FN / Actual Positives = 1 - True Positive Rate
	Positive Predictive Value / Precision :     TP  / (TP + FP)
	Negative Predictive Value : 		    TN  / (TN + FN)
	False Discovery Rate: 			    FP  / (FP + TP) = 1 - Positive Predictive Value	
	Accuracy : 				   (TP + TN)/ (TP  + TN + FP + FN)
	F1-Score : 2*TP/ (2TP + FP + FN) 	=   [2 * (Precision * Recall) / (Precision + Recall)]
		F1 score (also F-score or F-measure) is a measure of a test's accuracy. 
		The F1-score gives you the harmonic mean of precision and recall.
		The scores corresponding to every class will tell you the accuracy of the classifier in 
		classifying the data points in that particular class compared to all other classes.
		The F1 score is the harmonic average of the precision and recall, where an F1 score reaches 
		its best value at 1 (perfect precision and recall) and worst at 0.
		It considers both the precision and the recall of the test to compute the score: 
		 	- precision is the number of correct positive results divided by the number of all positive 
				results returned by the classifier, 
			- recall is the number of correct positive results divided by the number of all relevant 
				samples (all samples that should have been identified as positive). 
				F1-Score : 2*TP	/ (2TP + FP + FN) = [2 * (Precision * Recall) / (Precision + Recall)]
				
	Feature importances: 
		It is also known as the Gini importance. The importance of a feature is computed as the (normalized) 
		total reduction of the criterion brought by that feature.  That reduction or weighted information gain
		is defined as. The weighted impurity decrease equation is the following: 
			N_t / N * (impurity - N_t_R / N_t * right_impurity - N_t_L / N_t * left_impurity)
				N 	: Total number of samples
				N_t 	: No. of samples at the current node
				N_t_L 	: No. of samples in the left child 
				N_t_R 	: No. of samples in the right child
				
