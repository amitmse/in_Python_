# Decision Tree Algorithms

## Limitations of Decision Trees:

	-	Low-Performance
  
	-	Poor Resolution on Data With Complex Relationships Among the Variables
  
	-	Practically Limited to Classification							
  
	-	Poor Resolution With Continuous Expectation Variables
  
## Gini Index:

			Gini index says, if we select two items from a population at random then they must be of same class 
      and probability for this is 1 if population is pure.

			-	It works with categorical target variable “Success” or “Failure”.
			-	It performs only Binary splits
			-	Higher the value of Gini higher the homogeneity.
			-	CART (Classification and Regression Tree) uses Gini method to create binary splits.
											
			Steps to Calculate Gini for a split
			
				1. Calculate Gini for sub-nodes, using formula sum of square of probability for success and failure (p^2+q^2).
				2. Calculate Gini for split using weighted Gini score of each node of that split
				
				Total Student = 30 and only 15 play cricket(50%)
					Split on Gender:- Female, Male
						1.Female = 10 (2  play cricket, Play= 2/10 = 0.2,   No Play = 8/10 =0.8 )
						2.Male	 = 20 (13 play cricket, Play= 13/20= 0.65,  No Play = 7/20 =0.35)
						3.Calculate, 	Gini for sub-node Female 	= (0.2  )*(0.2 )+(0.8 )*(0.8 )=0.68				(p^2+q^2)
						4.Calculate, 	Gini for sub-node Male 		= (0.65 )*(0.65)+(0.35)*(0.35)=0.55				(p^2+q^2)
						5.Calculate weighted Gini for Split Gender  = (10/30)*0.68+(20/30)*0.55   =0.59

					Similar for Split on Class:- XII, X
						1.XII = 14  (6  play cricket, Play= 6/14 = 0.43,   No Play = 8/14 =0.57 )
						2.X   = 16  (9  play cricket, Play= 9/16 = 0.56,   No Play = 7/16 =0.44 )
						3.Gini for sub-node Class IX 				= (0.43 )*(0.43)+(0.57 )*(0.57)=0.51			(p^2+q^2)
						4.Gini for sub-node Class X 				= (0.56 )*(0.56)+(0.44 )*(0.44)=0.51			(p^2+q^2)
						5.Calculate weighted Gini for Split Class 	= (14/30)* 0.51 +(16/30)* 0.51 =0.51
						
					Above, we can see that Gini score for Split on Gender is higher (0.59> 0.51) than Class so node will split on Gender.	  
