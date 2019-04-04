
## ADA Boost

	1. Assign same weight to all obs ( 1/No. of obs )
  
	2. Develop a tree and pick node (decision stump) with lowest weighted training error
  
	3. Compute coefficient		(1/2 ln[{1 - weighted error(ft)}/weighted error(ft)])
  
	4. Recompute weights		( ai e^(-Wt))
  
	5. Normalize weights
  
 ## GBM
 
	1. Y 	  = M(x) + error 	(Logistic Regression if dep is binay). get the weight as well
  
	2. Error  = G(x) + error2 
		Regress error with other ind var. Apply linear regression as error is continuous. get the weight as well
                  
	3. Error2 = H(x) + error3	(continue the #2 until you get low error)
  
	4. Y 	  = M(x) + G(x) + H(x) + error3	(combine all model together)
  
	5. Y 	  = alpha * M(x) + beta * G(x) + gamma * H(x) + error4 	(alpha, beta, gamma are weight of each model)
  
  
