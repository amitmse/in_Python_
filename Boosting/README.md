
## ADA Boost

	--------------------------------------------------------------------------------------------------
	1. Assign same weight to all obs ( 1/No. of obs )
  
	2. Develop a tree and pick node (decision stump) with lowest weighted training error
  
	3. Compute coefficient		(1/2 ln[{1 - weighted error(ft)}/weighted error(ft)])
  
	4. Recompute weights		( ai e^(-Wt))
  
	5. Normalize weights
	
  	-------------------------------------------------------------------------------------

	y^ = sign(sum(Wt ft(X))) 	(Wt=Coefficient)
	
	first compute Wt (Coefficient)
		Wt = 1/2 ln[{1 - weighted error(ft)}/weighted error(ft)] 
			(weighted error	= total weight of mistake/total weight of all data point)
			
	second compute ai(weight)
		for first iteration ai = 1/N (N=no of observation)
		ai = "ai e^(-Wt)" if ft(Xi)=Yi OR "ai e^(Wt)" if ft(Xi) not equal to Yi	(ai=Weight)
		
	Normalize weight ai
		ai = ai/(sum of all a's)
		
	Function	: (1/sum (Wi)) sum(Wi exp(-2Yi -1)f(Xi))
	
	Initial Value	: 1/2 log [(sum (Yi Wi e^-oi))/sum ((1-Yi)Wi e^oi)]
	
	Gradient	: Zi = -(2Yi - 1) exp(-(2Yi-1)fXi))
  
  	-------------------------------------------------------------------------------------
  
 ## GBM
 
 	--------------------------------------
	1. Learn a regression predictor
	2. compute the error of residual
	3. learn to predict the residual
	4. identify good weights
	---------------------------------------	
 
	1. Y 	  = M(x) + error (Logistic Regression if dep is binay). get the weight as well
  
	2. Error  = G(x) + error2 (Regress error with other ind var. Apply linear regression as error is continuous. 
				   get the weight as well
                  
	3. Error2 = H(x) + error3 (continue the #2 until you get low error)
  
	4. Y 	  = M(x) + G(x) + H(x) + error3	(combine all model together)
  
	5. Y 	  = alpha * M(x) + beta * G(x) + gamma * H(x) + error4 	(alpha, beta, gamma are weight of each model)
  
  
https://stats.stackexchange.com/questions/204154/classification-with-gradient-boosting-how-to-keep-the-prediction-in-0-1

http://blog.kaggle.com/2017/01/23/a-kaggle-master-explains-gradient-boosting/

http://mccormickml.com/2013/12/13/adaboost-tutorial/

