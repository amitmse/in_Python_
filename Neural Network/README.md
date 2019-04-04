# Neural Networks

Algorithm: 

	1. Feed-forward computation: Calculate hidden layer nodes & output layer.
		a. Calculate all hidden layer nodes 	: Multiply input layer and their weights (random for first time). 
								And then apply sigmod function(Logistic function).
		b. Output layer node 			: Multiply hidden layer and their weights (random for first time).
								And then apply sigmod function(Logistic function).

	2. Back propagation to the output layer: calculate error of output layer & weights adjustment for hidden layer.
		Error in output layer node 		: Substract actual output and predicted Output layer node.
		Rate of change (weight for hidden) 	: Multiply Learning rate, Error in output layer node and 
								Hidden layer nodes.
		Adjusted weights for hidden layer 	: Add previous weights for hidden layer, Rate of change 
			(weight for hidden) and Momentum term into previous delta change of the weight 
			(will be 0 for 1st time).

	3. Back propagation to the hidden layer	: calculate error of hidden layer & weights adjustment input layer.
		Error in hidden layer node		: Multiply Error in output layer node and adjusted 
								weights for hidden layer.
		Rate of change (weight for input)	: Multiply Learning rate, Error in hidden layer nodes and 
								input layer nodes.
		Adjusted weights for input layer	: Add previous weights for input layer, 
			Rate of change (weight for input) and Momentum term into previous delta change of 
			the weight (will be 0 for 1st time).

	4. Weight updates: Use new weight to calculate hidden layer nodes, output layer & Error in output layer node.
		Updated hidden layer nodes		: Multiply input layer and their updated weights. 
								And then apply sigmod function(Logistic function).
		Updated output layer node		: Multiply hidden layer and their updated weights. 
								And then apply sigmod function(Logistic function).
		Updated Error in output layer node	: Substract actual output and Output layer node.
		Change in error				: Substract previous error and current error.
