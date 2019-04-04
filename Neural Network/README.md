# Neural Networks

## Algorithm

	1. Feed-forward computation: Calculate hidden layer nodes & output layer.
		a. Calculate all hidden layer nodes 	: Multiply input layer and their weights (random for 1st time)
								And then apply sigmod function(Logistic function)
		b. Output layer node 			: Multiply hidden layer and their weights (random for 1st time)
								And then apply sigmod function(Logistic function)

	2. Back propagation to the output layer: calculate error of output layer & weights adjustment for hidden layer
		Error in output layer node 		: Substract actual output and predicted Output layer node
		Rate of change (weight for hidden) 	: Multiply Learning rate, Error in output layer node and 
								Hidden layer nodes
		Adjusted weights for hidden layer 	: Add previous weights for hidden layer, Rate of change 
			(weight for hidden) and Momentum term into previous delta change of the weight 
			(will be 0 for 1st time)

	3. Back propagation to the hidden layer	: calculate error of hidden layer & weights adjustment input layer
		Error in hidden layer node		: Multiply Error in output layer node and adjusted 
								weights for hidden layer
		Rate of change (weight for input)	: Multiply Learning rate, Error in hidden layer nodes and 
								input layer nodes
		Adjusted weights for input layer	: Add previous weights for input layer, 
			Rate of change (weight for input) and Momentum term into previous delta change of 
			the weight (will be 0 for 1st time)

	4. Weight updates: Use new weight to calculate hidden layer nodes, output layer & Error in output layer node
		Updated hidden layer nodes		: Multiply input layer and their updated weights
								And then apply sigmod function(Logistic function)
		Updated output layer node		: Multiply hidden layer and their updated weights
								And then apply sigmod function(Logistic function)
		Updated Error in output layer node	: Substract actual output and Output layer node
		Change in error				: Substract previous error and current error

## Choosing the correct learning rate and momentum will help in weight adjustment

	Learning rate /step size:
		Setting right learning rate could be difficult task. The learning rate is a parameter that determines 
		how much an updating step influences the current value of the weights.If learning rate is too small, 
		algorithm might take long time to converges. choosing large learning rate could have opposite effect 
		algorithm could diverge. Sometimes in NN every weight has it’s own learning rate. Learning rate of 0.35 
		proved to be popular choice when training NN. This paper will use rate of 0.45 but this value is used 
		because of simple architecture of NN used in example.
						
	Momentum term: 
		It represents inertia. Large values of momentum term will influence the adjustment in the current weight 
		to move in same direction as previous adjustment. Easily get stuck in a local minima and the algorithm 
		may think you reach the global minima leading to sub-optimal results. Use a momentum term in the objective 
		function that increases the size of the steps taken towards the minimum by trying to jump from a local 
		minima. If the momentum term is large then the learning rate should be kept smaller. A large value of
		momentum also means that the convergence will happen fast. But if both the momentum and learning rate 
		are kept at large values, then you might skip the minimum with a huge step. A small value of momentum 
		cannot reliably avoid local minima, and can also slow down the training of the system. Momentum also 
		helps in smoothing out the variations, if the gradient keeps changing direction. A right value of 
		momentum can be either learned by hit and trial or through cross-validation.
												
		Momentum simply adds a fraction of the previous weight update to the current one. When the gradient 
		keeps pointing in the same direction, this will increase the size of the steps taken towards the minimum.
		It's necessary to reduce the global learning rate when using a lot of momentum (m close to 1). If you 
		combine a high learning rate with a lot of momentum, you will rush past the minimum with huge steps!
