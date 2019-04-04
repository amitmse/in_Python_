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
		may think to reach global minima leading to sub-optimal results. Use a momentum term in the objective 
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


## Multi Layer Perceptron (MLP): One or more non-linear hidden layers.
	Advantages:
		- Capability to learn non-linear models.
		- Capability to learn models in real-time (on-line learning) using partial_fit.
	Disadvantages:
		- MLP with hidden layers have a non-convex loss function where there exists more than 
		  one local minimum. Therefore different random weight initializations can lead to 
		  different validation accuracy.
		- MLP requires tuning a number of hyperparameters such as the number of hidden neurons, 
		  layers, and iterations.
		- MLP is sensitive to feature scaling.

	number of neurons in hidden layer (hidden_layer_sizes):
		- no of hidden layer is length of tuple. for below example hidden layers are  1,3,2:
			- len(7,) 	is 1 
			- len(10,10,10) is 3
			- len(5, 2) 	is 2
		- hidden unit is number inside tuple: 
			- hidden_layer_sizes = (7,) this refers to 1 hidden layer with 7 hidden units.
			- length = n_layers - 2 is because you have 1 input layer and 1 output layer.
			- 3 hidden layers with 10 hidden units each - (10,10,10)
			- (5, 2) i.e 1st hidden layer has 5 neurons. 2nd hidden layer has 2 neurons.
				
		Activation function for the hidden layer: (default is relu)
			identity: no-op activation, useful to implement linear bottleneck, returns f(x) = x
			logistic: logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
			tanh	: hyperbolic tan function, returns f(x) = tanh(x).
			relu	: rectified linear unit function, returns f(x) = max(0, x)

		The solver for weight optimization: (default is adam)
			lbfgs	: optimizer in the family of quasi-Newton methods.
			sgd	: stochastic gradient descent.
			adam	: stochastic gradient-based optimizer
			
		alpha 		: default 0.0001

		Learning rate: (default constant)
			constant   : constant learning rate given by ‘learning_rate_init’.
			invscaling : gradually decreases the learning rate learning_rate_ at each time step ‘t’ 
					using an inverse scaling exponent of ‘power_t’. 
					effective_learning_rate = learning_rate_init / pow(t, power_t)
			adaptive   : keeps the learning rate constant to ‘learning_rate_init’ as long as training 
					loss keeps decreasing. Each time two consecutive epochs fail to decrease 
					training loss by at least tol, or fail to increase validation score by at 
					least tol if ‘early_stopping’ is on, the current learning rate is divided by 5.
					Only used when solver='sgd'
								
			momentum 	: float, default 0.9
			

## Hidden layers
	- if data is linearly separable then don't need any hidden layers at all. Of course, you don't need an 
		NN to resolve your data either, but it will still do the job.
	- One issue within this subject on which there is a consensus is the performance difference from adding 
		additional hidden layers: the situations in which performance improves with a second (or third, etc.) 
		hidden layer are very small. One hidden layer is sufficient for the large majority of problems.
	- Get decent performance (even without a second optimization step) by setting the hidden layer configuration 
		using just two rules: 
			i. number of hidden layers equals one
			ii. number of neurons in that layer is the mean of the neurons in the input and output layers			
	- Pruning describes a set of techniques to trim network size (by nodes not layers) to improve computational 
		performance and sometimes resolution performance. Get a rough idea of which nodes are not important by 
		looking at your weight (weights very close to zero) matrix after training.
	- Using too many neurons in the hidden layers can result in several problems. 
		- First, too many neurons in the hidden layers may result in overfitting.
		- A second problem can occur even when the training data is sufficient. An inordinately large number of 
			neurons in the hidden layers can increase the time it takes to train the network. 
		- There are many rule-of-thumb methods for determining the correct number of neurons to use in the 
			hidden layers, such as the following:
		- number of hidden neurons should be between the size of the input layer and the size of the output layer
		- number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer
		- number of hidden neurons should be less than twice the size of the input layer
							
	- A model with zero hidden layers will resolve linearly separable data. So unless you already know your data 
		isn't linearly separable, it doesn't hurt to verify this.
	- Assuming your data does require separation by a non-linear technique, then always start with one hidden layer
