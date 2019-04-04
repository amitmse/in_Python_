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
		- #hidden neurons should be between the size of the input layer and the size of the output layer
		- #hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer
		- #hidden neurons should be less than twice the size of the input layer
							
	- A model with zero hidden layers will resolve linearly separable data. So unless you already know your data 
		isn't linearly separable, it doesn't hurt to verify this.
	- Assuming your data does require separation by a non-linear technique, then always start with one hidden layer



## Deep neural network

	-Deep learning works because of the architecture of the network AND the optimization routine applied 
		to that architecture.
	-Deep neural networks were a set of techniques that were discovered to overcome the vanishing gradient 
		problem which was severely limiting the depth of neural networks.
	-Neural networks are trained using backpropagation gradient descent. That is, you update the weights of 
		each layer as a function of the derivative of the previous layer. The problem is that the update 
		signal was lost as you increased the depth of the neural network. The math is pretty simple. 
		Therefore, in the old days, people pretty much only used neural networks with a single hidden-layer.
	-These new techniques include things like using RELU instead of sigmoids as activation functions. 
		RELU are of the form f(x)=max(0,x) and so they have non-vanishing derivative. But there are other
		techniques like using the sign of the derivative, rather than the magnitude, 
		in the backpropagation optimization problem.
	-The use of simple Rectified Linear Units (ReLU) instead of sigmoid and tanh functions is probably 
		the biggest building block in making training of DNNs possible. Note that both sigmoid and 
		tanh functions have almost zero gradient almost everywhere, depending on how fast they transit 
		from the low activation level to high; in the extreme case, when the transition is sudden, 
		we get a step function that has slope zero everywhere except at one point where the transition happens.
	-To expand on David Gasquez's answer, one of the main differences between deep neural networks and 
		traditional neural networks is that we don't just use backpropagation for deep neural nets. 
		Because backpropagation trains later layers more efficiently than it trains earlier layers--as you 
		go earlier and earlier in the network, the errors get smaller and more diffuse. So a ten-layer 
		network will basically be seven layers of random weights followed by three layers of fitted weights, 
		and do just as well as a three layer network.
		
	- Perceptron: 
		It's a Binary unit outputting Yes, No decisions with binary inputs. Perceptron function is a 
		step function. Transform the normal Perceptron into a sigmoid neuron using a sigmoid function 
		(y = 1/(1+e^-z). Sigmoid function is nothing but a smooth Step function and it gives us a range 
		of values between 0 and 1. Using this gradual output from the neuron, we can control our weight 
		learning and tweaking of the weights. This facilitates learning of weights in a neuron.
		
	-Neuron Saturation:
		If we use sigmoid function as the activation function, then the error term in the output layer 
		slows down as activation saturates. If you see as Z tends to grow higher 
		(refer to graph for logistic sigmod function) or smaller the function flattens either to 1 or 0. 
		This is called neuron saturation as it can't go beyond that. Which means the differential at 
		that time becomes very low or close to zero. This means if the output neuron saturates, 
		then the error for that neuron would be very low. Now error as defined above is just the change 
		of cost with respect to that neuron's weighted input (z), the cost would change slowly, 
		with respect to that weight and hence the gradient of that neuron is very low. This means the 
		weight update rule or weight learning will slow down. So once we calculate the error term of 
		the output layer , BP2 gives an equation of relating the error term in l+1th layer to 
		error terms in lth layer. Now we can calculate error terms of each neuron in each inside 
		layers as a function of error terms of the next layer neurons. But even the error term of 
		lth layer has the sigmoid function differential term in it. Which means when the activation or 
		input to the lth layer is saturated, then the weight incident on that neuron from previous 
		layer would learn slow too. Overall, weight learning will slow down if the input activation or 
		the output activation is saturated.
		
	-BackPropagation: 
		The errors are getting propagated in a backward fashion. We start by first finding the error terms 
		of the output layer using BP1. And then using BP2 we get error terms of the layer previous to that. 
		keep on using BP2 to get error terms of all neurons in all inside layer right till first input layer. 
		So this way we are propagating the error from the output layer all the way to the first layer. 
		Back Propagation is nothing but the chain rule of calculus. So the reason this error propagation 
		is happening backwards is because the initial cost function is used as a function of output
		activation and chain rule of calculus then causes it to behave that way.

	-Activation Function:
		Sigmoid is not only activation function right but we have other two most popular activation 
		functions which are also used are tanh and ReLu.
		TanH :  The advantage is that instead of having a range between 0 and 1, 
			tanh has a range between -1 to +1. 
			tanh(w*x + b) 
			tanh(z) = (e^z - e^-z)/(e^z + e^-z)						
		ReLU :	Rectified Linear Unit doesn't have the saturation problem where the output might get 
			constrained beyond a limit. max(0, w*x + b)

	- Problem in NN:
		Well due to Fully Connected Layer Architecture we experience two problems:
				-	Vanishing Gradient Problem 
				-	Exploding Gradient Problem 
		while training Deep Neural Networks which makes it hard to train deep neural networks.
		To prevent that we make use of a different architecture called as Convolutional Nets 
		which uses three main ideas:
				-	Local Receptive fields 
				-	Shared Weights 
				- 	Pooling
