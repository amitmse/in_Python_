# Neural Networks

	It consists of the input value and output value. Each input value is associated with its weight, 
	which passes on to next level, each perceptron will have an activation function. The weights and 
	input value forms a single perception. We use activation function and based on that, the value goes 
	to next well. And the process continues till it reaches output y’.
	
	In simple term: 
 		It uses logistic regression (or any other) and repeating it more than one times.
		In logistic regression, there are only two layers i.e. input and output but in neural network, 
		there is at least one hidden layer between input and output layer.
 	
## Problem in NN:
	Well due to Fully Connected Layer Architecture we experience two problems:
			-	Vanishing Gradient Problem 
			-	Exploding Gradient Problem 
	while training Deep Neural Networks which makes it hard to train deep neural networks.
	To prevent that we make use of a different architecture called as Convolutional Nets 
	which uses three main ideas:
			-	Local Receptive fields 
			-	Shared Weights 
			- 	Pooling
 ---------------------------------------------------------------------------------------------------------
 
 ## Cost function :
 
 	Sometimes the algorithm we create might predict the value incorrectly, so we need cost function. 
	It tried to quantify the error factor of neural network. It calculates how well the neural network 
	is performing based on the actual vs predicted value. Error factor = Predicted – Actual.
	
---------------------------------------------------------------------------------------------------------

## Activation Function:
 
 	It’s just a thing function that you use to get the output of node. It is also known as Transfer Function. 
	It is used to determine the output of neural network like yes or no. 
	
	Some examples: SIGMOID, RELU, LEAKYRELU, TANH, SOFTMAX, CUBE, ELU, RRELU, HARDSIGMOID, HARDTANH, IDENTITY,  
			RATIONALTANH, SOFTPLUS, SOFTSIGN, 
	
	The Activation Functions can be basically divided into 2 types:
	
### Linear Activation Function: 
	
	Range is -infinity to infinity. Not possible to use backpropagation as the derivative of the function is a 
	constant. no matter how many layers in the neural network, the last layer will be a linear function of the 
	first layer so a linear activation function turns the neural network into just one layer. A neural network
	with a linear activation function is simply a linear regression model. 
	
### Non-linear Activation Functions: 
	
	It makes it easy for the model to generalize or adapt with variety of 
	data and to differentiate between the output. It allows backpropagation because they have a derivative function
	which is related to the inputs. It allows “stacking” of multiple layers of neurons to create a deep neural network.
	Multiple hidden layers of neurons are needed to learn complex data sets with high levels of accuracy.
	Below are non-linear Activation Functions:
	
#### Sigmoid or Logistic Activation Function: 
	
	Sigmoid Function curve looks like a S-shape. The logistic sigmoid function can cause a neural network to get stuck 
	at the training time. The softmax function is a more generalized logistic activation function which is used for 
	multiclass classification. The main reason why we use sigmoid function is because it exists between (0 to 1). 
	Vanishing gradient problem.

![Function](https://github.com/amitmse/in_Python_/blob/master/Neural%20Network/Sigmoid.PNG)


#### Tanh or hyperbolic tangent Activation Function: 
	
	tanh is also like logistic sigmoid but better. The range of the tanh function is from (-1 to 1). 
	tanh is also sigmoidal (s - shaped). Both tanh and logistic sigmoid activation functions are used in 
	feed-forward nets. Vanishing gradient problem.
	
![Function](https://github.com/amitmse/in_Python_/blob/master/Neural%20Network/Tanh.PNG)
	
#### ReLU (Rectified Linear Unit) Activation Function: 
	
	The ReLU is the most used activation function in the world right now.Since, it is used in almost all 
	the convolutional neural networks or deep learning. As you can see, the ReLU is half rectified. f(z) is 
	zero when z is less than zero and f(z) is equal to z when z is above or equal to zero. Range is 0 to infinity. 
	But the issue is that all the negative values become zero immediately which decreases the ability of 
	the model to fit or train from the data properly. 
	That means any negative input given to the ReLU activation function turns the value into zero immediately in the 
	graph, which in turns affects the resulting graph by not mapping the negative values appropriately. It rectifies 
	vanishing gradient problem. Almost all deep learning Models use ReLu nowadays. it should only be used within Hidden 
	layers of a Neural Network Model. Hence for output layers we should use a Softmax function for a Classification 
	problem to compute the probabilites for the classes, and for a regression problem it should simply use a linear 
	function. Another problem with ReLu is that some gradients can be fragile during training and can die. It can 
	cause a weight update which will makes it never activate on any data point again. The draw backs of ReLU is when
	the gradient hits zero for the negative values, it does not converge towards the minima which will result in a 
	dead neuron while back propagation. To fix this problem another modification was introduced called Leaky ReLu to
	fix the problem of dying neurons. It introduces a small slope to keep the updates alive. 
	ReLU overcomes the vanishing gradient problem in the multi layer neural network.
	Rectified Linear Unit doesn't have the saturation problem where the output might get constrained beyond a limit. 
	
![Function](https://github.com/amitmse/in_Python_/blob/master/Neural%20Network/ReLU.PNG)
	
#### Leaky ReLU: 
	Range of the Leaky ReLU is -infinity to infinity.

![Function](https://github.com/amitmse/in_Python_/blob/master/Neural%20Network/Leaky_ReLU.PNG)

#### Softmax: 
	Softmax is a very interesting activation function because it not only maps our output to a [0,1] range but also 
	maps each output in such a way that the total sum is 1. The output of Softmax is therefore a probability distribution.
	The softmax function is often used in the final layer of a neural network-based classifier. Such networks are 
	commonly trained under a log loss (or cross-entropy) regime, giving a non-linear variant of multinomial logistic 
	regression. Softmax is used for multi-classification in logistic regression model whereas Sigmoid is used for binary
	classification in logistic regression model, the sum of probabilities is One for Softmax.
	It handles classification problems. Softmax is used only for the output layer, for neural networks that need to 
	classify inputs into multiple categories. 
	
![Function](https://github.com/amitmse/in_Python_/blob/master/Neural%20Network/softmax.PNG)


---------------------------------------------------------------------------------------------------------

# Types of neural networks and their applications:

## 01. Perceptron 
	The simplest and oldest model of Neuron, as we know it. Takes some inputs, sums them up, 
	applies activation function and passes them to output layer.

	It's a Binary unit outputting Yes, No decisions with binary inputs. Perceptron function is a 
	step function. Transform the normal Perceptron into a sigmoid neuron using a sigmoid function 
	(y = 1/(1+e^-z). Sigmoid function is nothing but a smooth Step function and it gives us a range 
	of values between 0 and 1. Using this gradual output from the neuron, we can control our weight 
	learning and tweaking of the weights. This facilitates learning of weights in a neuron.


## 02. Feedforward Neural Network

	In a feedforward neural network, the data passes through the different input nodes till it reaches 
	the output node. Data moves in only one direction from the first tier onwards until it reaches 
	the output node. This is also known as a front propagated wave which is usually achieved by 
	using a classifying activation function. There is no backpropagation and data moves in one direction only. 
	A feedforward neural network may have a single layer or it may have hidden layers.
	Feedforward neural networks are used in technologies like face recognition and computer vision. 
	This is because the target classes in these applications are hard to classify.
	A simple feedforward neural network is equipped to deal with data which contains a lot of noise. 
	Feedforward neural networks are also relatively simple to maintain.
	
## This one round of forward and back propagation iteration is known as one training iteration aka “Epoch“.
	
## 03. Convolutional Neural Networks
	It used back propagation in a feedforward net with many hidden layers, many maps of replicated units 
	in each layer, pooling of the outputs of nearby replicated units, a wide net that can cope with 
	several characters at once even if they overlap, and a clever way of training a complete system, 
	not just a recognizer. Later it is formalized under the name convolutional neural networks (CNNs). 
	They are primarily used for image processing but can also be used for other types of input such as as audio.

	A convolutional neural network(CNN) uses a variation of the multilayer perceptrons. 
	A CNN contains one or more than one convolutional layers. These layers can either be 
	completely interconnected or pooled. 
	Before passing the result to the next layer, the convolutional layer uses a convolutional operation 
	on the input. Due to this convolutional operation, the network can be much deeper but with much 
	fewer parameters. Due to this ability, convolutional neural networks show very effective results 
	in image and video recognition, natural language processing, and recommender systems. 
	Convolutional neural networks also show great results in semantic parsing and paraphrase detection.
	They are also applied in signal processing and image classification.
	CNNs are also being used in image analysis and recognition in agriculture where weather features 
	are extracted from satellites like LSAT to predict the growth and yield of a piece of land. 
	
	The neural nets exists and in addition to that an image is convoluted, converted in pixel level and studied, 
	converted and a max pooling, this entire thing is known as convolution + pooling layers.
	Convolution layer: Here we try to decompose RGB to multidimensional layer, and apply filter to each layer. 
	A filter tries to learn all the combinations present in the RGB layer. A strider is used to stride to each 
	matrix in the image. We try to understand these image using convolution strider. Steps to run a CNN :
		- Creating a model with mLP
		- Convolutional layer
		- Activation layer
		- Pooling layer
		- Dense (fully connected layer)
		- Model compile and train
	
## 04. Recurrent Neural Network(RNN):

	A Recurrent Neural Network is a type of artificial neural network in which the output of a particular 
	layer is saved and fed back to the input. This helps predict the outcome of the layer. The first layer 
	is formed in the same way as it is in the feedforward network. That is, with the product of the sum of 
	the weights and features. However, in subsequent layers, the recurrent neural network process begins.
	From each time-step to the next, each node will remember some information that it had in the previous 
	time-step. In other words, each node acts as a memory cell while computing and carrying out operations. 
	The neural network begins with the front propagation as usual but remembers the information it may 
	need to use later. If the prediction is wrong, the system self-learns and works towards making the right 
	prediction during the backpropagation. 
	This type of neural network is very effective in text-to-speech conversion technology.  

	So far the neural networks that we’ve examined have always had forward connections. The input layer always
	connects to the first hidden layer. Each hidden layer always connects to the next hidden layer. 
	The final hidden layer always connects to the output layer. This manner to connect layers is the 
	reason that these networks are called “feedforward.” Recurrent neural networks are not so rigid, 
	as backward connections are also allowed. A recurrent connection links a neuron in a layer to 
	either a previous layer or the neuron itself. Most recurrent neural network architectures maintain 
	state in the recurrent connections. Feedforward neural networks don’t maintain any state. 
	A recurrent neural network’s state acts as a sort of short-term memory for the neural network. 
	Consequently, a recurrent neural network will not always produce the same output for a given input.

	Recurrent neural networks do not force the connections to flow only from one layer to the next, 
	from input layer to output layer. A recurrent connection occurs when a connection is formed
	between a neuron and one of the following other types of neurons:

		- The neuron itself
		- A neuron on the same level
		- A neuron on a previous level

	Recurrent connections can never target the input neurons or the bias neurons.
	The processing of recurrent connections can be challenging. Because the recurrent links create endless
	loops, the neural network must have some way to know when to stop. A neural network that entered an 
	endless loop would not be useful. To prevent endless loops, we can calculate the recurrent connections 
	with the following three approaches:

		- Context neurons
		- Calculating output over a fixed number of iterations
		- Calculating output until neuron output stabilizes
		
	For sequence of events, languages, models, time series, Predicting stock prices, Speech recognition,
	Image captions, Word predictions, Language translation.
	
	Time series analysis such as stock prediction like price, price at time t1, t2 etc.. can be done using 
	Recurrent neural network. Predictions depend on earlier data, in order to predict time t2, we get the 
	earlier state information t1, this is known as recurrent neural network. Maintains memory from previous
	state. Length of the memory is very limited. 
	Variants of RNN :
		- Long Short Term Memory (LSTM)
		- GRU :Gated recurrent unit
		- End to end network
		- Memory network

## 05. Long Short Term Memory:
	
	Special type of RNN. They are explicitly designed to address the long term dependency problem, there are 
	gates to remember, where to forget in LSTM. RNN with LSTM prevents vanishing gradient effect by passing 
	errors recursively to the next NN. It controls the gradient flow & enable better preservation of “long-range 
	dependencies” by using gates. Maintains memory from previous and even other states. Length of the memory is 
	quite large. Long short-term memory (LSTM) is explicitly designed to address the long-term dependency problem, 
	by maintaining a state of what to remember and what to forget.
	Key components of LSTM : 
		- Gates
			- forget: Earlier gate which has data to be remembered are concatenated with the new data to 
					be remembered.
					
			- memory: Here it is used to determine how much information should be stored in the memory and 
					how much percentage to forget. Operations like dot product, 
					additions are performed here.
					
			- update: Forget from the early state and operations are performed and updated.
	
		- tanh(x): values from -1 to 1
		- sigmoid(x): values from 0 to 1
		
## 06. Multi Layer Perceptron (MLP): One or more non-linear hidden layers.

	A multilayer perceptron has three or more layers. It is used to classify data that cannot be
	separated linearly. It is a type of artificial neural network that is fully connected. 
	This is because every single node in a layer is connected to each node in the following layer.
	
	A multilayer perceptron uses a nonlinear activation function (mainly hyperbolic tangent or logistic function). 
	The activation functions are: Sigmoid (0 to 1) & Tanh (-1 to 1). 
	This type of neural network is applied extensively in speech recognition and machine translation technologies.
	
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
		- ReLU overcomes the vanishing gradient problem in the multi layer neural network.

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
		
## 07. Radial Basis Function Neural Network

	Actually FF (feed forward), that use radial basis function as activation function instead of 
	logistic function. What makes the difference? Logistic function map some arbitrary value 
	to a 0…1 range, answering a “yes or no” question. It is good for classification and decision 
	making systems, but works bad for continuous values. Contrary, radial basis functions answer 
	the question “how far are we from the target”? This is perfect for function approximation, 
	and machine control. 
	To be short, these are just FF networks with different activation function and appliance.
	
	A radial basis function considers the distance of any point relative to the centre. Such neural networks 
	have two layers. In the inner layer, the features are combined with the radial basis function.
	The radial basis function neural network is applied extensively in power restoration systems. 
	In recent decades, power systems have become bigger and more complex. This increases the risk 
	of a blackout. This neural network is used in the power restoration systems in order to restore 
	power in the shortest possible time.

## 08. Modular Neural Network:
	
	A modular neural network has a number of different networks that function independently and perform sub-tasks. 
	The different networks do not really interact with or signal each other during the computation process. 
	They work independently towards achieving the output.

	As a result, a large and complex computational process can be done significantly faster by breaking it down 
	into independent components. The computation speed increases because the networks are not interacting with 
	or even connected to each other.
	
## 09. Sequence-To-Sequence Models

	A sequence to sequence model consists of two recurrent neural networks. There’s an encoder that processes 
	the input and a decoder that processes the output. The encoder and decoder can either use the same or 
	different parameters. This model is particularly applicable in those cases where the length of the input 
	data is not the same as the length of the output data. Sequence-to-sequence models are applied mainly 
	in chatbots, machine translation, and question answering systems.	
	
## 10. DFF neural networks 
	These are just FF NNs, but with more than one hidden layer. training a traditional FF, we pass only a small
	amount of error to previous layer. Because of that stacking more layers led to exponential growth of 
	training times, making DFFs quite impractical. in early 2000s we developed a bunch of approaches 
	that allowed to train DFFs effectively; now they form a core of modern Machine Learning systems, 
	covering the same purposes as FFs, but with much better results.
	
## 11. Autoencoders are used for classification, clustering and feature compression.

	When you train FF neural networks for classification you mostly must feed then X examples in Y categories, 
	and expect one of Y output cells to be activated. This is called “supervised learning”.
	AEs, on the other hand, can be trained without supervision. Their structure — when number of hidden cells 
	is smaller than number of input cells (and number of output cells equals number of input cells), 
	and when the AE is trained the way the output is as close to input as possible, 
	forces AEs to generalise data and search for common patterns.

## 12. Dropout Regularization

	Most neural network frameworks implement dropout as a separate layer. Dropout layers function as a regular, 
	densely connected neural network layer. The only difference is that the dropout layers will periodically 
	drop some of their neurons during training. You can use dropout layers on regular feedforward neural networks.

	 A network with dropout means that some weights will be randomly set to zero. 
	 Imagine you have an array of weights [0.1, 1.7, 0.7, -0.9]. If the neural network has 
	a dropout, it will become [0.1, 0, 0, -0.9] with randomly distributed 0. The parameter that controls 
	the dropout is the dropout rate. The rate defines how many weights to be set to zeroes.

## 13. Deep neural network

	-Deep learning works because of the architecture of the network AND the optimization routine applied 
		to that architecture.
		
	-Deep neural networks were a set of techniques that were discovered to overcome the vanishing gradient 
		problem which was severely limiting the depth of neural networks.
		
	-The use of simple Rectified Linear Units (ReLU) instead of sigmoid and tanh functions is probably 
		the biggest building block in making training of DNNs possible. Note that both sigmoid and 
		tanh functions have almost zero gradient almost everywhere, depending on how fast they transit 
		from the low activation level to high; in the extreme case, when the transition is sudden, 
		we get a step function that has slope zero everywhere except at one point where the transition happens.
		
	- One of the main differences between deep neural networks and 
		traditional neural networks is that we don't just use backpropagation for deep neural nets. 
		Because backpropagation trains later layers more efficiently than it trains earlier layers--as you 
		go earlier and earlier in the network, the errors get smaller and more diffuse. So a ten-layer 
		network will basically be seven layers of random weights followed by three layers of fitted weights, 
		and do just as well as a three layer network.
		
 -----------------------------------------------------------------------------------------------------------
 
# Deep Learning: multi neural network architecture : 
	# Artificial neural network (ANN): Data in numeric format
	# Convolutional Neural Networks(CNN) : Imgaes data
	# Recurrent neural network(RNN): Time series data

-------------------------------------------------------------------------------------------------------------

![List of Neural Network](https://github.com/amitmse/in_Python_/blob/master/Neural%20Network/All%20Type.png)

Note: sourced from [Link](https://towardsdatascience.com/the-mostly-complete-chart-of-neural-networks-explained-3fb6f2367464)


---------------------------------------------------------------------------------------------------------

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
		
-------------------------------------------------------------------------------------------------------------
 
## Types of Gradient Descent:
	1. Batch Gradient Descent: 
		In batch gradient descent, we use the complete dataset available to compute the gradient of 
		the cost function. Batch gradient descent is very slow because we need to calculate the gradient 
		on the complete dataset to perform just one update, and if the dataset is large then it will 
		be a difficult task. 
		- Cost function is calculated after the initialization of parameters.
		- It reads all the records into memory from the disk.
		- After calculating sigma for one iteration, we move one step further, and repeat the process.
	
	2. Mini-batch Gradient Descent:
		It is a widely used algorithm that makes faster and accurate results. The dataset, here, 
		is clustered into small groups of ‘n’ training datasets. It is faster because it does not 
		use the complete dataset. In every iteration, we use a batch of ‘n’ training datasets to 
		compute the gradient of the cost function. It reduces the variance of the parameter updates, 
		which can lead to more stable convergence. It can also make use of a highly optimized matrix
		that makes computing of the gradient very efficient.

	3. Stochastic Gradient Descent:
		We use stochastic gradient descent for faster computation. The first step is to randomize 
		the complete dataset. Then, we use only one training example in every iteration to calculate 
		the gradient of the cost function for updating every parameter. It is faster for larger datasets 
		also because it uses only one training example in each iteration.

---------------------------------------------------------------------------------------------------------

## Hidden layers

	Why it is called hidden: Because hidden layer does not see inputs(training set)
	
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

---------------------------------------------------------------------------------------------------------

## Back-propagation
	
	When we feel that outputs are not correct, we back propagate the values to adjust the weights 
	to produce the right output. The architecture, activation functions remains the same in each 
	perceptron. Adjusts using gradient descent.

	Back-propagation is considered the standard method in artificial neural networks to calculate 
	the error contribution of each neuron after a batch of data is processed. However, there are 
	some major problems using back-propagation. Firstly, it requires labeled training data; 
	while almost all data is unlabeled. Secondly, the learning time does not scale well, 
	which means it is very slow in networks with multiple hidden layers. Thirdly, it can 
	get stuck in poor local optima, so for deep nets they are far from optimal.
	
	Neural networks are trained using backpropagation gradient descent. That is, you update the weights of 
	each layer as a function of the derivative of the previous layer. The problem is that the update 
	signal was lost as you increased the depth of the neural network. The math is pretty simple. 
	Therefore, in the old days, people pretty much only used neural networks with a single hidden-layer.
	
	The errors are getting propagated in a backward fashion. We start by first finding the error terms 
	of the output layer using BP1. And then using BP2 we get error terms of the layer previous to that. 
	keep on using BP2 to get error terms of all neurons in all inside layer right till first input layer. 
	So this way we are propagating the error from the output layer all the way to the first layer. 
	Back Propagation is nothing but the chain rule of calculus. So the reason this error propagation 
	is happening backwards is because the initial cost function is used as a function of output
	activation and chain rule of calculus then causes it to behave that way.
		
	Back propagation has some problems associated with it which include :
	
	 - Network paralysis: It occurs when the weights are adjusted to very large values during training, 
	 		large weights can force most of the units to operate at extreme values, in a region 
			where the derivative of the activation function is very small
	 
	 - Local minima : Perhaps the best known is called “Local Minima”. This occurs because 
	 		the algorithm always changes the weights in such a way as to cause 
			the error to fall. But the error might briefly have to rise as part of a more
			general fall, If this is the case, the algorithm will “get stuck” 
			(because it can‟t go uphill) and the error will not decrease further
			
	 - Slow convergence
	
	- A multilayer neural network requires many repeated presentations of the input patterns, 
			for which the weights need to be adjusted before the network is able to 
			settle down into an optimal solution
			
---------------------------------------------------------------------------------------------------------

## Choosing the correct learning rate and momentum will help in weight adjustment

## Learning rate /step size:
	The learning rate is how fast the network learns its parameters. 
	Setting right learning rate could be difficult task. The learning rate is a parameter that determines 
	how much an updating step influences the current value of the weights.If learning rate is too small, 
	algorithm might take long time to converges. choosing large learning rate could have opposite effect 
	algorithm could diverge. Sometimes in NN every weight has it’s own learning rate. Learning rate of 0.35 
	proved to be popular choice when training NN. This paper will use rate of 0.45 but this value is used 
	because of simple architecture of NN used in example.
						
## Momentum term: 
	It is a parameter that helps to come out of the local minima and smoothen the jumps while gradient descent.
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
	
## Number of epochs: 
	The number of times the entire training data is fed to the network while training is referred to as 
	the number of epochs. We increase the number of epochs until the validation accuracy starts decreasing, 
	even if the training accuracy is increasing (overfitting).

## Parameters vs Hyperparameters
	Parameters are learned by the model during the training time, while hyperparameters can be changed 
	before training the model. Parameters of a deep neural network are weight, Beta etc, which the model 
	updates during the backpropagation step. 
	On the other hand, there are a lot of hyperparameters for a deep NN, including:
		Learning rate – ⍺
		Number of iterations
		Number of hidden layers
		Units in each hidden layer
		Choice of activation function

## Neuron Saturation:
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
	
---------------------------------------------------------------------------------------------------------------

- The Turing test is a method to test a machine’s ability to match the human-level intelligence.

- F1 score is the weighted average of precision and recall. It considers both false positive and 
  false negative values into account. It is used to measure a model’s performance.

- A cost function is a scalar function that quantifies the error factor of the neural network.

- Dropout is a simple way to prevent a neural network from overfitting.

- Perceptron : Number of  Hidden Layer



https://www.analyticsvidhya.com/blog/2017/05/neural-network-from-scratch-in-python-and-r/

https://www.digitalvidya.com/blog/types-of-neural-networks/

https://towardsdatascience.com/the-mostly-complete-chart-of-neural-networks-explained-3fb6f2367464

https://towardsdatascience.com/types-of-neural-network-and-what-each-one-does-explained-d9b4c0ed63a1

https://analyticsindiamag.com/6-types-of-artificial-neural-networks-currently-being-used-in-todays-technology/

https://www.mygreatlearning.com/blog/types-of-neural-networks/

https://blog.statsbot.co/neural-networks-for-beginners-d99f2235efca


-----------------------------------------------------------------------------------
Gradient Descend:
https://towardsdatascience.com/a-comprehensive-guide-on-neural-networks-for-beginners-a4ca07cee1b7
https://becominghuman.ai/step-by-step-neural-network-tutorial-for-beginner-cc71a04eedeb
