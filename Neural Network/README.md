# Neural Networks

	It consists of the input value and output value. Each input value is associated with its weight, 
	which passes on to next level, each perceptron will have an activation function. The weights and 
	input value forms a single perception. We use activation function and based on that, the value goes 
	to next well. And the process continues till it reaches output y’.
	
	In simple term: 
 		It uses logistic regression (or any other) and repeating it more than one times.
		In logistic regression, there are only two layers i.e. input and output but in neural network, 
		there is at least one hidden layer between input and output layer.
 	
 ---------------------------------------------------------------------------------------------------------
 
 ## Cost function :
 
	It tried to quantify the error factor of neural network. It calculates how well the neural network 
	is performing based on the actual vs predicted value. Error factor = Predicted – Actual.
	
---------------------------------------------------------------------------------------------------------

## Activation Function:
 
 	It’s function to get the output of node. It is also known as Transfer Function. It is used to determine 
	the output of neural network like YES or NO. 
	
	Some examples: SIGMOID, RELU, LEAKYRELU, TANH, SOFTMAX, CUBE, ELU, RRELU, HARDSIGMOID, HARDTANH, IDENTITY,  
			RATIONALTANH, SOFTPLUS, SOFTSIGN, 
	
	The Activation Functions can be basically divided into 2 types:
	
### 1. Linear Activation Function: 
	
	A neural network with a linear activation function is simply a linear regression model. No matter how many 
	layers in the neural network, the last layer will be a linear function of the first layer so a linear 
	activation function turns the neural network into just one layer. Backpropagation can't be used as  
	the derivative of linear function is a constant.
	Range: -infinity to infinity.
	
### 2. Non-linear Activation Functions: 
	
	It makes it easy for the model to generalize or adapt with variety of data and to differentiate between 
	the output. It allows backpropagation because they have a derivative function which is related to the inputs. 
	It allows “stacking” of multiple layers of neurons to create a deep neural network. Multiple hidden layers 
	of neurons are needed to learn complex data sets with high levels of accuracy. 
	Below are few non-linear activation functions:
	
#### 2.1 Sigmoid or Logistic Activation Function: 
	
	Sigmoid Function looks like a S-shape. The logistic sigmoid function can cause a neural network to get stuck 
	at the training time. The softmax function is a more generalized logistic activation function which is used for 
	multiclass classification. The main reason why we use sigmoid function is because it exists between (0 to 1). 
	
	Problem: Vanishing gradient.

![Function](https://github.com/amitmse/in_Python_/blob/master/Neural%20Network/Sigmoid.PNG)


#### 2.2 Tanh or hyperbolic tangent Activation Function: 
	
	tanh is also like logistic sigmoid but better. The range of the tanh function is from (-1 to 1). 
	tanh is also sigmoidal (s - shaped). Both tanh and logistic sigmoid activation functions are used in 
	feed-forward nets. 
	
	Problem: Vanishing gradient.
	
![Function](https://github.com/amitmse/in_Python_/blob/master/Neural%20Network/Tanh.PNG)
	
#### 2.3 ReLU (Rectified Linear Unit) Activation Function: 
	
	The ReLU is the most used activation function. Since, it is used in almost all the convolutional neural 
	networks or deep learning. It rectifies vanishing gradient problem. Range is 0 to infinity. 
	
	It should only be used within Hidden layers of a Neural Network Model. Hence for output layers should a 
	Softmax function for a Classification problem to compute the probabilites for the classes, and for a 
	regression problem it should simply use a linear function. 
	
	Problem: 
	The issue is that all the negative values become zero immediately which decreases the ability of the model 
	to fit or train from the data properly.
	
	Another problem with ReLu is that some gradients can be fragile during training and can die. It can cause a
	weight update which will makes it never activate on any data point again. The draw backs of ReLU is when
	the gradient hits zero for the negative values, it does not converge towards the minima which will result in a 
	dead neuron while back propagate. To fix this problem another modification was introduced called Leaky ReLu to
	fix the problem of dying neurons. It introduces a small slope to keep the updates alive. ReLU overcomes the 
	vanishing gradient problem in the multi layer neural network. 
	
	Rectified Linear Unit doesn't have the saturation problem where the output might get constrained beyond a limit.
	
![Function](https://github.com/amitmse/in_Python_/blob/master/Neural%20Network/ReLU.PNG)
	
#### 2.4 Leaky ReLU: 
	Range : -infinity to infinity.

![Function](https://github.com/amitmse/in_Python_/blob/master/Neural%20Network/Leaky_ReLU.PNG)

#### 2.5 Softmax: 
	Softmax is a very interesting activation function because it not only maps our output to a [0,1] range but also 
	maps each output in such a way that the total sum is 1. The output of Softmax is therefore a probability distribution.
	
	The softmax function is often used in the final layer of a neural network-based classifier. Such networks are 
	commonly trained under a log loss (or cross-entropy) regime, giving a non-linear variant of multinomial logistic 
	regression. 
	
	Softmax is used for multi-classification in logistic regression model whereas Sigmoid is used for binary
	classification in logistic regression model, the sum of probabilities is One for Softmax. It handles classification
	problems. 
	
	Softmax is used only for the output layer, for neural networks that need to classify inputs into multiple 
	categories. 
	
![Function](https://github.com/amitmse/in_Python_/blob/master/Neural%20Network/softmax.PNG)


---------------------------------------------------------------------------------------------------------

# Types of neural networks and their applications:

## 01. Perceptron 
	The simplest and oldest model of Neuron, as we know it. Takes some inputs, sums them up, 
	applies activation function and passes them to output layer. It's a Binary unit outputting 
	"Yes/No" decisions with binary inputs. Logistic Regression is the best example with single 
	perceptron / neuron.
	
	Perceptron function is a step function. Transform the normal Perceptron into a sigmoid neuron 
	using a sigmoid function.

## 02. Feedforward Neural Network (FFNN) / Artificial Neural Network

	In a feedforward neural network, the data passes through the different input nodes till it reaches 
	the output node. Data moves in only one direction from the first tier onwards until it reaches 
	the output node. This is also known as a front propagated wave which is usually achieved by 
	using a classifying activation function. A simple feedforward neural network is equipped to deal with
	data which contains a lot of noise. Feedforward neural networks are also relatively simple to maintain.
	
	There is no backpropagation and data moves in one direction only. A feedforward neural network may 
	have a single layer or it may have hidden layers. 
	
	Use: face recognition, Simple classification, Speech Recognition and computer vision. 
	This is because the target classes in these applications are hard to classify.
	
### One round of Forward and Back propagation iteration is known as one training iteration aka "Epoch".
	
## 03. Convolutional Neural Networks:

	A convolutional neural network(CNN) contains one or more than one convolutional layers. These layers 
	can either be completely interconnected or pooled. Before passing the result to the next layer, 
	the convolutional layer uses a convolutional operation on the input. Due to this convolutional operation, 
	the network can be much deeper but with much fewer parameters. 
		
	Convolutional neural networks also show great results in semantic parsing and paraphrase detection.
	They are also applied in signal processing and image classification. CNNs are also being used in image 
	analysis and recognition in agriculture where weather features are extracted from satellites like 
	LSAT to predict the growth and yield of a piece of land. 
	
	The neural nets exists and in addition to that an image is convoluted, converted in pixel level and studied, 
	converted and a max pooling, this entire thing is known as convolution + pooling layers.
	Convolution layer: Decompose RGB to multidimensional layer, and apply filter to each layer. A filter tries 
	to learn all the combinations present in the RGB layer. A strider is used to stride to each matrix in the 
	image. We try to understand these image using convolution strider. Steps to run a CNN :
		- Creating a model with mLP
		- Convolutional layer
		- Activation layer
		- Pooling layer
		- Dense (fully connected layer)
		- Model compile and train
	
	Use: image and video recognition, natural language processing, Speech Recognition, Computer Vision
		Machine translation, and recommender systems. 

## 04. Recurrent Neural Network(RNN):

	RNNs are FFNNs with a time twist: they are not stateless; they have connections between passes, 
	connections through time. Neurons are fed information not just from the previous layer but also 
	from themselves from the previous pass. In this, the output of a particular layer is saved and 
	fed back to the input. This helps predict the outcome of the layer. The first layer is formed in 
	the same way as it is in the feedforward network (product of the sum of the weights and features). 
	From each time-step to the next, each node will remember some information that it had in the previous 
	time-step. In other words, each node acts as a memory cell while computing and carrying out operations. 
	The neural network begins with the front propagation as usual but remembers the information it may need 
	to use later. If the prediction is wrong, the system self-learns and works towards making the right prediction 
	during the backpropagation. 

	So far the neural networks that we’ve examined have always had forward connections. The input layer always
	connects to the first hidden layer. Each hidden layer always connects to the next hidden layer. 
	The final hidden layer always connects to the output layer. This manner to connect layers is the 
	reason that these networks are called “feedforward.” 
	
	Recurrent neural networks are not so rigid, as backward connections are also allowed. A recurrent connection 
	links a neuron in a layer to either a previous layer or the neuron itself. Most recurrent neural network 
	architectures maintain state in the recurrent connections. Feedforward neural networks don’t maintain any state.
	A recurrent neural network’s state acts as a sort of short-term memory for the neural network. 
	Consequently, a recurrent neural network will not always produce the same output for a given input.

	Recurrent neural networks do not force the connections to flow only from one layer to the next, 
	from input layer to output layer. A recurrent connection occurs when a connection is formed
	between a neuron and one of the following other types of neurons:

		- The neuron itself
		- A neuron on the same level
		- A neuron on a previous level

	Recurrent connections can never target the input neurons or the bias neurons. The processing of recurrent 
	connections can be challenging. Because the recurrent links create endless loops, the neural network must 
	have some way to know when to stop. A neural network that entered an endless loop would not be useful. 
	To prevent endless loops, we can calculate the recurrent connections with the following three approaches:

		- Context neurons
		- Calculating output over a fixed number of iterations
		- Calculating output until neuron output stabilizes
		
		
	Time series analysis such as stock prediction like price, price at time t1, t2 etc.. can be done using 
	Recurrent neural network. Predictions depend on earlier data, in order to predict time t2, we get the 
	earlier state information t1, this is known as recurrent neural network. Maintains memory from previous
	state. Length of the memory is very limited. 
	Variants of RNN :
		- Long Short Term Memory (LSTM)
		- GRU :Gated recurrent unit
		- End to end network
		- Memory network
		
	Use: For sequence of events, languages, models, time series, Predicting stock prices, Speech recognition,
	Image captions, Text to speech processing, Image tagger, Sentiment Analysis, Translation, Word predictions, 
	Language translation, Text processing i.e. auto suggest, grammar checks.
	
## 05. Long Short Term Memory:
	
	It's special type of RNN and explicitly designed to address the long term dependency problem, there are 
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
		
	Use: For sequence of events, languages, models, time series, Predicting stock prices, Speech recognition,
	Image captions, Word predictions, Language translation.
	
## 06. Multi Layer Perceptron (MLP): One or more non-linear hidden layers.

	A multilayer perceptron has three or more layers. It is used to classify data that cannot be separated linearly. 
	This is because every single node in a layer is connected to each node in the following layer. A multilayer 
	perceptron uses a nonlinear activation function (mainly hyperbolic tangent or logistic function).
	
	Advantages:
		- Capability to learn non-linear models.
		- Capability to learn models in real-time (on-line learning) using partial_fit.
		
	Disadvantages:
		- MLP with hidden layers have a non-convex loss function where there exists more than one local minimum. 
		  Therefore different random weight initializations can lead to different validation accuracy.
		- MLP requires tuning of hyperparameters i.e. number of hidden neurons, layers, and iterations.
		- MLP is sensitive to feature scaling.
		- ReLU overcomes the vanishing gradient problem in the multi layer neural network.
				
	Activation function for the hidden layer: (default is relu)
		
	Use: speech recognition, machine translation technologies, and Complex Classification.

	
## 07. Radial Basis Function Neural Network

	It's Feed Forward (FF) networks with different activation function (not logistic function). Logistic function is 
	good for classification and decision making systems, but works bad for continuous values. Contrary, radial 
	basis functions answer the question “how far are we from the target”? This is perfect for function approximation, 
	and machine control. 
	
	Use: Power restoration systems. In recent decades, power systems have become bigger and more complex and 
	high risk of a blackout. This neural network is used in the power restoration systems in order to restore 
	power in the shortest possible time.

## 08. Deep neural network

	Deep neural networks were a set of techniques that were discovered to overcome the vanishing gradient 
	problem which was severely limiting the depth of neural networks.
		
	The use of simple Rectified Linear Units (ReLU) instead of sigmoid and tanh functions is probably 
	the biggest building block in making training of DNNs possible. 
		
	One of the main differences between deep neural networks and traditional neural networks is that we don't 
	just use backpropagation for deep neural nets. Because backpropagation trains later layers more efficiently 
	than it trains earlier layers, the errors get smaller and more diffuse. 

-------------------------------------------------------------------------------------------------------------

![List of Neural Network](https://github.com/amitmse/in_Python_/blob/master/Neural%20Network/All%20Type.png)

Note: sourced from [Link](https://www.asimovinstitute.org/author/fjodorvanveen/)
			
---------------------------------------------------------------------------------------------------------

## Algorithm

	1. Feed-forward computation: Calculate hidden layer nodes & output layer.
		a. Calculate all hidden layer nodes: Multiply input layer and their weights (random for 1st time)
		And then apply sigmod function(Logistic function)
							  
		b. Output layer node: Multiply hidden layer and their weights (random for 1st time) and then apply 
		sigmod function(Logistic function)

	2. Back propagation to the output layer: calculate error of output layer & weights adjustment for hidden layer
		a. Error in output layer node: Substract actual output and predicted Output layer node
		
		b. Rate of change (weight for hidden): Multiply Learning rate, Error in output layer node and Hidden layer 
		nodes
							  
		c. Adjusted weights for hidden layer: Add previous weights for hidden layer, Rate of change (weight for 
		hidden) and Momentum term into previous delta change of the weight (will be 0 for 1st time)

	3. Back propagation to the hidden layer	: calculate error of hidden layer & weights adjustment input layer
		a. Error in hidden layer node: Multiply Error in output layer node and adjusted weights for hidden layer
								
		b. Rate of change (weight for input): Multiply Learning rate, Error in hidden layer nodes and input
		layer nodes
								
		c. Adjusted weights for input layer: Add previous weights for input layer, Rate of change (weight for 
		input) and Momentum term into previous delta change of the weight (will be 0 for 1st time)

	4. Weight updates: Use new weight to calculate hidden layer nodes, output layer & Error in output layer node
		a. Updated hidden layer nodes: Multiply input layer and their updated weights and then apply sigmod 
		function(Logistic function)
								
		b. Updated output layer node: Multiply hidden layer and their updated weights and then apply sigmod 
		function(Logistic function)
								
		c. Updated Error in output layer node: Substract actual output and Output layer node
		
		d. Change in error: Substract previous error and current error
		
-------------------------------------------------------------------------------------------------------------
 
## Types of Gradient Descent:
	1. Batch Gradient Descent: 
		It uses a complete dataset available to compute the gradient of the cost function hence and it's 
		very slow. 
		- Cost function is calculated after the initialization of parameters.
		- It reads all the records into memory from the disk.
		- After calculating sigma for one iteration, we move one step further, and repeat the process.
	
	2. Mini-batch Gradient Descent:
		It is a widely used algorithm that makes faster and accurate results. The dataset, here, 
		is clustered into small groups of ‘n’ training datasets hence it's faster. In every iteration, 
		it uses a batch of ‘n’ training datasets to compute the gradient of the cost function.
		It reduces the variance of the parameter updates, which can lead to more stable convergence. 
		It can also make use of a highly optimized matrix that makes computing of the gradient very efficient.

	3. Stochastic Gradient Descent:
		Stochastic gradient descent used for faster computation. First, it randomizes the complete dataset, 
		and then uses only one training example in every iteration to calculate the gradient. Its benifical 
		for huge datasets.

---------------------------------------------------------------------------------------------------------

## Hidden layers

	Why it is called hidden: Because hidden layer does not see inputs(training set)
	
	If data is linearly separable then don't need any hidden layers at all and no need to use NN. Should be verified.
	Assuming data does require separation by a non-linear technique, then always start with one hidden layer
		
	Additonal hidden layer creates major performance issue. Most of times, one hidden layer is sufficient for the 
	majority of problems.
		
	Using too many neurons in the hidden layers can result in several problems:
		- Overfitting.
		- Extra time to train the network. 
		- Number of neurons:
			- In between the size of the input layer and the size of the output layer
			- 2/3 of size of the input layer, plus the size of the output layer
			- Less than twice the size of the input layer

---------------------------------------------------------------------------------------------------------

## Back-propagation
	
	Back-propagation is a standard method in artificial neural networks to calculate the error contribution 
	of each neuron. However, there are some major problems using back-propagation:
	
		1. Requires labeled training data
		
		2. Slow convergence: Very slow in networks with multiple hidden layers. 
		
		3. Local minima: It happens because the algorithm always changes the weights to reduce the error. 
		But the error might briefly have to rise as part of a more general fall, If this is the case, 
		the algorithm will “get stuck” (because it can‟t go uphill) and the error will not decrease further.
		
	 	4. Network paralysis: It occurs when the weights are adjusted to very large values during training, 
	 	large weights can force most of the units to operate at extreme values, in a region where the derivative 
		of the activation function is very small.
		
		
	It updates the weights of each layer as a function of the derivative of the previous layer. The problem is 
	that the update signal was lost as depth increases.  
	
			
---------------------------------------------------------------------------------------------------------

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

### Choosing the correct learning rate and momentum will help in weight adjustment

## Learning rate /step size:
	The learning rate: how fast the network learns its parameters. Setting right learning rate could be 
	difficult task. The learning rate is a parameter that determines how much an updating step influences 
	the current value of the weights. If learning rate is too small, algorithm might take long time to 
	converges. Choosing large learning rate could have opposite effect.
						
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
	the number of epochs. Increase the number of epochs until the validation accuracy starts decreasing, 
	even if the training accuracy is increasing (overfitting).

## Parameters vs Hyperparameters
	Parameters are learned by the model during the training time, while hyperparameters can be changed 
	before training the model. 
	
	Parameters of a deep neural network are weight, Beta etc, which the model updates during the backpropagation step. 
	
	There are a lot of hyperparameters for a deep NN, including:
		1. Learning rate – ⍺
		2. Number of iterations
		3. Number of hidden layers
		4. Units in each hidden layer
		5. Choice of activation function

## Neuron Saturation:
	If we use sigmoid function as the activation function, then the error term in the output layer 
	slows down as activation saturates. If you see as Z tends to grow higher or smaller the function 
	flattens either to 1 or 0. This is called neuron saturation as it can't go beyond that. Which means 
	the differential at that time becomes very low or close to zero. This means if the output neuron saturates, 
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
	
# Use of neural network: 
	- Artificial neural network (ANN): Data in numeric format
	- Convolutional Neural Networks(CNN) : Imgaes data
	- Recurrent neural network(RNN): Time series data
	https://www.mygreatlearning.com/blog/types-of-neural-networks/
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

NN Zoo: https://www.asimovinstitute.org/author/fjodorvanveen/


-----------------------------------------------------------------------------------
Gradient Descend:

https://towardsdatascience.com/a-comprehensive-guide-on-neural-networks-for-beginners-a4ca07cee1b7

https://becominghuman.ai/step-by-step-neural-network-tutorial-for-beginner-cc71a04eedeb
