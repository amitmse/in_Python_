# Support Vector Machines

  - SVM construct linear separating hyperplanes in high-dimensional vector spaces.
  - kernels: 
	- A kernel is a similarity function. It is a function that you, as the domain expert, provide 
		to a machine learning algorithm. computing the kernel is easy, but computing the feature 
		vector corresponding to the kernel is really really hard. Many machine learning algorithms
		can be written to only use dot products, and then we can replace the dot products with kernels.
		By doing so, we don't have to use the feature vector at all. This means that we can work with 
		highly complex, efficient-to-compute, and yet high performing kernels without ever having to 
		write down the huge and potentially infinite dimensional feature vector. Thus if not for the 
		ability to use kernel functions directly, we would be stuck with relatively low dimensional, 
		low-performance feature vectors. This "trick" is called the kernel trick (Kernel trick). 
		A function that transforms one feature vector into a higher dimensional feature vector is 
		not a kernel function. Thus f(x) = [x, x^2 ] is not a kernel (becoz kernal doesn't create 
		feature and here we are creating feature). It is simply a new feature vector. You do not need 
		kernels to do this. You need kernels if you want to do this, or more complicated feature 
		transformations without blowing up dimensionality.
		
					
	- kernel is a shortcut that helps us do certain calculation faster which otherwise would involve 
		computations in higher dimensional space. kernels allow us to do stuff in infinite dimensions. 
		Sometimes going to higher dimension is not just computationally expensive, but also impossible. 
		kernels do not make the the data linearly separable. The feature vector makes the data linearly 
		separable. Kernel is to make the calculation process faster and easier, especially when 
		the feature vector is of very high dimension. The inner product means the projection of feature 
		vector [phi(x) onto phi(y)] or colloquially, how much overlap do x and y have in their 
		feature space. In other words, how similar they are.
					
	- It converts not separable problem to separable problem, these functions are called kernels. 
	- The kernel defines the similarity or a distance measure between new data and the support vectors. 
		The dot product is the similarity measure used for linear SVM or a linear kernel because 
		the distance is a linear combination of the inputs.
	- Kernels are internal functions which are tied up by the learning coefficients to form a cost 
		equation which you optimize. The most popular kernel types(in descending order) are -Gaussian, 
		Linear, Polynomial Kernels etc.
	- These are small-small functions which your model create and throw on your dataset to fit 
		the data accordingly (Just like a K means clustering works but in a supervised way). These functions 
		have shaped like gaussian , linear line, or polynomial function.
	- It is an equation that can pull data points apart into 3-dimensional space, and, instead of 
		using a line as a separator, it uses something called a hyperplane, that, from a vertical standpoint, 
		can take nonlinear forms. Nonlinear classification provides a more sophisticated way to classify
		complex data sets that can’t easily be separated by a straight line.
					
 - kernel trick:
	- In SVM, it is easy to have a linear hyper-plane between two classes. But, another burning question 
		which arises is, should we need to add this feature manually to have a hyper-plane. No, SVM has a 
		technique called the kernel trick. These are functions which takes low dimensional input space and 
		transform it to a higher dimensional space i.e. it converts not separable problem to separable problem, 
		these functions are called kernels. It is mostly useful in non-linear separation problem. Simply put, 
		it does some extremely complex data transformations, then find out the process to separate the data
		based on the labels or outputs you’ve defined.
	- Kernels can be used that transform the input space into higher dimensions such as a Polynomial Kernel 
		and a Radial Kernel. This is called the Kernel Trick. It is desirable to use more complex kernels 
		as it allows lines to separate the classes that are curved or even more complex. 
		This in turn can lead to more accurate classifiers.
			
 - Pros and Cons associated with SVM:
	Pros:
		- It works really well with clear margin of separation.
		- It is effective in high dimensional spaces.
		- It is effective in cases where number of dimensions is greater than the number of samples.
		- It uses a subset of training points in the decision function (called support vectors), 
			so it is also memory efficient.
	Cons:
		- It doesn’t perform well, when we have large data set because the required training time is higher
		- It also doesn’t perform very well, when the data set has more noise i.e. target classes are overlapping
		- SVM doesn’t directly provide probability estimates, these are calculated using an expensive 
			five-fold cross-validation. It is related SVC method of Python scikit-learn library.
			
 - sigmoid function :
	It is a mathematical function having an "S" shaped curve (sigmoid curve). Often, sigmoid function refers 
	to the special case of the logistic function shown in the first figure and defined by the formula.
			
 - activation function: 
	The role of the activation function in a neural network is to produce a non-linear decision boundary
	via non-linear combinations of the weighted inputs. Activation function is a function that transforms
	a set of input signals into an output signal.

 - Inner product: 	(U^T)V [U & V are vectors]
			(U^T)V  = projection .||U|| [length of projection of V onto U]
			
