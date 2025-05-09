
# Python
	- Object Oriented Programming (OOP- Abstraction : Process happens inside and not visible to public.
                It hides the inner workings of an object when it’s not necessary to see them.

	- Encapsulation	: Prevent data modification.
                  This prevent data from direct modification which is called encapsulation. 
                  Represent by prefix i.e single “ _ “ or double “ __“. 
                  It stores related variables and methods within objects and protects them.
                  Encapsulation is one of the ways that object-oriented programming creates abstraction. 
                  A programmer could make a change to the structure or contents of the object 
		  	without worrying about the public interface.

	- Inheritance :	Creating new class (child) using existing class (parent). 
                Inheritance is a way of creating new class for using details of existing class without modifying it. 
                It allows sub-classes to use attributes from parent classes. 
                Instead of creating new classes for everything, programmers can create a 
			base class and then extend it to new sub-classes when they need to.

	- Polymorphism :	Allows to define methods in the child class with the same name as defined in the parent class.
                  Polymorphism is an ability (in OOP) to use common interface for multiple form (data types).
                  It allows objects and methods to deal with multiple different situations with a single interface. 
                  Polymorphism is a result of inheritance.

---------------------------------------------------------------------------------------------------------


# Pandas
	The Python Pandas library is similar to SAS, providing equivalent functionalities.
  	
   	Limitation: 
		The Pandas loads datasets into memory, meaning the data size must not exceed the available RAM. 
  		This limitation arises because Pandas does not support distributed computing. 
		To manage larger datasets, libraries such as PySpark or Dask can be utilized 
  		as they provide distributed computing capabilities. Comparison is provided below.
	
 	The development of the pandas library relies on several other libraries. Some of the key libraries:
		- NumPy: pandas is built on top of numpy, leveraging its efficient array processing capabilities. 
			 NumPy data structure is ndarray while pandas structures are Series 
    			 (one-dimensional array,) and DataFrame (two-dimensional).
		- Python: The base language in which pandas is implemented.
		- Cython: Used to optimize performance-critical parts of the pandas codebase by compiling Python code to C.
		- dateutil: A library for parsing and handling dates, used extensively in pandas.
		- pytz: A library for working with time zones, utilized by pandas to handle time zone-aware datetime objects.
		- matplotlib: Often used in conjunction with pandas for data visualization, although not a direct dependency.
		- pytest: Used for testing the pandas codebase to ensure reliability and correctness.
		These libraries collectively enable pandas to offer its powerful data manipulation and analysis capabilities.

### SAS to Python: Basic SAS functions coded in python using Pandas

https://github.com/amitmse/in_Python_/blob/master/Data%20Prep/Basic%20Data%20check.py

    - Import, Export, Proc contents, Freq, Means
    - Create variable, duplicate, append, merge, Transpose, Correlation, Lag, first dot, 
    - Lift table/KS
    - PSI

# Distributed computing: Spark (Pyspark) vs Dask

	- Dask and Apache Spark are distributed computing tools.
 	- Spark is mature and all-inclusive.
	- Dask is smaller and lighter in weight than Spark. It is often faster than Spark.
	- Dask has limited features but integrates with other libraries like Pandas and Scikit-Learn for high-level functionality.

	Language:
	- Spark is written in Scala.
	- Dask is written in Python.

	Age:
	- Spark was introduced in 2010.
	- Dask was introduced in 2014.
 
	Ecosystem:
	- Spark is an all-in-one project that has own ecosystem.
	- Dask is a component of the Python ecosystem and depends on other libraries like NumPy, Pandas, and Scikit-Learn.

	Scope:
	- Spark is focused on SQL and lightweight ML.
	- Dask is focused on scientific.

	Internal Design:
	- Spark is an extension of the Map-Shuffle-Reduce paradigm and provides high level optimizations but lacking flexibility 
	- Dask is generic task scheduling and works on lower level, so lacks high level optimizations.

	Scale:
	- Spark scales from a single node to thousand-node clusters.
	- Dask is same as spark.

	DataFrames:
	- Spark DataFrame has its own API and memory model.
	- Dask DataFrame reuses the Pandas API and memory model.

	Machine Learning:
	- Spark uses MLLib along with JVM-based ML libraries like H2O.
	- Dask relies Scikit-learn and XGBoost.

	Arrays
	- Spark does support multi-dimensional arrays natively.
	- Dask fully supports multi-dimensional arrays.
 	  https://docs.dask.org/en/latest/spark.html

### Dask Sample Code:
	- Converted above pandas code into Dask.
https://github.com/amitmse/in_Python_/blob/master/Data%20Prep/Dask.py

### Pyspark
https://github.com/amitmse/in_Python_/tree/master/Pyspark

# Exploratory Data Analysis (EDA):

https://github.com/amitmse/in_Python_/blob/master/Data%20Prep/EDA.py

    Provides basic distribution of data i.e., count, missing, unique, sum, mean, STD, percentile


# Information Value:

https://github.com/amitmse/in_Python_/blob/master/Data%20Prep/Information_value_calculation.py

    Compute Information value of a variable


# Monotonic Binning / correlation / KS

https://github.com/amitmse/in_Python_/blob/master/Data%20Prep/Bin.py

    Monotonic Binning of a variable
	correlation
	KS    


# Charts in Python

https://github.com/amitmse/in_Python_/tree/master/plot


