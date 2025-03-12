
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

# Distributed computing: Pyspark vs Dask

	- Dask and Apache Spark is a distributed computing tool. Spark is mature and all-inclusive.
	- Dask is smaller and lighter weight than Spark. Dask is often faster than Spark.
	- Dask has limited  features but integrated with other libraries: Pandas, Scikit-Learn for high-level functionality. 

Language:

Spark is written in Scala with some support of Python and R, and work with JVM code.

Dask is written in Python, and work with C/C++/Fortran/LLVM.

Ecosystem:

Spark is an all-in-one project that has own ecosystem.

Dask is a component of the Python ecosystem and depends on other libraries: NumPy, pandas, and Scikit-learn.

Age:

Spark came into existence in 2010 and popular in Big Data world.

Dask came into existence in 2014.

Scope:

Spark is focused on SQL and lightweight ML.

Dask is focused on scientific and custom situations.

Internal Design:

Spark is an extension of the Map-Shuffle-Reduce paradigm and provides high level optimizations but lacking flexibility 

Dask is generic task scheduling and works on lower level, so lacks high level optimizations. Wroks well with sophisticated algorithms.

Scale:

Spark scales from a single node to thousand-node clusters.

Dask is same as spark.

DataFrames:

Spark DataFrame has its own API and memory model. It also implements a large subset of the SQL language. 

Dask DataFrame reuses the Pandas API and memory model. It implements neither SQL nor a query optimizer. 

Machine Learning:

Spark uses MLLib along with JVM-based ML libraries like H2O.

Dask relies Scikit-learn and XGBoost.

Arrays

Spark does support multi-dimensional arrays natively.

Dask fully supports multi-dimensional arrays.






Dask: Handle big size data processing which Panda unable to do

https://github.com/amitmse/in_Python_/blob/master/Data%20Prep/Dask.py




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

written in Python 2.7
