
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

# Dask: Handle big size data processing which Panda unable to do

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
