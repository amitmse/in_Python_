'''
- Python's name is derived from the television series Monty Python's Flying Circus. 
   It is a British sketch comedy series created by the comedy group Monty Python and broadcast by the BBC from 1969 to 1974.
- Python is a widely used high-level programming language for general-purpose programming.
- Created by Guido van Rossum and first released in 1991
- Python has a design philosophy that emphasizes code readability (notably using whitespace indentation), 
   and a syntax that allows programmers to express concepts in fewer lines of code.

- Python are open source.
- Python developed with the sole purpose of teaching non-programmers how to program. 
    Readability is one of the key features of Python.
- I can use any algorithm out there and choose the one that suits the challenge at hand best.
- data scientists use to manage unstructured data. 

Pro:
- Free
- Easy to learn
- Scalability
- Big community
- Good in Machine learning
- Growing
- Fast in adapting new technique
- Data Handling/Big Data
- Visualization 
   In terms of graphical output, Python edges SAS out.  Both are verbose and tedious to get it to work. 
     Python wins because the output is cleaner and there are way more options.
- Python is much a more versatile object oriented language that has capabilities that SAS doesn’t have.  

Cons:
- High cost
- Slow in adapting new technique
'''

######################## Python 2.7 #####################################################################

#********************************************************************************************************
	Set path		# set PATH=%PATH%;C:\python27\	
	http://pythoncentral.io/add-python-to-path-python-is-not-recognized-as-an-internal-or-external-command/
	
	packages installed 	# help("modules")
	
	Install Library 	# First install pip then 
				# http://stackoverflow.com/questions/28142839/pip-install-numpy-python-2-7-fails-with-errorcode-1
				# python get-pip.py  	- pip install pip-tools / python -m pip install 
				#			--upgrade pip  / 'pip install --upgrade pip
				#  https://bootstrap.pypa.io/get-pip.py
				# Download Library 	- http://www.lfd.uci.edu/~gohlke/pythonlibs
				# download package 	- go to folder open cmd to install / 'https://docs.python.org/2/install/						
				# pip install numpy 	- easy_install -U setuptools / easy_install -U statsmodels 
				#			  /easy_install scipy-0.16.0-cp26-none-win_amd64.whl	 
				#			  /python setup.py install 

	Graphviz 		# Install software and add path C:\Program Files (x86)\Graphviz2.38\bin 
	
	
#*********************************************************************************************************	
	version	Library		# print igraph.__version__	
	python program.py	# Run it from the command line/double click on code
	Ctrl-D			# Quit Python
	help("range")		# help on function 
	import library		# First import library before calling it. import numpy
	import os		# import module
	print os		# check location of source code of a library
	os.getcwd()		# check current directory	----> import os, os.getcwd(), 'C:\\Python27'
	os.chdir()		# Change directory		----> os.chdir("D:\\247NETWORKBACKUP\\Training")
	os.mkdir()		# create a directory		----> os.mkdir("D:\\test"), os.mkdir("test")
	type(var)		# type of data. type(9) 	----> int
	dir(string_l)		# inbuild function of a object. ----> x= [1,2,9] , dir(x)
				# import pdb / 	pdb.set_trace()
	
#***********************************************************************************************************	
	Int(num)		# convert to integer, 	int('9.8') 	----> 9
	float(num)		# Convert to float, 	float(9) 	----> 9.0
	complex()		# convert to a complex number with real part and imaginary part
	complex(x, y)		# convert to a complex number with real  and imaginary
	str()			# Convert to string,	str(9)	----> '9'
	raw_input()		# user input			----> fname = raw_input("enter file name"), raw_input("done")
	lower()			# string_l = string.lower()	----> 'example'	string = 'ExaMple'
	upper()			# string_u = string.upper()	----> 'EXAMPLE'			
	find()			# string.find('M') 		----> 3, string.find('o') ----> -1 
				# (if not able to findout then it will mention -1)
	replace()		# string.replace('Exa', 'EXA')	----> 'EXAMple'
	lstrip()		# Remove space. string.lstrip()	----> 'HeLooo TeSt ' string = ' HeLooo TeSt '
	rstrip()		# Remove space. string.rstrip()	----> ' HeLooo TeSt'			
	strip()			# Remove space. string.strip()	----> 'HeLooo TeSt'
	startswith()		# string.startswith(' HeLooo')		----> True
	range()			# create a list, x = range(3)	----> [0, 1, 2]
	Min			# tiny = min("AmitTest") 	----> A
	MAX			# big = max("amit xyz")  	----> z
	iter()			# it = iter(s) 			----> it.next()									s = 'abc', next will give next word 
	
#**************************************************************************************************************

	==			# Equal to			----> if x==1 then do
	=			# Assign Value			----> x=1
	#			# comment the code		----> #comment this part
	'''			# Multiple line comment. 	----> ''' should be at both places  start and end
	\n			# Newline 			----> stuff = 'HelOo\nWoRlD' [it will print in two lines] 
	!=			# Not Equal to			----> if x != 1 then do

#************************************************************************************************************
	import nltk
	import sklearn
	print('The nltk version is {}.'.format(nltk.__version__))
	print('The scikit-learn version is {}.'.format(sklearn.__version__))

#****Numpy***************************************************************************************************
	structured to regular array : t=np.array(df.tolist())
	
#************** PANDAS **************************************************************************************
	read data 		: train_df = pd.read_csv('train.csv', header=0)
	numpy to pandas		: b=pd.DataFrame(a) # a is numpy array
	pandas to numpy		: a=df1.as_matrix() / b=df1.values (df1 is pandas dataframe)

	Sample			: train_df.head()			/ train_df.head(20) #Print 20 obs
	Column & Row		: train_df.shape
	Var list 		: train_df.columns
	Num var distribution	: train_df.describe()
		
	Freq			: train_df['Sex'].value_counts() 	/ train_df.groupby('Sex').size()
	Mean			: mean = df3['float_col'].mean()
		
	Columns Missing		: train_df.isnull().sum()		/ train_df.apply(lambda x: sum(x.isnull().values), axis = 0)
	Count Missing 		: sum([True for idx,row in train_df.iterrows() if any(row.isnull())])	
	Row Missing		: train_df.apply(lambda x: sum(x.isnull().values), axis = 1)	
	Fill Missing		: df3['float_col'].fillna(mean)
	Drop Missing		: df2.dropna()
		
	Filter column		: train_df.loc[:, 'City']			/ 	train_df.loc[:, ['City', 'State']]
	Filter Row & Column	: train_df.loc[0:2,'repayment':'portfolio_code']
	Filter on var 		: x = train_df[train_df['A'] == 3]
		
	Correlation 		: a=input_data.corr()
	Append data		: old_data_frame = pd.concat([old_data_frame,new_record])
	Drop a column		: input_data=input_data.drop('reservation',1)
	Reset Index		: h.reset_index() #fresh index
	Use columnas  Index	: data2 = t.set_index('two')
	Change column position	: rearrage the column list and use the list (a) to subset the data (Transpose[a])
	New record		: a=pd.Series(TP, index=['Actual-1'], name='Predicted-1'), 
				  b=pd.Series(FN, index=['Actual-1'], name='Predicted-0'), 
				  c=pd.Series(FP, index=['Actual-0'], name='Predicted-1'), 
				  d=pd.Series(TN, index=['Actual-0'], name='Predicted-0')
	Add column 		: train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1) 
				  #Number will be available for Cabin
				  dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
	Merge by column		: first=pd.concat([a, b], axis=1), second=pd.concat([c, d], axis=1)
	Append			: third = pd.concat([first, second])
	
#****** Convert SAS data into Pandas data frame ****************************************************************************
pip install sas7bdat
from sas7bdat import SAS7BDAT

foo = SAS7BDAT('Y:\KR_PL_IFRS9_01\Data\Final\def_check1.sas7bdat')
#converts to data frame
ds = foo.to_data_frame()
ds.describe()

#******Conditionals ( IF/ELSE IF/ELSE) ****************************************************************************
	
	angle = 5
	if angle > 0:
			print("Turning clockwise")
	elif angle < 0:
			print("Turning anticlockwise")
	else:
			print("Not turning at all"
	
#****** TRY & EXCEPT *********************************************************************************************
'''			      
	Feature to handle any unexpected error in your Python programs.
	If you have some suspicious code that may raise an exception, you can defend your program by 
	placing the suspicious code in a try: block. 
	After the try: block, include an except: statement, followed by a block of code which handles 
	the problem as elegantly as possible.
'''
	astr =  'helpp'
	try:
		istr = int(astr)
	except :
		istr = -1	
	print istr
	-1

	try :   hr = raw_input("hour"); hour = float(hr) ; rt = raw_input("rate") ; rate  = float(rt) ; pay = hour*rate ; print pay;
	except:	print "error, please check"

	try :          	
		hr 	= raw_input("hour"); 
		hour 	= float(hr) ; 
		rt 	= raw_input("rate") ; 
		rate  	= float(rt) ; 
		pay 	= hour*rate ; 
		print 	pay;
	except:			
		print 	"error, please check"

	
	import os
	os.getcwd()
	os.chdir("D:\\247NETWORKBACKUP\\Training\\Tut Video\\Python - Programming for Everybody\\Week 7")
	fname = raw_input("enter file name")
	if len(fname) == 0:
		fname = "f.txt"
	try:
		fread  = open(fname)
		#print (fread.read())
		raw_input("read")
	except:
		print 'file not exist'
		raw_input("bad")
		exit()
	count = 0
	raw_input("count")
	for line in fread:
		if line.startswith('Received:'):
			count =  count + 1
	print "Total count", count
	raw_input("done")

#****** Break **********************************************************************************************************
'''			      
	Break 	: Terminates the loop statement and transfers execution to the statement immediately following the loop.
	continue: Causes the loop to skip the remainder of its body and immediately retest its condition prior to reiterating.
	pass : The pass statement in Python is used when a statement is required syntactically but you do not want any 
		command or code to execute.
'''	
	while True:
		line = raw_input('user input..... ')	#ask user input 
		if line == 'done':	#check user input 
			break		#if user input is 'done' then exit from loop
		print line		#if user input is not 'done' then print the user input
	print "DONE"			#if user input is 'done' then exit from loopif and print "DONE"
	
#****** Reading a file ***************************************************************************************************
	#every time open a file. once code is executed it will go to end of line.
	
	f= open("test.txt")	# open a text file
	print(f.read())		# print a file
	f_r = f.read()		# read a file
	print len(f_r)		# length of a file
	print file_r[:20]	# Print first 20 letter
	
	file_n = open('test_example.txt')
	for line in file_n:
		line = line.strip()
		if line.startswith('Received:'):
			print line
	
	file_n = open('test_example.txt')
	for line in file_n:
		line = line.strip()
		if not 'Received:' in line:
			continue #Causes the loop to skip the remainder of its body and 
			      	 #immediately retest its condition prior to reiterating.
	print line
	
	
	import csv
	csv_file = open('data.csv')
	print (csv_file.read())

	print open('data.csv').readline()		# column name.
	
	allrows=list(csv.reader(open('data.csv')))
	allrows[0]					# column name
	header = list(csv.reader(open('data.csv')))[0]	# column name. its combination above two lines of code
	
	with open('data.csv') as f:
	reader = csv.reader(f)
	row1 = next(reader) 				#header of CSV file
	column = next(csv.reader(open('data.csv')))	#column name. its combination above three lines of code
	
	r=sum(1 for row in (csv.reader(open('data.csv'))))	#total number of record with header
	
	text = 'Sample Text to Save\nNew line!'
	
	saveFile = open('exampleFile.txt','w')		# notifies Python that you are opening this file, with the intention to write
	saveFile.write(text)				# actually writes the information
	saveFile.close()				# It is important to remember to actually close the file, 
			      				# otherwise it will hang for a while and
							# could cause problems in your script

	''' so here, generally it can be a good idea to start with a newline, since otherwise it will append data 
	    on the same line as the file left off. you might want that, but I'll use a new line. another option 
	    used is to first append just a simple newline then append what you want. 
	'''
	appendMe 	= '\nNew bit of information'
	appendFile 	= open('exampleFile.txt','a')
	appendFile.write(appendMe)
	appendFile.close()
	
#****** List [] ***********************************************************************************************
'''			      
	The list is a most versatile datatype available in Python which can be written as a 
	list of comma-separated values (items) between square brackets. 
	Important thing about a list is that items in a list need not be of the same type.
'''
		HELP		: help(list), help("list"), help([])
		Empty list 	: shopping_list  = [], 				zz=list()
		append		: a=[-1, 1, 1, 2, 3, 3, 4, 4, 6.777, 'test']	a.append(10) #=> [-1,1,1,2,3,3,4,4,6.777,'test',10]
		extend		: 
		insert		: a.insert(2,-1) = [1, 2, -1, 3, 4, 6.777, 'test', 1, 2, 3, 4] 
		remove		: a.remove(2)    = [1,    -1, 3, 4, 6.777, 'test', 1, 2, 3, 4] 
		index		: [1, 2, -1, 3, 4, 6.777, 'test', 1, 2, 3, 4]	a.index('test')=6
		count		: a=[1, 2,3, 4, 6.777, 'test', 1, 2, 3, 4]	a.count(1)= 2, 	a.count(9)= 0,  				      			
		sort		: [4, 3,  2, 1,   'test', 6.777, 4, 3, -1, 1]	a.sort()= [-1, 1, 1, 2, 3, 3, 4, 4, 6.777, 'test']			
		reverse		: [1, -1,    3, 4, 6.777, 'test', 1, 2, 3, 4] 	a.reverse()= [4, 3, 2, 1, 'test', 6.777, 4, 3, -1, 1]
		Delete		: [-1, 2, 3, 3, 4, 4, 6.777, 'test']		del a[0]= [2, 3, 3, 4, 4, 6.777, 'test']
		split()		: xc = "test 123  ixcv ", 			xcx = xc.split(), xcx = ['test', '123', 'ixcv']

		len(s) 		: length of s	      s=[1, 2, 3, 4], len(s), 4
		min(s) 		: smallest item of s  s=[1, 2, 3, 4], min(s), 1
		max(s) 		: largest item of s   s=[1, 2, 3, 4], max(s), 4
		s.index(i) 	: index of the first occurrence of i in s s=[1, 2, 3, 4], s.index(3) , 2
		s.count(i) 	: total number of occurrences of i in s   s=[1, 2, 3, 4], s.count(2) , 1
		Tuple to list	: list(tupl) = [1, 2]	tupl=(1, 2) tuple 

		x in s 		: True if an item of s is equal to x, else False s=[1,2,3,4]   , 0 in s , True
		x not in s 	: False if an item of s is equal to x, else True s=[1,2,3,4]   , 2 not in s, False
		s + t 		: the concatenation of s and t 			 t=[0,9]       , s+t    , [1, 2, 3, 4, 0, 9]
		s * n, n * s 	: n shallow copies of s concatenated 		 s=[1,2,3,4]   , s*2    , [1, 2, 3, 4, 1, 2, 3, 4]
		s[i] 		: ith item of s, origin 0 			 s=[1,2,3,4]   , s[2]   , 3
		s[i:j] 		: slice of s from i to j-1			 s=[1,2,3,4]   , s[2:4] , [3, 4]
			          #(not inclde 4th position, start is 0)	
			

	Loop: 
		for item in a:				# a= [1, 2, 3, 4, 6.777, 'test', 1, 2, 3, 4]
				print(item)		# 1, 2, 3, 4, 6.777, 'test', 1, 2, 3, 4 (in colomn)
		
		if 'check10' not in a:
				print("not avaialbale") 
				a.append("check10")
			print(a)			# not avaialbale, [1, 2, 3, 4, 6.777, 'test', 1, 2, 3, 4, 'check10']
		
		list2 = list1 				# 2 names refer to 1 ref, Changing one affects both
		list2 = list1[:] 			# Two independent copies, two refs
		
		y = [x+x for x in [1,4,6,9]]		# list comprehensions
		
#######****** Set --- set()/Curly braces **************************************************************************************
	#A set is an unordered collection with no duplicate elements. It will create unique. 

	create Set from list 	: fruit = set(basket) 		set(['orange', 'pear', 'apple']), basket = ['apple', 'orange', 'apple', 'pear', 'orange']
	unique item		: a 	= set('abracadabra') 	set(['a', 'r', 'b', 'c', 'd'])
	unique item		: b 	= set('alacazam') 	set(['a', 'c', 'z', 'm', 'l'])
	a but not in b		: a-b	= set(['r', 'd', 'b'])
	either a or b		: a | b = set(['a', 'c', 'r', 'd', 'b', 'm', 'z', 'l'])
	both a and b		: a & b = set(['a', 'c'])
	a or b but not both	: a ^ b = set(['r', 'd', 'b', 'm', 'z', 'l'])	

###***** Tuples () ******************************************************************************************************
'''			      
	A tuple is a sequence of immutable Python objects. Tuples are sequences, just like lists. 
	The differences between tuples and lists are, the tuples cannot be changed unlike lists and 
	tuples use parentheses, whereas lists use square brackets.
'''
	help(tuple), help(())
		index		:tup[2]       = 3							  	  					(1, 2, 3, 4, 6.777, 'test')
		count		:tup.count(2) = 1								  					(1, 2, 3, 4, 6.777, 'test')
		list to tuple 	:tup          = tuple(a)  (1, 2, 3, 4, 6.777, 'test') a=[1, 2, 3, 4, 6.777, 'test']

###****** Dictionary/Hash ---- {} ----- (first key and then value. only works if hash is define) **************************
'''			      
	Each key is separated from its value by a colon (:), the items are separated by commas, 
	and the whole thing is enclosed in curly braces. 
	Keys are unique within a dictionary while values may not be. 
	The values of a dictionary can be of any type, but the keys must be of an immutable data type 
	such as strings, numbers, or tuples.
'''
		empty hash		 : d = {}, d=dict()
		Create hash 		 : d["key1"] = "value1" 	{'key1': 'value1'}
		print a value of key	 : d["key1"]			'value1'
		create hash 		 : d = {'Name': 'Zara', 'Age': 7};  {'Age': 7, 'Name': 'Zara'}
		length of hash		 : len(d)			2						
		Convert in string	 : str(d)			"{'Age': 7, 'Name': 'Zara'}"
		copy of hash		 : a =d.copy()			{'Age': 7, 'Name': 'Zara'}
		new hash with keys	 : d = dict.fromkeys(seq, 10)	seq = ('name', 'age', 'sex'){'age': 10, 'name': 10, 'sex': 10}
		Get value of a key	 : d.get(name, 'not available')	(key, default=None)				10
		for key return value/add : d.setdefault('test', 99)	{'test': 99, 'age': 10, 'name': 10, 'sex': 10}
		Check key		 : d.has_key(4)			False
		key/value in list/tuple  : d.items()			[('age', 10), ('name', 10), ('sex', 10)]
		Get key from in list	 : d.keys()			['age', 'name', 'sex']	
		add 1 hash to another	 : d.update(s)	s = {1:4, 6:0}	{1: 4, 'age': 10, 'name': 10, 6: 0, 'sex': 10}
		Get values in list	 : d.values()			[4, 10, 10, 0, 10]

		delete a key 		 : del d["Age"]			{'Name': 'Zara'}
		remove all entries	 : d.clear()			{}
		delete entire dictionary : del d				


	How to add key value mapping from a dataset:
		latitudes  = {}
		longitudes = {}
		f = open("airports.dat")
		for row in csv.reader(f):
			airport_id = row[0]				# Key
				latitudes[airport_id]  	= float(row[6])	# Map value for each Key
				longitudes[airport_id] 	= float(row[7])	# Map value for each Key

	#Frequency 
		count  = dict()
		names = ['amit', 'test', 1234, 'amit', 'test', 'test']
		for list in names:
			if list not in count :
				count[list] = 1
			else :
				count[list] = count[list] + 1
		print count	#---> {'test': 3, 'amit': 2, 1234:1}
	
	#Biggest count
		name = raw_input("file...")	#----> file...f.txt
		h = open(name)
		t = h.read()
		w = t.split()
		c=dict()
		for ws in w:
			c[ws] = c.get(ws,0)+1	
		bc = None
		bw = None
		for ws, ct in c.items():
			if bc is None or ct > bc :
			bw = ws
			bc = ct	
		print bw, bc	#---> Jan, 12

####******Loop**************************************************************
'''			      
	#while	: Repeats a statement or group of statements while a given condition is TRUE. 
		  It tests the condition before executing the loop body.
	#for	: Executes a sequence of statements multiple times and abbreviates the code that manages the loop variable. 		
	#nested : You can use one or more loop inside any another while, for or do..while loop.
'''		
	For:
		for i in 2, 4, 6, 8:
			print(i)

		for i in range(1,10):
			print(i)

		total = 0
		for i in 1, 3, 7:
			total = total + i
		print(total)


	While:
		f 		= open("months.txt")
		next 		= f.read(1)		# intital value of next. read one character
		while next 	!= "":		# loop end when it finds empty string
				print(next)
			next	= f.read(1)	# second onward value of next

			
	largest = -1
	print 'starting numnber ', largest
	for num in [9,41,12,1,2,3,74,15]:
		if num > largest :
			largest = num
		print largest, num

	print 'Final number ', largest
	raw_input("press enter to exit ")
			
	string = 'heloo'
		'o' in string #---> True
		
####******Functions*********************************************************************************************
'''			      
	#A function is a block of organized, reusable code that is used to perform a single, related action. 
	#Functions provide better modularity for your application and a high degree of code reusing.
	#Python gives many built-in functions like print(), etc. but we can also create your own functions. 
	#These functions are called user-defined functions.
'''	
		def say_hello_to(name):
				print("Hello " + name)
				
		say_hello_to("Miranda")
		say_hello_to("Fred")

		sum = lambda arg1, arg2: arg1 + arg2;
		print "Value of total : ", sum( 10, 20 )

		
		def heloo():
		print "first function"
		print "learn"

		def greet(lang):
			if lang == 'es':
				print 'Hola'
			elif lang == 'fr':
				print 'Bonjoor'
			else :
				print 'Helooo'

		greet('en')    #------> Helooo

		def gt():
			return "heloo"
		print gt(), "AMIT"   #----> heloo AMIT 

		def adi(a,b):
			added = a+b
			return added
		x= adi(2,7)
		print x             #------> 9

		# Function definition is here
		def sum( arg1, arg2 ):
		   # Add both the parameters and return them."
		   total = arg1 + arg2
		   print "Inside the function : ", total
		   return total;

		# Now you can call sum function
		total = sum( 10, 20 );
		print "Outside the function : ", total 
		
		y = map(lambda x: x*x, [1,2,3])

		# use in-built functions
		import __builtin__
		def open(path):
			f = __builtin__.open(path, 'r')
			return UpperCaser(f)

####****** Reading a text/csv file ***************************************************************************
	TEXT:
		f = open("months.txt")

	CSV:
		import csv
		f = open("airports.dat")
		for row in csv.reader(f):
			if row[3] == "Australia" or row[3] == "Russia":
				print(row[1])

###***** Class ***************************************************************************************************
	#A user-defined prototype for an object that defines a set of attributes that characterize any object of the class. 
	#The attributes are data members (class variables and instance variables) and methods, accessed via dot notation.

		class exClass:
			x1=123
			y1='test'
			def fnMethod(self): #self is place holder for object which is exObject
				return "working"

		exClass
		exObject = exClass()
		exObject.x1		#---> 123
		exObject.fnMethod()

		class swine:
			def apples(self):
				print "testing"		
		obj1 = swine()
		obj1.apples()	#---> testing

		print swine.__doc__ #--> documentation, it will print all text.
		
		class new:
			def __init__(self):		# __init__ is initialization. it will automatically run this
				print "testing -1"
				print "tsting  -2"
		obj2 = new()
		testing -1
		tsting  -2

		#interpreters main program executes 
		if __name__ == "__main__":
			main()
		
#****** Plot a chart *******************************************************************************
	import matplotlib.pyplot as plt
	vals = [3,2,5,0,1]
	plt.plot(vals)
	plt.show()

#******************************************************************************************************
	#list of locally installed Python modules:
		import pip
		sorted(["%s==%s" % (i.key, i.version) for i in pip.get_installed_distributions()])

#** Numpy : http://docs.scipy.org/doc/numpy#/***********************************************************
		arange() 	#will create arrays with regularly incrementing values	
			        #np.arange(10)	array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
		#Read data from text file and convert into numpy
			from StringIO import StringIO
			data = "1, 2, 3\n4, 5, 6"
			np.genfromtxt(StringIO(data), delimiter=",")
		
	prettytable import PrettyTable	
		
	from Python_logistic_regression_with_L2_regularization import LogisticRegression
	print inspect.getsource(LogisticRegression.negative_lik)

#**** Binning *************************************************************************************************
### http://www.analyticsvidhya.com/blog/2016/01/12-pandas-techniques-python-data-manipulation/?utm_content=buffer7ef30&utm_medium=social&utm_source=linkedin.com&utm_campaign=buffer
	def binning(col, cut_points, labels=None):
	  #Define min and max values:
	  minval = col.min()
	  maxval = col.max()

	  #create list by adding min and max to cut_points
	  break_points = [minval] + cut_points + [maxval]

	  #if no labels provided, use default labels 0 ... (n-1)
	  if not labels:
		labels = range(len(cut_points)+1)

	  #Binning using cut function of pandas
	  colBin = pd.cut(col,bins=break_points,labels=labels,include_lowest=True)
	  return colBin

	#Binning age:
	cut_points = [90,140,190]
	labels = ["low","medium","high","very high"]
	data["LoanAmount_Bin"] = binning(data["LoanAmount"], cut_points, labels)
	print pd.value_counts(data["LoanAmount_Bin"], sort=False)

#**** Restart your python program ***********************************************************************************
	import sys
	import os
	"""
	python = sys.executable
	os.execl(python, python, * sys.argv)
	"""
	def restart_program():
		"""Restarts the current program.
		Note: this function does not return. Any cleanup action (like
		saving data) must be done before calling this function."""
		python = sys.executable
		os.execl(python, python, * sys.argv)
	if __name__ == "__main__":
		answer = raw_input("Do you want to restart this program ? ")
		if answer.lower().strip() in "y yes".split():
			restart_program()
		
#**** Upgrading all packages with pip *************************************************************

import pip
from subprocess import call

for dist in pip.get_installed_distributions():
    call("pip install --upgrade " + dist.project_name, shell=True)


### Pandas *************************************************************
#Add data type in data
a=list(input_data.columns)
for i in range(len(a)):
	input_data[a[i]] = input_data[a[i]].astype(str)

new_record = pd.DataFrame([list(input_data.dtypes)],columns=list(input_data.columns))
new_record=new_record.replace({'object':'str', 'float64':'float', 'int64':'int'})
#new_record=new_record.replace({'float64':'float'})
old_record=input_data.head()
old_data_frame = pd.concat([new_record,old_record])

#read csv file with column name and remove row no=1
raw_file = pd.read_csv(data_folder+'/'+special_value_file_name[file_nm]+'.csv',names=list(pd.read_csv(data_folder+'/'+special_value_file_name[file_nm]+'.csv',nrows=0).columns),skiprows=2)


listA = ["a","b"]
listB = ["b", "c"]
#Substract list
listC = [item for item in listB if item not in listA] #listC = list(set(listB).difference(listA))
#intersection
list(set(listB). intersection(listA))

#Transpose
special_value_data=pd.read_csv('blackboxamount.csv',names=['variable','type','value'])
special_value_data_1 = special_value_data.drop('value',1)
Transpose=special_value_data_1.set_index('variable').T

### Replace "(xxx)" with none ##################################################################
import re
x = "This is a sentence. (once a day) [twice a day]"
re.sub("([\(\[]).*?([\)\]])", "\g<1>\g<2>", x)		#'This is a sentence. () []'
x = "This is a sentence. (once a day) [twice a day]"
re.sub("[\(\[].*?[\)\]]", "", x)					#'This is a sentence.  '
filename = "Example_file_(extra_descriptor).ext"
re.sub(r'\([^)]*\)', '', filename)					#'Example_file_.ext'

#in Pandas
d={'old_var':['(0.16_0.0003)1.5|', '(0.13_0.0003)2.5|', '(0.16_0.034)inf|', '(0.52_0.35)missing|']}
df = pd.DataFrame(d)
df['new_var']=df['old_var'].str.replace(r"\(.*\)","")
df['new_var']=df['old_var'].str.replace(r"\([^)]*\)","")
'''
               old_var   new_var
0    (0.16_0.0003)1.5|      1.5|
1    (0.13_0.0003)2.5|      2.5|
2     (0.16_0.034)inf|      inf|
3  (0.52_0.35)missing|  missing|
'''

################################################################################################
'''
OOP:
	Encapsulation	: restrict access to methods and variables. This can prevent the data from being 
			  modified by accident and is known as encapsulation.
	Abstraction	: Simplifying complex reality by modeling classes appropriate to the problem.
	Inheritance	: Instead of starting from scratch, you can create a class by deriving it from a 
			  pre existing class by listing the parent class in parentheses after the new class name.
	Polymorphism	: Process of using an operator or function in different ways for different data input. 
	Overloading	: It means more than one method shares the same name in the class but having different signature. 
			  It is to add or extend more to methods behavior. compile time polymorphism.
	Overriding	: It means method of base class is re-defined in the derived class having same signature.	
			  It is to Change existing behavior of method. run time polymorphism.

http://freefeast.info/difference-between/difference-between-method-overloading-and-method-overriding-method-overloading-vs-method-overriding/
'''								
################################################################################################

## suppress warning message
## http://stackoverflow.com/questions/41658568/chunkize-warning-while-installing-gensim


import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim


if os.name == 'nt':
    logger.info("detected Windows; aliasing chunkize to chunkize_serial")

    def chunkize(corpus, chunksize, maxsize=0, as_numpy=False):
        for chunk in chunkize_serial(corpus, chunksize, as_numpy=as_numpy):
            yield chunk
else:
    def chunkize(corpus, chunksize, maxsize=0, as_numpy=False):
    """
    Split a stream of values into smaller chunks.
    Each chunk is of length `chunksize`, except the last one which may be smaller.
    A once-only input stream (`corpus` from a generator) is ok, chunking is done
    efficiently via itertools.

    If `maxsize > 1`, don't wait idly in between successive chunk `yields`, but
    rather keep filling a short queue (of size at most `maxsize`) with forthcoming
    chunks in advance. This is realized by starting a separate process, and is
    meant to reduce I/O delays, which can be significant when `corpus` comes
    from a slow medium (like harddisk).

    If `maxsize==0`, don't fool around with parallelism and simply yield the chunksize
    via `chunkize_serial()` (no I/O optimizations).

    >>> for chunk in chunkize(range(10), 4): print(chunk)
    [0, 1, 2, 3]
    [4, 5, 6, 7]
    [8, 9]

    """
    assert chunksize > 0

    if maxsize > 0:
        q = multiprocessing.Queue(maxsize=maxsize)
        worker = InputQueue(q, corpus, chunksize, maxsize=maxsize, as_numpy=as_numpy)
        worker.daemon = True
        worker.start()
        while True:
            chunk = [q.get(block=True)]
            if chunk[0] is None:
                break
            yield chunk.pop()
    else:
        for chunk in chunkize_serial(corpus, chunksize, as_numpy=as_numpy):
            yield chunk
			
################################################################################################
## create all possible combination of words and count
import itertools
from itertools import islice, izip
from collections import Counter
s='ABC'
print list(itertools.combinations(s,2))
[('A','B'),('B','C'),('A','C')]
print Counter(list(itertools.combinations(s,2)))
Counter({('B', 'C'): 1, ('A', 'B'): 1, ('A', 'C'): 1})
#Sort decsending
print Counter(list(itertools.combinations(s,2))).most_common()

#https://stackoverflow.com/questions/12488722/counting-bigrams-pair-of-two-words-in-a-file-using-python
import re
from itertools import islice, izip
from collections import Counter
words = re.findall("\w+", "the quick person did not realize his speed and the quick person bumped")
print Counter(izip(words, islice(words, 1, None)))
Counter({('the', 'quick'): 2, ('quick', 'person'): 2, ('person', 'did'): 1, ('did', 'not'): 1, ('not', 'realize'): 1, ('and', 'the'): 1, ('speed', 'and'): 1, ('person', 'bumped'): 1, ('his', 'speed'): 1, ('realize', 'his'): 1})

#n-gram:
from itertools import tee, islice
def ngrams(lst, n):
	  tlst = lst
	  while True:
			a, b = tee(tlst)
			l = tuple(islice(a, n))
			if len(l) == n:
					yield l
					next(b)
					tlst = b
			else:
					break

Counter(ngrams(words, 3))

## Print string without u [u'String']
print [str(cachedStopWords[x]) for x in range(len(cachedStopWords))]
## Update stopword
from nltk.corpus import stopwords
cachedStopWords = set(stopwords.words("english"))
cachedStopWords.update((set([x.lower() for x in ['and','I','A','And','So','arnt','This','When','It','many','Many','so','cant','Yes','yes','No','no','These','these']]))

########################################################################################################################################
