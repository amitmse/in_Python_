################################## PYTHON 2.7 : Basic Code ##################################################
#while
		n= 5
		while n > 0 :
			print n 
			n = n-1
		print 'while loop done'
		print "now value of n is ", n

		while True:
			line = raw_input('user input..... ')	#ask user input 
			if line == 'done':	#check user input equal to 'done'
				break		#if user input is done then exit from loop
			print line		#if user input is not done then print the user input
		print "DONE"			#if user input is done then exit from loopif and print "DONE"

		while True:
			line = raw_input("user input ....")
			if line[0] =='#':	#check user input, if it starts with #
				continue 	#if user input does not start with # then run the loop again.
			if line == 'done':	#check user input equal to 'done'
				break		#if user input is 'done' then exit from loop
			print line		#if user input is not done and does not start with # then print the user input
		print 'DONE'			#Exit from loop and print "DONE"

#largest number
		largest = -1
		print 'starting numnber ', largest
		for num in [9,41,12,1,2,3,74,15]:
			if num > largest :
				largest = num
			print largest, num
		print 'Final number ', largest
		raw_input("press enter to exit ")

#Smallest number		
		small = None
		print 'first ', small
		for num in [9,99,3,2,1,88,5]:
			if small is None:
				small = num
			elif num < small :
				small = num		
			print small, num 
		print 'end', small 
		raw_input("close")
		
#sequence number
		zerok = 0
		print 'before ', zerok
		for list in [9,14,5,7,9,100]:
			zerok = zerok+1
			print zerok, list
		print 'after ', zerok
		raw_input("close ")

#Cumulative number
		start_point = 0
		print 'first number ', start_point
		for num_list in [9,7,10,15,1,3]:
			start_point = start_point + num_list
			print num_list, start_point
		print 'close ', start_point
		raw_input("close ")
		
#sequence number & Cumulative number
		s_n = 0
		cum = 0
		print 'start ' , s_n , cum
		for value in [9,4,5,7,100,4,0]:
			s_n = s_n + 1
			cum = cum + value
			print s_n, value, cum	
		print 'end ', s_n, cum, cum/s_n
		raw_input("close ")

#Sequence number & string 
		string = "heloo"
		index = 0
		while index < len(string):
			letter = string[index]
			print index, letter
			index  = index + 1

#Frequency 			
		string = "heloo"
		count = 0
		for letter in string :
			if letter =='o':
				count  = count +1
			print count

#string			
			string = "heloo"			
			print string 			#---> 'heloo'
			print string[:4]		#---> helo
			print string[3:]		#---> oo	#start with 3rd letter
			print string[:]			#---> heloo 	#all leter
			print string[0:5] 		#---> 'heloo'
			print string[0:4]		#---> helo 	# not 4, only 0 to 3
			first = string[1]		#---> 'e'	# it starts with 0.
			string.startswith('HELoo')	#---> True
			fin = string.find('o')		#---> 3
			string.replace('hel', 'HEL')	#---> 'HELoo'
			string_l = string.lower()	#---> 'heloo'
			string_u = string.upper()	#---> 'HELOO'
			'o' in string			#---> True
			
			data = 'From step.marq@gmail.com Sat Jan 5 09:14:16:2008'
			at = data.find('@')		#--> 14
			ed = data.find(' ',at)		#--> 24
			mail = data[at+1:ed]		#--> 'gmail.com'
			sp = data.find(' ')		#--> 4
			mail = data[sp+1:ed]		#--> step.marq@gmail.com

#read files
			fname  = raw_input("enter file name...") #---> enter file name...test_example.txt
			fread = open(fname)
			print (fread.read())

			text_f = open('test_example.txt')
			print(text_f.read())
			os.chdir("D:\\247NETWORKBACKUP\\Training\\Tut Video\\Python - Programming for Everybody\\Week 7")
			
			#total number of line
			file_n = open('test_example.txt')
			count = 0
			for line in file_n:
				count = count+1
			print 'total line', count	#---> 48

			#remove space and search for a word
			file_n = open('test_example.txt')
			for line in file_n:
				line = line.strip()
				if not 'Received:' in line:
					continue
			print line
			
			#remove space and search for a start word
			file_n = open('test_example.txt')
			for line in file_n:
				line = line.strip()
				if line.startswith('Received:'):
			print line
			
			fname = raw_input("enter file name ...") #--> enter file name ...test_example.txt
			fread = open(fname)
			for line in fread:
				if line.startswith("Received:"):
					count = count+1
			print count				#---> 9		

#Read file and count of a word			
			import os
			os.getcwd()
			os.chdir("D:\\247NETWORKBACKUP\\Training\\Tut Video\\Python - Programming for Everybody\\Week 7")
			fname = raw_input("enter file name")
			
			#IF name is not provided then use below file name
			if len(fname) == 0:
				fname = "f.txt"
			#Try : If code is able to open a file other wise it will move to except. 
			try:
				fread  = open(fname)
				#print (fread.read())
				raw_input("read")
			#Except: If code under try is not working then notify a message to user and exit from process.
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

#Most frequent word with count
			name = raw_input("file...")		#---> file...f.txt
			h = open(name)
			t = h.read()
			w = t.split()
			c = dict()
			for ws in w:
				c[ws] = c.get(ws,0)+1		#Get will give the value of key and it is not available 
								#then it will add 0 otherwise add 1.
			bc = None
			bw = None
			for ws, ct in c.items():	   	#items will provide the key, value. 
				if bc is None or ct > bc : 	#check value of key and compare with previous and 
								#logic is true then update the both variable.
					bw = ws
					bc = ct	
			print bw, bc				#---> Jan -> 12

#Convert into float			
			num_list = list()			#---> num_list	#---> []
			while True:
				input = raw_input("No. please...")
				if input == 'done':					
					break			#if input is done then exit from loop until then 
								#it will append all numbers
				value = float(input)
				num_list.append(value)
			print input				#---> done
			print value				#---> 4.0
			print sum(num_list)			#---> 109.0

			total = 0
			count = 0
			while True:
				input = raw_input("number please...")
				if input == 'done':
					break
				value = float(input)
				total = total + value
				count = count + 1	
			print total, count			#---> 6.0 2

#List			
			z= x+y					#---> [0, 1, 2, 4, 5, 6, 7, 8]
			x.append(111)				#---> [0, 1, 2, 'test', 111]
			0 in x					#---> True
			3 in x					#---> False
			3 not in x				#---> True
			x.sort()				#---> [0, 1, 2, 111, 'test']
			print len(x)				#---> 5
			print min(x)				#---> 0
			print max(x)				#---> test
			
			x = range(3)				#---> [0, 1, 2]
			y = range(4,9)				#---> [4, 5, 6, 7, 8]
			for i in range(len(x)):
				test = x[i]
				print test

#search a word				
			fname = open('f.txt')
			for line in fname:
				line = line.strip()
				word = line.split()
				print "working"		#debug. getting error due to blank line 
				if word[0] != 'From':
					continue
				print "end", word[2]
				
			fname = open('f.txt')
			for line in fname:
				line = line.strip()
				if line == '' :			
					continue 	#if line is blank then it will skip the operation 
							#and move to the top of the loop and reiterating the process.
				word = line.split()
				if word[0] != 'From':	#if first continue condition is not true then it will check this one.
					continue
				print "end", word[2]	#If both continue conditions are not true then it will execute this one.
				
			fname = open('f.txt')
			for line in fname:
				line = line.strip()
				word = line.split()
				if len(word) < 1 :
					continue
				if word[0] != 'From':
					continue
				print "end", word[2]

			fname = open('f.txt')
			for line in fname:
				line = line.strip()
				word = line.split()
				if word == [] :
					continue
				if word[0] != 'From':
					continue
				print "end", word[2]

#Frequency
			counts =dict()
			line =  raw_input("line please..") #---> line please..I am amit. going to delhi amit amit to
			words = line.split()
			print words	#---> ['I', 'am', 'amit.', 'going', 'to', 'delhi', 'amit', 'amit', 'to']
			for word in words:
				counts[word] = counts.get(word,0)+1 #Get will give the value of key and 
						#it is not available then it will add 0 otherwise add 1.
			print counts	#---> {'I': 1, 'delhi': 1, 'am': 1, 'to': 2, 'going': 1, 'amit': 2, 'amit.': 1}
				
			counts  = dict()
			name = ['amit', 'test', 1234, 'amit', 'test', 'test']
			for name in names:
				counts[name] = counts.get(name,0)+1
			print counts 	#---> {'test': 3, 'amit': 2}

			count  = dict()
			names = ['amit', 'test', 1234, 'amit', 'test', 'test']
			for list in names:
				if list not in count :
					count[list] = 1
				else :
					count[list] = count[list] + 1
			print count				#---> {'test': 3, 'amit': 2}
			print counts.get('name',0)		#---> 0
			print counts.get('amit','xxxxx')	#---> 2 
			print counts.get('amiT','xxxxx')	#---> xxxxx

			for key in counts:
				print key, counts[key]

#hash				
			p = dict()				#---> {}
			p['money'] = 12				#---> {'money': 12}
			p['xuz'] = 999				#---> {'money': 12, 'xuz': 999}
			p['money'] = p['money'] + 8		#---> {'money': 20, 'xuz': 999}
			p['text']='amit'			#---> {'money': 20, 'xuz': 999, 'text': 'amit'}
			p['text'] = p['text'] + 'xyz129'	#---> {'money': 20, 'xuz': 999, 'text': 'amitxyz129'}
			p['text']				#---> 'amitxyz129'


			d =dict()
			d['x1']=123
			d['x2'] = 999
			print d					#---> {'x2': 999, 'x1': 123}
			
			for (k,v) in d.items():
				print k,v
				
			#Sort
			tups=d.items()				#---> [('x2', 999), ('x1', 123)]
			t = sorted(d.items())			#---> [('x1', 123), ('x2', 999)]
			for (k,v) in sorted(d.items()):
				print k,v
				
			#append & sort
			temp = list()
			for k,v in d.items():
				temp.append((v,k))
			print temp				#---> [(999, 'x2'), (123, 'x1')]
			temp.sort(reverse=False)		#---> [(123, 'x1'), (999, 'x2')]

			#frequency of a word
			fname = open('f.txt')
			counts = dict()
			for line in fname :
				line = line.strip()
				words = line.split()
				for word in words:
					counts[word] = counts.get(word,0)+1
			lst = list()
			for key, val in counts.items():
				lst.append( (val, key) )
			lst.sort(reverse=True)
			for val, key in lst[:10]:
				print key, val

			print sorted( [ (v,k) for k,v in d.items() ]  )	#---> [(123, 'x1'), (999, 'x2')]
			
			
#Regex
			fname = open('f.txt')
			for line in fname:
				line = line.strip()
				if line.find('Received:')>=0:
					print line
					
			import re				# regex library
			fname = open('f.txt')
			for line in fname:
				line = line.strip()
				if re.search('Received:',line):	# call search function from regex 
					print line
					
			fname = open('f.txt')
			for line in fname:
				line = line.strip()
				if re.search('^Received:',line):
					print line		
					
			fname = open('f.txt')
			for line in fname:
				line = line.strip()
				if re.search('^M.*:',line):
					print line
					
			fname = open('f.txt')
			for line in fname:
				line = line.strip()
				if re.search('^M\S+:',line):
					print line
					
			fname = open('f.txt')
			for line in fname:
				line = line.strip()
				if re.findall('[0-9]+',line):
					print line
					
			fname = open('f.txt')
			for line in fname:
				line = line.strip()
				if re.search('@',line):
					y = re.findall('\S+@\S+', line)
					print y
					
			fname = open('f.txt')
			for line in fname:
				line = line.strip()
				if re.search('@',line):
					y = re.findall('^From (\S+@\S+)', line)
					print y
					
			txt = "From stephen.marquard@uct.ac.za Sat Jan  5 09:14:16 2008"	
			y = re.findall('@([^ ]*)',txt)		#---> ['uct.ac.za']			
			
#read the 4th column from a file and convert into number except column name
			num_list = list()
			coun = 0
			for line in open('pima-indians-diabetes.data.csv'):
				#print line
				words = line.split(',')
				#print words
				coun = coun + 1
				if coun > 1:
					col3 = int(words[3])
					num_list.append(col3)
				else :
					col3 = words[3]
				#print col3
				
				print coun
				print num_list
		
			#Min,MAX, Average, Sum
			num_list = list()
			coun = 0
			min_val = None
			max_val = None
			cum_val = 0
			for line in open('pima-indians-diabetes.data.csv'):
				words = line.split(',')
				coun = coun + 1
				
				if coun == 1:
					col_3 = words[3]		#Column Name
				else :
					col_3 = words[3]
					col_3 = int(words[3])

					cum_val = cum_val + col_3	#SUM

					if col_3 > max_val:
						max_val = col_3		#MAX

					if min_val is None :
						min_val = col_3		#MIN
					elif col_3 < min_val:
						min_val = col_3		#MIN
				num_list.append(col_3)			#List of value with column name
			#print everything
			print num_list, min_val, max_val, cum_val, coun, float(cum_val/(coun-1))

			#read column by line and put it in list
			#row_num = 0
			col_1 = list()
			col_2 = list()
			col_3 = list()
			col_4 = list()
			for line in open('pima-indians-diabetes.data.csv'):
				words = line.split(',')
				#row_num = row_num + 1
				#print 'current row number', row_num
				column_num  = 0
				for position  in range(len(words)):
					column_num = column_num + 1
					print 'current column number',  column_num
					if column_num == 1:
						col_position = words[position]
						col_1.append(col_position)
					elif column_num == 2:
						col_position = words[position]
						col_2.append(col_position)
					elif column_num == 3:
						col_position = words[position]
						col_3.append(col_position)
					elif column_num == 4:
						col_position = words[position]
						col_4.append(col_position)

			#Separate header name  
			row_num = 0
			col_0 = list()
			col_1 = list()
			col_2 = list()
			col_3 = list()
			col_4 = list()
			for line in open('pima-indians-diabetes.data.csv'):
				words = line.split(',')
				row_num = row_num + 1
				print 'current row number', row_num
				column_num  = 0
				for position  in range(len(words)):
					column_num = column_num + 1
					print 'current column number',  column_num
					if row_num == 1:
						col_position = words[position]
						col_0.append(col_position)
					else :
						if column_num == 1:
							col_position = words[position]
							col_1.append(col_position)
						elif column_num == 2:
							col_position = words[position]
							col_2.append(col_position)
						elif column_num == 3:
							col_position = words[position]
							col_3.append(col_position)
						elif column_num == 4:
							col_position = words[position]
							col_4.append(col_position)

			#Put it in Hash
			h=dict()
			row_num = 0
			col_0 = list()
			col_1 = list()
			col_2 = list()
			col_3 = list()
			col_4 = list()
			for line in open('pima-indians-diabetes.data.csv'):
				words = line.split(',')
				row_num = row_num + 1
				#print 'current row number', row_num
				column_num  = 0
				for position  in range(len(words)):
					column_num = column_num + 1
					#print 'current column number',  column_num
					if row_num == 1:
						col_position = words[position]
						col_0.append(col_position)
					else :
						if column_num == 1:
							col_position = words[position]
							col_1.append(col_position)
						elif column_num == 2:
							col_position = words[position]
							col_2.append(col_position)
						elif column_num == 3:
							col_position = words[position]
							col_3.append(col_position)
						elif column_num == 4:
							col_position = words[position]
							col_4.append(col_position)
				column_num  = 0
				for w in col_0:
					column_num = column_num + 1
					print w
					if column_num == 1:
						h[w] = col_1
					elif column_num == 2:
						h[w] = col_2
					elif column_num == 3:
						h[w] = col_3
					elif column_num == 4:
						h[w] = col_4
			#Tinku 
			record = {}
			header_read = False
			headers =[]
			for line in open('pima-indians-diabetes.data.csv'):
				line = line.strip().split(',')
				if not line:
					continue
				if not header_read:
					header_read = True
					headers = line
					record = dict([(k.strip(),[]) for k in line ])	#y = {a: a*a for a in x}
					continue
				for i, head in enumerate(headers):
					record[head].append(line[i].strip())
					#record.setdefault(head,[]).append(line[i].strip())


#################################################################################################################################
