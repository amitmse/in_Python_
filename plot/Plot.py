from __future__ import print_function
import subprocess
import sys
import pydotplus
import os
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from IPython.display import Image
import matplotlib.pylab as plt
from time import time
from operator import itemgetter
from collections import defaultdict

import matplotlib.pyplot as plt 

plt.style.use('seaborn')


#os.chdir(r"C:\Users\AMIT\Google Drive\Study\ML\07.Boosting")
#input_data = pd.read_csv('Dev1_Hilton_Model_Data.csv')
#df= input_data[:]
#df['target'] = np.where(df['reservation']>0, 'Yes', 'No')
#df = df.set_index('target')
#df.head()


os.chdir(r"C:\Users\AMIT\Google Drive\Study\ML\06.Random_Forest")
input_data = pd.read_csv('IV_Data.csv')
df = input_data[:]

df.head()

#####################################################################
###### Bar Graph using matplotlib ####################
#plt.bar(X Axis, Y Axis)
plt.bar(df.reservation_c,df['NOP_before_purchase'])
plt.xticks(rotation=70) 
plt.xlabel('Booking')
plt.ylabel('# Pages Visit in last time')
plt.title('Booking by #Page visits')
#plt.savefig(r"C:\Users\AMIT\Google Drive\Study\ML\07.Boosting\matplotlib_plotting_6.png", dpi=300,bbox_inches='tight') 
#plot plt.show();
plt.show();

########################################################################
###### Bar Graph using matplotlib ####################
#plt.bar(X Axis, Y Axis)
#plt.plot(df['NOP_before_purchase'])
#plt.plot(df['NOP_before_purchase'], 'ro')
#plt.plot(df['NOP_before_purchase'], 'bs')
#plt.scatter(df.reservation_c,df['NOP_before_purchase'])
plt.xticks(rotation=70) 
plt.xlabel('Booking')
plt.ylabel('# Pages Visit in last time')
plt.title('Booking by #Page visits')
#plt.savefig(r"C:\Users\AMIT\Google Drive\Study\ML\07.Boosting\matplotlib_plotting_6.png", dpi=300,bbox_inches='tight') 
#plot plt.show();
plt.show();
########################################################################
plt.hist(df['nop_last_visit'],rwidth=0.9,alpha=0.3,color='blue',bins=15,edgecolor='red')

#x and y-axis labels 
plt.xlabel('# Pages Visit in last time') 
plt.ylabel('# Visitors') 
#plot title 
plt.title('Booking by #Page visits') 
#save and display the plot 
#plt.savefig(r"C:\Users\AMIT\Google Drive\Study\ML\07.Boosting\matplotlib_plotting_10.png",dpi=300,bbox_inches='tight') 
plt.show();
########################################################################
fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(20,5)) 

#manipulating the first Axes 
ax[0].plot(week,df['nop_last_visit']) 
ax[0].set_xlabel('Week') 
ax[0].set_ylabel('Revenue') 
ax[0].set_title('Weekly income') 