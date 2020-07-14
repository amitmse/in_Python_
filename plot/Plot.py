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
###### matplotlib ####################################
X_axis='home_ownership'
Y_axis='recoveries'

plt.bar(df[X_axis],df[Y_axis])

plt.xticks(rotation=70) 
plt.xlabel(X_axis)
plt.ylabel(Y_axis)
plt.title(X_axis + ' by ' + Y_axis)
#plt.savefig(r"C:\Users\AMIT\Google Drive\Study\ML\07.Boosting\matplotlib_plotting_6.png", dpi=300,bbox_inches='tight') 
#plot plt.show();
plt.show();
######################
plt.bar(df[X_axis],df[Y_axis])
plt.plot(df[X_axis],df[Y_axis])
plt.scatter(df[X_axis],df[Y_axis])
plt.hist(df[Y_axis])
plt.pie(df[Y_axis])
########################################################


