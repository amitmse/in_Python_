##########################################
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
##########################################
import matplotlib.pyplot as plt 
##########################################
import seaborn as sns
##########################################
from plotnine import *
##########################################

##########################################
plt.style.use('seaborn')
##########################################

#os.chdir(r"C:\Users\AMIT\Google Drive\Study\ML\07.Boosting")
#input_data = pd.read_csv('Dev1_Hilton_Model_Data.csv')
#df= input_data[:]
#df['target'] = np.where(df['reservation']>0, 'Yes', 'No')
#df = df.set_index('target')
#df.head()
##########################################

os.chdir(r"C:\Users\AMIT\Google Drive\Study\ML\06.Random_Forest")
input_data = pd.read_csv('IV_Data.csv')
df = input_data[:]
df.head()

############################################################################################################
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
###############################################################################################################
######### Seaborn #####################################
#X_axis='home_ownership'
X_axis='annual_inc'
Y_axis='recoveries'

sns.barplot(x=X_axis, y=Y_axis, data = df)                        # Bar plot
sns.pointplot(x=X_axis, y=Y_axis, data = df)                      # Line with dot
sns.relplot(x=X_axis, y=Y_axis, data = df)                        # scatter plot
sns.distplot(df[Y_axis], kde=False, rug = True)                   # Histograms 
sns.countplot(x=X_axis, data = df)                                # Bar chart. Show the counts of observations in each categorical bin.
sns.jointplot(x=X_axis, y=Y_axis, data = df)                      # bivariate and univariate graphs
sns.catplot(x=X_axis, y=Y_axis, data = df)                        # scatterplots (Categorical)
sns.regplot(x=X_axis, y=Y_axis, data = df)                        # linear regression model fit
sns.lmplot(x=X_axis, y=Y_axis, data = df)                         # Combination of regplot & FacetGrid. 
sns.residplot(x=X_axis, y=Y_axis, data=df);                       # Residual values
############################################################################################################
######### ggplot ###########################################
X_axis='annual_inc'
Y_axis='recoveries'
##################
X_axis_cat_1='home_ownership'
X_axis_cat_2='income_category'
##################
#https://medium.com/@gscheithauer/data-visualization-in-python-like-in-rs-ggplot2-bc62f8debbf5
##################

#ggplot(df, aes(x=X_axis , y=Y_axis )) + geom_point()                               # Scatter
#ggplot(df, aes(x=X_axis_cat_1)) + geom_bar()                                       # Bar
#ggplot(df, aes(x=X_axis_cat_1)) + geom_bar() + coord_flip()                        # Bar: Flip
#ggplot(df, aes(x=X_axis_cat_1)) + geom_bar(stat = 'count')                         # Bar: Count
#ggplot(df, aes(x=X_axis_cat_1, fill = X_axis_cat_2)) + geom_bar(stat = 'count')    # Bar: Stack with count
#ggplot(df, aes(x=X_axis , y=Y_axis )) + geom_line()                                # Line
#ggplot(df, aes(x=X_axis )) + geom_histogram()                                      # Histogram
############################################################################################################

############################################################################################################

############################################################################################################

############################################################################################################

############################################################################################################

############################################################################################################
