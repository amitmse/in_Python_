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
import pandas_bokeh                               # This is Pandas-bokeh
pd.set_option('plotting.backend', 'pandas_bokeh') # This is Pandas-bokeh
#########################################
from bokeh.plotting import figure, output_file, show
from bokeh.io import output_notebook
from bokeh.models import HoverTool, BoxSelectTool, ColumnDataSource #For enabling tools
from bokeh.sampledata.autompg import autompg_clean as df
from bokeh.transform import factor_cmap
from bokeh.layouts import column
from bokeh.models import CustomJS, ColumnDataSource, Slider
# Go to => https://docs.bokeh.org/en/latest/docs/user_guide.html#userguide => click on "Tutorial"
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
######### Pandas bokeh ###########################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_bokeh                               # This is Pandas-bokeh
pd.set_option('plotting.backend', 'pandas_bokeh') # This is Pandas-bokeh

###################################
# Line Plots #
df = pd.DataFrame(np.random.randn(1000, 4), index=df.index, columns=list('ABCD'))
df = df.cumsum()
plt.figure()
df.plot()
###################################
# Scatter Plots with Pandas-bokeh #
df = pd.DataFrame(np.random.rand(50, 4), columns=['a', 'b', 'c', 'd'])
df.plot.scatter(x='a', y='b')
###################################
# Histogram #
df = pd.DataFrame({'a': np.random.randn(1000) + 1, 'b': np.random.randn(1000),'c': np.random.randn(1000) - 1}, columns=['a', 'b', 'c'])
plt.figure()
df.plot.hist(stacked=True, bins=20)
###################################
# Histogram with Pandas-bokeh #
df = pd.DataFrame({'a': np.random.randn(1000) + 1, 'b': np.random.randn(1000),'c': np.random.randn(1000) - 1}, columns=['a', 'b', 'c'])
plt.figure();
df.plot.hist(alpha=0.5)
############################################################################################################
######### bokeh ###########################################
X_axis='annual_inc'
Y_axis='recoveries'
##################
X_axis_cat_1='home_ownership'
X_axis_cat_2='income_category'
##################
# prepare some data
x= df[X_axis].values.tolist()
y= df[Y_axis].values.tolist()
#######################################
# output to static HTML file (with CDN resources)
output_notebook()
output_file("dot.html")

TOOLS = [BoxSelectTool(), HoverTool()]

# create a new plot with the tools above, and explicit ranges
#p = figure()
p = figure(tools=TOOLS)
# add a circle renderer with vectorized colors and sizes
####################################################
#p.line(x,y)                    #Line
p.circle(x,y,width=8)           #dot
#p.triangle(x,y)                #triangle
#p.vbar(x,top=y, width=2)       #bar
#p.hbar(x,right=y, height =2)   #bar
#p.quad(x,y)                    #triangle
#p.square(x, y, size=10)

# show the results
show(p)
####################################################
#### Slider
#https://hub.gke.mybinder.org/user/bokeh-bokeh-notebooks-zzw15vsm/notebooks/tutorial/06%20-%20Linking%20and%20Interactions.ipynb
#https://docs.bokeh.org/en/latest/docs/user_guide.html#userguide
#Linking and Interactions
from bokeh.layouts import column
from bokeh.models import CustomJS, ColumnDataSource, Slider

x = [x*0.005 for x in range(0, 201)]

source = ColumnDataSource(data=dict(x=x, y=x))

plot = figure(plot_width=400, plot_height=400)
plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)

slider = Slider(start=0.1, end=6, value=4, step=.1, title="power")

update_curve = CustomJS(args=dict(source=source, slider=slider), code="""
    var data = source.data;
    var f = slider.value;
    var x = data['x']
    var y = data['y']
    for (var i = 0; i < x.length; i++) {
        y[i] = Math.pow(x[i], f)
    }
    
    // necessary becasue we mutated source.data in-place
    source.change.emit();
""")
slider.js_on_change('value', update_curve)

show(column(slider, plot))

# Slider : https://s3.amazonaws.com/assets.datacamp.com/production/course_2244/slides/ch4_slides.pdf
# Real Time: https://stackoverflow.com/questions/37724660/streaming-two-line-graphs-using-bokeh

############################################################################################################

############################################################################################################

############################################################################################################

############################################################################################################
