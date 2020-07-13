# https://www.kaggle.com/fatmakursun/pyspark-ml-tutorial-for-beginners

!pip install pyspark

# ------------------------------------------------------------------------------------------------------------------------------------------------
# https://www.kaggle.com/fatmakursun/pyspark-ml-tutorial-for-beginners
    
import os
import pandas as pd
import numpy as np

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext

from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.sql.functions import udf, col

from pyspark.ml.regression import LinearRegression
from pyspark.mllib.evaluation import RegressionMetrics

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, CrossValidatorModel
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.evaluation import RegressionEvaluator

#########################################################

import seaborn as sns
import matplotlib.pyplot as plt

#######################################################

# Visualization
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

pd.set_option('display.max_columns', 200)
pd.set_option('display.max_colwidth', 400)

from matplotlib import rcParams
sns.set(context='notebook', style='whitegrid', rc={'figure.figsize': (18,4)})
rcParams['figure.figsize'] = 18,4

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

############################
# setting random seed for notebook reproducability
rnd_seed=23
np.random.seed=rnd_seed
np.random.set_state=rnd_seed

############### 2. Creating the Spark Session ####################
spark = SparkSession.builder.master("local[2]").appName("Linear-Regression-California-Housing").getOrCreate()
spark
############################################
sc = spark.sparkContext
sc
############################################
sqlContext = SQLContext(spark.sparkContext)
sqlContext
############################################
HOUSING_DATA = '../input/cal_housing.data'
#############################################
# define the schema, corresponding to a line in the csv data file.
schema = StructType([
    StructField("long", FloatType(), nullable=True),
    StructField("lat", FloatType(), nullable=True),
    StructField("medage", FloatType(), nullable=True),
    StructField("totrooms", FloatType(), nullable=True),
    StructField("totbdrms", FloatType(), nullable=True),
    StructField("pop", FloatType(), nullable=True),
    StructField("houshlds", FloatType(), nullable=True),
    StructField("medinc", FloatType(), nullable=True),
    StructField("medhv", FloatType(), nullable=True)]
)
###########################################################
sqlContext = SQLContext(spark.sparkContext)
sqlContext
############ Load The Data From a File Into a Dataframe #########
HOUSING_DATA = '../input/hausing-data/cal_housing.data'
######################################################################
# define the schema, corresponding to a line in the csv data file.
schema = StructType([
    StructField("long", FloatType(), nullable=True),
    StructField("lat", FloatType(), nullable=True),
    StructField("medage", FloatType(), nullable=True),
    StructField("totrooms", FloatType(), nullable=True),
    StructField("totbdrms", FloatType(), nullable=True),
    StructField("pop", FloatType(), nullable=True),
    StructField("houshlds", FloatType(), nullable=True),
    StructField("medinc", FloatType(), nullable=True),
    StructField("medhv", FloatType(), nullable=True)]
)

##################################################################
# Load housing data
housing_df = spark.read.csv(path=HOUSING_DATA, schema=schema).cache()
# read CSV
# test = spark.read.csv('../input/test.csv',header = True,inferSchema=True)

######################################################################
# Inspect first five rows
housing_df.take(5)
###################################################################
# Show first five rows
housing_df.show(5)
# In pandas dataframe
housing_df.limit(5).toPandas()

###################################################################
# show the dataframe columns
housing_df.columns
###################################################################
# show the schema of the dataframe
housing_df.printSchema()
######################## 4. Data Exploration ######################
# run a sample selection
housing_df.select('pop','totbdrms').show(10)
###################################################################
# group by housingmedianage and see the distribution
result_df = housing_df.groupBy("medage").count().sort("medage", ascending=False)
result_df.show(10)
result_df.toPandas().plot.bar(x='medage',figsize=(14, 6))
############# 4.2 Summary Statistics ##############################
(housing_df.describe().select(
                    "summary",
                    F.round("medage", 4).alias("medage"),
                    F.round("totrooms", 4).alias("totrooms"),
                    F.round("totbdrms", 4).alias("totbdrms"),
                    F.round("pop", 4).alias("pop"),
                    F.round("houshlds", 4).alias("houshlds"),
                    F.round("medinc", 4).alias("medinc"),
                    F.round("medhv", 4).alias("medhv"))
                    .show())
############## 5. Data Preprocessing ##############################
# Adjust the values of `medianHouseValue`
housing_df = housing_df.withColumn("medhv", col("medhv")/100000)
# Show the first 2 lines of `df`
housing_df.show(2)
############## 6. Feature Engineering #############################
housing_df.columns

# Add the new columns to `df`
housing_df = (housing_df.withColumn("rmsperhh", F.round(col("totrooms")/col("houshlds"), 2))
                       .withColumn("popperhh", F.round(col("pop")/col("houshlds"), 2))
                       .withColumn("bdrmsperrm", F.round(col("totbdrms")/col("totrooms"), 2)))

# Inspect the result
housing_df.show(5)

# Re-order and select columns
housing_df = housing_df.select("medhv", 
                              "totbdrms", 
                              "pop", 
                              "houshlds", 
                              "medinc", 
                              "rmsperhh", 
                              "popperhh", 
                              "bdrmsperrm")

featureCols = ["totbdrms", "pop", "houshlds", "medinc", "rmsperhh", "popperhh", "bdrmsperrm"]

# put features into a feature vector column
assembler = VectorAssembler(inputCols=featureCols, outputCol="features") 
assembled_df = assembler.transform(housing_df)
assembled_df.show(10, truncate=False)

# Initialize the `standardScaler`
standardScaler = StandardScaler(inputCol="features", outputCol="features_scaled")
# Fit the DataFrame to the scaler
scaled_df = standardScaler.fit(assembled_df).transform(assembled_df)
# Inspect the result
scaled_df.select("features", "features_scaled").show(10, truncate=False)
####################################################################################
### Building A Machine Learning Model With Spark ML ###############
# Split the data into train and test sets
train_data, test_data = scaled_df.randomSplit([.8,.2], seed=rnd_seed)
# Initialize `lr`
lr = (LinearRegression(featuresCol='features_scaled', labelCol="medhv", predictionCol='predmedhv',maxIter=10, regParam=0.3, elasticNetParam=0.8, standardization=False))
# Fit the data to the model
linearModel = lr.fit(train_data)

## Evaluating the Model ##
# Coefficients for the model
linearModel.coefficients
featureCols
# Intercept for the model
linearModel.intercept
coeff_df = pd.DataFrame({"Feature": ["Intercept"] + featureCols, "Co-efficients": np.insert(linearModel.coefficients.toArray(), 0, linearModel.intercept)})
coeff_df = coeff_df[["Feature", "Co-efficients"]]
coeff_df
# Generate predictions
predictions = linearModel.transform(test_data)
# Extract the predictions and the "known" correct labels
predandlabels = predictions.select("predmedhv", "medhv")
predandlabels.show()

# Get the RMSE: The RMSE measures how much error there is between two datasets comparing a predicted value and an observed or known value. 
# The smaller an RMSE value, the closer predicted and observed values are.
print("Root Mean Squared Error(RMSE): {0}".format(linearModel.summary.rootMeanSquaredError))
print("Mean Absolute Error(MAE): {0}".format(linearModel.summary.meanAbsoluteError))
# Get the R2: The R2 ("R squared") or the coefficient of determination is a measure that shows how close the data are to the fitted regression line. 
# This score will always be between 0 and a 100% (or 0 to 1 in this case), where 0% indicates that the model explains none of the variability of the response data around its mean, 
# and 100% indicates the opposite: it explains all the variability. That means that, in general, the higher the R-squared, the better the model fits our data.
print("R Squared (R2): {0}".format(linearModel.summary.r2))
###### Using the RegressionEvaluator from pyspark.ml package:
evaluator = RegressionEvaluator(predictionCol="predmedhv", labelCol='medhv', metricName='rmse')
print("RMSE: {0}".format(evaluator.evaluate(predandlabels)))
evaluator = RegressionEvaluator(predictionCol="predmedhv", labelCol='medhv', metricName='mae')
print("MAE: {0}".format(evaluator.evaluate(predandlabels)))
evaluator = RegressionEvaluator(predictionCol="predmedhv", labelCol='medhv', metricName='r2')
print("R2: {0}".format(evaluator.evaluate(predandlabels)))
# mllib is old so the methods are available in rdd
metrics = RegressionMetrics(predandlabels.rdd)
print("RMSE: {0}".format(metrics.rootMeanSquaredError))
print("MAE: {0}".format(metrics.meanAbsoluteError))
print("R2: {0}".format(metrics.r2))
#spark.stop()
###################################################################
         # END OF CODE #
###################################################################
# ------------------------------------------------------------------------------------------------------------------------------------------------
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import numpy as np
import pandas as pd 

# Read data from a competition: 
#https://www.kaggle.com/c/hawaiiml0/discussion/53539
#https://www.kaggle.com/getting-started/25930
df = pd.read_csv('../input/loandata/Loan payments data.csv') # load data from csv
df.head() # Gives first 5 rows

'''from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
spark = SparkSession.builder.getOrCreate()
'''
# ------------------------------------------------------------------------------------------------------------------------------------------------

# https://www.kaggle.com/suyashgulati/using-pyspark-randomforest-crossvalidatn-gridsrch
# https://www.kaggle.com/vchulski/tutorial-collaborative-filtering-with-pyspark
# https://www.kdnuggets.com/2018/02/google-colab-free-gpu-tutorial-tensorflow-keras-pytorch.html
# https://dzone.com/articles/introduction-to-spark-with-python-pyspark-for-begi
# https://www.guru99.com/pyspark-tutorial.html


# ------------------------------------------------------------------------------------------------------------------------------------------------

# https://www.kaggle.com/suyashgulati/using-pyspark-randomforest-crossvalidatn-gridsrch

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
#print(os.listdir("../input"))
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

sc = SparkContext(appName = "forest_cover")
spark = SparkSession.Builder().getOrCreate()

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

train = spark.read.csv('../input/train.csv',header = True,inferSchema=True)
test = spark.read.csv('../input/test.csv',header = True,inferSchema=True)

train.limit(5).toPandas()
test.count()

###############################################################3
train_mod = train.withColumn("HF1", train.Horizontal_Distance_To_Hydrology + train.Horizontal_Distance_To_Fire_Points) \
.withColumn("HF2", abs(train.Horizontal_Distance_To_Hydrology - train.Horizontal_Distance_To_Fire_Points)) \
.withColumn("HR1", abs(train.Horizontal_Distance_To_Hydrology + train.Horizontal_Distance_To_Roadways)) \
.withColumn("HR2", abs(train.Horizontal_Distance_To_Hydrology - train.Horizontal_Distance_To_Roadways)) \
.withColumn("FR1", abs(train.Horizontal_Distance_To_Fire_Points + train.Horizontal_Distance_To_Roadways)) \
.withColumn("FR2", abs(train.Horizontal_Distance_To_Fire_Points - train.Horizontal_Distance_To_Roadways)) \
.withColumn("ele_vert", train.Elevation - train.Vertical_Distance_To_Hydrology) \
.withColumn("slope_hyd", pow((pow(train.Horizontal_Distance_To_Hydrology,2) + pow(train.Vertical_Distance_To_Hydrology,2)),0.5)) \
.withColumn("Mean_Amenities", (train.Horizontal_Distance_To_Fire_Points + train.Horizontal_Distance_To_Hydrology + train.Horizontal_Distance_To_Roadways)/3) \
.withColumn("Mean_Fire_Hyd", (train.Horizontal_Distance_To_Fire_Points + train.Horizontal_Distance_To_Hydrology)/2)

test_mod = test.withColumn("HF1", test.Horizontal_Distance_To_Hydrology + test.Horizontal_Distance_To_Fire_Points) \
.withColumn("HF2", abs(test.Horizontal_Distance_To_Hydrology - test.Horizontal_Distance_To_Fire_Points)) \
.withColumn("HR1", abs(test.Horizontal_Distance_To_Hydrology + test.Horizontal_Distance_To_Roadways)) \
.withColumn("HR2", abs(test.Horizontal_Distance_To_Hydrology - test.Horizontal_Distance_To_Roadways)) \
.withColumn("FR1", abs(test.Horizontal_Distance_To_Fire_Points + test.Horizontal_Distance_To_Roadways)) \
.withColumn("FR2", abs(test.Horizontal_Distance_To_Fire_Points - test.Horizontal_Distance_To_Roadways)) \
.withColumn("ele_vert", test.Elevation - test.Vertical_Distance_To_Hydrology) \
.withColumn("slope_hyd", pow((pow(test.Horizontal_Distance_To_Hydrology,2) + pow(test.Vertical_Distance_To_Hydrology,2)),0.5)) \
.withColumn("Mean_Amenities", (test.Horizontal_Distance_To_Fire_Points + test.Horizontal_Distance_To_Hydrology + test.Horizontal_Distance_To_Roadways)/3) \
.withColumn("Mean_Fire_Hyd", (test.Horizontal_Distance_To_Fire_Points + test.Horizontal_Distance_To_Hydrology)/2)
#######################################################
train_columns = test_mod.columns[1:]
train_mod.printSchema
#######################################################
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler()\
.setInputCols(train_columns)\
.setOutputCol("features")
train_mod01 = assembler.transform(train_mod)
##################################################
train_mod02 = train_mod01.select("features","Cover_Type")
test_mod01 = assembler.transform(test_mod)
test_mod02 = test_mod01.select("Id","features")
###################################################
from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel
rfClassifer = RandomForestClassifier(labelCol = "Cover_Type", numTrees = 100)
from pyspark.ml import Pipeline
pipeline = Pipeline(stages = [rfClassifer])
##################################################################
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

paramGrid = ParamGridBuilder()\
   .addGrid(rfClassifer.maxDepth, [1, 2, 4, 5, 6, 7, 8])\
   .addGrid(rfClassifer.minInstancesPerNode, [1, 2, 4, 5, 6, 7, 8])\
   .build()

evaluator = MulticlassClassificationEvaluator(labelCol = "Cover_Type", predictionCol = "prediction", metricName = "accuracy") 

crossval = CrossValidator(estimator = pipeline,
                          estimatorParamMaps = paramGrid,
                          evaluator = evaluator,
                          numFolds = 10)

cvModel = crossval.fit(train_mod02)
cvModel.avgMetrics


# ------------------------------------------------------------------------------------------------------------------------------------------------

df['watching_percentage'].hist()

# https://nbviewer.jupyter.org/gist/vykhand/1f2484ff14fbbf805234160cf90668b4
# https://www.kaggle.com/getting-started/160043#893381
# https://www.kaggle.com/amalshajiprof/handling-huge-datasets-using-pyspark-time
# https://www.kaggle.com/kkhandekar/pyspark-vs-scikit
# https://www.kaggle.com/filipnowak/pyspark-abc-bike-trips-data-analysis

# ------------------------------------------------------------------------------------------------------------------------------------------------
