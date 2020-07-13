# Pyspark

- Spark is another execution framework with MapReduce.
- Spark is based on computational engine.
- Lazy Evaluation. It has inbuilt process to optimize the execution. 
- Faster than Hadoop (Map Reduce frameworks). 
- Cost is high as it requires a lot of RAM for in-memory computation.

- Resilient Distributed Datasets (RDDs): 

      - It's a data structures that are the core building blocks of Spark. 
      - A RDD is an immutable.
      - No columnar structure (RDDs do not have a schema) and records are similar to a list
      
- DataFrames: 
      
      - Different then Pandas DataFrames
      - Spark DataFrame have all of the features of RDDs but also have a schema
      - Spark DataFrames comes with optimalization (distributed memory) unlike pandas.
      - Spark DataSets: Similar to DataFrames but not used in PySpark.
      
- Best Practices in Spark

      - Use Spark DataFrames (Not RDD)
      - Don’t call collect() on large RDDs
      - The join operation is one of the most expensive operations
      - Avoid groupByKey() on large RDDs
      - 
      -


Sample Code : https://github.com/amitmse/in_Python_/blob/master/Pyspark/Try_Pyspark.py

Cheat Sheet : https://s3.amazonaws.com/assets.datacamp.com/blog_assets/PySpark_Cheat_Sheet_Python.pdf

Tutorial    : https://www.guru99.com/pyspark-tutorial.html

Spark site  : http://spark.apache.org/docs/latest/api/python/

Kaggle      : https://www.kaggle.com/amitmse/pyspark/edit?rvi=1
