# Pyspark
- Spark is based on computational engine.
- Lazy Evaluation. It has inbuilt process to optimize the execution. 
- Faster than Hadoop (Map Reduce frameworks). 
- Cost is high as it requires a lot of RAM for in-memory computation.

- Resilient Distributed Datasets (RDDs): 

      - It's a data structures that are the core building blocks of Spark. 
      - A RDD is an immutable.
      
- Spark DataFrames comes with optimalization (distributed memory) unlike pandas.
- Best Practices in Spark

      - Use Spark DataFrames (Not RDD)
      - Don’t call collect() on large RDDs
      - The join operation is one of the most expensive operations
      - Avoid groupByKey() on large RDDs
      - 

https://s3.amazonaws.com/assets.datacamp.com/blog_assets/PySpark_Cheat_Sheet_Python.pdf

https://www.guru99.com/pyspark-tutorial.html

http://spark.apache.org/docs/latest/api/python/
