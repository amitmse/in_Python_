# Pyspark

- Lazy Evaluation. It has inbuilt process to optimize the execution. 
- Faster than Hadoop (Map Reduce frameworks). 
- Cost is high as it requires a lot of RAM for in-memory computation.
- RDDs
- Spark DataFrames comes with optimalization (distributed memory) unlike pandas.
- Best Practices in Spark
      - Use Spark DataFrames (Not RDD)
      - Don’t call collect() on large RDDs
      - The join operation is one of the most expensive operations
      - 
