#!/usr/bin/env python
# coding: utf-8

# # Big Data(Task 1)
# 
# 
# 
# ## Are there missing values? 
# There are currently no missing values in the dataset.
# Having filtered null rows using isNULL() function.
# 
# ### Discuss how you will deal with missing values, even if there are no missing values in this data set.
# 
# In order to deal with missing values, we can either utilize the replace(), fill() and drop() methods.. 
# 
# However within this sequence, I have implented the count for missing values using the isnan() Function. 
# The Column name (c) df.column is passed to isnan() function to output missing values.

# In[1]:


#import libraries
import pyspark
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#import spark session
from pyspark.sql import SparkSession
from pyspark.sql.functions import col,isnan, when, count,sum

#create spark session 
spark = SparkSession.builder.getOrCreate()


# In[2]:


#reads dataset
df = spark.read.csv("nuclear_plants_small_dataset.csv",inferSchema=True,header=True)


# In[4]:


#displays content of spark dataframe
df.printSchema()


# In[6]:


df.select(*(sum(col(c).isNull().cast("int")).alias(c) for c in df.columns)).show(vertical=True)


# In[ ]:


#Find missing values within a dataframe

from pyspark.sql.functions import lit

#df.count finds total value for rows
rows = df.count()

#df.describe returns description of dataframe 
# it can also be interpreted as returning statistical summary of dataframe 
summary = df.describe().filter(col("summary") == "count")
summary.select(*((lit(rows)-col(c)).alias(c) for c in df.columns)).show()


# In[ ]:


#Here we are finding the filtering rows with NULL values within nuclear dataset. 
#select() or where() functions of Spark dataframe performs the filtering of rows with NULL values when utilising isNULL(). 
#The statement below returns all rows that have null values on the state column.
df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]).show()


# In[ ]:


#We can also use .drop to dropping rows having null values
df.na.drop().show()

