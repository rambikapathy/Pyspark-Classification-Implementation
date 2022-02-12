#!/usr/bin/env python
# coding: utf-8

# ## Task 8 (Mapreduce)

# In[1]:


#import required libraries
import numpy
import pyspark


# In[2]:


##define the dataset to a variable
df_big = "nuclear_plants_big_dataset.csv"


# In[3]:


sc = pyspark.SparkContext()


# In[4]:


from pyspark import SparkContext


# In[5]:


#Import sprk context and session from pysprk
from pyspark.sql import SparkSession
from pyspark import SparkContext


# In[6]:


#Spark builder
spark = SparkSession.builder     .master("local")     .appName("nuclearBig")     .getOrCreate()
    #.master specifies local url to run on one thread
    #.appname appoints unique name identfier for  application, to be shown within Spark UI


# #### Map & Split

# In[7]:


#Here the dataset is imported into rdd.
#map function transforms using lambda on each of the 12 features within the RDD
#Splitting is occured prior to
#Comma is used to separate each of the 12 features column elements.
df_big2 = spark.sparkContext.textFile(df_big).map(lambda x: x.split(","))
#lambda constructs an inline function within rdd.
# .Map apply  transformation lambda on each element within dataset


# In[8]:


#We need to remove the status column, hence it is in string value and not integer
#This limits the lambda transformation taking place.
df_big_big2 = df_big2.map(lambda x: x[1:])
#convert_rdd variable take/obtains the title of each feature within the rdd
#[1,0] refering to the first column
convert_rdd = df_big_big2.take(1)[0]
#now the Status column is filtered using .filter and dropped from the rdd
convert2_rdd = df_big_big2.filter(lambda line: line != convert_rdd)


# In[12]:


#the for loop transposes each of the 12 features elements to compute mean,min and max within the rdd

for mprdce, en in enumerate(convert_rdd):
#%s the start to first column on dataset
#%en end of the datset
    print('%s:'  %en)
    
#lambda function implements to the elements with the 0th index(Power_range_sensor1), then the elements of 1st index until the n-th(11th) index (Vibrate_Sensor_4)    
    data_column = convert2_rdd.map(lambda x : float(x[mprdce]))
    mean = data_column.mean()
    print(' The mean is ',  mean)    
    minimum = data_column.min()
    print(" minimum value is ", minimum)
    maximum = data_column.max()
    print(" maximum value is ", maximum)


# In[ ]:




