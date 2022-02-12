#!/usr/bin/env python
# coding: utf-8

# # Big Data(Task 2)
# 

# In[1]:


#import libraries
import pyspark
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler

#import spark session
from pyspark.sql import SparkSession
from pyspark.sql.functions import col,isnan, when, count,sum

#create spark session 
spark = SparkSession.builder.getOrCreate()


# In[2]:


#reads dataset
df = spark.read.csv("nuclear_plants_small_dataset.csv",inferSchema=True,header=True)


# In[3]:


#We need to seperate 2 group of new subjects

#subsets Normal columns of dataframe according to its labels in the specified index
dfNormal = df.filter(df.Status == 'Normal' )

#Abnormal
dfAbnormal = df.filter(df.Status == 'Abnormal' )


# In[8]:


#Getting median using pandas

df.where(df.Status=="Normal").select('Power_range_sensor_1').mean()
df.where(df.Status=="Normal").select('Power_range_sensor_2').median()
df.where(df.Status=="Normal").select('Power_range_sensor_3 ').median()
df.where(df.Status=="Normal").select('Power_range_sensor_4').median()
df.where(df.Status=="Normal").select('Pressure _sensor_1').median()
df.where(df.Status=="Normal").select('Pressure _sensor_2').median()
df.where(df.Status=="Normal").select('Pressure _sensor_3').median()
df.where(df.Status=="Normal").select('Pressure _sensor_4').median()
df.where(df.Status=="Normal").select('Vibration_sensor_1').median()
df.where(df.Status=="Normal").select('Vibration_sensor_2').median()
df.where(df.Status=="Normal").select('Vibration_sensor_3').median()
df.where(df.Status=="Normal").select('Vibration_sensor_4').median()


# In[9]:


#Normal: Mean using while loop

i = 1
print('Mean for subject: Normal: Power_range_sensor_\n')
while i < 3:
#.agg function used to combine multiple Aggregate functions together to analyze the result, in this case the mean
  print(dfNormal.agg({'Power_range_sensor_'+ str(i): 'mean'}).show())
  i += 1
    
i = 3
print('Mean for subject: Normal: Power_range_sensor_\n')
while i < 4:

  print(dfNormal.agg({'Power_range_sensor_'+ str(i) + ' ': 'mean'}).show())
  i += 1          
    
i = 4
print('Mean for subject: Normal: Power_range_sensor_\n')
while i < 5:

  print(dfNormal.agg({'Power_range_sensor_'+ str(i): 'mean'}).show())
  i += 1    


i = 1
print('Mean for subject: Normal: Pressure _sensor\n')
while i < 5:

  print(dfNormal.agg({'Pressure _sensor_'+ str(i): 'mean'}).show())
  i += 1
    
i = 1
print('Mean for subject: Normal: Vibration_sensor_\n')
while i < 5:

  print(dfNormal.agg({'Vibration_sensor_'+ str(i): 'mean'}).show())
  i += 1


# In[21]:


#Abnormal: Mean

i = 1
print('Mean for subject: Abnormal: Power_range_sensor_\n')
while i < 3:

  print(dfAbnormal.agg({'Power_range_sensor_'+ str(i): 'mean'}).show())
  i += 1
    
i = 3
print('Mean for subject: Abnormal: Power_range_sensor_\n')
while i < 4:

  print(dfAbnormal.agg({'Power_range_sensor_'+ str(i) + ' ': 'mean'}).show())
  i += 1          
    
i = 4
print('Mean for subject: Abnormal: Power_range_sensor_\n')
while i < 5:

  print(dfAbnormal.agg({'Power_range_sensor_'+ str(i): 'mean'}).show())
  i += 1    


i = 1
print('Mean for subject: Abnormal: Pressure _sensor\n')
while i < 5:

  print(dfAbnormal.agg({'Pressure _sensor_'+ str(i): 'mean'}).show())
  i += 1
    
i = 1
print('Mean for subject: Abnormal: Vibration_sensor_\n')
while i < 5:

  print(dfAbnormal.agg({'Vibration_sensor_'+ str(i): 'mean'}).show())
  i += 1


# Variance

# In[23]:


#Normal: Variance

i = 1
print('Variance for subject: Normal: Power_range_sensor_\n')
while i < 3:

  print(dfNormal.agg({'Power_range_sensor_'+ str(i): 'variance'}).show())
  i += 1
    
i = 3
print('Variance for subject: Normal: Power_range_sensor_3\n')
while i < 4:

  print(dfNormal.agg({'Power_range_sensor_'+ str(i) + ' ': 'variance'}).show())
  i += 1          
    
i = 4
print('Variance for subject: Normal: Power_range_sensor_4\n')
while i < 5:

  print(dfNormal.agg({'Power_range_sensor_'+ str(i): 'variance'}).show())
  i += 1    


i = 1
print('Variance for subject: Normal: Pressure _sensor\n')
while i < 5:

  print(dfNormal.agg({'Pressure _sensor_'+ str(i): 'variance'}).show())
  i += 1
    
i = 1
print('Variance for subject: Normal: Vibration_sensor_\n')
while i < 5:

  print(dfNormal.agg({'Vibration_sensor_'+ str(i): 'variance'}).show())
  i += 1


# In[24]:


#AbNormal: Variance

i = 1
print('Variance for subject: AbNormal: Power_range_sensor_\n')
while i < 3:

  print(dfAbnormal.agg({'Power_range_sensor_'+ str(i): 'variance'}).show())
  i += 1
    
i = 3
print('Variance for subject: AbNormal: Power_range_sensor_3\n')
while i < 4:

  print(dfAbnormal.agg({'Power_range_sensor_'+ str(i) + ' ': 'variance'}).show())
  i += 1          
    
i = 4
print('Variance for subject: AbNormal: Power_range_sensor_4\n')
while i < 5:

  print(dfAbnormal.agg({'Power_range_sensor_'+ str(i): 'variance'}).show())
  i += 1    


i = 1
print('Variance for subject: AbNormal: Pressure _sensor\n')
while i < 5:

  print(dfAbnormal.agg({'Pressure _sensor_'+ str(i): 'variance'}).show())
  i += 1
    
i = 1
print('Variance for subject: AbNormal: Vibration_sensor_\n')
while i < 5:

  print(dfAbnormal.agg({'Vibration_sensor_'+ str(i): 'variance'}).show())
  i += 1


# In[25]:


#Normal: Minimum

i = 1
print('Minimum for subject: Normal: Power_range_sensor_\n')
while i < 3:

  print(dfNormal.agg({'Power_range_sensor_'+ str(i): 'min'}).show())
  i += 1
    
i = 3
print('Minimum for subject: Normal: Power_range_sensor_3\n')
while i < 4:

  print(dfNormal.agg({'Power_range_sensor_'+ str(i) + ' ': 'min'}).show())
  i += 1          
    
i = 4
print('Minimum for subject: Normal: Power_range_sensor_4\n')
while i < 5:

  print(dfNormal.agg({'Power_range_sensor_'+ str(i): 'min'}).show())
  i += 1    


i = 1
print('Minimum for subject: Normal: Pressure _sensor\n')
while i < 5:

  print(dfNormal.agg({'Pressure _sensor_'+ str(i): 'min'}).show())
  i += 1
    
i = 1
print('Minimum for subject: Normal: Vibration_sensor_\n')
while i < 5:

  print(dfNormal.agg({'Vibration_sensor_'+ str(i): 'min'}).show())
  i += 1


# In[ ]:


#AbNormal: Minimum

i = 1
print('Minimum for subject: AbNormal: Power_range_sensor_\n')
while i < 3:

  print(dfAbnormal.agg({'Power_range_sensor_'+ str(i): 'min'}).show())
  i += 1
    
i = 3
print('Minimum for subject: AbNormal: Power_range_sensor_3\n')
while i < 4:

  print(dfAbnormal.agg({'Power_range_sensor_'+ str(i) + ' ': 'min'}).show())
  i += 1          
    
i = 4
print('Minimum for subject: AbNormal: Power_range_sensor_4\n')
while i < 5:

  print(dfAbnormal.agg({'Power_range_sensor_'+ str(i): 'min'}).show())
  i += 1    


i = 1
print('Minimum for subject: AbNormal: Pressure _sensor\n')
while i < 5:

  print(dfAbnormal.agg({'Pressure _sensor_'+ str(i): 'min'}).show())
  i += 1
    
i = 1
print('Minimum for subject: AbNormal: Vibration_sensor_\n')
while i < 5:

  print(dfAbnormal.agg({'Vibration_sensor_'+ str(i): 'min'}).show())
  i += 1


# In[11]:


#Normal: Maximum

i = 1
print('Maximum for subject: Normal: Power_range_sensor_\n')
while i < 3:

  print(dfNormal.agg({'Power_range_sensor_'+ str(i): 'max'}).show())
  i += 1
    
i = 3
print('Maximum for subject: Normal: Power_range_sensor_3\n')
while i < 4:

  print(dfNormal.agg({'Power_range_sensor_'+ str(i) + ' ': 'max'}).show())
  i += 1          
    
i = 4
print('Maximum for subject: Normal: Power_range_sensor_4\n')
while i < 5:

  print(dfNormal.agg({'Power_range_sensor_'+ str(i): 'max'}).show())
  i += 1    


i = 1
print('Maximum for subject: Normal: Pressure _sensor\n')
while i < 5:

  print(dfNormal.agg({'Pressure _sensor_'+ str(i): 'max'}).show())
  i += 1
    
i = 1
print('Maximum for subject: Normal: Vibration_sensor_\n')
while i < 5:

  print(dfNormal.agg({'Vibration_sensor_'+ str(i): 'max'}).show())
  i += 1


# In[12]:


#AbNormal: Maximum

i = 1
print('Maximum for subject: AbNormal: Power_range_sensor_\n')
while i < 3:

  print(dfAbnormal.agg({'Power_range_sensor_'+ str(i): 'max'}).show())
  i += 1
    
i = 3
print('Maximum for subject: AbNormal: Power_range_sensor_3\n')
while i < 4:

  print(dfAbnormal.agg({'Power_range_sensor_'+ str(i) + ' ': 'max'}).show())
  i += 1          
    
i = 4
print('Maximum for subject: AbNormal: Power_range_sensor_4\n')
while i < 5:

  print(dfAbnormal.agg({'Power_range_sensor_'+ str(i): 'max'}).show())
  i += 1    


i = 1
print('Maximum for subject: AbNormal: Pressure _sensor\n')
while i < 5:

  print(dfAbnormal.agg({'Pressure _sensor_'+ str(i): 'max'}).show())
  i += 1
    
i = 1
print('Maximum for subject: AbNormal: Vibration_sensor_\n')
while i < 5:

  print(dfAbnormal.agg({'Vibration_sensor_'+ str(i): 'max'}).show())
  i += 1


# In[35]:


#convert to pandas
df = df.toPandas()
df


# In[14]:


#median for Normal grouped features
df.where(df.Status == "Normal").median()


# In[15]:


#median for Abnormal grouped features
df.where(df.Status == "Abnormal").median()


# In[16]:


#mode for Normal grouped features
df.where(df.Status == "Normal").mode()


# In[17]:


#mode for Normal grouped features
df.where(df.Status == "Abnormal").mode()


# In[30]:


#box plot using pandas for each group and set of features
rishi = pd.DataFrame(df.where(df.Status == "Normal"),

                  columns=['Power_range_sensor_1', 'Power_range_sensor_2', 'Power_range_sensor_3 ', 'Power_range_sensor_4'])

boxplot = rishi.boxplot(column=['Power_range_sensor_1', 'Power_range_sensor_2', 'Power_range_sensor_3 ', 'Power_range_sensor_4'], figsize=(12,8))


# In[29]:


rishi = pd.DataFrame(df.where(df.Status == "Normal"),

                  columns=['Pressure _sensor_1', 'Pressure _sensor_2', 'Pressure _sensor_3', 'Pressure _sensor_4'])

boxplot = rishi.boxplot(column=['Pressure _sensor_1', 'Pressure _sensor_2', 'Pressure _sensor_3', 'Pressure _sensor_4'], figsize=(12,8))


# In[28]:


rishi = pd.DataFrame(df.where(df.Status == "Normal"),

                  columns=['Vibration_sensor_1', 'Vibration_sensor_2', 'Vibration_sensor_3', 'Vibration_sensor_4'])

boxplot = rishi.boxplot(column=['Vibration_sensor_1', 'Vibration_sensor_2', 'Vibration_sensor_3', 'Vibration_sensor_4'], figsize=(12,8))


# In[27]:


rishi = pd.DataFrame(df.where(df.Status == "Abnormal"),

                  columns=['Power_range_sensor_1', 'Power_range_sensor_2', 'Power_range_sensor_3 ', 'Power_range_sensor_4'])

boxplot = rishi.boxplot(column=['Power_range_sensor_1', 'Power_range_sensor_2', 'Power_range_sensor_3 ', 'Power_range_sensor_4'], figsize=(12,8))


# In[26]:


rishi = pd.DataFrame(df.where(df.Status == "Abnormal"),

                  columns=['Pressure _sensor_1', 'Pressure _sensor_2', 'Pressure _sensor_3', 'Pressure _sensor_4'])

boxplot = rishi.boxplot(column=['Pressure _sensor_1', 'Pressure _sensor_2', 'Pressure _sensor_3', 'Pressure _sensor_4'], figsize=(12,8))


# In[25]:


rishi = pd.DataFrame(df.where(df.Status == "Abnormal"),

                  columns=['Vibration_sensor_1', 'Vibration_sensor_2', 'Vibration_sensor_3', 'Vibration_sensor_4'])

boxplot = rishi.boxplot(column=['Vibration_sensor_1', 'Vibration_sensor_2', 'Vibration_sensor_3', 'Vibration_sensor_4'], figsize=(12,8))


# In[ ]:




