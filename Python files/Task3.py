#!/usr/bin/env python
# coding: utf-8

# # Big Data(Task 3)
# 

# ### Import neccessary libraries and spark session

# In[8]:


from pyspark.sql import SparkSession
from pyspark.ml.stat import Correlation
import pyspark.sql.functions as F
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import seaborn as sns
import matplotlib.pyplot as plt


# ### Create spark session & read dataset

# In[9]:


#Spark session carries implicits imported within this scope.
spark = SparkSession.builder.getOrCreate()
data = spark.read.csv("nuclear_plants_small_dataset.csv",inferSchema=True,header=True)


# ### Drop status column before implementing Vector Assembler
# Status column is strings.
# 
# Implementing the vector assembler would require all columns to be in interger values.

# In[10]:


df = data.drop("Status")

#Convert to vector column
vector_col = "corr_features"

#Vector Assembler is a transformer that combines input column (features) into single vector column, which you can see as output column
assembler = VectorAssembler(inputCols=df.columns, outputCol=vector_col)

#Here the assembler is transforming columns into single vector column
df_vector = assembler.transform(df).select(vector_col)


# ### Correlation Matrix

# In[20]:


matrix = Correlation.corr(df_vector, vector_col).collect()[0][0]
# Calculates Pearson correlation coefficient of features as double value. 
#Here we have the transformed vector columns 
corrmatrix = matrix.toArray().tolist()
# .toArray; array with the shape represented by  sparse matrix 
#tolist is defined as an extension method to converting collections into list instance,as seen below.
print(corrmatrix)


# In[12]:


#spark.sql("show corrmatrix").show()


# In[13]:


df.columns #columns


# In[14]:


columns = ["Power_range_sensor_1","Power_range_sensor_2","Power_range_sensor_3 ","Power_range_sensor_4","Pressure _sensor_1","Pressure _sensor_2","Pressure _sensor_3","Pressure _sensor_4","Vibration_sensor_1","Vibration_sensor_2","Vibration_sensor_3","Vibration_sensor_4"]


# In[15]:


#create spark dataframe for correaltion matrix and columns
df_corr = spark.createDataFrame(corrmatrix,columns)
df_corr.show()


# In[16]:


df.columns


# In[17]:


df_corr.select('Power_range_sensor_1','Power_range_sensor_2','Power_range_sensor_3 ','Power_range_sensor_4','Pressure _sensor_1','Pressure _sensor_2','Pressure _sensor_3','Pressure _sensor_4','Vibration_sensor_1','Vibration_sensor_2','Vibration_sensor_3','Vibration_sensor_4').show()


# ### Correlation Heatmap

# In[18]:


#Import seaborn to plot heatmap for corerlation matrix of features
plt.figure(figsize=(16,12))#heatmap size
sns.heatmap(corrmatrix, xticklabels=df_corr.columns, 
           yticklabels=df_corr.columns, annot=True)
plt.show()

