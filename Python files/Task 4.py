#!/usr/bin/env python
# coding: utf-8

# # Big Data (Task 4) <br>
# 

# In[6]:


from pyspark.sql import SparkSession
from pyspark.ml.stat import Correlation
import pyspark.sql.functions as F
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import IntegerType,BooleanType,DateType
from pyspark.ml.classification import DecisionTreeClassifier, LinearSVC, MultilayerPerceptronClassifier

from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# In[2]:


spark = SparkSession.builder.getOrCreate()


# In[3]:


df = spark.read.csv("nuclear_plants_small_dataset.csv",inferSchema=True,header=True)


# In[18]:


# Filter the groups into Normal and abnormal
dfNormal = df.filter(df.Status == 'Normal' )


# In[19]:


#Currently the data type is double, this function converts all the feature columns into integer
dfNormal= dfNormal.withColumn("Power_range_sensor_1",dfNormal["Power_range_sensor_1"].cast(IntegerType()))
dfNormal= dfNormal.withColumn("Power_range_sensor_2",dfNormal["Power_range_sensor_2"].cast(IntegerType()))
dfNormal= dfNormal.withColumn("Power_range_sensor_3 ",dfNormal["Power_range_sensor_3 "].cast(IntegerType()))
dfNormal= dfNormal.withColumn("Power_range_sensor_4",dfNormal["Power_range_sensor_4"].cast(IntegerType()))

dfNormal= dfNormal.withColumn("Pressure _sensor_1",dfNormal["Pressure _sensor_1"].cast(IntegerType()))
dfNormal= dfNormal.withColumn("Pressure _sensor_2",dfNormal["Pressure _sensor_2"].cast(IntegerType()))
dfNormal= dfNormal.withColumn("Pressure _sensor_3",dfNormal["Pressure _sensor_3"].cast(IntegerType()))
dfNormal= dfNormal.withColumn("Pressure _sensor_4",dfNormal["Pressure _sensor_4"].cast(IntegerType()))

dfNormal= dfNormal.withColumn("Vibration_sensor_1",dfNormal["Vibration_sensor_1"].cast(IntegerType()))
dfNormal= dfNormal.withColumn("Vibration_sensor_2",dfNormal["Vibration_sensor_2"].cast(IntegerType()))
dfNormal= dfNormal.withColumn("Vibration_sensor_3",dfNormal["Vibration_sensor_3"].cast(IntegerType()))
dfNormal= dfNormal.withColumn("Vibration_sensor_4",dfNormal["Vibration_sensor_4"].cast(IntegerType()))


# In[20]:


assembler= VectorAssembler(inputCols=['Power_range_sensor_1',
 'Power_range_sensor_2',
 'Power_range_sensor_3 ',
 'Power_range_sensor_4',
 'Pressure _sensor_1',
 'Pressure _sensor_2',
 'Pressure _sensor_3',
 'Pressure _sensor_4',
 'Vibration_sensor_1',
 'Vibration_sensor_2',
 'Vibration_sensor_3',
 'Vibration_sensor_4'],outputCol='features')


# In[29]:


output= assembler.transform(dfNormal)


# In[22]:


#Stringindexer encodes a column's string label into indices.
indexer=StringIndexer(inputCol="Status",outputCol="Status_ind")
indexed=indexer.fit(output).transform(output)


# In[23]:


df_fix = indexed.select("features", "Status_ind")


# In[24]:


#output indexed columns
indexed.columns


# Shuffle the dataset using randomSplit then split the dataset into 70% training and 30% test set

# In[25]:


training_df,test_df=df_fix.randomSplit([0.7,0.3], seed=50)


# ### How many examples in each group for the training dataset? (Normal Group:338)

# In[26]:


print(training_df.count())


# ### How many examples in each group for the test dataset? (Normal Group:160)

# In[27]:


print(test_df.count())


# In[ ]:





# In[28]:


#fILTER for Abnormal
dfAbnormal = df.filter(df.Status == 'Abnormal' )


# In[30]:


dfAbnormal= dfAbnormal.withColumn("Power_range_sensor_1",dfAbnormal["Power_range_sensor_1"].cast(IntegerType()))
dfAbnormal= dfAbnormal.withColumn("Power_range_sensor_2",dfAbnormal["Power_range_sensor_2"].cast(IntegerType()))
dfAbnormal= dfAbnormal.withColumn("Power_range_sensor_3 ",dfAbnormal["Power_range_sensor_3 "].cast(IntegerType()))
dfAbnormal= dfAbnormal.withColumn("Power_range_sensor_4",dfAbnormal["Power_range_sensor_4"].cast(IntegerType()))

dfAbnormal= dfAbnormal.withColumn("Pressure _sensor_1",dfAbnormal["Pressure _sensor_1"].cast(IntegerType()))
dfAbnormal= dfAbnormal.withColumn("Pressure _sensor_2",dfAbnormal["Pressure _sensor_2"].cast(IntegerType()))
dfAbnormal= dfAbnormal.withColumn("Pressure _sensor_3",dfAbnormal["Pressure _sensor_3"].cast(IntegerType()))
dfAbnormal= dfAbnormal.withColumn("Pressure _sensor_4",dfAbnormal["Pressure _sensor_4"].cast(IntegerType()))

dfAbnormal= dfAbnormal.withColumn("Vibration_sensor_1",dfAbnormal["Vibration_sensor_1"].cast(IntegerType()))
dfAbnormal= dfAbnormal.withColumn("Vibration_sensor_2",dfAbnormal["Vibration_sensor_2"].cast(IntegerType()))
dfAbnormal= dfAbnormal.withColumn("Vibration_sensor_3",dfAbnormal["Vibration_sensor_3"].cast(IntegerType()))
dfAbnormal= dfAbnormal.withColumn("Vibration_sensor_4",dfAbnormal["Vibration_sensor_4"].cast(IntegerType()))


# In[31]:


output= assembler.transform(dfAbnormal)


# In[32]:


indexer=StringIndexer(inputCol="Status",outputCol="Status_ind")
indexed=indexer.fit(output).transform(output)


# In[33]:


df_fix = indexed.select("features", "Status_ind")


# In[34]:


training_df,test_df=df_fix.randomSplit([0.7,0.3], seed=50)


# ### How many examples in each group for the training dataset? (Abnormal Group:338)

# In[36]:


print(training_df.count())


# ### How many examples in each group for the test dataset? (Abnormal Group:160)

# In[37]:


print(test_df.count())


# In[ ]:





# ## Total dataset

# In[38]:


from pyspark.sql.types import IntegerType,BooleanType,DateType

df= df.withColumn("Power_range_sensor_1",df["Power_range_sensor_1"].cast(IntegerType()))
df= df.withColumn("Power_range_sensor_2",df["Power_range_sensor_2"].cast(IntegerType()))
df= df.withColumn("Power_range_sensor_3 ",df["Power_range_sensor_3 "].cast(IntegerType()))
df= df.withColumn("Power_range_sensor_4",df["Power_range_sensor_4"].cast(IntegerType()))

df= df.withColumn("Pressure _sensor_1",df["Pressure _sensor_1"].cast(IntegerType()))
df= df.withColumn("Pressure _sensor_2",df["Pressure _sensor_2"].cast(IntegerType()))
df= df.withColumn("Pressure _sensor_3",df["Pressure _sensor_3"].cast(IntegerType()))
df= df.withColumn("Pressure _sensor_4",df["Pressure _sensor_4"].cast(IntegerType()))

df= df.withColumn("Vibration_sensor_1",df["Vibration_sensor_1"].cast(IntegerType()))
df= df.withColumn("Vibration_sensor_2",df["Vibration_sensor_2"].cast(IntegerType()))
df= df.withColumn("Vibration_sensor_3",df["Vibration_sensor_3"].cast(IntegerType()))
df= df.withColumn("Vibration_sensor_4",df["Vibration_sensor_4"].cast(IntegerType()))


# In[39]:


output= assembler.transform(df)


# In[40]:


indexer=StringIndexer(inputCol="Status",outputCol="Status_ind")
indexed=indexer.fit(output).transform(output)


# In[41]:


df_fix = indexed.select("features", "Status_ind")


# In[42]:


training_df,test_df=df_fix.randomSplit([0.7,0.3], seed=50)


# ### How many examples in each group for the training dataset? (total:686)

# In[43]:


print(training_df.count())


# ### How many examples in each group for the test dataset? (total:310)

# In[44]:


print(test_df.count())


# In[ ]:




