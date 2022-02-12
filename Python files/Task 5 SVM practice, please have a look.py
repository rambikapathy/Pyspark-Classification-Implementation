#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark.ml.classification import LinearSVC


# In[2]:


from pyspark.sql import SparkSession
from pyspark.ml.stat import Correlation
import pyspark.sql.functions as F
from pyspark.ml.feature import StringIndexer
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import Normalizer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# In[ ]:


spark = SparkSession.builder.getOrCreate()


# In[ ]:


df = spark.read.csv("nuclear_plants_small_dataset.csv",inferSchema=True,header=True)


# In[ ]:


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


# In[ ]:


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


# In[ ]:


splits = df.randomSplit([0.7,0.3])


# In[ ]:


df_train = splits[0]
df_test = splits[1]


# In[ ]:


df_fix = indexed.select("features", "Status_ind")


# In[ ]:


training_df,test_df=df_fix.randomSplit([0.7,0.3], seed=50)


# In[ ]:


output= assembler.transform(df)


# In[ ]:


indexer=StringIndexer(inputCol="Status",outputCol="Status_ind")
indexed=indexer.fit(output).transform(output)


# In[ ]:


encoder = OneHotEncoder(inputCol="Status_ind", outputCol="Status_indVec")


# In[ ]:


vectorAssembler = VectorAssembler(inputCols=['Power_range_sensor_1',
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
 'Vibration_sensor_4'], outputCol = "features")


# In[ ]:


normalizer = Normalizer(inputCol = "features", outputCol="features_norm", p=1.0)


# In[ ]:


#from new program

from pyspark.ml import Pipeline


# In[ ]:


pipeline = Pipeline(stages=[indexer,encoder,vectorAssembler, normalizer, LinearSVC])


# In[ ]:


lsvc = LinearSVC(maxIter=10, regParam=0.1)


# In[ ]:


from pyspark.ml import Pipeline


# In[ ]:


pipeline = Pipeline(stages=[indexer,encoder,vectorAssembler, normalizer, lsvc])


# In[ ]:


model = pipeline.fit(df_train)


# In[ ]:


prediction = model.transform(df_train)


# In[ ]:


from pyspark.ml.evaluation import BinaryClassificationEvaluator


# In[ ]:


evaluator = BinaryClassificationEvaluator(rawPredictionCol = "rawPrediction")


# In[ ]:


evaluator.evaluate(prediction)


# In[ ]:


prediction = model.transform(df_train)


# In[ ]:


evaluator.evaluate(prediction)


# In[ ]:




