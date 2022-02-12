#!/usr/bin/env python
# coding: utf-8

# ###### Big Data (Task 5) Artificial Neural Networks <br>
# 

# In[26]:


from pyspark.sql import SparkSession
from pyspark.ml.stat import Correlation
import pyspark.sql.functions as F
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier, LinearSVC, MultilayerPerceptronClassifier

from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# In[27]:


spark = SparkSession.builder.getOrCreate()


# In[28]:


df = spark.read.csv("nuclear_plants_small_dataset.csv",inferSchema=True,header=True)


# In[29]:


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


# In[30]:


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


# In[31]:


output= assembler.transform(df)


# In[32]:


indexer=StringIndexer(inputCol="Status",outputCol="Status_ind")
indexed=indexer.fit(output).transform(output)


# In[33]:


df_fix = indexed.select("features", "Status_ind")


# In[34]:


indexed.columns


# Shuffle the dataset using randomSplit then split the dataset into 70% training and 30% test set

# In[35]:


training_df,test_df=df_fix.randomSplit([0.7,0.3], seed=50)


# In[36]:


df.printSchema()


# ### ANN

# In[37]:


#ANN
ann = MultilayerPerceptronClassifier(layers=[12, 2, 2], seed=123, labelCol="Status_ind", featuresCol="features").fit(training_df)


# In[38]:


ann_pred = ann.transform(test_df)


# In[39]:


dfann_accuracy = MulticlassClassificationEvaluator(labelCol="Status_ind",predictionCol= "prediction",metricName="accuracy").evaluate(ann_pred)


# ### Predicted Label

# In[40]:


from pyspark.ml.classification import LogisticRegression
log_reg=LogisticRegression(labelCol="Status_ind").fit(training_df)

results=log_reg.evaluate(test_df).predictions
results.printSchema()


# In[41]:


results.select(["features","prediction"]).show(20,False)


# ### ANN Accuracy

# In[44]:


dfann_accuracy = MulticlassClassificationEvaluator(labelCol="Status_ind",predictionCol= "prediction",metricName="accuracy").evaluate(ann_pred)


# In[45]:


dfann_accuracy*100


# ### ANN Error Rate

# In[46]:


df_error = 100 - dfann_accuracy*100


# In[47]:


df_error


# ### Confusion Matrix

# #### True Positive

# In[48]:


# confusion matrix
tp = ann_pred[(ann_pred.Status_ind== 1) & (ann_pred.prediction== 1)].count()


# In[49]:


tp


# #### True Negative

# In[50]:


tn = ann_pred[(ann_pred.Status_ind == 0) & (ann_pred.prediction ==0)].count()


# In[51]:


tn


# #### False Positive

# In[52]:


fp = ann_pred[(ann_pred.Status_ind == 0) & (ann_pred.prediction == 1)].count()


# In[53]:


fp


# #### False Negative

# In[54]:


fn = ann_pred[(ann_pred.Status_ind == 1) & (ann_pred.prediction ==0)].count()


# In[55]:


fn


# #### Accuracy

# In[56]:


# accuracy
accuracy=float((tp+tn)/(results.count()))


# In[57]:


accuracy*100


# In[58]:


# recall
recall = float(tn)/(tp + tn)


# ### ANN rate  (`Incorrectly Classified Samples’ divided by Classified Sample’)

# ### Error rate (FP + FN) /(p+n) <br>

# In[59]:


# total rows in test dataset
totaltestrows = df.count()
print(totaltestrows)


# In[60]:


#Calculate total positives from dataset
positive = fp + tp

#Calculate total negatives from dataset
negative = fn + tn


# In[61]:


total = positive + negative
total


# In[62]:


totalFalse = fp + fn
totalFalse


# In[63]:


Errorrate = totalFalse / total
Errorrate*100


# ## Specificity Neural Network (tn/n)  <br>
# 
# Specificity (SP) is calculated as the number of correct negative predictions divided by the total number of negatives. It is also called true negative rate (TNR). The best specificity is 1.0, whereas the worst is 0.0.

# In[64]:


specificity = tn / negative
specificity


# ## Sensitivity Neural network (tp/p)  <br>
# 
# Sensitivity (Recall or True positive rate)
# Sensitivity (SN) is calculated as the number of correct positive predictions divided by the total number of positives. It is also called recall (REC) or true positive rate (TPR). The best sensitivity is 1.0, whereas the worst is 0.0.

# In[65]:


sensitivity = tp / positive
sensitivity


# In[ ]:




