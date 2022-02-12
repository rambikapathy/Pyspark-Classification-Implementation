#!/usr/bin/env python
# coding: utf-8

# # Big Data (Task 5) <br>
# 

# In[57]:


from pyspark.sql import SparkSession
from pyspark.ml.stat import Correlation
import pyspark.sql.functions as F
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier, LinearSVC, MultilayerPerceptronClassifier

from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# In[58]:


spark = SparkSession.builder.getOrCreate()


# In[59]:


df = spark.read.csv("nuclear_plants_small_dataset.csv",inferSchema=True,header=True)


# In[60]:


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


# In[61]:


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


# In[62]:


output= assembler.transform(df)


# In[63]:


indexer=StringIndexer(inputCol="Status",outputCol="Status_ind")
indexed=indexer.fit(output).transform(output)


# In[64]:


df_fix = indexed.select("features", "Status_ind")


# In[65]:


indexed.columns


# Shuffle the dataset using randomSplit then split the dataset into 70% training and 30% test set

# In[66]:


training_df,test_df=df_fix.randomSplit([0.7,0.3], seed=50)


# In[67]:


df.printSchema()


# ### Decision Tree

# In[68]:


#Decision Tree classfier used to interpret categorical features.
#It can break down dataset into smaller subsets while an associated tree is developed.
df_classifier = DecisionTreeClassifier(labelCol="Status_ind", featuresCol="features").fit(training_df)


# In[69]:


df_pred = df_classifier.transform(test_df)


# In[70]:


#MullticlassClassificationEvaluator is used to classifies data into a specific class, in this case expects input index, prediction and accuracy for evaulating the prediction label
df_accuracy = MulticlassClassificationEvaluator(labelCol="Status_ind",predictionCol= "prediction",metricName="accuracy").evaluate(df_pred)


# ### Predicted Label

# In[94]:


from pyspark.ml.classification import LogisticRegression
log_reg=LogisticRegression(labelCol="Status_ind").fit(training_df)
#LogisticRegression used to predict categorical response to help classification type machine learning problems.

results=log_reg.evaluate(test_df).predictions
results.printSchema()


# In[96]:


results.select(["features","prediction"]).show(20,False)


# ### Decision Tree Accuracy

# In[71]:


df_accuracy = MulticlassClassificationEvaluator(labelCol="Status_ind",predictionCol= "prediction",metricName="accuracy").evaluate(df_pred)


# In[72]:


df_accuracy*100


# ### Decision Tree Error Rate

# In[73]:


df_error = 100 - df_accuracy*100


# In[74]:


df_error


# ### Confusion Matrix

# #### True Positive

# In[75]:


# confusion matrix
tp = results[(results.Status_ind== 1) & (results.prediction== 1)].count()


# In[76]:


tp


# #### True Negative

# In[78]:


tn = results[(results.Status_ind == 0) & (results.prediction ==0)].count()


# In[79]:


tn


# #### False Positive

# In[80]:


fp = results[(results.Status_ind == 0) & (results.prediction == 1)].count()


# In[81]:


fp


# #### False Negative

# In[82]:


fn = results[(results.Status_ind == 1) & (results.prediction ==0)].count()


# In[83]:


fn


# #### Accuracy

# In[84]:


# accuracy
accuracy=float((tp+tn)/(results.count()))


# In[86]:


accuracy*100


# In[35]:


# recall
recall = float(tn)/(tp + tn)


# ### Decision Tree Error rate  (`Incorrectly Classified Samples’ divided by Classified Sample’)

# ### Error rate (FP + FN) /(p+n) <br>

# In[87]:


# total rows in test dataset
totaltestrows = df.count()
print(totaltestrows)


# In[88]:


#Calculate total positives from dataset
positive = fp + tp

#Calculate total negatives from dataset
negative = fn + tn


# In[89]:


total = positive + negative
total


# In[90]:


totalFalse = fp + fn
totalFalse


# In[91]:


Errorrate = totalFalse / total
Errorrate*100


# ## Specificity (tn/n)  <br>
# 
# Specificity (SP) is calculated as the number of correct negative predictions divided by the total number of negatives. It is also called true negative rate (TNR). The best specificity is 1.0, whereas the worst is 0.0.

# In[92]:


specificity = tn / negative
specificity


# ## Sensitivity (tp/p)  <br>
# 
# Sensitivity (Recall or True positive rate)
# Sensitivity (SN) is calculated as the number of correct positive predictions divided by the total number of positives. It is also called recall (REC) or true positive rate (TPR). The best sensitivity is 1.0, whereas the worst is 0.0.

# In[93]:


sensitivity = tp / positive
sensitivity

