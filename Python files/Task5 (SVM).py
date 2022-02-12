#!/usr/bin/env python
# coding: utf-8

# # Big Data (Task 5) <br>
# 

# In[1]:


from pyspark.sql import SparkSession
from pyspark.ml.stat import Correlation
import pyspark.sql.functions as F
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier, LinearSVC, MultilayerPerceptronClassifier

from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# In[2]:


spark = SparkSession.builder.getOrCreate()


# In[3]:


df = spark.read.csv("nuclear_plants_small_dataset.csv",inferSchema=True,header=True)


# In[4]:


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


# In[5]:


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


# In[6]:


output= assembler.transform(df)


# In[7]:


indexer=StringIndexer(inputCol="Status",outputCol="Status_ind")
indexed=indexer.fit(output).transform(output)


# In[8]:


df_fix = indexed.select("features", "Status_ind")


# In[9]:


indexed.columns


# Shuffle the dataset using randomSplit then split the dataset into 70% training and 30% test set

# In[10]:


training_df,test_df=df_fix.randomSplit([0.7,0.3], seed=50)


# In[11]:


df.printSchema()


# ### Support Vector Machine

# In[12]:


#SVM
lsvc = LinearSVC(labelCol="Status_ind", featuresCol="features", maxIter=10, regParam=0.1).fit(training_df)


# In[13]:


svm_pred = lsvc.transform(test_df)


# In[23]:


#dfsvm_accuracy = MulticlassClassificationEvaluator(labelCol="Status_ind",predictionCol= "prediction",metricName="accuracy").evaluate(svm_pred)


# ### Predicted Label

# In[24]:


from pyspark.ml.classification import LogisticRegression
log_reg=LogisticRegression(labelCol="Status_ind").fit(training_df)

results=log_reg.evaluate(test_df).predictions
results.printSchema()


# In[25]:


results.select(["features","prediction"]).show(20,False)


# ### SVM Accuracy

# In[54]:


dfsvm_accuracy = MulticlassClassificationEvaluator(labelCol="Status_ind",predictionCol= "prediction",metricName="accuracy").evaluate(svm_pred)


# In[55]:


dfsvm_accuracy*100


# ### SVM Error Rate

# In[56]:


df_error = 100 - dfsvm_accuracy*100


# In[57]:


df_error


# ### Confusion Matrix

# #### True Positive

# In[58]:


# confusion matrix
tp = svm_pred[(svm_pred.Status_ind== 1) & (svm_pred.prediction== 1)].count()


# In[59]:


tp


# #### True Negative

# In[60]:


tn = svm_pred[(svm_pred.Status_ind == 0) & (svm_pred.prediction ==0)].count()


# In[61]:


tn


# #### False Positive

# In[62]:


fp = svm_pred[(svm_pred.Status_ind == 0) & (svm_pred.prediction == 1)].count()


# In[63]:


fp


# #### False Negative

# In[64]:


fn = svm_pred[(svm_pred.Status_ind == 1) & (svm_pred.prediction ==0)].count()


# In[65]:


fn


# #### Accuracy

# In[76]:


# accuracy
accuracy=float((tp+tn)/(svm_pred.count()))


# In[77]:


accuracy*100


# In[78]:


# recall
recall = float(tn)/(tp + tn)


# ### SVM Error rate  (`Incorrectly Classified Samples’ divided by Classified Sample’)

# ### Error rate (FP + FN) /(p+n) <br>

# In[79]:


# total rows in test dataset
totaltestrows = df.count()
print(totaltestrows)


# In[80]:


#Calculate total positives from dataset
positive = fp + tp

#Calculate total negatives from dataset
negative = fn + tn


# In[81]:


total = positive + negative
total


# In[82]:


totalFalse = fp + fn
totalFalse


# In[83]:


Errorrate = totalFalse / total
Errorrate*100


# ## Specificity (tn/n)  <br>
# 
# Specificity (SP) is calculated as the number of correct negative predictions divided by the total number of negatives. It is also called true negative rate (TNR). The best specificity is 1.0, whereas the worst is 0.0.

# In[84]:


specificity = tn / negative
specificity


# ## Sensitivity (tp/p)  <br>
# 
# Sensitivity (Recall or True positive rate)
# Sensitivity (SN) is calculated as the number of correct positive predictions divided by the total number of positives. It is also called recall (REC) or true positive rate (TPR). The best sensitivity is 1.0, whereas the worst is 0.0.

# In[85]:


sensitivity = tp / positive
sensitivity


# In[ ]:





# In[ ]:





# In[ ]:





# In[86]:



#ann = MultilayerPerceptronClassifier(layers=[12, 2, 2], seed=123, labelCol="Status_ind", featuresCol="features").fit(training_df)


#ann_pred = ann.transform(test_df)



#dfann_accuracy = MulticlassClassificationEvaluator(labelCol="Status_ind",predictionCol= "prediction",metricName="accuracy").evaluate(ann_pred)



#dfann_accuracy*100

