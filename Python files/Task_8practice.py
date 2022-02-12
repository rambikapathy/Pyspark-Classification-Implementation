#!/usr/bin/env python
# coding: utf-8

# ## Method2: ReduceBy 

# #### convert dataset to RDD

# In[ ]:


#####  This  implementation method was a trial, may not work
df_big1 = sc.textFile(df_big).map(lambda x: x.split(','))#split between commas


# #### Drop status 

# In[ ]:


drop_status = df_big11.take(1)[0]#(1,0) refers to first column(Status)
drop_status = df_big11.filter(lambda line: line != drop_status)


# In[ ]:


#TUPLE groups 2 or more elements, in this scenario 12 during the lambda implementation.
df_big1 = df_big1.map(lambda x: (x[0], tuple(x[1:])))

#merges each elements with dataset utilizing reduce function, by shuffling dataset across multiple partitions, due to being a large dataset...
#With RDD to compute maximum, mean and minimum
df_big1_maximum = df_big1.reduceByKey(lambda x, y: max(x[0],y[0]))
df_big1_maximum.collect()

df_big1_minimum = df_big1.reduceByKey(lambda x, y: min(x[0],y[0]))
df_big1_minimum.collect()

#df_big1_mean = df_big1.reduceByKey(lambda x, y: mean(x[0],y[0]))
#df_big1_mean.collect()


# In[ ]:


#sc.parallelize assigns the big data across multiple nodes 
parrallelize = sc.parallelize(df_big)

#reduce function
def reduce_data(comp, n):
    print(comp, n)
    if comp[1] > n[1]:
        return(n)
    else: return(comp)
    
#map function
def map_thedata(column):
    return (column[0], column[1])
# this would return the mapped values for each column within dataset, starting with Power_range_sensor1 [0,1]

parrallelize.map(map_thedata).keyBy(lambda x: x[0]).reduceByKey(reduce_data).map(lambda x : x[1]).collect()


# In[ ]:





# In[ ]:





# In[ ]:




