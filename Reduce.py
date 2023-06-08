import pandas as pd
import numpy as np

from dis import dis
from pyspark.conf import SparkConf

from pyspark.context import SparkContext

conf = SparkConf()
conf.setMaster("local").setAppName("My app")

sc = SparkContext(conf=conf)

cols = ['Power_range_sensor_1',
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
    'Vibration_sensor_4',]

#Loading in dataset
df = pd.read_csv("nuclear_plants_big_dataset.csv", names=cols)
#Individual feature retrieval
data = df['Power_range_sensor_1'].values
new_data= np.delete(data, 0)

#Creating RDD on a column
rdd = sc.parallelize(new_data)

#Mapping each RDD element to (x,1)
mapped_rdd = rdd.map(lambda x: (x,1))

##Calculating Average
total_sum = mapped_rdd.reduce(lambda x,y: x + y).collect()

#total sum would then contain two elements in an RDD

#[(feature[0]+[feature][1]...+feature[n]), [1 + 1... + 1]
# ([Sum of feature, Total count])

#Assign variables sum_of_features, count = total_sum[0][1] (psuedocode)
#Average would then be = sum_of_features / count 

##Caluclating max/max
minimum = mapped_rdd.reduceByKey(min).collect()
maxmimum = mapped_red.reduceByKey(max).collect()






