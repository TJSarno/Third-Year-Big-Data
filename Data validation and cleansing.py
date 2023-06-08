#Pyspark import
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import isnan, when, count, col

#Graphs/Sytling imports
import pandas as pd
from pandas.plotting import table
import matplotlib as mp
import matplotlib.pyplot as plt
import seaborn as sns


spark = SparkSession.builder.getOrCreate()
##Pyspark test block##
#df = spark.sql("select 'spark' as hello")
#df.show()

filepath = r"C:\Users\Computing\Desktop\Big\nuclear_plants_small_dataset.csv" #Can be changed depending on file location
#df_small = spark.read.csv(filepath, inferSchema=True,header=True)


##Task 1##   (Have commented out as only used to output values for screenshot)
#Show missing values
#df_small.select([count(when(isnan(c), c)).alias(c) for c in df_small.columns]).show()

#Show null values
#df_small.select([count(when(col(c).isNull(), c)).alias(c) for c in df_small.columns]).show()

##Task 2##
#Importing csv as pandas (just for tabling/plotting purposes)
#df = pd.read_csv(filepath)
df = pd.read_csv('nuclear_plants_small_dataset.csv')

#Splitting data based on Status
df1 = df[df['Status'] == 'Normal']
df2 = df[df['Status'] == 'Abnormal']

##Getting data in csv format for display later, no longer needed ##
#df1.describe().to_csv("normal")
#df2.describe().to_csv("abnormal")


##Box plots
df1.plot(kind='box', fontsize=10,vert=False, title="Nuclear Plants Small Dataset: Normal Features") #rot=90 for vertical labels
plt.show()

df2.plot(kind='box', fontsize=10,vert=False, title="Nuclear Plants Small Dataset: Abormal Features") #rot=90 for vertical labels
plt.show()





