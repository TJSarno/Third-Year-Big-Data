#Pyspark import
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import isnan, when, count, col

#Graphs/Sytling imports
import pandas as pd
import matplotlib as mp
import matplotlib.pyplot as plt
import seaborn as sns

#Task 3 specific imports
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.stat import Statistics

spark = SparkSession.builder.getOrCreate()
filepath = r"C:\Users\Computing\Desktop\Big\nuclear_plants_small_dataset.csv" #Can be changed depending on file location

#Using pandas dataframe and correlation calculaton
data = pd.read_csv('nuclear_plants_small_dataset.csv')

df = pd.DataFrame(data)
#Pandas built in correlation function.
corrMatrix = df.corr()

#Plotting
#sns.set(font_scale=0.9)
heatmap = sns.heatmap(corrMatrix, annot=True, xticklabels=True, yticklabels=True).set_title("Nuclear plants small feature correlation matrix")
heatmap.figure.tight_layout()
plt.show()
