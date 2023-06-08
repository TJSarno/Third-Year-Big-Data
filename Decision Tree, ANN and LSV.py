#Pyspark import
from distutils.log import error
from tkinter import font
from typing import final
import pyspark
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.functions import isnan, when, count, col

#Graphs/Sytling imports
import pandas as pd
import matplotlib as mp
import matplotlib.pyplot as plt
import seaborn as sns

#Task 5 Decision tree imports
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.mllib.util import MLUtils

#Task 5 Support vector machine model #
from pyspark.ml.classification import LinearSVC

#Task 5 ANN imports#
from pyspark.ml.classification import MultilayerPerceptronClassifier

#Other imports
import pyspark.sql.functions as F
from pyspark.sql.types import FloatType
from pyspark.mllib.evaluation import MulticlassMetrics

spark = SparkSession.builder.getOrCreate()
sq=SQLContext(sparkContext=spark.sparkContext, sparkSession=spark)

#Importing csv
df_small = spark.read.csv('nuclear_plants_small_dataset.csv', inferSchema=True,header=True)

##Task 4## (COMMENTED OUT AS TASK 4 ONLY INCLUDED DISPLAYING VALUES, LATER ON THE SAME TRAIN/TEST SPLIT IS USED ON THE VECTOR ASSEMBLED DATAFRAME)

#Splitting the data
#train,test = df_small.randomSplit([0.7,0.3], 40) #Added seed to keep consistent results

##Task 4 Displaying totals ## (not needed for later tasks)

#Getting totals for the train set
#normal_train_total = train.select('Status').where(train.Status=='Normal').count()
#abnormal_train_total = train.select('Status').where(train.Status=='Abnormal').count()
#Displaying Values
#print("Normal train total: ", normal_train_total)
#print("Abnormal train total: ", abnormal_train_total)

#Getting totals for test set
#normal_test_total = test.select('Status').where(test.Status=='Normal').count()
#abnormal_test_total = test.select('Status').where(test.Status=='Abnormal').count()

#print("Normal test total: ", normal_test_total)
#print("Abnormal test total: ", abnormal_test_total)

##End of Task 4##


#Vector Assembler to sort pyspark features and labels

assembler = VectorAssembler(
    inputCols=['Power_range_sensor_1',
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
    'Vibration_sensor_4',
    ],
    outputCol="features"
)

output = assembler.transform(df_small)


#Indexing Status 
indexer = StringIndexer(inputCol='Status', outputCol='StatusIndexer')
output_fixed = indexer.fit(output).transform(output)

#Final prepped dataframe
df_small_final = output_fixed.select('features', 'StatusIndexer')


#Splitting data for training and testing (SAME AS TASK 4 JUST ON NEW DATAFRAME)
train,test = df_small_final.randomSplit([0.7,0.3], 40)

##Evaluators (NOT USED IN FINAL OUTPUT BUT LEFT IN FOR METRIC COMPARISONS)##
accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="StatusIndexer", predictionCol="prediction", metricName="accuracy")
tp_evaluator = MulticlassClassificationEvaluator(labelCol="StatusIndexer", predictionCol="prediction", metricName="truePositiveRateByLabel")
test_evaluator = BinaryClassificationEvaluator(labelCol="StatusIndexer", rawPredictionCol="prediction", metricName='precisionByLabel')
precision_evaluator = MulticlassClassificationEvaluator(labelCol="StatusIndexer", predictionCol="prediction", metricName="precisionByLabel")


##Decision Tree##
#Train Decision Tree
dt = DecisionTreeClassifier(labelCol="StatusIndexer", featuresCol="features")
dt_model = dt.fit(train)

#Fitting the model
dt_pred = dt_model.transform(test)

#Not used accuracy evaluation using pyspark
dt_accuracy = accuracy_evaluator.evaluate(dt_pred)
print("Decision Tree Accuracy: ", dt_accuracy)



##ANN##
#Prepping the model
layers = [12, 8, 4, 2]
trainer = MultilayerPerceptronClassifier(labelCol="StatusIndexer", featuresCol="features", maxIter=100, layers=layers, blockSize=128, seed=1234)

#Training and fitting the model
ann_model = trainer.fit(train)
ann_pred = ann_model.transform(test)

#Not used accuracy evaluation using pyspark
ann_accuracy = accuracy_evaluator.evaluate(ann_pred)
print("ANN accuracy: ", ann_accuracy)



##Support Vector##
#Prepping the model
lsvc = LinearSVC(maxIter=100, regParam=0.1, labelCol="StatusIndexer", featuresCol="features")

#Training and fitting the model
lsvc_model = lsvc.fit(train)
lsvc_pred = lsvc_model.transform(test)

#Not used accuracy evaluation using pyspark
lsvc_accuracy = accuracy_evaluator.evaluate(lsvc_pred)
print("LSVC accuracy: ", lsvc_accuracy)



##Metric generation##
#Getting Y True and Y Pred into dataframes for later analysis
dt_df = dt_pred.select('StatusIndexer', 'prediction')
ann_df = ann_pred.select('StatusIndexer', 'prediction')
lsvc_df = lsvc_pred.select('StatusIndexer', 'prediction')


#Metric functon to calculate error rate, sensitivity and specifcity
def metric_output(df):
    tp = df[(df.StatusIndexer== 1) & (df.prediction == 1)].count()
    tn = df[(df.StatusIndexer == 0) & (df.prediction == 0)].count()
    fp = df[(df.StatusIndexer == 0) & (df.prediction == 1)].count()
    fn = df[(df.StatusIndexer == 1) & (df.prediction == 0)].count()
    total = df.count()

    error_rate = 1 - float(tp + tn)/(total)
    print("Error rate: ", error_rate)

    sensitivity = float(tp)/(tp + fn)
    print("Sensitivity: ", sensitivity)

    specificity = float(tn)/(tn + fp)
    print("Specificity: ",specificity)

    return error_rate, sensitivity, specificity

#Calculating error rate, sensitivity and specificity for each model
dt_er, dt_sens, dt_spec = metric_output(dt_df)
ann_er, ann_sens, ann_spec = metric_output(ann_df)
lsvc_er, lsvc_sens, lsvc_spec = metric_output(lsvc_df)

#Putting these values into a pandas dataframe
final_outputs = pd.DataFrame({'Error Rate': [dt_er, ann_er, lsvc_er],
                   'Sensitivity': [dt_sens, ann_sens, lsvc_sens],
                   'Specificity': [dt_spec, ann_spec, lsvc_spec]})

print(final_outputs)



#Outpitting as Table
fig, ax = plt.subplots()
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

#Labels
row_labels=("Decision Tree", "ANN", "Linear Support Vector")
col_labels=("Error Rate", "Sensitivity", "Specificity")

ax.table(cellText=final_outputs.values, colLabels=col_labels, rowLabels=row_labels, loc='center')
fig.tight_layout()

plt.show()

