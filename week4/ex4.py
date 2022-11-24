"""Exercise 4 for Data-Intensive Programming"""

from typing import List
import os
import sys

from pyspark.sql import SparkSession, DataFrame
from pyspark.rdd import RDD
from pyspark.sql.types import StructType, StructField, DoubleType, Row
from pyspark.ml.regression import LinearRegression, LinearRegressionModel
from pyspark.ml.feature import VectorAssembler
import pyspark.sql.functions as f

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

def main():
    # Create the Spark session
    spark: SparkSession = SparkSession.builder \
                                      .appName("ex4") \
                                      .config("spark.driver.host", "localhost") \
                                      .master("local") \
                                      .getOrCreate()

    # suppress informational log messages related to the inner working of Spark
    spark.sparkContext.setLogLevel("ERROR")

    spark.conf.set("spark.sql.shuffle.partitions", "5")

    # Wikipedia defines: Simple Linear Regression
    # In statistics, simple linear regression is a linear regression model with a single explanatory variable.
    # That is, it concerns two-dimensional sample points with one independent variable and one dependent variable
    # (conventionally, the x and y coordinates in a Cartesian coordinate system) and finds a linear function (a non-vertical straight line)
    # that, as accurately as possible, predicts the dependent variable values as a function of the independent variables. The adjective simple
    # refers to the fact that the outcome variable is related to a single predictor.

    # You are given an dataRDD of Rows (first element is x and the other y). We are aiming at finding simple linear regression model
    # for the dataset using MLlib. I.e. find function f so that y ~ f(x)

    hugeSequenceOfXYData = [Row(0.0, 0.0), Row(0.3, 0.5), Row(0.9, 0.8), Row(1.0, 0.8),
                            Row(2.0, 2.2), Row(2.2, 2.4), Row(3.0, 3.7), Row(4.0, 4.3),
                            Row(1.5, 1.4), Row(3.2, 3.9), Row(3.5, 4.1), Row(1.2, 1.1)]

    dataRDD: RDD[Row] = spark.sparkContext.parallelize(hugeSequenceOfXYData)

    printTaskLine(1)
    # Task 1: Transform dataRDD to a DataFrame dataDF, with two columns "X" (of type Double) and "label" (of type Double).
    #         (The default dependent variable name is "label" in MLlib)
    schema = StructType([
        StructField('X', DoubleType(), True),
        StructField('label', DoubleType(), True)
    ])

    dataDF: DataFrame = spark.createDataFrame(dataRDD, schema=schema)

    # Let's split the data into training and testing datasets
    trainTest: List[DataFrame] = dataDF.randomSplit([0.7, 0.3])
    trainingDF: DataFrame = trainTest[0]
    trainingDF.show()

    printTaskLine(2)
    # Task 2: Create a VectorAssembler for mapping input column "X" to "features" column and
    #         apply it to trainingDF in order to create assembled training data frame
    vectorAssembler: VectorAssembler = VectorAssembler(inputCols=['X'], outputCol='features')
    assembledTrainingDF: DataFrame = vectorAssembler.transform(trainingDF)
    assembledTrainingDF.show()

    printTaskLine(3)
    # Task 3: Create a LinearRegression object and fit using the training data to get a LinearRegressionModel object
    lr: LinearRegression = LinearRegression(labelCol='label')

    print(lr.explainParams())

    lrModel: LinearRegressionModel = lr.fit(assembledTrainingDF)
    lrModel.summary.predictions.show()

    printTaskLine(4)
    # Task 4: Apply the model to the whole dataDF
    allPredictions: DataFrame = vectorAssembler.transform(dataDF)
    allPredictions.show()

    printTaskLine(5)
    # Task 5: Use the LinearRegressionModel to predict y for values [-0.5, 3.14, 7.5]
    testlist: List = [Row(-0.5), Row(3.14), Row(7.5)]

    testDF: DataFrame = spark.createDataFrame(testlist, schema=StructType([StructField('X', DoubleType(), True)]))
    assembledTestDF: DataFrame = vectorAssembler.transform(testDF)
    assembledTestDF.show()

    testlr: LinearRegression = LinearRegression(featuresCol='features')
    lrpreds = lrModel.transform(assembledTestDF)
    lrpreds.select('features','prediction').show()

    printTaskLine(6)
    # Task 6: File "data/numbers.csv" contains one column "X" with several more x values.
    #         Use the LinearRegressionModel to predict the corresponding y values for them.
    numberPredictionsDF: DataFrame = spark.read.csv('data/numbers.csv', header=True, schema=StructType([StructField('X', DoubleType(), True)]))
    numberPredictionsDF.show()

    assembledTDF: DataFrame = vectorAssembler.transform(numberPredictionsDF)
    testlr2: LinearRegression = LinearRegression(featuresCol='features')
    lrpreds2 = lrModel.transform(assembledTDF)
    lrpreds2.select('features','prediction').show()

    printTaskLine(7)
    # Task 7: Store the resulting DataFrame from task 6 into the folder "results" in CSV format.
    #         NOTE: It is ok if you get multiple files with long file names
    lrpreds2.withColumn('prediction', f.col('prediction').cast('string'))\
    .withColumn('features', f.col('features').cast('string'))\
    .withColumn('X', f.col('X').cast('string'))\
    .write.csv('Results')

    # Stop the Spark session
    spark.stop()

# Helper function to separate the task outputs from each other
def printTaskLine(taskNumber: int) -> None:
    print(f"======\nTask {taskNumber}\n======")


if __name__ == "__main__":
    main()
