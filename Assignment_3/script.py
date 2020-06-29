# PDP assignment 3 by Lennart de Waart (563079@student.inholland.nl)
# Import Python-packages
import findspark  # To initialize Spark
# To initialize and store the Spark session
from pyspark.sql import SparkSession
from pyspark.sql.types import *  # To support datatype conversions
# To create Dense Vectors for scaling and regression
from pyspark.ml.linalg import DenseVector
# To scale values to prepare for regression
from pyspark.ml.feature import StandardScaler
# To calculate a regression model
from pyspark.ml.regression import LinearRegression
from pyspark.sql.functions import when  # To support classification
# You should also have installed Pandas and Numpy to process this script successfully

# Data preparation
# Initialize path
findspark.init("C:/Users/ldewa/Downloads/PDP/spark-3.0.0-bin-hadoop2.7")

# Initialize a Spark session
spark = SparkSession.builder.master("local").appName(
    "Assignment 3").config("spark.executor.memory", "1gb").getOrCreate()

# Read the dataset (with headers, infer datatypes)
df = spark.read.csv("C:/Users/ldewa/Downloads/PDP/titanic.csv",
                    inferSchema=True, header=True)

# Columns Age and Sex are not fit for analysis, so lets change that
# Column Age should have an integer datatype
df = df.withColumn("Age", df.Age.cast(IntegerType()))
# Column Sex should be classified as numeric values to support analysis
df = df.withColumn("Sex", when(
    df.Sex == "male", 0).when(df.Sex == "female", 1))

# Question 1
# Calculate the probability of individual variables
male_prob = len(df.toPandas()[df.toPandas().Sex == 0]) / len(df.toPandas())
female_prob = len(df.toPandas()[df.toPandas().Sex == 1]) / len(df.toPandas())
print('Male probability in dataset: ' + str(male_prob))
print('Female probability in dataset: ' + str(female_prob))

pc1_prob = len(df.toPandas()[df.toPandas().Pclass == 1]) / len(df.toPandas())
pc2_prob = len(df.toPandas()[df.toPandas().Pclass == 2]) / len(df.toPandas())
pc3_prob = len(df.toPandas()[df.toPandas().Pclass == 3]) / len(df.toPandas())
print('Passenger class 1 probability in dataset: ' + str(pc1_prob))
print('Passenger class 2 probability in dataset: ' + str(pc2_prob))
print('Passenger class 3 probability in dataset: ' + str(pc3_prob))

survived_prob = len(
    df.toPandas()[df.toPandas().Survived == 1]) / len(df.toPandas())
print('Survived probability in dataset: ' + str(survived_prob))

print('\n------------- Question 1 -------------')
# Answer probability questions
q1a = round((survived_prob * female_prob * pc1_prob) *
            100, 2)  # round output to 2 decimals
q1b = round((survived_prob * female_prob * pc2_prob) * 100, 2)
q1c = round((survived_prob * female_prob * pc3_prob) * 100, 2)
q1d = round((survived_prob * male_prob * pc1_prob) * 100, 2)
q1e = round((survived_prob * male_prob * pc2_prob) * 100, 2)
q1f = round((survived_prob * male_prob * pc3_prob) * 100, 2)
print('a)   P(S = true | G = female, C = 1) = ' + str(q1a) + '%')
print('b)   P(S = true | G = female, C = 2) = ' + str(q1b) + '%')
print('c)   P(S = true | G = female, C = 3) = ' + str(q1c) + '%')
print('d)   P(S = true | G = male, C = 1) = ' + str(q1d) + '%')
print('e)   P(S = true | G = male, C = 2) = ' + str(q1e) + '%')
print('f)   P(S = true | G = male, C = 3) = ' + str(q1f) + '%')
print('\n')  # format output

# Question 2
# Calculate the probability of individual variables
age10oryounger_prob = len(
    df.toPandas()[df.toPandas().Age <= 10]) / len(df.toPandas())
print('Age 10 or younger probability: ' + str(age10oryounger_prob))

survived_prob = len(
    df.toPandas()[df.toPandas().Survived == 1]) / len(df.toPandas())
print('Survived probability in dataset: ' + str(survived_prob))

print('\n------------- Question 2 -------------')
# Answer
q2 = round((survived_prob * age10oryounger_prob * pc3_prob)
           * 100, 2)  # round output to 2 decimals
print('P(S = true | A <= 10, C = 3) = ' + str(q2) + '%')

# Question 3
# Create a Linear Regression model
# Create a dataframe with only columns we need
df_select = df.select("Fare", "Pclass")
input_data = df_select.rdd.map(lambda x: (x[0], DenseVector(x[1:])))
df_vector = spark.createDataFrame(
    input_data, ["fare", "features"])  # Create a vector dataframe
# Scale the Pclass values to make it more fit for analysis
standardScaler = StandardScaler(
    inputCol="features", outputCol="features_scaled")
scaler = standardScaler.fit(df_vector)
df_scaled = scaler.transform(df_vector)

# Create train and test data for the regression model
train_data, test_data = df_scaled.randomSplit([.8, .2], seed=1234)
# Create a Linear Regression model
lr = LinearRegression(labelCol="fare", maxIter=10,
                      regParam=0.3, elasticNetParam=0.8)
model = lr.fit(train_data)

print('\n------------- Question 3 -------------')
# Print some important statistics from the regression model
print('Linear Regression model statistics for dependent Fare and independent Pclass:')
print("Coefficient(s): %s" % str(model.coefficients))
print("Intercept: %s" % str(model.intercept))
print("RMSE: %f" % model.summary.rootMeanSquaredError)
print("r2: %f" % model.summary.r2)
print('\n')

# Answer y = b + ax or fare = intercept + coefficient * pclass
# round output to 2 decimals
q3a = round(model.intercept + (model.coefficients[0] * 1), 2)
q3b = round(model.intercept + (model.coefficients[0] * 2), 2)
q3c = round(model.intercept + (model.coefficients[0] * 3), 2)
print('a) Passenger class 1 predicted fare: ' + str(q3a) + ' pound')
print('b) Passenger class 2 predicted fare: ' + str(q3b) + ' pound')
print('c) Passenger class 3 predicted fare: ' + str(q3c) + ' pound')
print('\n')  # format output
