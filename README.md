# Scala/Spark nearest neighbours

This project contains a simple implementation of a K-nearest neighbours algorithm
written in Scala using Spark.

The dependencies of the project are:
- spark-core
- spark-sql

### Simple usage example with a dataset in csv format
```scala
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DoubleType
import com.neighbours.KNNRegressor


val spark = SparkSession
  .builder()
  .master("local")
  .appName("Nearest Neighbours")
  .getOrCreate()

import spark.implicits._

// Reading the dataset in csv format
val df = spark
  .read
  .option("header", false)
  .option("inferSchema", true)
  .csv(...)

// Casting all values to Double type
val newDf = df.select(df.columns.map(c => df(c).cast(DoubleType)):_*)

// Splitting data in train & test set
val Array(train, test) = newDf.randomSplit(Array(0.7, 0.3))

// Converting Dataset[Row] to Dataset[Double]
val targs = spark.createDataset(
  train.select(train("_c5")).collect.map { case Row(v: Double) => v }
)
val feats = train.drop("_c5")

feats.show
targs.show

val knn = KNNRegressor.trainRegressor(1, feats, targs)

val testFeats = test.drop("_c5")
val testTargs = spark.createDataset(
  test.select(test("_c5")).collect.map { case Row(v: Double) => v}
)

// Printing predicted & actual value for each test example
(testFeats.collect, testTargs.collect).zipped.foreach { (f,t) =>
  println(knn.predict(f), t)
}

spark.stop
```
