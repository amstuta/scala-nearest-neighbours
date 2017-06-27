# Scala/Spark nearest neighbours

This project contains a simple implementation of a K-nearest neighbours algorithm
written in Scala using Spark.

The dependencies of the project are:
- spark-core
- spark-sql

### Simple usage example with a random dataset
```scala
import scala.language.postfixOps
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions._
import org.apache.spark.sql.SparkSession

val session = SparkSession
  .builder()
  .master("local")
  .appName("Nearest Neighbours")
  .getOrCreate()

import session.implicits._

val r = scala.util.Random
val feats = (1 to 100).map { i => (r.nextDouble, r.nextDouble) } toSeq
val targs = (1 to 100).map { i => r.nextDouble * 100 } toSeq
val data = session.createDataset(feats)
val labels = session.createDataset(targs)

data.show
labels.show

val knn = new KNNRegressor[(Double, Double)](k=3)
knn.fit(data, labels)

val row = session.createDataset(Seq((0.387, 0.876)))
println(knn.predict(row))
```
