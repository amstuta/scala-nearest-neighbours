package com.neighbours

import org.apache.spark.sql.{Dataset,Row}
import org.apache.spark.sql.functions._
import scala.language.implicitConversions
import scala.language.reflectiveCalls
import Utils._


/** Helper object to add an average function to numeric iterables */
object Utils {

  def average[T](ts: Iterable[T])(implicit num: Numeric[T]) = {
    num.toDouble(ts.sum) / ts.size
  }

  implicit def iterebleWithAvg[T:Numeric](data:Iterable[T]) = new {
    def avg = average(data)
  }
}


object KNN {

  /** Returns a trained KNNRegressor algorithm
   *
   * @param k         Number of neighbours to use for prediction
   * @param features  Dataset[Row] containing the training features
   * @param targets   Dataset[T:Numeric] containing the training targets
   */
  def trainRegressor[T:Numeric](k: Int, features: Dataset[Row], targets: Dataset[T]) =
    new KNNRegressor(k, features, targets)


  /** Returns a trained KNNClassifier algorithm
   *
   * @param k         Number of neighbours to use for prediction
   * @param features  Dataset[Row] containing the training features
   * @param targets   Dataset[T] containing the training targets
   */
  def trainClassifier[T](k: Int, features: Dataset[Row], targets: Dataset[T]) =
    new KNNClassifier(k, features, targets)
}


/** K-nearest neighbours algorithm for classification problems.
 *
 * @param k         Number of neighbours to use for prediction
 * @param features  Dataset[Row] containing the training features
 * @param labels    Dataset[T] containing the training targets
 */
sealed class KNNClassifier[T](protected val k: Int,
                              protected val features: Dataset[Row],
                              protected val labels: Dataset[T])
                              extends KNN[T] {

   /** Predicts a class given a particular feature.
    *
    * @param feature Row composed of Double objects
    * @return  Predicted value of type T
    */
  def predict(feature: Row): T = {
    val distances = getDistances(feature).zipWithIndex.sortWith {
      case ((valA, idxA), (valB, idxB)) => valA < valB
    }
    val indices = distances.take(k).map(_._2)
    labels.collect.zipWithIndex.filter { case (v, idx) =>
      indices.contains(idx)
    }.map(_._1).toSeq.groupBy(identity).maxBy(_._2.size)._1
  }

}


/** K-nearest neighbours algorithm for regression problems.
 *
 * @param k         Number of neighbours to use for prediction
 * @param features  Dataset[Row] containing the training features
 * @param labels    Dataset[T:Numeric] containing the training targets
 */
sealed class KNNRegressor[T:Numeric](protected val k: Int,
                                     protected val features: Dataset[Row],
                                     protected val labels: Dataset[T])
                                     extends KNN[T] {

  /** Predicts a value given a particular feature.
   *
   * @param feature Row composed of Double objects
   * @return  Predicted value of type T
   */
  def predict(feature: Row): T = {
    val distances = getDistances(feature).zipWithIndex.sortWith {
      case ((valA, idxA), (valB, idxB)) => valA < valB
    }
    val indices = distances.take(k).map(_._2)
    labels.collect.zipWithIndex.filter { case (v, idx) =>
      indices.contains(idx)
    }.map(_._1).toSeq.avg.asInstanceOf[T]
  }

}


sealed trait KNN[T] {

  protected val k: Int
  protected val features: Dataset[Row]
  protected val labels:   Dataset[T]


  def predict(feature: Row): T


  /** Calculates the distances between an example and the training features.
   *
   * @param feature Row composed of Double objects
   * @return  A Seq[Double] containing the distance between each training
   *          example and the given row
   */
  def getDistances(feature: Row): Seq[Double] = {
    val feat = feature.toSeq
    val pSquared = feat.map(_.asInstanceOf[Double]).map(v => v * v).reduce(_+_)
    val qSquared = features
      .select(features.columns.map(col => pow(features(col), 2)).reduce((c1, c2) => c1 + c2) as "sq")
    val prod = features.collect.map { row =>
      row.toSeq.zipWithIndex.map {
        case (e, idx) => e.asInstanceOf[Double] * feat(idx).asInstanceOf[Double]
      }.reduce(_+_) * -2
    }

    (qSquared
      .select(qSquared("sq") + pSquared)
      .collect.map(_(0).asInstanceOf[Double]), prod).zipped.map((l,r) => l + r)
  }

}
