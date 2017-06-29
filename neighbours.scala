package com.neighbours

import org.apache.spark.sql.{Dataset,Row}
import org.apache.spark.sql.functions._
import scala.language.implicitConversions
import scala.language.reflectiveCalls
import Utils._


object Utils {

  def average[T](ts: Iterable[T])(implicit num: Numeric[T]) = {
    num.toDouble(ts.sum) / ts.size
  }

  implicit def iterebleWithAvg[T:Numeric](data:Iterable[T]) = new {
    def avg = average(data)
  }
}


object KNNRegressor {

  def trainRegressor(k: Int, features: Dataset[Row], targets: Dataset[Double]) =
    new KNNRegressor(k, features, targets)

}


private[neighbours] class KNNRegressor(private val k: Int,
                                      private val features: Dataset[Row],
                                      private val labels: Dataset[Double]) {


  def predict(feature: Row): Double = {
    val distances = getDistances(feature).zipWithIndex.sortWith {
      case ((valA, idxA), (valB, idxB)) => valA < valB
    }
    val indices = distances.take(k).map(_._2)
    labels.collect.zipWithIndex.filter { case (v, idx) =>
      indices.contains(idx)
    }.map(_._1).toSeq.avg
  }


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
