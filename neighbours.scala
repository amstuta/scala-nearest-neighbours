package com.neighbours

import org.apache.spark.sql.Dataset
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


class KNNRegressor[T <: Product](private var k: Int) {

  private var features: Dataset[T] = _
  private var labels:   Dataset[Double] = _


  def fit(features: Dataset[T], targets: Dataset[Double]): Unit = {
    this.features = features
    this.labels   = targets
  }


  def predict(feature: Dataset[T]): Double = {
    val distances = getDistances(feature).zipWithIndex.sortWith { case (a, b) =>
      val (valA, idxA) = a
      val (valB, idxB) = b
      valA < valB
    }
    distances.take(k).map(_._2).avg
  }


  def getDistances(feature: Dataset[T]): Seq[Double] = {
    val feat = feature.collect.last.productIterator.toList
    val pSquared = feature
      .select(feature.columns.map(col => pow(feature(col), 2)).reduce((c1, c2) => c1 + c2))
      .head.toSeq.last.asInstanceOf[Double]
    val qSquared = features
      .select(features.columns.map(col => pow(features(col), 2)).reduce((c1, c2) => c1 + c2) as "sq")
    val prod = features.collect.map { row =>
      row.productIterator.toList.zipWithIndex.map {
        case (e, idx) => e.asInstanceOf[Double] * feat(idx).asInstanceOf[Double]
      }.reduce(_+_) * -2
    }

    (qSquared
      .select(qSquared("sq") + pSquared)
      .collect.map(_(0).asInstanceOf[Double]), prod).zipped.map((l,r) => l + r)
  }

}
