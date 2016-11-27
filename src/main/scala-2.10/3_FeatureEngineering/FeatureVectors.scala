package FeatureEngineering

import FeatureEngineering.FeatureVectors.Customer
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext

object FeatureVectors {

  case class Customer(churn: Int, sessions: Int, revenue: Double, recency: Int)


  def main(args: Array[String]): Unit = {

    val sc = new SparkContext("local[*]", "StatisticsAndSampling")

    val sqlContext = new SQLContext(sc)
    sc.setLogLevel("WARN")
    import sqlContext.implicits._
    import org.apache.spark.sql.functions._


    val customers = {
      sqlContext.sparkContext.parallelize(
            Customer(1, 20, 61.24, 103) ::
            Customer(1, 8, 80.64, 23) ::
            Customer(0, 4, 100.94, 42) ::
            Customer(0, 8, 99.48, 26) ::
            Customer(1, 17, 120.56, 47) :: Nil
      ).toDF()
    }

    customers.show()

    import org.apache.spark.ml.feature.VectorAssembler

    val assembler = new VectorAssembler()
      .setInputCols(Array("sessions", "revenue", "recency"))
      .setOutputCol("features")

    val dfWithFeatures = assembler.transform(customers)

    dfWithFeatures.show()

    // VECTOR SLICERS
    import org.apache.spark.ml.feature.VectorSlicer

    val slicer = new VectorSlicer()
      .setInputCol("features")
      .setOutputCol("some_features")

    slicer.setIndices(Array(0,1)).transform(dfWithFeatures).show()


    sc.stop()
  }
}
