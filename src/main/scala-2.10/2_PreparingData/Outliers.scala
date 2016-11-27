package PreparingData

/**
  * Compute the inverse of covariance matrix given of a dataset
  * Compute Mahalanobis Distance for all elements in a dataset
  * Remove outliers from a dataset
  */

import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext

object Outliers {

  def main(args: Array[String]): Unit = {

    val sc = new SparkContext("local[*]", "StatisticsAndSampling")

    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._

    sc.setLogLevel("WARN")


    import org.apache.spark.ml.feature.VectorAssembler
    import org.apache.spark.mllib.linalg.{Vectors, Vector}
    import org.apache.spark.ml.feature.StandardScaler
    import org.apache.spark.sql.functions._

    val dfRandom = sqlContext.range(0, 10).select("id")
      .withColumn("uniform", rand(10L))
      .withColumn("normal1", randn(10L))
      .withColumn("normal2", randn(11L))

    dfRandom.show()

    val assembler = new VectorAssembler()
      .setInputCols(Array("uniform", "normal1", "normal2"))
      .setOutputCol("features")

    val dfVec = assembler.transform(dfRandom)

    // append an outlier
    val dfOutlier = dfVec.select("id", "features")
      .unionAll(sqlContext.createDataFrame(Seq((10, Vectors.dense(3,3,3)))))

    dfOutlier.sort(dfOutlier("id").desc).show()

    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeat")
      .setWithMean(true)
      .setWithStd(true)

    val scalerModel = scaler.fit(dfOutlier.select("id", "features"))

    val dfScaled = scalerModel.transform(dfOutlier).select("id", "scaledFeat")

    dfScaled.sort(dfScaled("id").desc).show()


    // INVERSE  OF COVARIANCE MATRIX

    import org.apache.spark.mllib.stat.Statistics
    import  breeze.linalg._

    val rddVec = dfScaled.select("scaledFeat")
      .rdd.map(_(0).asInstanceOf[org.apache.spark.mllib.linalg.Vector])

    val colCov = Statistics.corr(rddVec)
    println(colCov)

    val invColCovB = inv(new DenseMatrix(3, 3, colCov.toArray))
    println(invColCovB)


    // COMPUTING MAHALANOBIS DISTANCE
    import org.apache.spark.sql.functions.udf

    /*
    val mahalanobis = udf[Double, org.apache.spark.mllib.linalg.Vector]{
      v =>
        val vB =  DenseVector(v.toArray());
    }*/

    sc.stop()

  }
}
