package PreparingData

/**
  * Compute column summary statistics
  * Compute pairwise statistics between series/columns
  * Perform standard sampling on any DataFrame
  * Split any DataFrame randomly into subsets
  * Perform stratified sampling on DataFrames
  * Generate Random Data from Uniform and Normal Distributions
  */

import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.linalg.{Matrix, Vector, Vectors}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary



object StatisticsAndSampling {

  case class Record(desc: String, value1: Int, value2: Double)


  def main(args: Array[String]): Unit = {


    val sc = new SparkContext("local[*]", "StatisticsAndSampling")

    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._
    sc.setLogLevel("WARN")



    val recDF = sc.parallelize(Array(Record("first", 1, 3.7),
      Record("second", -2, 2.1), Record("third", 6, 0.7))).toDF()


    val recStats = recDF.describe()

    recStats.show()


    import org.apache.spark.sql.functions._
    val aaa = recDF.groupBy().agg(min("value1"), max("value2"))
    aaa.columns.foreach(x=>println(x))

    aaa.first().toSeq.toArray.map(_.toString.toDouble).foreach(x=>println(x))

    val recDFStat = recDF.stat

    println("Pearson Correlation: " + recDFStat.corr("value1", "value2"))
    println("Covariance: " + recDFStat.cov("value1", "value2"))
    // 0.3 is the minimum proportion of rows
    // may yield false positive
    println(recDFStat.freqItems(Seq("value1"), 0.3).show())

    // SAMPLING
    println("--- Sampling ---")
    val df = sqlContext.createDataFrame(Seq((1, 10), (1, 20), (2, 10),
      (2, 20), (2, 30), (3, 20), (3, 30))).toDF("key", "value")

    val dfSampled = df.sample(withReplacement = false, fraction = 0.3, seed = 11L)

    println(dfSampled.show())

    // RANDOM SPLIT
    println("---- Random Split ----")
    val dfSplit = df.randomSplit(weights=Array(0.7, 0.3), seed = 11L)
    println(dfSplit(0).show())
    println(dfSplit(1).show())

    // STRATIFIED SAMPLING
    println("--- Stratified Sampling ---")
    val dfStrat = df.stat.sampleBy(col="key",
      fractions=Map(1 -> 0.7, 2 -> 0.7, 3-> 0.7),
      seed=11L)
    println(dfStrat.show())

    // RANDOM VALUES
    println("--- Random Data ---")
    import org.apache.spark.sql.functions.{rand, randn}

    val dff = sqlContext.range(0, 10)
    val randDf = dff.select("id")
      .withColumn("uniform", rand(10L))
      .withColumn("normal", randn(10L))

    println(randDf.show())




    sc.stop()
  }
}
