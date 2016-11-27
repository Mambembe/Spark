package ML

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.linalg.{Matrix, Matrices}
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary

object basicStatistics {

  def main(args: Array[String]): Unit = {

    val sc = new SparkContext("local[*]", "Word Count")
    sc.setLogLevel("WARN")

    // STATISTICAL SUMMARY
    val observations: RDD[Vector] = sc.parallelize(Array(
      Vectors.dense(1, 2),
      Vectors.dense(4, 5),
      Vectors.dense(7, 8)))

    val summary: MultivariateStatisticalSummary = Statistics.colStats(observations)

    println(summary.mean)
    println(summary.variance)
    println(summary.numNonzeros)
    println(summary.normL1)
    println(summary.normL2)

    // CORRELATION
    val X: RDD[Double] = sc.parallelize(Array(2.0, 9.0, -7.0))
    val Y: RDD[Double] = sc.parallelize(Array(1.0, 3.0, 5.0))
    val correlation1: Double = Statistics.corr(X, Y, "pearson")
    val correlation2: Double = Statistics.corr(X, Y, "spearman")

    println("Correlation:")
    println(correlation1)
    println(correlation2)

    // CORRELATION MATRIX
    val data: RDD[Vector] = sc.parallelize(Array(
      Vectors.dense(2, 9, -7),
      Vectors.dense(1, -3, 5),
      Vectors.dense(4, 0, -5)))

    val corrMatrix: Matrix = Statistics.corr(data, "pearson")

    println("Correlation Matrix:")
    println(corrMatrix)

    // RANDOM DATA GENERATION
    import org.apache.spark.mllib.random.RandomRDDs._

    val data2 = normalVectorRDD(sc, numRows = 10000L, numCols = 3, numPartitions = 10)
    val stats: MultivariateStatisticalSummary = Statistics.colStats(data2)

    println("Random Numbers Generations:")
    println(stats.mean)
    println(stats.variance)

    sc.stop()

  }

}
