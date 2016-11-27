package ML

/**
  * Goodness of fit
  * Independence
  * Equality of probability distributions
  * Kernel density estimation
  */

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.linalg.{Matrix, Matrices}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.stat.Statistics

object hypothesisTesting {

  def main(args: Array[String]): Unit = {

    val sc = new SparkContext("local[*]", "Word Count")
    sc.setLogLevel("WARN")

    // Goodness of fit
    // tested against a uniform distribution
    val vec: Vector = Vectors.dense(0.3, 0.2, 0.15, 0.1, 0.1, 0.05)
    val goodnessOfFitResult = Statistics.chiSqTest(vec)
    println(goodnessOfFitResult)

    // Independence
    val mat: Matrix = Matrices.dense(3, 2, Array(13, 47, 40, 80, 11, 9))
    val independenceResult = Statistics.chiSqTest(mat)
    println("__________________")
    println(independenceResult)

    import org.apache.spark.mllib.regression.LabeledPoint
    import org.apache.spark.mllib.stat.test.ChiSqTestResult

    val obs: RDD[LabeledPoint] = sc.parallelize(Array(
      LabeledPoint(0, Vectors.dense(1, 2)),
      LabeledPoint(0, Vectors.dense(0.5, 1.5)),
      LabeledPoint(1, Vectors.dense(1, 8))))

    val featureTestResults: Array[ChiSqTestResult] = Statistics.chiSqTest(obs)
    println("__________________")
    featureTestResults.foreach(x=>print(x))

    // Kolmogorov-Smirnof
    import org.apache.spark.mllib.random.RandomRDDs.normalRDD
    import org.apache.spark.mllib.random.RandomRDDs.uniformRDD


    val data: RDD[Double] = normalRDD(sc, size=100, numPartitions = 1, seed=13L)
    val testResult = Statistics.kolmogorovSmirnovTest(data, "norm", 0, 1)

    println("\n__________________")
    println(testResult)

    val data2: RDD[Double] = uniformRDD(sc, size=100, numPartitions = 1, seed=13L)
    val testResult2 = Statistics.kolmogorovSmirnovTest(data2, "norm", 0, 1)

    println("\n__________________")
    println(testResult2)

    // Kernel Density Estimation
    import org.apache.spark.mllib.stat.KernelDensity

    val data3: RDD[Double] = normalRDD(sc, size = 1000, numPartitions = 1, seed = 17L)
    val kd = new KernelDensity().setSample(data3).setBandwidth(0.1)
    val densities = kd.estimate(Array(-1.5, -1, -0.5, 0, 0.5, 1, 1.5))

    println("\n__________________")
    densities.foreach(x=>println(x))

    val data4: RDD[Double] = uniformRDD(sc, size = 1000, numPartitions = 1, seed = 17L)
    val kd2 = new KernelDensity().setSample(data4).setBandwidth(0.1)
    val densities2 = kd2.estimate(Array(-1.5, -1, -0.5, 0, 0.5, 1, 1.5))

    println("\n__________________")
    densities2.foreach(x=>println(x))

    sc.stop()


  }

}
