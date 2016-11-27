package ML
/**
  * Sampling
  * Random Split
  * Stratified Sampling
 */

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}

object sampling {

  def main(args: Array[String]): Unit = {

    val sc = new SparkContext("local[*]", "Word Count")
    sc.setLogLevel("WARN")

    val elements: RDD[Vector] = sc.parallelize(Array(
      Vectors.dense(4, 7, 13),
      Vectors.dense(-2, 8, 4),
      Vectors.dense(3, -11, 19)))

    val res00 = elements.sample(withReplacement=false, fraction=0.5, seed=10L).collect()
    val res01 = elements.sample(withReplacement=false, fraction=0.5, seed=7L).collect()
    val res02 = elements.sample(withReplacement=false, fraction=0.5, seed=64L).collect()

    println("res00")
    res00.foreach(x=>println(x))
    println("res01")
    res01.foreach(x=>println(x))
    println("res02")
    res02.foreach(x=>println(x))

    // RANDOM SPLIT
    val data = sc.parallelize(1 to 1000000)
    val splits = data.randomSplit(Array(0.6, 0.2, 0.2), seed = 13L)
    val training = splits(0)
    val test = splits(1)
    val validation = splits(2)

    println("Random Split")
    val res03 = splits.map(_.count())
    res03.foreach(x=>println(x))

    // STRATIFIED SAMPLING
    import org.apache.spark.mllib.linalg.distributed.IndexedRow

    // 2 rows with the same index
    val rows: RDD[IndexedRow] = sc.parallelize(Array(
      IndexedRow(0, Vectors.dense(1, 2)),
      IndexedRow(1, Vectors.dense(4, 5)),
      IndexedRow(1, Vectors.dense(7, 8))))

    // The keys in the map are matched with the keys in the RDD that I'm sampling from
    // The values in the map are the probabilities of drawing an element that is
    // associated with a given key.
    val fractions: Map[Long, Double] = Map(0L -> 1.0, 1L -> 0.5)

    val approxSample = rows.map{
      case IndexedRow(index, vec) => (index, vec)
    }.sampleByKey(withReplacement = false, fractions, 9L)

    val res04 = approxSample.collect()
    println("Stratified Sampling:")
    res04.foreach(x=>println(x))

    sc.stop()


  }

}
