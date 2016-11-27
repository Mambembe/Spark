package PreparingData

/**
  * Understand, create, and use a Transformer
  * Understand, create, and use an Estimator (Logistic Regression)
  * Set parameters of Transformers and Estimators
  * Create a feature Vector with VectorAssembler
  */

import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext

object TransformersAndEstimators {

  def main(args: Array[String]): Unit = {

    val sc = new SparkContext("local[*]", "StatisticsAndSampling")

    val sqlContext = new SQLContext(sc)
    sc.setLogLevel("WARN")

    // TRANSFORMERS
    import org.apache.spark.ml.feature.{Tokenizer, RegexTokenizer}

    val sentenceDataFrame = sqlContext.createDataFrame(
      Seq((0, "Hi I heard about Spark"),
        (1, "I wish Java could use case classes"),
        (2, "Logistic, regression, models"))).toDF("label", "sentence")

    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")

    val tokenized = tokenizer.transform(sentenceDataFrame)

    tokenized.show()

    // ESTIMATORS (Logistic Regression)
    println("ESTIMATORS")

    import org.apache.spark.ml.classification.LogisticRegression
    import org.apache.spark.mllib.linalg.{Vector, Vectors}

    val training = sqlContext.createDataFrame(Seq(
      (1.0, Vectors.dense(0.0, 1.1, 0.1)),
      (0.0, Vectors.dense(2.0, 1.0, -1.0)),
      (0.0, Vectors.dense(2.0, 1.3, 1.0)),
      (1.0, Vectors.dense(0.0, 1.2, -0.5)))).toDF("label", "features")

    val lr = new  LogisticRegression()
    lr.setMaxIter(10).setRegParam(0.01)
    val model1 = lr.fit(training)

    model1.transform(training).show()

    // PARAMETER SETTING
    println("PARAMETER SETTING")

    import org.apache.spark.ml.param.ParamMap

    val paramMap = ParamMap(lr.maxIter -> 20, lr.regParam -> 0.01)

    val model2 = lr.fit(training, paramMap)

    model2.transform(training).show()

    // VECTOR ASSEMBLER
    println(" --- VECTOR ASSEMBLER ---")

    import org.apache.spark.ml.feature.VectorAssembler
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

    dfVec.select("id", "features").show()




  }
}
