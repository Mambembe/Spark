package PreparingData


import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext


object Normalization {

  def main(args: Array[String]): Unit = {

    val sc = new SparkContext("local[*]", "StatisticsAndSampling")

    val sqlContext = new SQLContext(sc)
    sc.setLogLevel("WARN")

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

    import org.apache.spark.ml.feature.Normalizer

    val scaler1 = new Normalizer()
      .setInputCol("features").setOutputCol("scaledFeat").setP(1.0)

    scaler1.transform(dfVec.select("id", "features")).show()


    // STANDARD SCALER
    println(" --- STANDARD SCALER ---")

    import org.apache.spark.ml.feature.StandardScaler

    val scaler2 = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeat")
      .setWithStd(true)
      .setWithMean(true)

    val scaler2Model = scaler2.fit(dfVec.select("id", "features"))

    scaler2Model.transform(dfVec.select("id", "features")).show()


    // MIN-MAX SCALER
    println(" --- MIN-MAX SCALER ---")

    import org.apache.spark.ml.feature.MinMaxScaler

    val scaler3 = new MinMaxScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeat")
      .setMin(-1.0)
      .setMax(1.0)

    val scaler3Model = scaler3.fit(dfVec.select("id", "features"))

    scaler3Model.transform(dfVec.select("id", "features")).show()







  }
}
