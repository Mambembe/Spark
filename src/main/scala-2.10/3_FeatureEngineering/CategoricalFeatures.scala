package FeatureEngineering

/**
  * encode categorical features with Spark's StringIndexer
  * encode categorical features with Spark's OneHotEncoder
  * know when to use each of these
  */

import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext

object CategoricalFeatures {

  def main(args: Array[String]): Unit = {

    val sc = new SparkContext("local[*]", "StatisticsAndSampling")

    val sqlContext = new SQLContext(sc)
    sc.setLogLevel("WARN")
    import sqlContext.implicits._
    import org.apache.spark.sql.functions._
    import org.apache.spark.ml.feature.StringIndexer

    val df = sqlContext.createDataFrame(
      Seq((0, "US"), (1, "UK"), (2, "FR"), (3, "US"), (4, "US"), (5, "FR")))
        .toDF("id", "nationality")

    val indexer = new StringIndexer()
      .setInputCol("nationality")
      .setOutputCol("nIndex")

    val indexed = indexer.fit(df).transform(df)

    indexed.show()


    // INDEX-TO-STRING
    import org.apache.spark.ml.feature.IndexToString

    val converter = new IndexToString()
      .setInputCol("predictedIndex")
      .setOutputCol("predictedNationality")

    val predictions = indexed
      .selectExpr("nIndex as predictedIndex")

    converter.transform(predictions).show()


    // ONE-HOT-ENCODER
    // Suppose we want to fit a linear regression that uses nationality as a
    // feature. It would be impossible to learn a weight for this one feature
    // that can distinguish between the 3 nationalities in our data-set.
    // --> better have a separate boolean feature for each nationality
    //     and learn weights for those features independently

    import org.apache.spark.ml.feature.OneHotEncoder

    val encoder = new OneHotEncoder()
      .setInputCol("nIndex")
      .setOutputCol("nVector")

    val encoded = encoder.transform(indexed)

    encoded.show()

    // the  OneHotEncoder creates a sparse vector
    // convert it to a dense one
    import org.apache.spark.mllib.linalg._

    encoded.foreach{ c =>
      val dv = c.getAs[SparseVector]("nVector").toDense
      println(s"${c(0)} ${c(1)} ${dv}")
    }

    /**
      * There are only 2 values in each vector and not 3 as expected
      * This is because the last category is not included by default
      * Because if it were , it would make the vector entries linearly dependent
      */

    // To include all categories set DropLast to false

    val encoder2 = new OneHotEncoder()
      .setInputCol("nIndex")
      .setOutputCol("nVector")
      .setDropLast(false)

    val encoded2 = encoder2.transform(indexed)

    val toDense = udf[DenseVector, SparseVector] (_.toDense)

    encoded2.withColumn("denseVector", toDense(encoded2("nVector"))).show()


    sc.stop()
  }
}
