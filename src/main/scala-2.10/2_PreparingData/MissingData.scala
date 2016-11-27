package PreparingData

/**
  * "na"-method of DataFrames provides functionality for missing values
  * Returns an instance of DataFrameNAFunctions
  * 3 methods available:
  * -  drop
  * -  fill
  * -  replace
  */

import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.linalg.{Matrix, Vector, Vectors}
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.sql.functions._

object MissingData {

  def main(args: Array[String]): Unit = {

    val sc = new SparkContext("local[*]", "StatisticsAndSampling")

    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._
    sc.setLogLevel("WARN")

    val df = sqlContext.range(0, 10)
      .select("id")
      .withColumn("uniform", rand(10L))
      .withColumn("normal", randn(10L))

    // udf takes a whole dataset column as input, performs operations over each
    // value and outputs a new dataframe column
    val halfToNaN = udf[Double, Double] (x => if (x > 0.5) Double.NaN else x)
    val oneToNaN = udf[Double, Double] (x => if (x > 1.0) Double.NaN else x)

    val dfNaN = df.withColumn("nanUniform", halfToNaN(df("uniform")))
      .withColumn("nanNormal", oneToNaN(df("normal")))
      .drop("uniform")
      .withColumnRenamed("nanUniform", "uniform")
      .drop("normal")
      .withColumnRenamed("nanNormal", "normal")

    println(dfNaN.show())

    // ------------------ DROP ---------------------
    println("--- DROP ---")

    // Drops rows with less then 3 non-null values
    dfNaN.na.drop(minNonNulls = 3).show()

    // Drops rows with both values as null
    dfNaN.na.drop("all", Array("uniform", "normal")).show()

    // Drops rows with at least one null value
    dfNaN.na.drop("any", Array("uniform", "normal")).show()


    // ------------------ FILL ---------------------
    println("--- FILL ---")

    dfNaN.na.fill(0.0).show()

    // Compute the mean for the uniform column and uses it as a
    // default value for filling null values
    val uniformMean = dfNaN.filter("uniform <> 'NaN'")
          .groupBy()
          .agg(mean("uniform")).first()(0)

    dfNaN.na.fill(Map("uniform" -> uniformMean)).show()

    // Does the same thing for both columns at once
    val dfCols = dfNaN.columns.drop(1)

    val dfMeans = dfNaN.na.drop().groupBy()
      .agg(mean("uniform"), mean("normal"))
      .first().toSeq

    val meansMap = (dfCols.zip(dfMeans)).toMap

    println("Means:")
    dfCols.zip(dfMeans).foreach(println)

    dfNaN.na.fill(meansMap).show()


    // ------------ REPLACE -------------
    println("--- REPLACE ---")

    dfNaN.na.replace("uniform", Map(Double.NaN -> 0.0)).show()

    // ------------ DUPLICATES ----------
    println("--- DUPLICATES ---")

    val dfDuplicates = df.unionAll(sc.parallelize(Seq((10, 1, 1), (11, 1, 1))).toDF())

    println("Dataframe with duplicates")
    println(dfDuplicates.show())

    val dfCols2 = dfNaN.columns.drop(1)

    dfDuplicates.dropDuplicates(dfCols2).show()





  }

}
