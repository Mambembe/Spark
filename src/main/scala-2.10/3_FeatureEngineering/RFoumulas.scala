package FeatureEngineering

import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext

object RFoumulas {

  def main(args: Array[String]): Unit = {

    val sc = new SparkContext("local[*]", "RFormulas")

    val sqlContext = new SQLContext(sc)
    sc.setLogLevel("WARN")
    import sqlContext.implicits._
    import org.apache.spark.sql.functions._

    val crimes = sqlContext.read
      .format("com.databricks.spark.csv")
      .option("delimiter", ",")
      .option("header", "true") // Use first line of all files as header
      .option("inferSchema", "true") // Automatically infer data types
      .load("data/UScrime2-colsLotsOfNAremoved.csv")

    import org.apache.spark.ml.feature.RFormula

    val formula = new RFormula()
      .setFormula("ViolentCrimesPerPop ~ householdsize + PctLess9thGrade + pctWWage")
      .setFeaturesCol("features")
      .setLabelCol("label")

    val output = formula.fit(crimes).transform(crimes)

    output.select("features", "label").show(10)


    sc.stop()
  }
}
