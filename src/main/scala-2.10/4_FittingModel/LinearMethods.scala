package FittingModel

/**
  * Logistic Regression
  * Linear Least Squares
  */

import org.apache.spark.SparkContext
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.BinaryLogisticRegressionSummary
import org.apache.spark.ml.regression.LinearRegression

object LinearMethods {

  def main(args: Array[String]): Unit = {

    val sc = new SparkContext("local[*]", "LinearMethods")

    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._
    sc.setLogLevel("WARN")


    val data = MLUtils.loadLibSVMFile(sc, "data/sample_libsvm_data.txt").toDF()
    data.show(5)

    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

    //---------------------------------------------------------------

    // ------- LOGISTIC REGRESSION ------
    println(" ------- LOGISTIC REGRESSION --------")

    val logr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    val logrModel = logr.fit(trainingData)

    println(s"Weights: ${logrModel.coefficients} \nIntercept: ${logrModel.intercept}")

    val trainingSummaryLR = logrModel.summary

    // OBJECTIVE HISTORY: Error of the model at every iteration
    val objectiveHistoryLR = trainingSummaryLR.objectiveHistory

    println(trainingSummaryLR.toString)
    println(objectiveHistoryLR.toList)

    // Model Evaluation
    val binarySummary = trainingSummaryLR
      .asInstanceOf[BinaryLogisticRegressionSummary]

    println("------- > Evaluation of the model\n areaUnderRoc: " + binarySummary.areaUnderROC)

    val fMeasure = binarySummary.fMeasureByThreshold

    fMeasure.show()

    val maxFMeasure = fMeasure.agg("F-Measure" -> "max")
      .head().getDouble(0)

    val bestThreshold = fMeasure.where($"F-Measure" === maxFMeasure)
      .select("threshold")
      .head().getDouble(0)

    println("maxFMeasure = " + maxFMeasure + "\nbestThreshold = " + bestThreshold)
    binarySummary.pr.show(3)
    binarySummary.precisionByThreshold.show(3)
    binarySummary.recallByThreshold.show(3)
    binarySummary.roc.show(3)

    // ------- LINEAR REGRESSION ------
    println("\n\n ------- LINEAR REGRESSION --------")

    val lr = new LinearRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    val lrModel = lr.fit(trainingData)

    println(s"Weights: ${lrModel.coefficients}\nIntercept: ${lrModel.intercept}")

    // Model Evaluation
    val trainingSummaryLLS = lrModel.summary

    println(s"numIterations: ${trainingSummaryLLS.totalIterations}")

    println(s"objectiveHistory: ${trainingSummaryLLS.objectiveHistory.toList}")

    println("RESIDUALS")
    trainingSummaryLLS.residuals.show(5)

    println("Explained Varience: " + trainingSummaryLLS.explainedVariance)
    println("Mean Absolute Error: " + trainingSummaryLLS.meanAbsoluteError)
    println("Mean Squared Error: " + trainingSummaryLLS.meanAbsoluteError)
    println("r2: " + trainingSummaryLLS.r2)



    sc.stop()
  }
}
