package FittingModel

/**
  * Evaluate binary classification algorithms using area under the ROC curve
  * Evaluate multi-class classification algorithms using several metrics
  * Evaluate regression algorithms using several metrics
  * Evaluate logistic and linear regression algorithms using several metrics
  * Evaluate logistic and linear regression algorithms using summaries
  */

import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SQLContext

object Evaluation {

  def main(args: Array[String]): Unit = {

    val sc = new SparkContext("local[*]", "Evaluation")

    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._
    sc.setLogLevel("WARN")


    //---------------------------------------------------------------

    println("BINARY EVALUATION - Logistic Regression")
    import org.apache.spark.ml.classification.LogisticRegression
    import org.apache.spark.ml.classification.BinaryLogisticRegressionSummary
    import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

    val data = MLUtils.loadLibSVMFile(sc, "data/sample_libsvm_data.txt").toDF()
    data.show(5)
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

    val logr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    val logrModel = logr.fit(trainingData)

    println(s"Weights: ${logrModel.coefficients} \nIntercept: ${logrModel.intercept}")

    val predictionsLogR = logrModel.transform(testData)

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("rawPrediction")
      .setMetricName("areaUnderROC")

    val roc = evaluator.evaluate(predictionsLogR)
    println(roc)

    println("Press Enter to continue...")
    Console.in.read()

    //---------------------------------------------------------------

    println("MULTI-CLASS EVALUATION - Random Forests")
    import org.apache.spark.ml.classification.RandomForestClassifier
    import org.apache.spark.ml.classification.RandomForestClassificationModel
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

    val data2 = MLUtils.loadLibSVMFile(sc, "data/sample_libsvm_data.txt").toDF()
    data2.show(5)

    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(data2)

    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(data2)

    val rfC = new RandomForestClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setNumTrees(3)

    val Array(trainigData2, testData2) = data2.randomSplit(Array(0.7, 0.3))

    val pipelineRFC = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, rfC, labelConverter))

    val modelRFC = pipelineRFC.fit(trainigData2)

    val predictionsRFC = modelRFC.transform(testData2)

    val evaluator2 = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("precision")

    val accuracy = evaluator2.evaluate(predictionsRFC)

    println("Test Error = " + (1.0 - accuracy))

    println("Press Enter to continue...")
    Console.in.read()

    //---------------------------------------------------------------

    println("REGRESSION EVALUATION - Random Forests")
    import org.apache.spark.ml.classification.RandomForestClassifier
    import org.apache.spark.ml.classification.RandomForestClassificationModel
    import org.apache.spark.ml.evaluation.RegressionEvaluator

    val rfR = new RandomForestRegressor()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures")

    val pipelineRFR = new Pipeline()
      .setStages(Array(featureIndexer, rfR))

    val modelRFR = pipelineRFR.fit(trainigData2)

    val predictionsRFR = modelRFR.transform(testData2)

    val evaluator3 = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    val rmse = evaluator3.evaluate(predictionsRFR)

    println("Root Mean Squared Error = " + rmse)






    sc.stop()
  }
}
