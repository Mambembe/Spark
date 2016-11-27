package FittingModel

import org.apache.spark.SparkContext
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SQLContext

import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.ml.regression.RandomForestRegressionModel
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.feature.{StringIndexer, IndexToString, VectorIndexer}

object RandomForests {

  def main(args: Array[String]): Unit = {

    val sc = new SparkContext("local[*]", "RandomForests")

    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._
    sc.setLogLevel("WARN")

    val data = MLUtils.loadLibSVMFile(sc, "data/sample_libsvm_data.txt").toDF()
    data.show(5)

    //---------------------------------------------------------------

    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(data)

    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(data)

    //-----------------------------------------------------------------

    // ------- RANDOM FOREST CLASSIFIER ------
    println(" ------- RANDOM FOREST CLASSIFIER --------")

    val rfC = new RandomForestClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setNumTrees(3)

    val Array(trainigData, testData) = data.randomSplit(Array(0.7, 0.3))

    val pipelineRFC = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, rfC, labelConverter))

    val modelRFC = pipelineRFC.fit(trainigData)

    // Predict
    val predictionsRFC = modelRFC.transform(testData)

    predictionsRFC.select("predictedLabel", "label", "features").show(5)

    // We can derive the features classification model and look at feature importance
    println("Features importance")

    val rfModelC = modelRFC.stages(2).asInstanceOf[RandomForestClassificationModel]

    println(rfModelC.featureImportances)

    println("Learned classification forest model:\n" + rfModelC.toDebugString)


    // ------- RANDOM FOREST REGRESSION ------
    println(" ------- RANDOM FOREST REGRESSION --------")

    val rfR = new RandomForestRegressor()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures")

    val pipelineRFR = new Pipeline()
      .setStages(Array(featureIndexer, rfR))

    val modelRFR = pipelineRFR.fit(trainigData)

    val predictionsRFR = modelRFR.transform(testData)

    predictionsRFR.select("prediction", "label", "features").show(5)


    sc.stop()
  }
}
