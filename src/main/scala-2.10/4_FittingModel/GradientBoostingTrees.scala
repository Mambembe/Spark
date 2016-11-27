package FittingModel

import org.apache.spark.SparkContext
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SQLContext

import org.apache.spark.ml.classification.GBTClassificationModel
import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.regression.GBTRegressionModel
import org.apache.spark.ml.regression.GBTRegressor
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.feature.{StringIndexer, IndexToString, VectorIndexer}

object GradientBoostingTrees {

  def main(args: Array[String]): Unit = {

    val sc = new SparkContext("local[*]", "GradientBoostingTrees")

    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._
    sc.setLogLevel("WARN")

    val data = MLUtils.loadLibSVMFile(sc, "data/sample_libsvm_data.txt").toDF()
    data.show(5)

    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))


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

    // ------- GRADIENT-BOOSTING CLASSIFIER ------
    println(" ------- GRADIENT-BOOSTING CLASSIFIER --------")

    val gbtC = new GBTClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setMaxIter(10)

    val pipelineGBTC = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, gbtC, labelConverter))

    val modelGBTC = pipelineGBTC.fit(trainingData)

    val predictionsGBTC = modelGBTC.transform(testData)

    predictionsGBTC.select("predictedLabel", "label", "features").show(5)

    val gbtModelC = modelGBTC
      .stages(2)
      .asInstanceOf[GBTClassificationModel]

    println("Learned classification GBT model:\n" + gbtModelC.toDebugString)



    // ------- GRADIENT-BOOSTING REGRESSION ------
    println(" ------- GRADIENT-BOOSTING REGRESSION --------")

    val gbtR = new GBTRegressor()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures")
      .setMaxIter(10)

    val pipelineGBTR = new Pipeline()
      .setStages(Array(featureIndexer, gbtR))

    val modelGBTR = pipelineGBTR.fit(trainingData)

    val predictionsGBTR = modelGBTR.transform(testData)

    predictionsGBTR.select("prediction", "label", "features").show(5)

    sc.stop()

  }
}
