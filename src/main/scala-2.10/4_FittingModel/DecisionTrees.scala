package FittingModel

import org.apache.spark.SparkContext
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SQLContext

import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.feature.{StringIndexer, IndexToString, VectorIndexer}


object DecisionTrees {

  def main(args: Array[String]): Unit = {

    val sc = new SparkContext("local[*]", "DecisionTrees")

    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._
    sc.setLogLevel("WARN")

    // Load Data
    val data = MLUtils.loadLibSVMFile(sc, "data/sample_libsvm_data.txt").toDF()
    data.show(5)

    // ------- DECISION TREE CLASSIFIER ------
    println(" ------- DECISION TREE CLASSIFIER --------")

    // Decision Tree Model
    val dtC = new DecisionTreeClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")

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

    val pipelineClass = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, dtC, labelConverter))

    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

    val modelClassifier = pipelineClass.fit(trainingData)

    // Create the treeModels by casting the data to an instance
    // of a decision tree classification model
    val treeModel = modelClassifier.stages(2)
      .asInstanceOf[DecisionTreeClassificationModel]

    println("Learned classification tree model:\n" + treeModel.toDebugString)

    // Getting results from the model
    /**
      * Notice that "prediction" and "predictedLabel" have opposite values.
      * This is because Spark assigns its own values while handling categorical features
      */
    val predictionsClass = modelClassifier.transform(testData)

    predictionsClass.show(5)


    // ------- DECISION TREE REGRESSION ------
    println(" ------- DECISION TREE REGRESSION --------")

    val dtR = new DecisionTreeRegressor()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures")

    val pipelineReg = new Pipeline()
      .setStages(Array(featureIndexer, dtR))


    val modelRegressor = pipelineReg.fit(trainingData)

    val treeModel2 = modelRegressor
      .stages(1)
      .asInstanceOf[DecisionTreeRegressionModel]

    println("Learned regression ree model:\n" +  treeModel2.toDebugString)

    val predictionsReg  = modelRegressor.transform(testData)

    predictionsReg.show(5)



    sc.stop()

  }
}
