package PipelineGridSearch

/**
  * https://www.kaggle.com/c/unimelb
  * Dataset of grant applications, only a subset successful
  * Problem: predict if a grant application will be successful
  */

import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.CrossValidatorModel


object PredictingGrantApplications {

  implicit class BestParamMapCrossValidatorModel(cvModel: CrossValidatorModel){

    // Pairs each parameter map with its corresponding score
    // Finds the pair with the highest score and returns the parameter map
    // from the winning score
    def bestEstimatorParamMap: ParamMap = {
      cvModel.getEstimatorParamMaps
        .zip(cvModel.avgMetrics)
        .maxBy(_._2)
        ._1
    }
  }


  def main(args: Array[String]): Unit = {

    val sc = new SparkContext("local[*]", "LinearMethods")

    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._
    import org.apache.spark.sql.functions._
    sc.setLogLevel("WARN")

    val data = sqlContext.read
      .format("com.databricks.spark.csv")
      .option("delimiter", "\t")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("data/grantsPeople.csv")

    data.printSchema()
    data.show(3)

    println("\n ------ RESEARCHERS DATABASE ------")

    // Transforms string features into numerical ones
    val researchers = data
      .withColumn("phd", data("With_PHD").equalTo("Yes").cast("Int"))
      .withColumn("CI", data("Role").equalTo("CHIEF_INVESTIGATOR").cast("Int"))
      // More weight to A2 and A jurnal publications
      .withColumn("paperscore", data("A2") * 4 + data("A") *3)

    researchers.printSchema()
    researchers.show(3)


    println("\n ------ GRANT DATABASE ------")

    // groupBy of the researchers DF on the Grant Application ID: we'll aggregate
    // team by team combining information about the researcher for each grant
    val grants = researchers
      .groupBy("Grant_Application_ID")
      .agg(
        // Information about the grant
        max("Grant_Status").as("Grant_Status"),
        max("Grant_Category_Code").as("Category_Code"),
        max("Contract_Value_Band").as("Value_Band"),

        // Number of PhDs of the team
        sum("phd").as("PHDs"),

        // Only take the maximum paperscore of any of the chief investigators of the team
        when(max(expr("paperscore * CI")).isNull, 0)
          .otherwise(max(expr("paperscore * CI")))
          .as("paperscore"),

        // Num of researchers associated with each grant application
        count("*").as("teamsize"),

        // Add up number of successful and unsuccessful grant applications that each team
        // members has been involved in
        when(sum("Number_of_Successful_Grant").isNull, 0)
          .otherwise(sum("Number_of_Successful_Grant"))
          .as("successes"),
        when(sum("Number_of_Unsuccessful_Grant").isNull, 0)
          .otherwise(sum("Number_of_Unsuccessful_Grant"))
          .as("failures")
      )

    grants.printSchema()
    grants.show(3)

    // -------------------------------------------------------------

    val value_band_indexer = new StringIndexer()
      .setInputCol("Value_Band")
      .setOutputCol("Value_index")
      .fit(grants)

    val category_indexer = new StringIndexer()
      .setInputCol("Category_Code")
      .setOutputCol("Category_index")
      .fit(grants)

    val label_indexer = new StringIndexer()
      .setInputCol("Grant_Status")
      .setOutputCol("status")
      .fit(grants)

    val assembler = new VectorAssembler()
      .setInputCols(Array(
        "Value_index",
        "Category_index",
        "PHDs",
        "paperscore",
        "teamsize",
        "successes",
        "failures"
      ))
      .setOutputCol("assembled")

    val rf = new RandomForestClassifier()
      .setFeaturesCol("assembled")
      .setLabelCol("status")
      .setSeed(42)

    println(rf.explainParams)

    val pipeline = new Pipeline()
      .setStages(Array(
        value_band_indexer,
        category_indexer,
        label_indexer,
        assembler,
        rf
      ))

    val auc_eval = new BinaryClassificationEvaluator()
      .setLabelCol("status")
      .setRawPredictionCol("rawPrediction")

    println("\nMetric: " + auc_eval.getMetricName)

    val training = grants.filter("Grant_Application_ID < 6635")
    val test = grants.filter("Grant_Application_ID >= 6635")
    println("\n(training_count, test_count) = " + (training.count, test.count))

    val model = pipeline.fit(training)
    val pipeline_results = model.transform(test)

    println("\n\nModel Evaluation: " + auc_eval.evaluate(pipeline_results))



    // PARAMETERS TUNING
    println("\n" + rf.extractParamMap)

    val paramGrid = new ParamGridBuilder()
      .addGrid(rf.maxDepth, Array(14, 15))
      .addGrid(rf.numTrees, Array(99, 100))
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(auc_eval)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)

    val cvModel = cv.fit(training)

    val cv_results = cvModel.transform(test)

    println("\n\nModel Evaluation after Grid Search: " + auc_eval.evaluate(cv_results))

    println(cvModel.avgMetrics)

    println(cvModel.bestEstimatorParamMap)

    val bestPipelineModel = cvModel.bestModel
      .asInstanceOf[org.apache.spark.ml.PipelineModel]

    println("\nBest model stages: " + bestPipelineModel.stages)

    val bestRandomForest = bestPipelineModel.stages(4)
      .asInstanceOf[RandomForestClassificationModel]

    bestRandomForest.toDebugString

    println("\nBest Random Forest num nodes: " + bestRandomForest.totalNumNodes)

    // Listed in the Vector Assembler input columns order
    println("\nBest Random Forest Features Importance: " + bestRandomForest.featureImportances)



  }
}
