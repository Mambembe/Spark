import sbt.complete.Parsers._

name := "SparkML"

version := "1.0"

scalaVersion := "2.10.6"

//default Spark version
val sparkVersion = "1.6.0"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core"              % sparkVersion withSources(),
  "org.apache.spark" %% "spark-streaming"         % sparkVersion withSources(),
  "org.apache.spark" %% "spark-sql"               % sparkVersion withSources(),
  "org.apache.spark" %% "spark-hive"              % sparkVersion withSources(),
  "org.apache.spark" %% "spark-streaming-twitter" % sparkVersion withSources(),
  "org.apache.spark" %% "spark-mllib"             % sparkVersion withSources(),
  //"com.typesafe.play" %% "play-json"              % "2.4.4" withSources(),
  "org.twitter4j"    %  "twitter4j-core"          % "4.0.4" withSources(),
  "com.databricks"   %% "spark-csv"               % "1.3.0"      withSources(),
  "org.scala-lang" % "scala-reflect" % "2.10.6",
  "org.scala-lang" % "scala-compiler" % "2.10.6"
)

    