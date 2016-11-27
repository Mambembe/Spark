package course2.module4

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.streaming.StreamingContext
import org.apache.spark.streaming.StreamingContext._
import org.apache.spark.streaming.dstream.DStream
import org.apache.spark.streaming.Seconds
import org.apache.spark.streaming.scheduler.{
  StreamingListener, StreamingListenerReceiverError,
  StreamingListenerReceiverStopped
}
import org.apache.spark.sql.hive.HiveContext
import data.Flight
import util.Files
import scala.util.control.NonFatal
import java.net.{Socket, ServerSocket}
import java.io.{File, PrintWriter}

/**
  * We'll ingest the flights data into Hive table partitions using the same streaming
  * socket technique we used in Streaming.
  * The streaming job will use the HiveContext to create the table, if necessary, and
  * for each year, month and day in the flight data, construct a corresponding partition
  */
object StreamsHive {
/*
  val defaultPort = 9000
  val interval = Seconds(5)
  val pause = 10  // milliseconds
  val server = "127.0.0.1"
  val hiveETLDir = "output/hive-etl"
  val checkpointDir = "output/checkpoint_dir"
  val runtime = 10 * 1000   // run for N*1000 milliseconds
  val numRecordsToWritePerBlock = 10000

  def main(args: Array[String]): Unit = {
    val port = if (args.size > 0) args(0).toInt else defaultPort
    val conf = new SparkConf()
    conf.setMaster("local[*]")
    conf.setAppName("ETL with Spark Streaming and Hive")
    conf.set("spark.sql.shuffle.partitions", "4")
    conf.set("spark.app.id", "HiveETL")
    val sc = new SparkContext(conf)

    // Delete the Hive metadata and table data, as well as the streaming
    // checkpoint directory from previous runs

    def createContext(): StreamingContext = {
      val ssc = new StreamingContext(sc, interval)
      ssc.checkpoint(checkpointDir)

      val dstream = readSocket(ssc, server, port)
      processDStream(ssc, dstream)
    }

    // Declared as a var so we can see it in the following try/catch/finally blocks.
    var ssc: StreamingContext = null
    var dataThread: Thread = null

    try {

      println("Creating source socket:")
      dataThread = startSocketDataThread(port)

      // The preferred way to construct a StreamingContext, because if the
      // program is restarted, e.g., due to a crash, it will pick up where it
      // left off. Otherwise, it starts a new job.
      ssc = StreamingContext.getOrCreate(checkpointDir, createContext _)
      ssc.addStreamingListener(new EndOfStreamListener(ssc, dataThread)      )
      ssc.start()
      ssc.awaitTerminationOrTimeout(runtime)

    } finally {
      shutdown(ssc, dataThread)
    }
  }

  def readSocket(ssc: StreamingContext, server: String, port: Int): DStream[String] =
    try {
      banner(s"Connecting to $server:$port...")
      ssc.socketTextStream(server, port)
    } catch {
      case th: Throwable =>
        ssc.stop()
        throw new RuntimeException(
          s"Failed to initialize server:port socket with $server:$port:",
          th)
    }

  import java.net.{Socket, ServerSocket}
  import java.io.PrintWriter

  def makeRunnable(port: Int) = new Runnable {
    def run() = {
      val listener = new ServerSocket(port);
      var socket: Socket = null
      try {
        val socket = listener.accept()
        val out = new PrintWriter(socket.getOutputStream(), true)
        val  inputPath = "data/airline-flights/alaska-airlines/2008.csv"
        var lineCount = 0
        var passes = 0
        scala.io.Source.fromFile(inputPath).getLines()
          .foreach {line =>
            out.println(line)
            if (lineCount % numRecordsToWritePerBlock == 0)
              Thread.sleep(pause)
            lineCount += 1
        }
      } finally {
        listener.close();
        if (socket != null) socket.close();
      }
    }
  }

  def startSocketDataThread(port: Int): Thread = {
    val dataThread = new Thread(makeRunnable(port))
    dataThread.start()
    dataThread
  }

  // Create a Hive context and obtain the full path for the HTL dir.
  def processDStream(ssc: StreamingContext, dstream: DStream[String]): StreamingContext = {
    val hiveContext = new HiveContext(ssc.sparkContext)
    import hiveContext.implicits._
    import hiveContext.sql
    import org.apache.spark.sql.functions._

    val hiveETLFile = new java.io.File(hiveETLDir)
    val hiveETLPath = hiveETLFile.getCanonicalPath

    println("Create an 'external' Hive table for the flights data:")
    sql(s"""
        CREATE EXTERNAL TABLE IF NOT EXISTS flights2 (
          depTime  INT,
          arrTime  INT,
          uniqueCarrier     STRING,
          flightNUm    INT,
          origin  STRING,
          dest    STRING)
        PARTITIONED BY (
          depYear   STRING,
          depMonth  STRING,
          depDay    STRING)
        ROW FORMAT DELIMITED FIELDS TERMINATED BY '|'
        LOCATION '$hiveETLPath'
        """).show
    println("Tables: ")
    sql("SHOW TABLES").show

    dstream.foreachRDD{ (rdd, timestamp) =>
      try{

        // For each RDD (datastream iteration), parse the read text lines into
        // Flight instances, convert to a DataFrame and cache
        val flights = rdd.flatMap(line => Flight.parse(line)).toDF().cache

        // Find the distinct year-month-day combinations and "collect"
        // in a Driver-based Scala-collection
        val uniqueYMDs = flights
          .select("date.year", "date.month", "date.dayOfMonth")
          .distinct.collect

        uniqueYMDs.foreach { row =>
          val year          = row.getInt(0)
          val month         = row.getInt(1)
          val day           = row.getInt(2)
          val yearStr       = "%04d".format(year)
          val monthStr      = "%02d".format(month)
          val dayStr        = "%02d".format(day)
          val partitionPath = "%s/%s-%s-%s".format(
            hiveETLPath, yearStr, monthStr, dayStr)
          // Loop over the year-month-days. For each one, add
          // a partition with the directory name we want
          sql(
            s"""
               ALTER TABLE flights2 ADD IF NOT EXISTS PARTITION (
                depYear='$yearStr',
                depMonth='$monthStr',
                depDay='$dayStr')
               LOCATION '$partitionPath'
             """.stripMargin)


          // While still looping over YMDs, find the corresponding records
          flights
            .where(
              $"date.year" === year and
              $"date.month" === month and
              $"date.dayOfMonth" === day)
          // Don't write the partition columns
            .select($"times.depTime", $"timesarrTime",
              $"uniqueCarrier", $"flightNum", $"origin", $"dest")
            .map(row => row.mkString("|"))
            .saveAsTextFile(partitionPath)
        }

        val show = sql("SHOW PARTITIONS flights2")
        val showCount = show.count
        println(s"Partitions (${showCount}):")
        showp.foreach(p => println("  "+p))
      } catch {
        case NonFatal(ex) => sys.exit(1)
      }
    }
    ssc
  }

  protected def shutdown(ssc: StreamingContext, dataThread: Thread) = {
    banner("Shutting down...")
    if (dataThread != null) dataThread.interrupt()
    else "The dataThread is null!"
    if (ssc != null)
      ssc.stop(stopSparkContext = true, stopGracefully = true)
    else "The StreamingContext is null!"
  }

  class EndOfStreamListener(ssc: StreamingContext, dataThread) extends StreamingListener {
    override def onReceiverError(
                                  error: StreamingListenerReceiverError): Unit = {
      banner(s"Receiver Error: $error. Stopping...")
      shutdown(ssc, dataThread)
    }

    override def onReceiverStopped(stopped: StreamingListenerReceiverStopped): Unit = {
      banner(s"Receiver Stopped: $stopped. Stopping...")
      shutdown(ssc, dataThread)
    }

  }
*/
}
