package FeatureEngineering

/**
  * Input: DB of deals closed by members of a sales team.
  * Output: new DB showing the commission earned by each sales rep during each year
  * note: if 2 people earn an amount of money, that the latter must be split in two
  */

import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext

object ExplodeFunctionsAndPivot {

  case class Sales(
                    id: Int,
                    account: String,
                    year: String,
                    commission: Int,
                    sales_reps: Seq[String])

  def main(args: Array[String]): Unit = {

    val sc = new SparkContext("local[*]", "ExplodeFunctionsAndPivot")

    val sqlContext = new SQLContext(sc)
    sc.setLogLevel("WARN")
    import sqlContext.implicits._
    import org.apache.spark.sql.functions._



    val sales = Seq(
      Sales(1, "Acme", "2013", 1000, Seq("Jim", "Tom")),
      Sales(2, "Lumos", "2013", 1100, Seq("Fred", "Ann")),
      Sales(2, "Acme", "2014", 2800, Seq("Jim")),
      Sales(4, "Lumos", "2014", 1200, Seq("Ann")),
      Sales(5, "Acme", "2014", 4200, Seq("Fred", "Sally"))
    ).toDF()

    println("DB:")
    sales.show()


    // EXPLODE
    // Split up the cells where we have a list of sales people

    println("Exploded DB")
    sales.select($"id", $"account", $"year", $"commission",
      explode($"sales_reps").as("sales_rep")).show()

    // Split the commission... we need to know how many ways to divide each commission
    // to do this we need to define a function which computes the column length

    val len: (Seq[String] => Int) = (arg: Seq[String]) => {arg.length}

    val column_len = udf(len)

    val exploded = sales.select(
      $"id", $"account", $"year", $"commission",
      ($"commission" / column_len($"sales_reps")).as("share"),
      explode($"sales_reps").as("sales_rep")
    )

    println("splitted commission DB")
    exploded.show()

    // PIVOT TABLES

    // GroupBy on 1 column
    println("Pivot tables")
    exploded
      .groupBy($"sales_rep")
      .pivot("year")
      .agg(sum("share"))
      .orderBy("sales_rep")
      .show()

    // GroupBy on 2 columns
    exploded
      .groupBy($"account", $"sales_rep")
      .pivot("year")
      .agg(sum("share"))
      .orderBy("account", "sales_rep")
      .show()

    sc.stop()

  }
}
