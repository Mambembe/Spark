package ML

import org.apache.spark.SparkContext

object basicTypes {

  def main(args: Array[String]): Unit = {

    val sc = new SparkContext("local[*]", "Word Count")
    sc.setLogLevel("WARN")

    // VECTORS
    import org.apache.spark.mllib.linalg.{Vector, Vectors}

    val v0: Vector = Vectors.dense(44.0, 0.0, 55.0)
    // Define the same vector in two different ways
    val v1: Vector = Vectors.sparse(3,Array(0,2), Array(44.0, 55.0))
    val v2: Vector = Vectors.sparse(3,Seq((0,44.0), (2, 55.0)))

    // LABELED POINTS
    import org.apache.spark.mllib.regression.LabeledPoint
    val label1 = LabeledPoint(1, Vectors.dense(44.0, 0.0, 55.0))
    val label2 = LabeledPoint(1, v1)

    println("VECTORS")
    println(v0)
    println(v1)
    println(v2)
    println("LABELS")
    println(label1)
    println(label2)

    // MATRICES
    import org.apache.spark.mllib.linalg.{Matrix, Matrices}

    val m0: Matrix = Matrices.dense(3, 2, Array(1, 3, 5, 2, 4, 6))
    val m1: Matrix = Matrices.sparse(5, 4,
      Array(0,0,1,2,2),
      Array(1,3),
      Array(34,35))

    println("MATRICES")
    println(m0)
    println(m1)

    // RowMatrix
    import org.apache.spark.mllib.linalg.distributed.RowMatrix
    import org.apache.spark.rdd.RDD

    val rows: RDD[Vector] = sc.parallelize(Array(
      Vectors.dense(1.0, 2.0),
      Vectors.dense(4.0, 5.0),
      Vectors.dense(7.0, 8.0)))

    val mat: RowMatrix = new RowMatrix(rows)
    val m = mat.numRows()
    val n = mat.numCols()

    println(mat)
    println(m)
    println(n)

    // Indexed Row Matrix
    import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}

    val rows2: RDD[IndexedRow] = sc.parallelize(Array(
      IndexedRow(0, Vectors.dense(1.0, 2.0)),
      IndexedRow(0, Vectors.dense(4.0, 5.0)),
      IndexedRow(0, Vectors.dense(7.0, 8.0))))

    val idxMat: IndexedRowMatrix = new IndexedRowMatrix(rows2)

    println("Indexed Row Matrix:")
    println(idxMat)

    // COORDINATE MATRIX
    import org.apache.spark.mllib.linalg.distributed.MatrixEntry
    import org.apache.spark.mllib.linalg.distributed.CoordinateMatrix

    val entries: RDD[MatrixEntry] = sc.parallelize(Array(
      MatrixEntry(0, 0, 9),
      MatrixEntry(1, 1, 8),
      MatrixEntry(2, 1, 69)
    ))
    val coordMat: CoordinateMatrix = new CoordinateMatrix(entries)

    println("Coordinate Matrix:")
    println(coordMat)

  }

}
