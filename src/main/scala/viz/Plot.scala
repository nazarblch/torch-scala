package viz

import java.io.FileWriter
import java.util.Random

import breeze.plot._
import breeze.linalg._

import scala.collection.mutable.ArrayBuffer

class Plot(xlabel: String, ylabel: String) {

  val f = Figure()
  val p = f.subplot(0)
  p.xlabel = xlabel
  p.ylabel = ylabel
  p.legend = true
  p.setYAxisDecimalTickUnits()

  p.setXAxisDecimalTickUnits()
  // p.logScaleX = true

  val lines = ArrayBuffer[(String, DenseVector[Double], DenseVector[Double])]()


  var lc = 0

  def addline(x: DenseVector[Double], y: DenseVector[Double], name: String) {
    p += plot(x, y, name = name, shapes = true)
    lc += 1
    // if (lc > 1) p.legend = true

    lines += Triple(name, x, y)
  }

  def addline(y: DenseVector[Double], name: String) {
    val x:  DenseVector[Double] = DenseVector(Range(0, y.length).map(_.toDouble).toArray)
    p += plot(x, y, name = name, shapes = true)
    lines += Triple(name, x, y)
  }

  def addline(y: Array[Double], name: String) {
    addline(DenseVector[Double](y), name)
  }

  def addline(x: Array[Double], y: Array[Double], name: String) {
    addline(DenseVector[Double](x), DenseVector[Double](y), name)
  }

  def addline(y: Array[(Int, Double)], name: String) {
    addline(DenseVector(y.map(_._1.toDouble)), DenseVector[Double](y.map(_._2)), name)
  }

  def addNoizedLine(x1: Double, x2: Double, y1: Double, y2: Double, k: Int): Unit = {
    val r = new Random()

    val xarr = (0 until k).map(i => x1 + i * (x2 - x1) / k).toArray

    val yarr = (0 until k).map(i => y1 + i * (y2 - y1) / k + r.nextGaussian() / 10).toArray

    p += plot(xarr, yarr, name = "", shapes = true, lines=false)

  }


  val buffer: ArrayBuffer[(Double, Double)] = ArrayBuffer()

  def addDot(x: Double, y: Double): Unit = {
    buffer += Pair(x, y)
  }

  def print(path: String) {

    // addline(buffer.result().toArray.map(_._1), buffer.result().toArray.map(_._2), "")

    f.saveas(path)
    val fw = new FileWriter(path + ".txt")

    lines.result().foreach({case(s, x, y) =>
      fw.write(s + "\n")
      x.toArray.zip(y.toArray).foreach({case(i,j) => fw.write(i + "," + j + "\n")})
    })

    fw.close()

  }
}