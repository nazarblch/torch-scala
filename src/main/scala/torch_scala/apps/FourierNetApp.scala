package torch_scala.apps


import breeze.stats.distributions.Poisson
import torch_scala.api.aten._
import torch_scala.api._
import torch_scala.api.aten.functions.{Basic, Functions}
import torch_scala.api.aten.functions.Math._
import torch_scala.api.aten.functions.Basic._
import torch_scala.autograd.Variable
import torch_scala.nn.Linear
import torch_scala.optim.Adam
import torch_scala.autograd.MathVariable._
import viz.Plot


class FourierNet(size: Int) {

  val fc1: Linear[Double, CUDA] = Linear[Double, CUDA](1, size)
  val fc2: Linear[Double, CUDA] = Linear[Double, CUDA](1, 5)
  val c1 = Variable(Tensor.randn[Double, CUDA](Shape(size, 1)))
  val c2 = Variable(Tensor.randn[Double, CUDA](Shape(5, 1)).abs())
  val c = Variable(Tensor.randn[Double, CUDA](Shape(1, 1)) + 1)
  val sigma = Variable(Tensor.arange[Double, CUDA](-6, -1).reshape(Shape(1, 5)))

  val optimizer: Adam[CUDA] = Adam[CUDA]((fc1.parameters ++ fc2.parameters ++ Seq(c1, c2, c, sigma)).asInstanceOf[Seq[Variable[Any, CUDA]]], 0.003)

  def forward(x: Variable[Double, CUDA]): Variable[Double, CUDA] = {
    (fc1(x) * 10).cos().mm(c1 / size) + c +
      ((fc2(x)  ).cos().pow(2) * sigma ).exp().mm(c2 / 10)
  }

  def weighted_mse(y1: Variable[Double, CUDA], y2: Variable[Double, CUDA], weights: Variable[Double, CUDA]): Variable[Double, CUDA] = {
    val lseq = (y1 - y2).abs() + (y1 - y2).pow(2)
    lseq dot weights
  }

  def l1_pen(): Variable[Double, CUDA] = {
    (c1.abs().mean() * 5) + (c2.abs().mean() * 0.1)
  }

  def train(y: Array[Double], weights: Array[Double], itersCount: Int): (Double, Array[Double]) = {

    val n = y.length

    val xs = Tensor.arange[Double, CUDA](0, y.length).reshape(Shape(y.length, 1))
    val ys = Variable( Tensor[Double, CUDA](y) )
    val weights = Variable(Tensor.ones_like(ys.data))

    for (iter <- 0 to itersCount) {
      val y_pred = forward(Variable(xs))
      val loss = weighted_mse(ys, y_pred.reshape(n), weights) + l1_pen()
      optimizer.zeroGrad()
      loss.backward()
      optimizer.step()
      // println(loss.data.cpu().item())
    }

    val y_pred = forward(Variable(xs))
    val loss = weighted_mse(ys, y_pred.reshape(n), weights)
    (loss.data.cpu().item(), y_pred.data.cpu().data())
  }

}




object DataLoader {
  def load_ecg(path: String): Array[Array[Double]] = {
    scala.io.Source.fromFile(path).getLines.toArray.map(line => {
      line.split(",").map(_.toDouble)
    })
  }
}


object LRT {
  def apply(y: Array[Double], h: Int): Array[Double] = {

    val ws: Array[Double] = Poisson(1.0).sample(y.length).toArray.map(_.toDouble)

    val y_slides = y.sliding(2*h, 5)
    val w_slides = ws.sliding(2*h, 5)

    val model = new FourierNet(30)

    model.train(y_slides.next(), w_slides.next(), 1000)
    model.train(y_slides.next(), w_slides.next(), 1000)

    y_slides.zip(w_slides).toArray.map({case(y12, w12) =>
      val y1 = y12.slice(0, h)
      val y2 = y12.slice(h, 2 * h)

      val w1 = w12.slice(0, h)
      val w2 = w12.slice(h, 2 * h)

      val (loss12, y_pred12) = model.train(y12, w12, 300)
      val (loss1, y_pred1) = model.train(y1, w1, 300)
      val (loss2, y_pred2) = model.train(y2, w2, 300)

      val lrt = loss12 - loss1 - loss2

      println(lrt)

      lrt
    })

  }
}


object FourierModel extends App {

  val net = new FourierNet(50)

  val data = DataLoader.load_ecg("/home/nazar/PycharmProjects/torch/data/ptbdb_normal.csv")



  val row = data(10).slice(0, 105)
  val row1 = data(12).slice(0, 105)

  val rows = row ++ row
  val pl1 = new Plot("time", "LRT")
  pl1.addline(rows, "model interpolation")

  val (loss, y_pred) = net.train(rows, Array.fill(rows.length)(1.0d), 1000)

  // val lrt = LRT.apply(rows, row.length + 30)

  val pl = new Plot("time", "LRT")
  // pl.addline(lrt, "model interpolation")
  pl.addline(y_pred.map(_.toDouble), "123")
  pl.addline(data(10).slice(10, 140) ++ data(10).slice(10, 140), "123")

}