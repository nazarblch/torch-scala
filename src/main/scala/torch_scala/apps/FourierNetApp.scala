package torch_scala.apps


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

object FourierNetApp extends App {

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
      val lseq = (y1 - y2).abs() + (y1 - y2).pow(2) + (y1 - y2).abs().*(2).exp() * 0.05
      (lseq dot weights) + (c1.abs().mean() * 3)
    }

  }

  val data = io.Source.fromFile("/home/nazar/ptbdb_normal.csv").getLines().next().split(",")
    .map(_.toDouble).take(120)


  //print(data.mkString(","))

  val plot = new Plot("x", "y")
  //plot.addline(data, "data")
  //plot.print("lines.pdf")

  val net = new FourierNet(30)
  val n = data.length

  val xs = Tensor.arange[Double, CUDA](0, data.length).reshape(Shape(data.length, 1))
  val ys = Variable( Tensor[Double, CUDA](data) )
  val weights = Variable(Tensor.ones_like(ys.data))

  for (iter <- 0 to 2000) {
    val y_pred = net.forward(Variable(xs))
    val loss = net.weighted_mse(ys, y_pred.reshape(n), weights)
    net.optimizer.zeroGrad()
    loss.backward()
    net.optimizer.step()
    println(loss.data.cpu().item())
  }

  val y_pred = net.forward(Variable(xs))

  plot.addline(ys.data.cpu().data(), "data")
  plot.addline(y_pred.data.cpu().data(), "pred")



}