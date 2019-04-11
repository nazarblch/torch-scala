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

    val fc1: Linear[Double, CPU] = Linear[Double, CPU](1, size)
    val fc2: Linear[Double, CPU] = Linear[Double, CPU](1, 5)
    val c1 = Variable(Tensor.randn[Double, CPU](Shape(size, 1)))
    val c2 = Variable(Tensor.randn[Double, CPU](Shape(5, 1)))
    val c = Variable(Tensor.randn[Double, CPU](Shape(1, 1)))
    val sigma = Variable(Tensor.arange[Double, CPU](-6, -1).reshape(Shape(1, 5)))

    val optimizer: Adam[CPU] = Adam[CPU]((fc1.parameters ++ fc2.parameters ++ Seq(c1, c2, c, sigma)).asInstanceOf[Seq[Variable[Any, CPU]]], 0.003)

    def forward(x: Variable[Double, CPU]): Variable[Double, CPU] = {
       (fc1(x)  ).mm(c1 ) // + c +
       //((fc2(x) * 10).cos().pow(2) * sigma / 10).exp().mm(c2 / 10.0)
    }

    def weighted_mse(y1: Variable[Double, CPU], y2: Variable[Double, CPU], weights: Variable[Double, CPU]): Variable[Double, CPU] = {
      val lseq = (y1 - y2).pow(4)  + (y1 - y2).pow(2)
      lseq dot weights
    }

  }

  val data = io.Source.fromFile("/home/nazar/Downloads/ptbdb_normal.csv").getLines().next().split(",")
    .map(_.toDouble).take(120)


  //print(data.mkString(","))

  //val plot = new Plot("x", "y")
  //plot.addline(data, "data")
  //plot.print("lines.pdf")

  val net = new FourierNet(10)
  val n = data.length

  val xs = Tensor.arange[Double, CPU](0, data.length).reshape(Shape(data.length, 1))
  val ys = Variable( Tensor[Double, CPU](data).reshape(Shape(n, 1)) )
  val weights = Variable(Tensor.ones_like(xs))

  for (iter <- 0 to 1000) {
    val y_pred = net.forward(Variable(xs))
    val loss = net.weighted_mse(ys.reshape(n), y_pred.reshape(n), weights.reshape(n))
    net.optimizer.zeroGrad()
    loss.backward()
    net.optimizer.step()
    println(loss.data.item())
  }

  val y_pred = net.forward(Variable(xs))

  //plot.addline(ys.data.data(), "data")
  //plot.addline(y_pred.data.data(), "pred")



}