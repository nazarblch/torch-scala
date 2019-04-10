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

object FourierNetApp extends App {

  class FourierNet(size: Int) {

    val fc1: Linear[Double, CPU] = Linear[Double, CPU](1, size)
    val fc2: Linear[Double, CPU] = Linear[Double, CPU](1, 5)
    val c1 = Variable(Tensor.randn[Double, CPU](Shape(size, 1)))
    val c2 = Variable(Tensor.randn[Double, CPU](Shape(5, 1)))
    val c = Variable(Tensor.randn[Double, CPU](Shape(1, 1)))
    val sigma = Variable(Tensor.arange[Double, CPU](-6, -1).reshape(Shape(1, 5)))

    val optimizer: Adam[CPU] = Adam[CPU]((fc1.parameters ++ fc2.parameters ++ Seq(c1, c2, c, sigma)).asInstanceOf[Seq[Variable[Any, CPU]]], 0.01)

    def forward(x: Variable[Double, CPU]): Variable[Double, CPU] = {
      ((fc2(x) * 0.1).cos().pow(2) * sigma).exp().mm(c2 / 10.0) + (fc1(x) * 0.1).cos().mm(c1 / 10.0) + c
    }

    def weighted_mse(y1: Variable[Double, CPU], y2: Variable[Double, CPU], weights: Variable[Double, CPU]): Variable[Double, CPU] = {
      val lseq = (y1 - y2).pow(2)
      lseq dot weights
    }

  }

  val net = new FourierNet(30)

  val xs = (Tensor.arange[Double, CPU](0, 500) / 100.0).reshape(Shape(500, 1))
  val ys = Variable(xs.cos() + (xs / 10).cos()  )
  val weights = Variable(Tensor.ones_like(xs))

  for (iter <- 0 to 2000) {
    val y_pred = net.forward(Variable(xs))
    val loss = net.weighted_mse(ys.reshape(500), y_pred.reshape(500), weights.reshape(500))
    net.optimizer.zeroGrad()
    loss.backward()
    net.optimizer.step()
    println(loss.data.item())
  }



}