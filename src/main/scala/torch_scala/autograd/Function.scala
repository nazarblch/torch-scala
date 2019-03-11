package torch_scala.autograd


import torch_scala.api.aten.{Shape, Tensor, TensorType}

import scala.language.postfixOps
import torch_scala.api.aten.functions.Math._
import torch_scala.api.{intToScalar, doubleToScalar}

trait Function[T, TT <: TensorType[T]] {
  def forward(): Variable[T, TT]
  def backward(gradOutput: Variable[T, TT]): Unit

  def unbroadcast(v: Variable[T, TT], oldShape: Shape): Variable[T, TT] = {
    unbroadcast(v.data, oldShape)
  }

  def unbroadcast(data: Tensor[T, TT], oldShape: Shape): Variable[T, TT] = {
    val t = oldShape.asArray.zip(data.shape.asArray).zipWithIndex.foldLeft(data) {
      case (d: Tensor[T, TT], ((oi, ni), i)) =>
        if (oi == ni)
          d
        else if (oi == 1)
          d.sum(i)
        else
          throw new Exception(
            s"unable to unbroadcast shape ${data.shape} to $oldShape")
    }
    Variable(t)
  }
}

case class Add[T, TT <: TensorType[T]](v1: Variable[T, TT], v2: Variable[T, TT]) extends Function[T, TT] {
  def forward(): Variable[T, TT] = Variable[T, TT](v1.data + v2.data, gradFn = Some(this))
  def backward(gradOutput: Variable[T, TT]): Unit = {
    v1.backward(unbroadcast(gradOutput, v1.shape))
    v2.backward(unbroadcast(gradOutput, v2.shape))
  }
}

case class AddConstant[T, TT <: TensorType[T]](v: Variable[T, TT], d: T) extends Function[T, TT] {
  override def forward(): Variable[T, TT] = Variable[T, TT](v.data + d, Some(this))
  override def backward(gradOutput: Variable[T, TT]): Unit = {
    v.backward(gradOutput)
  }
}

case class Sub[T, TT <: TensorType[T]](v1: Variable[T, TT], v2: Variable[T, TT]) extends Function[T, TT] {
  def forward(): Variable[T, TT] = Variable[T, TT](v1.data - v2.data, gradFn = Some(this))
  def backward(gradOutput: Variable[T, TT]): Unit = {
    v1.backward(unbroadcast(gradOutput, v1.shape))
    v2.backward(unbroadcast(-gradOutput.data, v2.shape))
  }
}

case class SubConstant[T, TT <: TensorType[T]](v: Variable[T, TT], d: T) extends Function[T, TT] {
  override def forward(): Variable[T, TT] = Variable[T, TT](v.data + d, Some(this))
  override def backward(gradOutput: Variable[T, TT]): Unit = {
    v.backward(gradOutput)
  }
}

case class Mul[T, TT <: TensorType[T]](v1: Variable[T, TT], v2: Variable[T, TT]) extends Function[T, TT] {
  override def forward(): Variable[T, TT] = Variable[T, TT](v1.data * v2.data, Some(this))
  override def backward(gradOutput: Variable[T, TT]): Unit = {
    val dv1 = v2.data * gradOutput.data
    val vdv1 = unbroadcast(dv1, v1.shape)
    val dv2 = v1.data * gradOutput.data
    val vdv2 = unbroadcast(dv2, v2.shape)
    v1.backward(vdv1)
    v2.backward(vdv2)
  }
}

case class MulConstant[T, TT <: TensorType[T]](v: Variable[T, TT], d: T) extends Function[T, TT] {
  override def forward(): Variable[T, TT] = Variable[T, TT](v.data * d, Some(this))
  override def backward(gradOutput: Variable[T, TT]): Unit = {
    val dv = gradOutput.data * d
    v.backward(Variable[T, TT](dv))
  }
}

case class Div[T, TT <: TensorType[T]](v1: Variable[T, TT], v2: Variable[T, TT]) extends Function[T, TT] {
  override def forward(): Variable[T, TT] = Variable[T, TT](v1.data / v2.data, Some(this))
  override def backward(gradOutput: Variable[T, TT]): Unit = {
    val rv2 = Tensor.ones_like(v2.data) / v2.data
    val gv1 = gradOutput.data * rv2
    val gv2 = -gradOutput.data * v1.data * (rv2 ** 2)

    val vgv1 = unbroadcast(gv1, v1.shape)
    val vgv2 = unbroadcast(gv2, v2.shape)
    v1.backward(vgv1)
    v2.backward(vgv2)
  }
}

case class DivConstant[T, TT <: TensorType[T]](v: Variable[T, TT], d: T) extends Function[T, TT] {
  override def forward(): Variable[T, TT] = Variable[T, TT](v.data / d, Some(this))
  override def backward(gradOutput: Variable[T, TT]): Unit = {
    val dv = gradOutput.data / d
    v.backward(Variable[T, TT](dv))
  }
}

/* Pow of 2 tensors is currently not implemented in numsca
case class Pow(a: Variable[T, TT], b: Variable[T, TT]) extends Function[T, TT] {


  override def forward(): Variable[T, TT] = {
    Variable[T, TT](a.data ** b.data, Some(this))
  }

  override def backward(gradOutput: Variable[T, TT]): Unit = {
     val ga = gradOutput.data * b.data * (a.data ** (b.data - 1))
     val gb = gradOutput.data * (a.data ** b.data) * ns.log(a.data)

    val vga = unbroadcast(ga, a.shape)
    val vgb = unbroadcast(gb, b.shape)

    logger.debug(s"pow backward, ga.shape=${vga.shape}, gb.shape=${vgb.shape}")
    a.backward(vga)
    b.backward(vgb)
  }
}
 */

case class PowConstant[T, TT <: TensorType[T]](v: Variable[T, TT], d: Double) extends Function[T, TT] {
  override def forward(): Variable[T, TT] = Variable[T, TT](v.data.**(d), Some(this))
  val cache: Tensor[T, TT] = v.data.**(d - 1) * v.data.dataType.cast(d)
  override def backward(gradOutput: Variable[T, TT]): Unit = {
    val dv = cache * gradOutput.data
    v.backward(Variable[T, TT](dv))
  }
}

case class Sqrt[T, TT <: TensorType[T]](v: Variable[T, TT]) extends Function[T, TT] {
  val cache: Tensor[T, TT] = v.data.sqrt()
  override def forward(): Variable[T, TT] = Variable[T, TT](cache, Some(this))
  override def backward(gradOutput: Variable[T, TT]): Unit = {
    val dv = (Tensor.ones_like(cache) / (cache * cache.dataType.cast(2))) * gradOutput.data
    v.backward(Variable[T, TT](dv))
  }
}

case class Abs[T, TT <: TensorType[T]](v: Variable[T, TT]) extends Function[T, TT] {
  val cache: Tensor[T, TT] = v.data.abs()
  override def forward(): Variable[T, TT] = Variable[T, TT](cache, Some(this))
  override def backward(gradOutput: Variable[T, TT]): Unit = {
    val dv = (v.data / cache) * gradOutput.data
    v.backward(Variable[T, TT](dv))
  }
}

case class Negate[T, TT <: TensorType[T]](v: Variable[T, TT]) extends Function[T, TT] {
  override def forward(): Variable[T, TT] = Variable[T, TT](-v.data, Some(this))
  override def backward(gradOutput: Variable[T, TT]): Unit = {
    val dv = -gradOutput.data
    v.backward(Variable[T, TT](dv))
  }
}

case class Dot[T, TT <: TensorType[T]](v1: Variable[T, TT], v2: Variable[T, TT]) extends Function[T, TT] {
  val w: Tensor[T, TT] = v1.data
  val x: Tensor[T, TT] = v2.data

  override def forward(): Variable[T, TT] = Variable[T, TT](w dot x, Some(this))
  override def backward(gradOutput: Variable[T, TT]): Unit = {
    val dd = gradOutput.data
    val dw = dd dot x.T
    val dx = w.T dot dd
    v1.backward(Variable[T, TT](dw))
    v2.backward(Variable[T, TT](dx))
  }
}

case class Transpose[T, TT <: TensorType[T]](v: Variable[T, TT]) extends Function[T, TT] {
  override def forward(): Variable[T, TT] = Variable[T, TT](v.data.T, Some(this))
  override def backward(gradOutput: Variable[T, TT]): Unit =
    v.backward(Variable[T, TT](gradOutput.data.T))
}

// todo test this
case class Reshape[T, TT <: TensorType[T]](v: Variable[T, TT], shape: Shape) extends Function[T, TT] {
  val oldShape: Shape = v.shape
  override def forward(): Variable[T, TT] =
    Variable[T, TT](v.data.reshape(shape), Some(this))
  override def backward(gradOutput: Variable[T, TT]): Unit = {
    val dv = gradOutput.data.reshape(oldShape)
    v.backward(Variable[T, TT](dv))
  }
}

//case class Concat[T, TT <: TensorType[T]](v1: Variable[T, TT], v2: Variable[T, TT], axis: Int = 0) extends Function[T, TT] {
//
//  require(axis == 0 || axis == 1, "axis must be either 0 or 1")
//
//  override def forward(): Variable[T, TT] = {
//    Variable[T, TT](ns.concatenate(Seq(v1.data, v2.data), axis), Some(this))
//  }
//
//  override def backward(gradOutput: Variable[T, TT]): Unit =
//    if (axis == 0) {
//      val d = gradOutput.data.data
//      val (d1, d2) = d.splitAt(v1.shape.product)
//      val dv1 = Tensor(d1).reshape(v1.shape: _*)
//      val dv2 = Tensor(d2).reshape(v2.shape: _*)
//      v1.backward(Variable[T, TT](dv1))
//      v2.backward(Variable[T, TT](dv2))
//    } else {
//      val dv1 = gradOutput.data(:>, 0 :> v1.shape(axis))
//      val dv2 = gradOutput.data(:>, v1.shape(axis) :>)
//      v1.backward(Variable[T, TT](dv1))
//      v2.backward(Variable[T, TT](dv2))
//    }
//}

//===============================================================

case class Exp[T, TT <: TensorType[T]](v: Variable[T, TT]) extends Function[T, TT] {
  val cache: Tensor[T, TT] = v.data.exp()
  def forward() = Variable[T, TT](data = cache, gradFn = Some(this))
  def backward(gradOutput: Variable[T, TT]): Unit = {
    v.backward(Variable[T, TT](gradOutput.data * cache))
  }
}

case class Cos[T, TT <: TensorType[T]](v: Variable[T, TT]) extends Function[T, TT] {
  val cache: Tensor[T, TT] = v.data.sin() * v.data.dataType.cast(-1)
  def forward() = Variable[T, TT](data = v.data.cos(), gradFn = Some(this))
  def backward(gradOutput: Variable[T, TT]): Unit = {
    v.backward(Variable[T, TT](gradOutput.data * cache))
  }
}

case class Sin[T, TT <: TensorType[T]](v: Variable[T, TT]) extends Function[T, TT] {
  val cache: Tensor[T, TT] = v.data.cos()
  def forward() = Variable[T, TT](data = v.data.sin(), gradFn = Some(this))
  def backward(gradOutput: Variable[T, TT]): Unit = {
    v.backward(Variable[T, TT](gradOutput.data * cache))
  }
}

//case class Tanh[T, TT <: TensorType[T]](v: Variable[T, TT]) extends Function[T, TT] {
//  val cache: Tensor = ns.tanh(v.data)
//  override def forward(): Variable[T, TT] = Variable[T, TT](cache, Some(this))
//  override def backward(gradOutput: Variable[T, TT]): Unit =
//    v.backward(Variable[T, TT]((1 - ns.square(cache)) * gradOutput.data))
//}
//
//case class Sigmoid[T, TT <: TensorType[T]](v: Variable[T, TT]) extends Function[T, TT] {
//  lazy val sigmoid: Tensor = ns.sigmoid(v.data)
//  override def forward(): Variable[T, TT] = Variable[T, TT](sigmoid, Some(this))
//  override def backward(gradOutput: Variable[T, TT]): Unit =
//    v.backward(Variable[T, TT](gradOutput.data * sigmoid * (1 - sigmoid)))
//}
//
//case class Softmax[T, TT <: TensorType[T]](v: Variable[T, TT]) extends Function[T, TT] {
//  lazy val softmax: Tensor = ns.softmax(v.data)
//  override def forward(): Variable[T, TT] = Variable[T, TT](softmax, Some(this))
//  override def backward(gradOutput: Variable[T, TT]): Unit = {
//    // from https://stackoverflow.com/questions/33541930/how-to-implement-the-softmax-derivative-independently-from-any-loss-Function[T, TT]
//    val y = softmax
//    val dy = gradOutput.data
//
//    val dx = y * dy
//    val s = ns.sum(dx, axis = dx.shape.length - 1)
//    dx -= y * s
//    v.backward(Variable[T, TT](dx))
//  }
//}
//
//case class Mean[T, TT <: TensorType[T]](v: Variable[T, TT]) extends Function[T, TT] {
//  def forward() = Variable[T, TT](data = ns.mean(v.data), gradFn = Some(this))
//  def backward(gradOutput: Variable[T, TT]): Unit = {
//    val n = v.shape.product
//    v.backward(Variable[T, TT](gradOutput.data / n))
//  }
//}
//
//case class MeanByAxis[T, TT <: TensorType[T]](v: Variable[T, TT], axis: Int) extends Function[T, TT] {
//  def forward() = Variable[T, TT](data = ns.mean(v.data, axis), gradFn = Some(this))
//  def backward(gradOutput: Variable[T, TT]): Unit = {
//    val n = v.shape(axis)
//    v.backward(Variable[T, TT](gradOutput.data / n))
//  }
//}
//
//case class Variance[T, TT <: TensorType[T]](v: Variable[T, TT]) extends Function[T, TT] {
//  override def forward(): Variable[T, TT] = ((v - v.mean()) ** 2).mean()
//  override def backward(gradOutput: Variable[T, TT]): Unit =
//    v.backward(gradOutput)
//}
//
//case class VarianceByAxis[T, TT <: TensorType[T]](v: Variable[T, TT], axis: Int) extends Function[T, TT] {
//  override def forward(): Variable[T, TT] = ((v - v.mean(axis)) ** 2).mean(axis)
//  override def backward(gradOutput: Variable[T, TT]): Unit =
//    v.backward(gradOutput)
//}
//
//case class Max[T, TT <: TensorType[T]](x: Variable[T, TT], y: Variable[T, TT]) extends Function[T, TT] {
//  def forward(): Variable[T, TT] = {
//    val max: Tensor = ns.maximum(x.data, y.data)
//    Variable[T, TT](max, Some(this))
//  }
//  override def backward(gradOutput: Variable[T, TT]): Unit = {
//    x.backward(Variable[T, TT]((x.data >= y.data) * gradOutput.data))
//    y.backward(Variable[T, TT]((x.data <= y.data) * gradOutput.data))
//  }
//}
//
//case class Threshold[T, TT <: TensorType[T]](x: Variable[T, TT], d: Double) extends Function[T, TT] {
//  override def forward(): Variable[T, TT] = Variable[T, TT](ns.maximum(x.data, d), Some(this))
//  override def backward(gradOutput: Variable[T, TT]): Unit = {
//    x.backward(Variable[T, TT](gradOutput.data * (x.data > d)))
//  }
//}
//
////============================================
//// Loss Function[T, TT]s
//case class SoftmaxLoss[T, TT <: TensorType[T]](actual: Variable[T, TT], target: Variable[T, TT]) extends Function[T, TT] {
//  val x: Tensor = actual.data
//  val y: Tensor = target.data.T
//
//  val shiftedLogits: Tensor = x - ns.max(x, axis = 1)
//  val z: Tensor = ns.sum(ns.exp(shiftedLogits), axis = 1)
//  val logProbs: Tensor = shiftedLogits - ns.log(z)
//  val n: Int = x.shape.head
//  val loss: Double = -ns.sum(logProbs(ns.arange(n), y)) / n
//
//  override def forward(): Variable[T, TT] = Variable[T, TT](Tensor(loss), Some(this))
//
//  override def backward(gradOutput: Variable[T, TT] /* not used */ ): Unit = {
//    val dx = ns.exp(logProbs)
//    dx(ns.arange(n), y) -= 1
//    dx /= n
//
//    actual.backward(Variable[T, TT](dx))
//  }
//}
//
///**
//  * Computes the cross-entropy loss
//  * @param actuals sequence of yHat Variable[T, TT]s
//  * @param targets sequence of Y indices (ground truth)
//  */
//case class CrossEntropyLoss[T, TT <: TensorType[T]](actuals: Seq[Variable[T, TT]], targets: Seq[Int])
//    extends Function[T, TT] {
//
//  /**
//    * Computes the cross entropy loss, and wraps it in a Variable[T, TT].
//    * The Variable[T, TT] can be back propped into, to compute the gradients of the parameters
//    * @return the cross entropy loss Variable[T, TT]
//    */
//  override def forward(): Variable[T, TT] = {
//    val seqLoss = actuals.zip(targets).foldLeft(0.0) {
//      case (loss, (yht, y)) =>
//        loss - ns.log(yht.data(y, 0)).squeeze()
//    }
//    Variable[T, TT](Tensor(seqLoss), Some(this))
//  }
//
//  /**
//    * Compute the loss of each generated character, and back prop from last to first
//    * @param gradOutput not used
//    */
//  override def backward(gradOutput: Variable[T, TT]): Unit = {
//    actuals.zip(targets).reverse.foreach {
//      case (yh, y) =>
//        val dy = ns.copy(yh.data)
//        dy(y, 0) -= 1
//        yh.backward(Variable[T, TT](dy))
//    }
//  }
//}
