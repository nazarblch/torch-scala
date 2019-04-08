package torch_scala.autograd


import torch_scala.api.aten.functions.{Basic, MathBackward}
import torch_scala.api.aten.functions.Basic._
import torch_scala.api.aten._
import torch_scala.api._

import scala.language.postfixOps
import torch_scala.api.aten.functions.Math._
import torch_scala.api.{doubleToScalar, intToScalar}
import torch_scala.autograd.MathVariable._

import scala.reflect.ClassTag

import scala.reflect.runtime.universe._

case class VariableTypeDetector[T: TypeTag, TT <: TensorType](related: Variable[T, TT]) {
    def getType(): Type = {
      typeOf[T]
    }
}


case class VariableWithGradient[T, TT <: TensorType](variable: Variable[T, TT], grad: Tensor[T, TT]) {
}


abstract class Function[T: ClassTag, TT <: TensorType](val name: String, val args: Variable[_, TT]*) {

  def varName: String = args.length match {
    case 0 => name
    case 2 => s"(${args(0).name} $name ${args(1).name})"
    case _ => s"$name(${args.map(_.name).mkString(",")})"
  }

  def forward(): Variable[T, TT] = {
    Variable[T, TT](forwardImpl(), Some(this), Some(varName))
  }

  protected def forwardImpl(): Tensor[T, TT]

  def backward(gradOutput: Tensor[T, TT]): Seq[VariableWithGradient[_, TT]] = {
    val grads = backwardImpl(gradOutput)
    require(args.length equals grads.length, "Each argument should have gradient")
    args.zip(grads).map({case(v, g) =>
      val t = VariableTypeDetector(v).getType()
      VariableWithGradient(v.asInstanceOf[Variable[t.type, TT]], g.asInstanceOf[Tensor[t.type, TT]])
    })
  }

  protected def backwardImpl(gradOutput: Tensor[T, TT]): Seq[Tensor[_, TT]]

  def unbroadcast(v: Variable[T, TT], oldShape: Shape): Variable[T, TT] = {
    Variable(unbroadcast(v.data, oldShape))
  }

  def unbroadcast(data: Tensor[T, TT], oldShape: Shape): Tensor[T, TT] = {
    data.shape.asArray.zipWithIndex.reverse.foldLeft(data) {
      case (d: Tensor[T, TT], (ni, i)) =>
        if (oldShape.rank <= i)
          d.sum(i)
        else if (oldShape(i) == ni)
          d
        else if (oldShape(i) == 1)
          d.sum(Array(i), keepdim = true)
        else
          throw new Exception(
            s"unable to unbroadcast shape ${data.shape} to $oldShape")
    }
  }
}


abstract class UnaryFunction[T: ClassTag, TT <: TensorType](name: String, v: Variable[_, TT]) extends Function[T, TT](name, v) {
}

abstract class BinaryFunction[T: ClassTag, TT <: TensorType](name: String, v1: Variable[_, TT], v2: Variable[_, TT]) extends Function[T, TT](name, v1, v2) {
}

case class Add[T: ClassTag, TT <: TensorType](v1: Variable[T, TT], v2: Variable[T, TT]) extends Function[T, TT]("+", v1, v2) {
  def forwardImpl(): Tensor[T, TT] = v1.data + v2.data
  override def backwardImpl(gradOutput: Tensor[T, TT]): Seq[Tensor[_, TT]]  = {
    val g1 = unbroadcast(gradOutput, v1.shape)
    val g2 = unbroadcast(gradOutput, v2.shape)
    Seq(g1, g2)
  }
}

case class AddConstant[T: ClassTag, TT <: TensorType](v: Variable[T, TT], d: Scalar[T]) extends Function[T, TT]("+", v) {
  override def forwardImpl(): Tensor[T, TT] = v.data + d
  override def varName: String = "(" + v.name.getOrElse("") + " + " + d.getValue.toString + ")"
  override def backwardImpl(gradOutput: Tensor[T, TT]): Seq[Tensor[_, TT]] = {
    Seq(gradOutput)
  }
}

case class Sub[T: ClassTag, TT <: TensorType](v1: Variable[T, TT], v2: Variable[T, TT]) extends Function[T, TT]("-", v1, v2) {
  def forwardImpl(): Tensor[T, TT] =  v1.data - v2.data
  def backwardImpl(gradOutput: Tensor[T, TT]): Seq[Tensor[_, TT]] = {
    val g1 = unbroadcast(gradOutput, v1.shape)
    val g2 = unbroadcast(-gradOutput, v2.shape)
    Seq(g1, g2)
  }
}

case class SubConstant[T: ClassTag, TT <: TensorType](v: Variable[T, TT], d: Scalar[T]) extends Function[T, TT]("-", v) {
  override def forwardImpl(): Tensor[T, TT] = v.data - d
  override def varName: String = "(" + v.name.getOrElse("") + " - " + d.getValue.toString + ")"
  override def backwardImpl(gradOutput: Tensor[T, TT]): Seq[Tensor[_, TT]] = {
    Seq(gradOutput)
  }
}

case class Mul[T: ClassTag, TT <: TensorType](v1: Variable[T, TT], v2: Variable[T, TT]) extends Function[T, TT]("*", v1, v2) {
  override def forwardImpl(): Tensor[T, TT] = v1.data * v2.data
  override def backwardImpl(gradOutput: Tensor[T, TT]): Seq[Tensor[_, TT]] = {
    val dv1 = v2.data * gradOutput
    val vdv1 = unbroadcast(dv1, v1.shape)
    val dv2 = v1.data * gradOutput
    val vdv2 = unbroadcast(dv2, v2.shape)
    Seq(vdv1, vdv2)
  }
}

case class MulConstant[T: ClassTag, TT <: TensorType](v: Variable[T, TT], d: Scalar[T]) extends Function[T, TT]("*", v) {
  override def forwardImpl(): Tensor[T, TT] = v.data * d
  override def varName: String = "(" + v.name.getOrElse("") + " * " + d.getValue.toString + ")"
  override def backwardImpl(gradOutput: Tensor[T, TT]): Seq[Tensor[_, TT]] = {
    Seq(gradOutput * d)
  }
}

case class Div[T: ClassTag, TT <: TensorType](v1: Variable[T, TT], v2: Variable[T, TT])(implicit num: Numeric[T]) extends Function[T, TT]("/", v1, v2) {
  override def forwardImpl(): Tensor[T, TT] = v1.data / v2.data
  override def backwardImpl(gradOutput: Tensor[T, TT]): Seq[Tensor[_, TT]] = {
    val rv2 = Tensor.ones_like(v2.data) / v2.data
    val gv1 = gradOutput * rv2
    val gv2 = -gradOutput * v1.data * (rv2 ** num.fromInt(2))

    val vgv1 = unbroadcast(gv1, v1.shape)
    val vgv2 = unbroadcast(gv2, v2.shape)
    Seq(vgv1, vgv2)
  }
}

case class DivConstant[T: ClassTag, TT <: TensorType](v: Variable[T, TT], d: Scalar[T]) extends Function[T, TT]("/", v) {
  override def forwardImpl(): Tensor[T, TT] = v.data / d
  override def varName: String = "(" + v.name.getOrElse("") + " / " + d.getValue.toString + ")"
  override def backwardImpl(gradOutput: Tensor[T, TT]): Seq[Tensor[_, TT]] = {
    val dv = gradOutput / d
    Seq(dv)
  }
}

/* Pow of 2 tensors is currently not implemented in numsca
case class Pow(a: Variable[T, TT], b: Variable[T, TT]) extends Function[T, TT] {


  override def forwardImpl(): Tensor[T, TT] = {
    Variable[T, TT](a.data ** b.data, Some(this))
  }

  override def backwardImpl(gradOutput: Tensor[T, TT]): Seq[Tensor[_, TT]] = {
     val ga = gradOutput * b.data * (a.data ** (b.data - 1))
     val gb = gradOutput * (a.data ** b.data) * ns.log(a.data)

    val vga = unbroadcast(ga, a.shape)
    val vgb = unbroadcast(gb, b.shape)

    logger.debug(s"pow backward, ga.shape=${vga.shape}, gb.shape=${vgb.shape}")
    a.backward(vga)
    b.backward(vgb)
  }
}
 */

case class PowConstant[T: ClassTag, TT <: TensorType](v: Variable[T, TT], d: Scalar[T])(implicit num: Numeric[T]) extends Function[T, TT]("pow", v) {
  override def forwardImpl(): Tensor[T, TT] = v.data.**(d)
  override def varName: String = v.name + s".pow($d)"
  val cache: Tensor[T, TT] = v.data.**(num.minus(d.getValue, num.fromInt(1))) * d
  override def backwardImpl(gradOutput: Tensor[T, TT]): Seq[Tensor[_, TT]] = {
    val dv = cache * gradOutput
    Seq(dv)
  }
}

case class Sqrt[T: ClassTag, TT <: TensorType](v: Variable[T, TT])(implicit num: Numeric[T]) extends Function[T, TT]("sqrt", v) {
  val cache: Tensor[T, TT] = v.data.sqrt()
  override def forwardImpl(): Tensor[T, TT] = cache
  override def backwardImpl(gradOutput: Tensor[T, TT]): Seq[Tensor[_, TT]] = {
    val dv = (num.one / (cache * num.fromInt(2))) * gradOutput
    Seq(dv)
  }
}

case class Abs[T: ClassTag, TT <: TensorType](v: Variable[T, TT]) extends Function[T, TT]("abs", v) {
  val cache: Tensor[T, TT] = v.data.abs()
  override def forwardImpl(): Tensor[T, TT] = cache
  override def backwardImpl(gradOutput: Tensor[T, TT]): Seq[Tensor[_, TT]] = {
    val dv = (v.data / cache) * gradOutput
    Seq(dv)
  }
}

case class Negate[T: ClassTag, TT <: TensorType](v: Variable[T, TT]) extends Function[T, TT]("-", v) {
  override def forwardImpl(): Tensor[T, TT] = -v.data
  override def backwardImpl(gradOutput: Tensor[T, TT]): Seq[Tensor[_, TT]] = {
    val dv = -gradOutput
    Seq(dv)
  }
}

case class Dot[T: ClassTag, TT <: TensorType](v1: Variable[T, TT], v2: Variable[T, TT]) extends Function[T, TT]("dot", v1, v2) {
  val w: Tensor[T, TT] = v1.data
  val x: Tensor[T, TT] = v2.data

  override def forwardImpl(): Tensor[T, TT] = w dot x
  override def backwardImpl(gradOutput: Tensor[T, TT]): Seq[Tensor[_, TT]] = {
    val dd = gradOutput
    val dw = x * dd
    val dx = w * dd
    Seq(dw, dx)
  }
}

case class Matmul[T: ClassTag, TT <: TensorType](v1: Variable[T, TT], v2: Variable[T, TT]) extends Function[T, TT]("matmul", v1, v2) {
  val w: Tensor[T, TT] = v1.data
  val x: Tensor[T, TT] = v2.data

  override def forwardImpl(): Tensor[T, TT] = w matmul x
  override def backwardImpl(gradOutput: Tensor[T, TT]): Seq[Tensor[_, TT]] = {
    val dd = gradOutput
    val dw = dd matmul x.T
    val dx = w.T matmul dd
    Seq(dw, dx)
  }
}

case class Transpose[T: ClassTag, TT <: TensorType](v: Variable[T, TT]) extends Function[T, TT]("T", v) {
  override def forwardImpl(): Tensor[T, TT] = v.data.T
  override def backwardImpl(gradOutput: Tensor[T, TT]): Seq[Tensor[_, TT]] = {
    Seq(gradOutput.T)
  }
}

// todo test this
case class Reshape[T: ClassTag, TT <: TensorType](v: Variable[T, TT], shape: Shape) extends Function[T, TT]("reshape", v) {
  val oldShape: Shape = v.shape
  override def forwardImpl(): Tensor[T, TT] = v.data.reshape(shape)
  override def backwardImpl(gradOutput: Tensor[T, TT]): Seq[Tensor[_, TT]] = {
    val dv = gradOutput.reshape(oldShape)
    Seq(dv)
  }
}

case class Concat[T: ClassTag, TT <: TensorType](v1: Variable[T, TT], v2: Variable[T, TT], axis: Int = 0) extends Function[T, TT]("concat", v1, v2) {

  override def forwardImpl(): Tensor[T, TT] = Basic.cat(v1.data, v2.data, axis)

  override def backwardImpl(gradOutput: Tensor[T, TT]): Seq[Tensor[_, TT]] = {
      val d = gradOutput
      val d12 = d.split_with_sizes(Array(v1.shape(axis), v2.shape(axis)), axis)
      val dv1 = d12(0)
      val dv2 = d12(1)
      Seq(dv1, dv2)
  }

}

//===============================================================

case class Exp[T: ClassTag, TT <: TensorType](v: Variable[T, TT]) extends Function[T, TT]("exp", v) {
  val cache: Tensor[T, TT] = v.data.exp()
  def forwardImpl(): Tensor[T, TT] = cache
  def backwardImpl(gradOutput: Tensor[T, TT]): Seq[Tensor[_, TT]] = {
    val g = gradOutput * cache
    Seq(g)
  }
}

case class Cos[T: ClassTag, TT <: TensorType](v: Variable[T, TT]) extends Function[T, TT]("cos", v) {
  val cache: Tensor[T, TT] = v.data.sin() * v.data.dataType.cast(-1)
  def forwardImpl(): Tensor[T, TT] = v.data.cos()
  def backwardImpl(gradOutput: Tensor[T, TT]): Seq[Tensor[_, TT]] = {
    val g = gradOutput * cache
    Seq(g)
  }
}

case class Sin[T: ClassTag, TT <: TensorType](v: Variable[T, TT]) extends Function[T, TT]("sin", v) {
  val cache: Tensor[T, TT] = v.data.cos()
  def forwardImpl(): Tensor[T, TT] = v.data.sin()
  def backwardImpl(gradOutput: Tensor[T, TT]): Seq[Tensor[_, TT]] = {
    val g = gradOutput * cache
    Seq(g)
  }
}

case class Tanh[T: ClassTag, TT <: TensorType](v: Variable[T, TT]) extends Function[T, TT]("tanh", v) {
  val cache: Tensor[T, TT] = v.data.tanh()
  override def forwardImpl(): Tensor[T, TT] = cache
  override def backwardImpl(gradOutput: Tensor[T, TT]): Seq[Tensor[_, TT]] = {
    val g = MathBackward.tanh_backward(gradOutput, cache)
    Seq(g)
  }
    //v.backward(Variable[T, TT]((1 - cache.**(2)) * gradOutput))
}

case class Sigmoid[T: ClassTag, TT <: TensorType](v: Variable[T, TT])(implicit num: Numeric[T]) extends Function[T, TT]("sigmoid", v) {
  lazy val sigmoid: Tensor[T, TT] = v.data.sigmoid()
  override def forwardImpl(): Tensor[T, TT] = sigmoid
  override def backwardImpl(gradOutput: Tensor[T, TT]): Seq[Tensor[_, TT]] = {
    val g = gradOutput * sigmoid * (num.one - sigmoid)
    Seq(g)
  }
}

case class Softmax[T: ClassTag, TT <: TensorType](v: Variable[T, TT], dim: Long) extends Function[T, TT]("softmax", v) {
  lazy val softmax: Tensor[T, TT] = v.data.softmax(dim)
  override def forwardImpl(): Tensor[T, TT] = softmax
  override def backwardImpl(gradOutput: Tensor[T, TT]): Seq[Tensor[_, TT]] = {
    // from https://stackoverflow.com/questions/33541930/how-to-implement-the-softmax-derivative-independently-from-any-loss-Function[T, TT]
    val y = softmax
    val dy = gradOutput

    val dx = y * dy
    val s = dx.sum(Array(dim.toInt), true)
    dx -= y * s
    Seq(dx)
  }
}

case class Mean[T: ClassTag, TT <: TensorType](v: Variable[T, TT])(implicit num: Numeric[T]) extends Function[T, TT]("mean", v) {
  def forwardImpl(): Tensor[T, TT] = v.data.mean()
  def backwardImpl(gradOutput: Tensor[T, TT]): Seq[Tensor[_, TT]] = {
    val n: T = v.shape.numElements.asInstanceOf[T]
    val g = gradOutput / n
    Seq(g)
  }
}

case class MeanByAxis[T: ClassTag, TT <: TensorType](v: Variable[T, TT], axis: Int)(implicit num: Numeric[T]) extends Function[T, TT]("mean", v) {
  def forwardImpl(): Tensor[T, TT] = v.data.mean(Array(axis), true)
  override def varName: String = v.name + s".mean($axis)"
  def backwardImpl(gradOutput: Tensor[T, TT]): Seq[Tensor[_, TT]] = {
    val n: T = num.fromInt(v.shape(axis))
    val g = gradOutput / n
    Seq(g)
  }
}

case class Variance[T: ClassTag, TT <: TensorType](v: Variable[T, TT])(implicit num: Numeric[T]) extends Function[T, TT]("var", v) {
  override def forwardImpl(): Tensor[T, TT] = (v - v.mean()).pow(num fromInt 2).mean().data
  override def backwardImpl(gradOutput: Tensor[T, TT]): Seq[Tensor[_, TT]] = {
    (v - v.mean()).pow(num fromInt 2).mean().backward(Variable(gradOutput))
    Seq(v.grad.data)
  }
}

case class VarianceByAxis[T: ClassTag, TT <: TensorType](v: Variable[T, TT], axis: Int)(implicit num: Numeric[T]) extends Function[T, TT]("var", v) {
  override def forwardImpl(): Tensor[T, TT] = (v - v.mean(axis)).**(num.fromInt(2)).mean(axis).data
  override def backwardImpl(gradOutput: Tensor[T, TT]): Seq[Tensor[_, TT]] = {
    (v - v.mean(axis)).**(num.fromInt(2)).mean(axis).backward(Variable(gradOutput))
    Seq(v.grad.data)
  }
}

case class Max[T: ClassTag, TT <: TensorType](x: Variable[T, TT], y: Variable[T, TT]) extends Function[T, TT]("max", x, y) {
  def forwardImpl(): Tensor[T, TT] = x.data.maximum(y.data)
  override def backwardImpl(gradOutput: Tensor[T, TT]): Seq[Tensor[_, TT]] = {
    val gx = (x.data ge y.data).to(x.data) * gradOutput
    val gy = (x.data le y.data).to(y.data) * gradOutput
    Seq(gx, gy)
  }
}

case class Threshold[T: ClassTag, TT <: TensorType](x: Variable[T, TT], d: Scalar[T]) extends Function[T, TT]("threshold", x) {
  override def forwardImpl(): Tensor[T, TT] = x.data.threshold(d, d)
  override def varName: String = x.name + s".threshold($d)"
  override def backwardImpl(gradOutput: Tensor[T, TT]): Seq[Tensor[_, TT]] = {
    val g = gradOutput * (x.data gt d).to(x.data)
    Seq(g)
  }
}

//============================================
// Loss Function[T, TT]s
case class SoftmaxLoss[T: ClassTag, TT <: TensorType](actual: Variable[T, TT], target: Variable[Long, TT])(implicit num: Numeric[T]) extends Function[T, TT]("softmaxLoss", actual) {
  val x: Tensor[T, TT] = actual.data
  val n: Int = x.shape(0)
  implicit val long_opt: TensorOptions[Long, TT] = x.long_options()
  val y: Tensor[Long, TT] = target.data.reshape(Shape(n))

  val shiftedLogits: Tensor[T, TT] = x - x.maximum(1, keepdim = true)
  val z: Tensor[T, TT] = shiftedLogits.exp().sum(Array(1), true)
  val logProbs: Tensor[T, TT] = shiftedLogits - z.log()
  val p: Int = x.shape(1)
  val m = Tensor.arange[Long, TT](0, n) * p + y
  val loss = - logProbs.take(m).sum() / num.fromInt(n)


  override def forwardImpl(): Tensor[T, TT] = loss

  def backwardImpl(gradOutput: Tensor[T, TT] = null /* not used */ ): Seq[Tensor[_, TT]]  = {
    val dx: Tensor[T, TT] = logProbs.exp()
    val m = Tensor.arange[Long, TT](0, n) * p + y
    dx.put(m, dx.take(m) - num.one)

    val g = dx / num.fromInt(n)
    Seq(g)
  }
}

///**
//  * Computes the cross-entropy loss
//  * @param actuals sequence of yHat Variable[T, TT]s
//  * @param targets sequence of Y indices (ground truth)
//  */
//case class CrossEntropyLoss[T: ClassTag, TT <: TensorType](actuals: Seq[Variable[T, TT]], targets: Seq[Int])
//    extends Function[T, TT]("") {
//
//  /**
//    * Computes the cross entropy loss, and wraps it in a Variable[T, TT].
//    * The Variable[T, TT] can be back propped into, to compute the gradients of the parameters
//    * @return the cross entropy loss Variable[T, TT]
//    */
//  override def forwardImpl(): Tensor[T, TT] = {
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
//  override def backwardImpl(gradOutput: Tensor[T, TT]): Seq[Tensor[_, TT]] = {
//    actuals.zip(targets).reverse.foreach {
//      case (yh, y) =>
//        val dy = ns.copy(yh.data)
//        dy(y, 0) -= 1
//        yh.backward(Variable[T, TT](dy))
//    }
//  }
//}
