package torch_scala.autograd



import torch_scala.api.aten.{Shape, Tensor, TensorType}

import scala.language.implicitConversions
import scala.reflect.ClassTag

case class Variable[T, TT <: TensorType[T]](data: Tensor[T, TT],
                                                      gradFn: Option[Function[T, TT]] = None,
                                                      name: Option[String] = None) {

  override def toString: String =
    if (name.isDefined) s"name: ${name.get}, data: $data" else s"data: $data"

  lazy val grad: Variable[T, TT] =
    Variable(Tensor.zeros_like(data), name = name.map(n => s"g_$n"))
  def shape: Shape = data.shape

  def backward(): Unit = {
    backward(Variable(Tensor.ones_like(data)))
  }

  def backward(gradOutput: Variable[T, TT]): Unit = {
    grad.data += gradOutput.data
    for (gf <- gradFn) gf.backward(gradOutput)
  }

  def detach(name: Option[String] = None) = Variable(data, name = name)

  // chain operator
  def ~>[T1, TT1 <: TensorType[T1]](f: Variable[T, TT] => Variable[T1, TT1]): Variable[T1, TT1] = f(this)
}
