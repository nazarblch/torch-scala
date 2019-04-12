package torch_scala.autograd

import torch_scala.api.aten.{Tensor, TensorType}

import scala.collection.mutable
import scala.reflect.ClassTag


class GradientsMap(data: mutable.HashMap[Variable[Any, TensorType], Tensor[Any, TensorType]],
                   pendingCount: mutable.HashMap[Variable[Any, TensorType], Int]) {

  def add(v2g: VariableWithGradient[Any, TensorType]): Unit = {
    val v = v2g.variable
    val g = v2g.grad
    if (data.contains(v)) {
      data.put(v, data(v) + g)
    } else {
      data.put(v, g)
    }
    pendingCount.update(v, pendingCount(v) - 1)
    assert(pendingCount(v) >= 0)
  }

  def get(v: Variable[Any, TensorType]): Tensor[Any, TensorType] = data(v)

  def isPending(v: Variable[Any, TensorType]): Boolean = {
    pendingCount(v) > 0
  }

  def getPendingCount(v: Variable[Any, TensorType]): Int = {
    pendingCount(v)
  }

  def result: Map[Variable[Any, TensorType], Tensor[Any, TensorType]] = {
    if(!pendingCount.values.forall(_ == 0)) {
      pendingCount.filter(_._2 != 0).keys.foreach(println)
    }
    assert(pendingCount.values.forall(_ == 0))
    data.toMap
  }
}


trait BackwardGraphFilter {
  def filterArgs(variable: Variable[Any, TensorType]): Seq[Variable[Any, TensorType]]
}

object DefaultBackwardGraphFilter extends BackwardGraphFilter {
  def filterArgs(variable: Variable[Any, TensorType]): Seq[Variable[Any, TensorType]] = if (variable.gradFn.isEmpty) {
    Seq()
  } else {
    variable.gradFn.get.args.map(_.asInstanceOf[Variable[Any, TensorType]])
  }
}

class BackwardGraphFilterWithLeaves(leaves: Set[Variable[Any, TensorType]]) extends BackwardGraphFilter {

  private val included = new mutable.HashMap[Variable[Any, TensorType], Boolean]()
  leaves.foreach(v => included.put(v, true))

  def contains(variable: Variable[Any, TensorType]): Boolean = included.get(variable) match {
    case Some(true) => true
    case Some(false) => false
    case None => if (variable.gradFn.isEmpty) {
      included.put(variable, false)
      false
    } else {
      val res = variable.gradFn.get.args.exists(vi => contains(vi.asInstanceOf[Variable[Any, TensorType]]))
      included.put(variable, res)
      res
    }
  }

  def filterArgs(variable: Variable[Any, TensorType]): Seq[Variable[Any, TensorType]] = if (variable.gradFn.isEmpty) {
    Seq()
  } else {
    variable.gradFn.get.args.map(_.asInstanceOf[Variable[Any, TensorType]]).filter(contains)
  }

}


object Gradient {

  def backward[T: ClassTag, TT <: TensorType](head_v: Variable[T, TT],
               head_g: Tensor[T, TT],
               arguments: Set[Variable[_ , TT]]): Map[Variable[Any, TensorType], Tensor[Any, TensorType]] = {
    backward(head_v.asInstanceOf[Variable[Any, TensorType]],
      head_g.asInstanceOf[Tensor[Any, TensorType]],
      arguments.map(_.asInstanceOf[Variable[Any, TensorType]]))
  }

  def backward(head_v: Variable[Any, TensorType],
               head_g: Tensor[Any, TensorType],
               arguments: Set[Variable[Any, TensorType]] = Set()): Map[Variable[Any, TensorType], Tensor[Any, TensorType]] = {

    val backwardGraphFilter = arguments.size match {
      case 0 => DefaultBackwardGraphFilter
      case _ => new BackwardGraphFilterWithLeaves(arguments)
    }

    val grad = new GradientsMap(mutable.HashMap(), variablePendingCount(head_v, backwardGraphFilter))
    grad.add(VariableWithGradient(head_v, head_g))
    val stack = new mutable.ArrayStack[Variable[Any, TensorType]]()
    stack.push(head_v)

    while (stack.nonEmpty) {
      val top_v = stack.pop()
      assert(grad.getPendingCount(top_v) == 0)
      val top_g = grad.get(top_v)
      val args_set = backwardGraphFilter.filterArgs(top_v).toSet
      for (fn <- top_v.gradFn) { if (args_set.nonEmpty) {
        fn.backward(top_g)
          .map(v2g => v2g.asInstanceOf[VariableWithGradient[Any, TensorType]])
          .filter(v2g => args_set.contains(v2g.variable))
          .foreach({v2g =>
            grad.add(v2g)
            if(!grad.isPending(v2g.variable)) {
              stack.push(v2g.variable)
            }
        })
      }}
    }

    arguments.size match {
      case 0 => grad.result
      case _ => grad.result.filterKeys(arguments.contains)
    }
  }

  private def variablePendingCount(v: Variable[Any, TensorType], filter: BackwardGraphFilter): mutable.HashMap[Variable[Any, TensorType], Int] = {

    val stack = new mutable.ArrayStack[Variable[Any, TensorType]]()
    stack.push(v)
    val data = new mutable.HashMap[Variable[Any, TensorType], Int]()
    data.put(v, 1)

    while (stack.nonEmpty) {
      val top_v = stack.pop()
      val args = filter.filterArgs(top_v)
      args.foreach(vi => {
          data.put(vi, data.getOrElse(vi, 0) + 1)
          if (data(vi) == 1) {
            stack.push(vi)
          }
      })
    }

    data
  }

}
