package torch_scala.autograd

import torch_scala.api.aten.{Tensor, TensorType}

import scala.collection.mutable


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

  def result: Map[Variable[Any, TensorType], Tensor[Any, TensorType]] = {
    assert(pendingCount.values.forall(_ == 0))
    data.toMap
  }
}


class SubGraphBuilder(leaves: Set[Variable[Any, TensorType]]) {
  def build(root: Variable[Any, TensorType]): Map[Variable[Any, TensorType], Boolean] = {
    val included = new mutable.HashMap[Variable[Any, TensorType], Boolean]()
  }
}


object Gradient {

  def backward(head_v: Variable[Any, TensorType], head_g: Tensor[Any, TensorType]): Map[Variable[Any, TensorType], Tensor[Any, TensorType]] = {
    val grad = new GradientsMap(mutable.HashMap(), variablePendingCount(head_v))
    grad.add(VariableWithGradient(head_v, head_g))
    val stack = new mutable.ArrayStack[Variable[Any, TensorType]]()
    stack.push(head_v)

    while (stack.nonEmpty) {
      val top_v = stack.pop()
      val top_g = grad.get(top_v)
      for (fn <- top_v.gradFn) {
        fn.backward(top_g)
          .map(v2g => v2g.asInstanceOf[VariableWithGradient[Any, TensorType]])
          .foreach({v2g =>
            grad.add(v2g)
            if(!grad.isPending(v2g.variable)) {
              stack.push(v2g.variable)
            }
        })
      }
    }

    grad.result
  }

  private def variablePendingCount(v: Variable[Any, TensorType]): mutable.HashMap[Variable[Any, TensorType], Int] = {

    val stack = new mutable.ArrayStack[Variable[Any, TensorType]]()
    stack.push(v)
    val data = new mutable.HashMap[Variable[Any, TensorType], Int]()
    data.put(v, 1)
    val visited = mutable.Set[Variable[Any, TensorType]]()
    visited.add(v)

    while (stack.nonEmpty) {
      val top_v = stack.pop()
      for (fn <- top_v.gradFn) {
        fn.args.map(_.asInstanceOf[Variable[Any, TensorType]]).foreach(vi => {
          data.put(vi, data.getOrElse(vi, 0) + 1)
          if (!visited.contains(vi)) {
            stack.push(vi)
            visited.add(vi)
          }
        })
      }
    }

    data
  }

}
