package torch_scala.autograd

import torch_scala.api.aten.{Tensor, TensorType}

import scala.collection.mutable


class GradientsMap(data: mutable.HashMap[Variable[Any, TensorType], Tensor[Any, TensorType]]) {
  def add(v: Variable[Any, TensorType], g: Tensor[Any, TensorType]): Unit = {
    if (data.contains(v)) {
      data.put(v, data(v) + g)
    } else {
      data.put(v, g)
    }
  }

  def get(v: Variable[Any, TensorType]): Tensor[Any, TensorType] = data(v)

  def result = data.toMap
}



object Gradient {

  def backward(v: Variable[Any, TensorType], g: Tensor[Any, TensorType]): Map[Variable[Any, TensorType], Tensor[Any, TensorType]] = {
    val grad = new GradientsMap(mutable.HashMap())
    grad.add(v, g)
    val stack = new mutable.ArrayStack[(Variable[Any, TensorType], Tensor[Any, TensorType])]()
    stack.push((v, g))
    while (stack.nonEmpty) {
      val (top_v, top_g) = stack.pop()
      for (fn <- top_v.gradFn) {
        fn.backward(top_g).foreach({case(vi, gi) =>
          // println(vi.name)
          // println(Tensor.summarize(gi.asInstanceOf[Tensor[Any, TensorType]]))
          grad.add(vi.asInstanceOf[Variable[Any, TensorType]], gi.asInstanceOf[Tensor[Any, TensorType]])
          stack.push((vi.asInstanceOf[Variable[Any, TensorType]], gi.asInstanceOf[Tensor[Any, TensorType]]))
        })
      }
    }

    grad.result
  }

  private def variablePendingCount(v: Variable[Any, TensorType]): Map[Variable[Any, TensorType], Int] = {

    val stack = new mutable.ArrayStack[Variable[Any, TensorType]]()
    stack.push(v)
    val data = new mutable.HashMap[Variable[Any, TensorType], Int]()

    while (stack.nonEmpty) {
      val top_v = stack.pop()
      for (fn <- top_v.gradFn) {
//        fn.backward(top_g).foreach({case(vi, gi) =>
//          // println(vi.name)
//          // println(Tensor.summarize(gi.asInstanceOf[Tensor[Any, TensorType]]))
//          grad.add(vi.asInstanceOf[Variable[Any, TensorType]], gi.asInstanceOf[Tensor[Any, TensorType]])
//          stack.push((vi.asInstanceOf[Variable[Any, TensorType]], gi.asInstanceOf[Tensor[Any, TensorType]]))
//        })
      }
    }

    data.toMap

  }

}
