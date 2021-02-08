package examples

import torch_scala.NativeLoader
import torch_scala.api.aten._
import torch_scala.api.aten.functions.NativeFunctions






object FourierModelApp extends App {

  val loader = NativeLoader

  val t = Tensor[Double, CUDA](1,2,3,4)

  println(t)

  val tt = NativeFunctions.add(t, t)
  println(tt)



}
