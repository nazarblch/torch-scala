package torch_scala.apps


import torch_scala.NativeLoader
import torch_scala.api.aten._






object FourierModelApp extends App {

  val loader = NativeLoader

  val t = Tensor[Int, CPU](1,2,3,4)

  println(t)

//  val t1 = NativeFunctions.clamp(t, new Scalar[Int](0), new Scalar[Int](2))

//  println(t1)

}
