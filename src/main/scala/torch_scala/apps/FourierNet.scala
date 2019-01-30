package torch_scala.apps

import org.bytedeco.javacpp.{FloatPointer, IntPointer, Loader, LongPointer}
import org.bytedeco.javacpp.tools.{Builder, Generator, Logger}
import torch_scala.api.Functions.Deallocator_Pointer
import torch_scala.api._
import torch_scala.api.nn.Module
import torch_scala.examples.FourierNet


object FourierNetApp extends App {


  //System.setProperty("org.bytedeco.javacpp.loadlibraries", "false")

//  val net = new FourierNet(20)
//  val pred = net.train(Array[Float](1, 2, 3, 3, 4, 5, 6, 7, 6, 4, 3, 2, 4, 4, 5), 500, Array[Float](1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1))
//  val loss = net.loss(pred, Array[Float](1, 2, 3, 3, 4, 5, 6, 7, 6, 4, 3, 2, 4, 4, 5), Array[Float](1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1))
//  System.out.println("loss = " + loss)

  val data = Array(4l, 5l)
  val list = new IntList(data)

  println(list.data().get(1))

  val t: Tensor[Float] = Functions.from_blob[Float](new FloatPointer(2f,3f,5f,6f), new IntList(Array(2, 2)), new Deallocator_Pointer(new FloatPointer()))
  t.print()

//
//  println(t.toString, t.dim, t.scalar_type())
//  println(t.data.mkString(","))


  val arrayRef = new FloatList(Array(1f,4f,6f))

  val t1: Tensor[CudaTensorType[Float]] = Functions.tensor[CudaTensorType[Float]](arrayRef)
  println(t1.toString, t1.dim, t1.scalar_type())

  println(t1.cpu().data())

  val dev = CudaDevice
  println(dev.has_index(), dev.is_cuda())


}