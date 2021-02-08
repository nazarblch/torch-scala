package examples

import org.bytedeco.javacpp.Loader
import org.bytedeco.javacpp.tools.Builder
//import javacpp_tools.{Generator, Logger}
import torch_scala.api.aten.functions.NativeFunctions

object GenerateCpp extends App {

//  val gen = new Generator(Logger.create(GenerateCpp.getClass), Loader.loadProperties())
//  gen.generate("", "", "", "", "", NativeFunctions.getClass)
  Builder.main(
    Array("torch_scala.api.aten.ArrayRefDouble",
           "torch_scala.api.aten.Tensor",
                "torch_scala.api.aten.TensorVector",
                 "torch_scala.api.aten.TensorOptions",
                "torch_scala.api.aten.TensorTuple4AndLong",
                 "torch_scala.api.aten.DoubleLong",
                 "torch_scala.api.std.ArrayBool4",
        "torch_scala.api.aten.Scalar",
         "torch_scala.api.aten.TensorTupleAndDoubleLong",
         "torch_scala.api.aten.TensorOptions$",
         "torch_scala.api.aten.IntList",
             "torch_scala.api.aten.TensorTripleAndLong",
             "torch_scala.api.aten.ArrayRefByte",
             "torch_scala.api.aten.functions.Functions$",
         "torch_scala.api.aten.functions.Basic$" ,
         "torch_scala.api.aten.functions.Math$" ,
         "torch_scala.api.aten.functions.NativeFunctions$"
    )
  )
}
