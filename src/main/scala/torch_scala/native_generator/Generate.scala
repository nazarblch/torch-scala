package torch_scala.native_generator

import generate.Builder
import jep.Jep
import org.bytedeco.javacpp.Loader
import org.bytedeco.javacpp.tools.{Generator, Logger}
import torch_scala.api.nn.Module
import torch_scala.examples.FourierNet

object Generate extends App {

//  val gen = new Generator(Logger.create(classOf[Generator]), Loader.loadProperties)

  //  val res = gen.generate("/home/nazar/java_torch_2/src/native/java_torch_lib.cpp", "/home/nazar/java_torch_2/src/native/java_torch_lib.h", "", "", "",
  //    classOf[FourierNet],
  //    classOf[Module]
  //  )

//  val gen = new Builder()
//  gen.outputDirectory("/home/nazar/java_torch_2/src/native")
//  gen.classesOrPackages("torch_scala.api.nn.Module", "torch_scala.examples.FourierNet")
//  gen.build()

//  println(res)

  val jep: Jep = new Jep()

    jep.eval("import CppHeaderParser")
    // any of the following work, these are just pseudo-examples

    // using eval(String) to invoke methods
    jep.set("arg", -1)
    jep.eval("cppHeader = CppHeaderParser.CppHeader(\"/home/nazar/libtorch/include/ATen/Functions.h\")")
    jep.eval("functions = cppHeader.functions")
    val obj = jep.getValue("cppHeader", classOf[Object])
  println(obj)





}
