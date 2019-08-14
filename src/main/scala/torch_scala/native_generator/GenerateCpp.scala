package torch_scala.native_generator

import generate.{Builder, Generator, Logger}
import org.bytedeco.javacpp.Loader


object GenerateCpp extends App {

  val gen = new Generator(Logger.create(GenerateCpp.getClass), Loader.loadProperties())
//  gen.generate("", "", "", "", "",
//    NativeFunctions.getClass)

}
