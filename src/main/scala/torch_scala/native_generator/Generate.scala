package torch_scala.native_generator

import java.io.FileInputStream

import com.github.javaparser.JavaParser
import com.github.javaparser.ast.body.{FieldDeclaration, MethodDeclaration}
import generate.Builder
import jdk.internal.org.objectweb.asm.MethodVisitor
import org.bytedeco.javacpp.{Loader, Pointer}
import org.bytedeco.javacpp.tools.{Generator, Logger}
import torch_java.mapping.FunctionsMapper
import torch_scala.api.nn.Module
import torch_scala.examples.FourierNet

object Generate extends App {

  case class ParameterDeclaration(
                                   decorator: Option[String],
                                   name: String,
                                   typeName: String
                                 )

  class ScalaMethodDeclaration(
                                val isNative: Boolean,
                                val typeDecorator: Array[String],
                                val returnType: String,
                                val name: String,
                                val templates: Option[String]
                                   )

  def toScalaType(name: String, methodOrVarName: String): String = name.trim match {
    case "Tensor<T,TT>" =>
      if (methodOrVarName.contains("index")) "Tensor[Long, TT]"
      else "Tensor[T, TT]"
    case _ => name

  }


  object ScalaMethodDeclaration {
    def apply(md: MethodDeclaration): ScalaMethodDeclaration = {
      new ScalaMethodDeclaration(
        md.isNative,
        md.getAnnotations.toArray().map(_.toString),
        toScalaType(md.getType.asString(), md.getName.toString),
        md.getName.asString(),
        None
      )
    }
  }




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

  //val gen = new Builder()
  //gen.classesOrPackages("torch_java.mapping.FunctionsMapper")
  //gen.build()


  val in = new FileInputStream("NativeFunctions.java")

  val cu = new JavaParser().parse(in).getResult.get()
  in.close()

  import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration
  import java.util.Optional

  val classA = cu.getClassByName("NativeFunctions")

  classA.get().getMethods.forEach(md => {
    val smd = ScalaMethodDeclaration(md)
    println(smd)
  })




}
