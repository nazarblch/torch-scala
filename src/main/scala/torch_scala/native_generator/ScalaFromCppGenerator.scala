package torch_scala.native_generator

import java.io.{FileInputStream, FileWriter}

import com.github.javaparser.JavaParser
import com.github.javaparser.ast.body.{MethodDeclaration, Parameter}
import generate.{Builder, UserClassLoader}
import org.bytedeco.javacpp.{ClassProperties, Loader}


class ScalaFromCppGenerator(val mappingClass: String) {

  val loader = new UserClassLoader(Thread.currentThread.getContextClassLoader)
  val JAVA_TMP_DIR = "java_generated_tmp"
  val SCALA_SRC_DIR = "src/main/scala"
  val METHODS_TO_IGNORE = Set(
    "from_blob",
    "range",
    "size",
    "q_zero_point",
    "numel",
    "_cufft_get_plan_cache_max_size",
    "_cufft_get_plan_cache_size",
    "stride",
    "_debug_has_internal_overlap"

  )
  val TYPES_TO_IGNORE = Set("Dimname", "DimnameList", "FunctionPointer")

  val mapping: java.lang.Class[_] = loadClass(mappingClass)
  val properties: ClassProperties = Loader.loadProperties(mapping, Loader.loadProperties(), false)
  val target: String = properties.get("target").get(0)
  val target_name: String = target.split('.').last
  val target_pkg: String = target.split('.').dropRight(1).mkString(".")
  val java_src: String = JAVA_TMP_DIR + "/" + target.replace('.', '/') + ".java"
  val scala_src: String = SCALA_SRC_DIR + "/" + target.replace('.', '/') + ".scala"

  def loadClass(className: String): Class[_] = {
    Class.forName(className, false, loader)
  }

  def cppToJava(): Unit = {
    val gen = new Builder()
    gen.property("platform.includepath", "/home/nazar/pytorch/torch/include:/home/nazar/torch_scala/src/native")
    gen.outputDirectory(JAVA_TMP_DIR)
    gen.classesOrPackages(mappingClass)
    gen.build()
  }

  private val scalaHead: String =
    s"""
       |package $target_pkg
       |
       |import java.nio.LongBuffer
       |import org.bytedeco.javacpp._
       |import org.bytedeco.javacpp.annotation._
       |import annotations._
       |import torch_scala.api.aten._
       |import torch_scala.api.std._
       |import torch_scala.{NativeLoader, Torch}
       |import scala.reflect.ClassTag
       |
       |@Platform(include = Array( ${properties.get("platform.include").toArray().map(h => "\"" + h.toString + "\"").mkString(", ") } ))
       |@NoOffset object $target_name {
     """.stripMargin

  def javaToScala(): Unit = {


    val in = new FileInputStream(java_src)
    val cu = new JavaParser().parse(in).getResult.get()
    in.close()

    val classA = cu.getClassByName(target_name)

    val fw = new FileWriter(scala_src)

    fw.write(scalaHead + "\n")

    classA.get().getMethods.forEach(md => {
      val smd = ScalaMethodDeclaration(md)
      val types = smd.parameterDeclaration.map(_.typeName)
      if (!METHODS_TO_IGNORE.contains(smd.name) && !types.exists(TYPES_TO_IGNORE.contains)) {
        fw.write("\t" + smd + "\n")
      }
    })

    fw.write("\n}\n")

    fw.close()

  }

}

object ScalaFromCppGenerator extends App {
  def generate(mapping: String): Unit = {
    val gen = new ScalaFromCppGenerator(mapping)
    gen.cppToJava()
    gen.javaToScala()
  }

  generate("torch_java.mapping.FunctionsMapper")
}
