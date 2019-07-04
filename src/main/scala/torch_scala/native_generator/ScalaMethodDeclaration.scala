package torch_scala.native_generator

import com.github.javaparser.ast.body.{MethodDeclaration, Parameter}
import com.github.javaparser.ast.expr.AnnotationExpr

class ScalaMethodDeclaration(
                              val isNative: Boolean,
                              val typeDecorator: Array[String],
                              val returnType: String,
                              val name: String,
                              val templates: Option[String],
                              val parameterDeclaration: Array[ParameterDeclaration]) {
  override def toString: String = {
    val nativeDec = if (isNative) "@native " else ""
    val typeDec = typeDecorator.mkString(" ") + (if (typeDecorator.nonEmpty) " " else "")
    val params = parameterDeclaration.mkString(", ")
    val templ = if (templates.isDefined) "[" + templates.get + "]" else ""
    val badNames = Set("val", "var", "def", "type")
    val sname = if (badNames.contains(name)) "`" + name + "`" else name
    s"$nativeDec${typeDec}def $sname$templ($params): $returnType"
  }
}

object ScalaMethodDeclaration {
  def apply(md: MethodDeclaration): ScalaMethodDeclaration = {

    val params = md.getParameters.toArray().map(_.asInstanceOf[Parameter]).map(p => {
      ParameterDeclaration(
        p.getAnnotations.toArray.map(_.asInstanceOf[AnnotationExpr]).map(toScalaAnnotation),
        p.getName.toString,
        toScalaType(p.getType.asString(), md.getName.toString, Some(p.getName.toString), "")
      )
    })

    val templates = makeTemplates(params.map(_.typeName).toSet)
    val stip = toScalaType(md.getType.asString(), md.getName.toString, None, templates.getOrElse(""))

    new ScalaMethodDeclaration(
      md.isNative,
      md.getAnnotations.toArray().map(_.asInstanceOf[AnnotationExpr]).map(toScalaAnnotation),
      stip,
      md.getName.asString(),
      templates,
      params
    )
  }

  def toScalaAnnotation(ann: AnnotationExpr): String = {
    if(ann.getName.toString equals "Cast") {
      "@" + ann.getName + s"(Array(${ann.asSingleMemberAnnotationExpr().getMemberValue.toString}))"
    } else {
      ann.toString
    }
  }

  def detectMethodDataType(methodName: String): String = {
    val longIndicators: Set[String] = Set("indices")
    val byteIndicators: Set[String] = Set("\\_and", "\\_or", "\\_le", "\\_lt", "\\_eq", "\\_gt", "\\_ge")
    if (longIndicators.exists(ind => methodName.contains(ind))) {
      "Long"
    } else if (byteIndicators.exists(ind => methodName.contains(ind))) {
      "Byte"
    } else {
      "T"
    }
  }

  def detectValDataType(methodName: String, valName: String): String = {
    val longIndicators: Set[String] = Set("index", "indices")
    val byteIndicators: Set[String] = Set("mask")
    if (longIndicators.exists(ind => valName.contains(ind))) {
      "Long"
    } else if (byteIndicators.exists(ind => valName.contains(ind))) {
      "Byte"
    } else {
      "T"
    }
  }

  def detectDataType(methodName: String, valName: Option[String]): String = {
    if (valName.isDefined) detectValDataType(methodName, valName.get)
    else detectMethodDataType(methodName)
  }

  def toScalaType(name: String, methodName: String, valName: Option[String], templates: String): String = {
    val templates_set = templates.split(',').map(_.trim).map(_.split(":|<:").head.trim).toSet

    var dt = detectDataType(methodName, valName)
    if (valName.isEmpty && (dt equals "T") && (!templates_set.contains("T"))) {
      dt = "Double"
    }

    name.trim match {
      case "Tensor<T,TT>" => s"Tensor[$dt, ${if (valName.isDefined || templates_set.contains("TT")) "TT" else "CPU" }]"
      case "TensorOptions<T,TT>" => s"TensorOptions[${detectDataType(methodName, valName)}, TT]"
      case "TensorList<T,TT>" => s"TensorList[${detectDataType(methodName, valName)}, TT]"
      case "Scalar<T>" => s"Scalar[${detectDataType(methodName, valName)}]"
      case "TensorTuple<T1,T2,TT>" => "TensorTuple[T,T,TT]"
      case "TensorTuple5<T,TT>" => "TensorTuple5[T,TT]"
      case "TensorTuple4<T,TT>" => "TensorTuple4[T,TT]"
      case "TensorRefTuple<T1,T2,TT>" => "TensorRefTuple[T,T,TT]"
      case "TensorTriple<T1,T2,T3,TT>" => "TensorTriple[T,T,T,TT]"
      case "TensorRefTriple<T1,T2,T3,TT>" => "TensorRefTriple[T,T,T,TT]"
      case "TensorTripleAndVector<T,TT>" => "TensorTripleAndVector[T,TT]"
      case "long" => "Long"
      case "int" => "Int"
      case "double" => "Double"
      case "void" => "Unit"
      case "byte" => "Byte"
      case "boolean" => "Boolean"
      case "long[]" => "Array[Long]"
      case "int[]" => "Array[Int]"
      case "byte[]" => "Array[Byte]"
      case "boolean[]" => "Array[Boolean]"
      case _ => name

    }
  }

  def makeTemplates(typeNames: Set[String]): Option[String] = {

    val terminalMap = Map(
      "T" -> "T :ClassTag",
      "T1" -> "T1 :ClassTag",
      "T2" -> "T2 :ClassTag",
      "T3" -> "T3 :ClassTag",
      "TT" -> "TT <: TensorType"
    )

    val terminals: Set[String] = typeNames.flatMap(name => {
      if (name.contains("[") && name.contains("]") && name.split("\\[|\\]").length > 1) {
        name.split("\\[|\\]")(1).split(",").map(_.trim)
      }
      else None
    })
    if (terminals.isEmpty) {
      None
    } else {
      Some(terminals.toArray.map(t => terminalMap.getOrElse(t, "")).filter(_.nonEmpty).mkString(", "))
    }
  }

}