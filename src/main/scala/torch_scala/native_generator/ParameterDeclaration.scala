package torch_scala.native_generator

case class ParameterDeclaration(
                                 decorator: Array[String],
                                 name: String,
                                 typeName: String
                               ) {
  override def toString: String = {

    val badNames = Set("val", "var", "def", "type")
    val sname = if (badNames.contains(name)) "`" + name + "`" else name

    decorator.mkString(" ") +
      (if (decorator.nonEmpty) " " else "") +
      sname + ": " + typeName
  }
}