import sbt._
import sbt.Keys._


version := "1.0"

scalaVersion := "2.12.7"


// https://mvnrepository.com/artifact/org.bytedeco/javacpp
libraryDependencies += "org.bytedeco" % "javacpp" % "1.4.3"
libraryDependencies += "org.scala-lang" % "scala-reflect" % "2.12.7"

enablePlugins(JniGeneratorPlugin, JniBuildPlugin)
//sourceDirectory in nativeCompile := sourceDirectory.value / "native"
//target in nativeCompile :=target.value / "native" / nativePlatform.value








