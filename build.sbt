import sbt._
import sbt.Keys._


version := "1.0"

scalaVersion := "2.12.7"


// https://mvnrepository.com/artifact/org.bytedeco/javacpp
libraryDependencies += "org.bytedeco" % "javacpp" % "1.4.3"
libraryDependencies += "org.scala-lang" % "scala-reflect" % "2.12.7"

enablePlugins(JniGeneratorPlugin, JniBuildPlugin)
JniBuildPlugin.autoImport.torchLibPath in jniBuild := "/home/nazar/pytorch"
//sourceDirectory in nativeCompile := sourceDirectory.value / "native"
//target in nativeCompile :=target.value / "native" / nativePlatform.value


libraryDependencies += "com.typesafe.scala-logging" %% "scala-logging" % "3.7.2"
libraryDependencies += "ch.qos.logback" % "logback-classic" % "1.2.3"

lazy val scalaTest = "org.scalatest" %% "scalatest" % "3.0.3"

libraryDependencies += scalaTest % Test

libraryDependencies  ++= Seq(
  // Last stable release
  "org.scalanlp" %% "breeze" % "0.13.2",

  // Native libraries are not included by default. add this if you want them (as of 0.7)
  // Native libraries greatly improve performance, but increase jar sizes. 
  // It also packages various blas implementations, which have licenses that may or may not
  // be compatible with the Apache License. No GPL code, as best I know.
  "org.scalanlp" %% "breeze-natives" % "0.13.2",

  // The visualization library is distributed separately as well.
  // It depends on LGPL code
  "org.scalanlp" %% "breeze-viz" % "0.13.2"
)


resolvers += "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"


