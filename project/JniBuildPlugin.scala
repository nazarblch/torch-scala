
import java.io.File

import sbt._
import sbt.Keys._

import sys.process._


object JniBuildPlugin extends AutoPlugin {

  override val trigger: PluginTrigger = noTrigger

  override val requires: Plugins = plugins.JvmPlugin

  object autoImport extends JniGeneratorKeys {

    lazy val targetBuildDir = settingKey[File]("target directory to store so files.")
    lazy val jniBuild = taskKey[Unit]("Builds so lib")
  }

  import autoImport._

  override lazy val projectSettings: Seq[Setting[_]] =Seq(
    
    targetBuildDir in jniBuild :=  (Compile / resourceDirectory).value / "torch_lib",

    targetGeneratorDir in jniBuild := sourceDirectory.value / "native" ,

    targetLibName in jniBuild := "java_torch_lib",

    jniBuild := {
      val src_directory = (targetGeneratorDir in jniBuild).value
      val target_directory = (targetBuildDir in jniBuild).value
      val cmake_prefix = (torchLibPath in jniBuild).value
      val log = streams.value.log

      Process("rm -rf " + target_directory.getAbsolutePath + "/*") ! log

      log.info("Build to " + target_directory.getAbsolutePath)
      val command = s"cmake -H$src_directory -B$target_directory -DCMAKE_PREFIX_PATH=$cmake_prefix"
      log.info(command)
      val exitCode = Process(command) ! log
      if (exitCode != 0) sys.error(s"An error occurred while running cmake. Exit code: $exitCode.")
      val command1 = s"make -C$target_directory"
      log.info(command1)
      val exitCode1 = Process(command1) ! log
      if (exitCode1 != 0) sys.error(s"An error occurred while running make. Exit code: $exitCode1.")
    },

    jniBuild := jniBuild.dependsOn(jniGen).value,
    compile := (compile in Compile).dependsOn(jniBuild).value,

  )


}
