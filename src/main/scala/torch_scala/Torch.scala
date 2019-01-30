package torch_scala

import org.bytedeco.javacpp.annotation.Platform

trait NativeLoader {
  val workingDir = System.getProperty("user.dir")
  System.load(workingDir + "/src/native/libjava_torch_lib0.so")

}

object Torch {


}
