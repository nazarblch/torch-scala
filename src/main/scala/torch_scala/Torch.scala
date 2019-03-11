package torch_scala

import org.bytedeco.javacpp.annotation.{Cast, Namespace, Platform}

trait NativeLoader {
  val workingDir = System.getProperty("user.dir")
  System.load(workingDir + "/src/native/libjava_torch_lib0.so")

}


@Platform(include = Array("torch/all.h"))
@Namespace("torch::cuda") object Torch {

  @native @Cast(Array("size_t")) def device_count(): Int

  /// Returns true if at least one CUDA device is available.
  @native def is_available(): Boolean

  /// Returns true if CUDA is available, and CuDNN is available.
  @native def cudnn_is_available(): Boolean

}
