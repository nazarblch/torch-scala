package torch_scala.api.nn

import org.bytedeco.javacpp._
import org.bytedeco.javacpp.annotation._


@Platform(include = Array(
  "torch/all.h",
  "torch/nn/module.h"
))
@Namespace("torch::nn") @NoOffset class Module() extends Pointer {

  allocate()

  @native def allocate(): Unit
}