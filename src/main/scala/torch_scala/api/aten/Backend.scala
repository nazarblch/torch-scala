package torch_scala.api.aten

import org.bytedeco.javacpp.Pointer
import org.bytedeco.javacpp.annotation.{Namespace, NoOffset, Platform}

@Platform(include = Array("torch/all.h"))
@Namespace("at") @NoOffset class Backend extends Pointer {
}
