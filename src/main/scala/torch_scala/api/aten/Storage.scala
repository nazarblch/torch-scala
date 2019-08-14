package torch_scala.api.aten

import org.bytedeco.javacpp.Pointer
import org.bytedeco.javacpp.annotation.{Name, Namespace, NoOffset, Platform}

@Platform(include = Array("torch/all.h"))
@Namespace("at") @NoOffset class Storage extends Pointer{
  allocate()
  @native private def allocate(): Unit
}
