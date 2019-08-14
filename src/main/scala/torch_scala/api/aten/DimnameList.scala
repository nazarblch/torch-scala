package torch_scala.api.aten

import org.bytedeco.javacpp.{Pointer, PointerPointer}
import org.bytedeco.javacpp.annotation._

@Platform(include = Array("ATen/Dimname.h", "torch/all.h"))
@NoOffset @Name(Array("at::ArrayRef<at::Dimname>")) class DimnameList extends Pointer(null.asInstanceOf[Pointer])  {
  @native def allocate(@Cast(Array("at::Dimname*")) d: PointerPointer[Dimname], @Cast(Array("size_t")) length: Int): Unit
}


@Platform(include = Array("ATen/Dimname.h"))
@Namespace("at") @Opaque class Dimname extends Pointer(null.asInstanceOf[Pointer])  {
}
