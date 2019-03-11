package torch_scala.api.aten

import org.bytedeco.javacpp.Pointer
import org.bytedeco.javacpp.annotation._
import torch_scala.NativeLoader

@Platform(include = Array("c10/Device.h", "stdint.h"))
@Namespace("c10") @NoOffset class Device(name: String) extends Pointer with NativeLoader {
  allocate(name)
  @native def allocate(@ByRef string: String): Unit

  @native @Name(Array("operator==")) def ==(@Const @ByRef other: Device): Boolean

  @native @Name(Array("operator!=")) def !=(@Const @ByRef other: Device): Boolean

  @native @Cast(Array("int16_t")) def `type`(): Short

  @native @Cast(Array("int16_t")) def index(): Short

  @native def has_index(): Boolean

  @native def is_cuda(): Boolean

  @native def is_cpu(): Boolean

}


case class CudaDevice(cuda_index: Int) extends Device("cuda:" + cuda_index)
object CudaDevice extends Device("cuda")
object CPUDevice extends Device("cpu")

