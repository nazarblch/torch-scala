package torch_scala.api.aten

trait TensorType {
   def dtype: Short
}

class CUDA(val dtype: Short) extends TensorType
class CUDA1(val dtype: Short) extends TensorType
class CUDA2(val dtype: Short) extends TensorType
class CUDA3(val dtype: Short) extends TensorType
class CPU(val dtype: Short) extends TensorType



