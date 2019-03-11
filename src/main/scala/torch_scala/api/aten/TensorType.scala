package torch_scala.api.aten

trait TensorType[T] {
   def dtype: Short
}

class CudaTensorType[T](val index: Short, val dtype: Short) extends TensorType[T]
class CPUTensorType[T](val dtype: Short) extends TensorType[T]


