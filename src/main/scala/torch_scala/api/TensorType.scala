package torch_scala.api

trait TensorType[T] {

}

class CudaTensorType[T](index: Int) extends TensorType[T]
class CPUTensorType[T]() extends TensorType[T]
