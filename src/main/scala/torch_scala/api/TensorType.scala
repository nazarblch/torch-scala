package torch_scala.api

import scala.reflect.ClassTag
import scala.reflect._
import scala.reflect.runtime.universe._

trait TensorType[T] {
   def dtype: Short
}

class CudaTensorType[T](val index: Short, val dtype: Short) extends TensorType[T]
class CPUTensorType[T](val dtype: Short) extends TensorType[T]


