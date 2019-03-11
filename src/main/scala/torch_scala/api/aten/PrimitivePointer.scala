package torch_scala.api.aten

import org.bytedeco.javacpp._

import scala.reflect.ClassTag


object PrimitivePointer {
  implicit class PrimitivePointer[PT <: Pointer](data: PT) {
    def asArray[T: ClassTag](num_elements: Int): Array[T] = data match {
      case dd: IntPointer => Array.range(0, num_elements).map(i => dd.get(i).asInstanceOf[T]).toArray
      case dd: LongPointer => Array.range(0, num_elements).map(i => dd.get(i).asInstanceOf[T]).toArray
      case dd: FloatPointer => Array.range(0, num_elements).map(i => dd.get(i).asInstanceOf[T]).toArray
      case dd: DoublePointer => Array.range(0, num_elements).map(i => dd.get(i).asInstanceOf[T]).toArray
      case dd: ShortPointer => Array.range(0, num_elements).map(i => dd.get(i).asInstanceOf[T]).toArray
      case dd: BytePointer => Array.range(0, num_elements).map(i => dd.get(i).asInstanceOf[T]).toArray
    }
  }
}