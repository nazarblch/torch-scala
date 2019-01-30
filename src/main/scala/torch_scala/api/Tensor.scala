package torch_scala.api

import org.bytedeco.javacpp.Pointer
import torch_scala.NativeLoader
import org.bytedeco.javacpp._
import org.bytedeco.javacpp.annotation._
import torch_scala.api.Functions.Deallocator_Pointer

import scala.reflect.ClassTag




@Platform(include = Array("ATen/ATen.h"))
@Namespace("at") @NoOffset class Tensor[T, TT <: TensorType[T] : ClassTag](nt: Tensor[T, TT], val options: TensorOptions[TT]) extends Pointer with NativeLoader {
  allocate(nt)

  val tensorType: TT = is_cuda() match {
    case true => new CudaTensorType[T](if(device().has_index()) device().index() else 0, scalar_type()).asInstanceOf[TT]
    case false => new CPUTensorType[T](scalar_type()).asInstanceOf[TT]
  }

  @native def allocate(@ByRef other: Tensor[T, TT]): Unit

  @native @Cast(Array("long")) def dim: Long

  @native @Name(Array("operator+=")) @ByRef def add(@ByRef other: Tensor[T, TT]): Tensor[T, TT]

  @native @Cast(Array("const char *")) override def toString: String

  @native @Cast(Array("long")) def storage_offset: Long

  @native def defined: Boolean

  @native def reset(): Unit

  @native def is_same(@ByRef tensor: Tensor[T, TT]): Boolean

  @native @Cast(Array("size_t")) def use_count: Long

  @native @Cast(Array("size_t")) def weak_use_count: Long

  @native def print(): Unit

  @native @ByVal def sizes: IntList
  def num_elements: Long = {
    val ss = sizes.data()
    (0 until dim.toInt).map(ss.get(_)).product
  }

  @native @ByVal def strides: IntList

  @native @Cast(Array("long")) def ndimension: Long

  @native def is_contiguous: Boolean

  @native @ByRef def `type`: Functions.Type

  @native @Cast(Array("int8_t")) def scalar_type(): Short


  @native @ByVal def cpu(): Tensor[T, CPUTensorType[T]]
  @native @ByVal def cuda(): Tensor[T, CudaTensorType[T]]

  @native @Name(Array("data<int>")) private def data_int(): IntPointer
  @native @Name(Array("data<float>")) private def data_float(): FloatPointer
  def data(): Array[T] = scalar_type() match {
    case 3 =>
      val dd = data_int()
      Array.range(0, num_elements.toInt).map(i => dd.get(i.toLong)).asInstanceOf[Array[T]]
    case 6 =>
      val dd = data_float()
      Array.range(0, num_elements.toInt).map(i => dd.get(i.toLong)).asInstanceOf[Array[T]]
  }

  @native @Name(Array("item<float>")) private def item_float(): Float
  @native @Name(Array("item<int>")) private def item_int(): Int

  @native @ByVal @Name(Array("operator[]")) def apply(@ByVal index: Tensor[Long, TensorType[Long]]): Tensor[T, TT]
  @native @ByVal @Name(Array("operator[]")) def apply(@Cast(Array("long")) index: Long): Tensor[T, TT]

  @native def is_cuda(): Boolean

  @native @ByVal def device(): Device


}


object Tensor {

  def apply[T](data: Array[T], shape: Array[Long])(implicit options: TensorOptions[CPUTensorType[T]]): Tensor[T, CPUTensorType[T]] = {
    val pt = data.head match {
      case h: Float => new FloatPointer(data.asInstanceOf[Array[Float]]:_*)
      case h: Int => new IntPointer(data.asInstanceOf[Array[Int]]:_*)
    }

    val nt = Functions.from_blob[T](pt, new IntList(shape), new Deallocator_Pointer(null))

    new Tensor[T, CPUTensorType[T]](nt, options)

  }

}
