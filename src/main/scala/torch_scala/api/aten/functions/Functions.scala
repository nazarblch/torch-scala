package torch_scala.api.aten.functions

import org.bytedeco.javacpp.annotation._
import org.bytedeco.javacpp.{FunctionPointer, LongPointer, Pointer}
import torch_scala.NativeLoader
import torch_scala.api.aten._

@Platform(include = Array("/home/nazar/CLionProjects/torch_app/helper.h",
                          "torch/all.h",
                          "<complex>"))
@Namespace("at")
@NoOffset object Functions extends NativeLoader {


  @Opaque class Type() extends Pointer(null.asInstanceOf[Pointer]) {
    allocate()
    @native def allocate(): Unit
  }

  @native @ByVal def make_ones[T](dtype: Int, @StdVector data: LongPointer): Tensor[T, CPUTensorType[T]]

  @native @ByVal def ones[T, TT <: TensorType[T]](@ByVal size: IntList, @Const @ByRef options: TensorOptions[TT]): Tensor[T, TT]
  @native @ByVal def ones[T](@ByVal size: IntList): Tensor[T, CPUTensorType[T]]

  @native def int_list(@Cast(Array("size_t")) size: Int, data: Array[Int]): IntList

  @native @ByVal def arange[T, TT <: TensorType[T]](@ByVal start: Scalar[Long], @ByVal end: Scalar[Long], @ByRef options: TensorOptions[TT]): Tensor[T, TT]
  @native @ByVal def arange[T, TT <: TensorType[T]](@ByVal start: Scalar[Long], @ByVal end: Scalar[Long], @ByVal step: Scalar[Long], @Const @ByRef options: TensorOptions[TT]): Tensor[T, TT]
  @native @ByVal def arange(@ByVal start: Scalar[Long], @ByVal end: Scalar[Long]): Tensor[Float, CPUTensorType[Float]]
  @native @ByVal def arange(@ByVal start: Scalar[Long], @ByVal end: Scalar[Long], @ByVal step: Scalar[Long]): Tensor[Float, CPUTensorType[Float]]

  @native @ByVal def tensor[TT <: TensorType[Int]](@ByVal values: ArrayRefInt)(implicit @Const @ByRef options: TensorOptions[TT]): Tensor[Int, TT]
  @native @ByVal def tensor[TT <: TensorType[Float]](@ByVal values: ArrayRefFloat)(implicit @Const @ByRef options: TensorOptions[TT]): Tensor[Float, TT]
  @native @ByVal def tensor[TT <: TensorType[Long]](@ByVal values: ArrayRefLong)(implicit @Const @ByRef options: TensorOptions[TT]): Tensor[Long, TT]
  @native @ByVal def tensor[TT <: TensorType[Double]](@ByVal values: ArrayRefDouble)(implicit @Const @ByRef options: TensorOptions[TT]): Tensor[Double, TT]

  @native @ByVal def zeros[T, TT <: TensorType[T]](@ByVal size: IntList)(implicit @Const @ByRef options: TensorOptions[TT]): Tensor[T, TT]
  @native @ByVal def zeros_like[T, TT <: TensorType[T]](@Const @ByRef self: Tensor[T, TT]): Tensor[T, TT]
  @native @ByVal def zeros_like[T, TT <: TensorType[T]](@Const @ByRef self: Tensor[T, TT], @Const @ByRef options: TensorOptions[TT]): Tensor[T, TT]

  class Deallocator_Pointer(p: Pointer) extends FunctionPointer(p) {
    @Name(Array("deleter")) def call(data: Pointer): Unit = {
      Pointer.free(data)
      println("delete tensor")
    }
  }

  @native @ByVal def from_blob[T](
    data: Pointer,
    @ByVal sizes: IntList,
    @Cast(Array("const std::function<void(void*)>")) deleter: Deallocator_Pointer)(implicit @Const @ByRef options: TensorOptions[CPUTensorType[T]]): Tensor[T, CPUTensorType[T]]

  @native @ByVal def from_blob[T](
                                   data: Pointer,
                                   @ByVal sizes: IntList,
                                   @ByVal strides: IntList,
                                   deleter: Deallocator_Pointer)(implicit @Const @ByRef options: TensorOptions[CPUTensorType[T]]): Tensor[T, CPUTensorType[T]]

}
