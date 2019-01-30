package torch_scala.api

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, ObjectInputStream, ObjectOutputStream}
import java.nio.LongBuffer

import org.bytedeco.javacpp._
import org.bytedeco.javacpp.annotation._
import torch_scala.NativeLoader

import scala.reflect.ClassTag

import scala.reflect.runtime.universe.{typeOf, TypeTag}

@Platform(include = Array("c10/util/ArrayRef.h"))
@Namespace("c10") @Name(Array("ArrayRef<int64_t>")) class IntList(list_data: Array[Long]) extends Pointer(null.asInstanceOf[Pointer]) with NativeLoader {
  @native def allocate(@Cast(Array("long*")) d: LongPointer, @Cast(Array("size_t")) length: Int): Unit
  allocate(new LongPointer(list_data:_*), list_data.length)

  @native @Cast(Array("long*")) def data(): LongPointer
}


@Platform(include = Array("c10/util/ArrayRef.h"))
@Namespace("c10") @Name(Array("ArrayRef<float>")) class FloatList(list_data: Array[Float]) extends Pointer(null.asInstanceOf[Pointer]) with NativeLoader {
  @native def allocate(@Cast(Array("float*")) d: FloatPointer, @Cast(Array("size_t")) length: Int): Unit
  allocate(new FloatPointer(list_data:_*), list_data.length)

  @native @Cast(Array("float*")) def data(): FloatPointer
}


@Platform(include = Array("torch/all.h"))
@Namespace("c10") class ScalarType() extends Pointer(null.asInstanceOf[Pointer]) {
  allocate()
  @native def allocate(): Unit
}

@Platform(include = Array("torch/all.h"))
@Namespace("at") class TensorOptions[T: TypeTag]() extends Pointer(null.asInstanceOf[Pointer]) {
  allocate()
  @native def allocate(): Unit
  @native @ByVal def device(@ByRef d: Device): TensorOptions[T]
  @native @ByVal def device_index(@Cast(Array("int16_t")) device_index: Short): TensorOptions[T]
}


object TensorOptions {

  implicit val intTensorOptions: TensorOptions[CPUTensorType[Int]] = Functions.create_options[Int, CPUTensorType[Int]](0)
  implicit val floatTensorOptions: TensorOptions[CPUTensorType[Float]] = Functions.create_options[Float, CPUTensorType[Float]](1)
  implicit val cudaFloatTensorOptions: TensorOptions[CudaTensorType[Float]] = Functions.create_options[Float, CudaTensorType[Float]](1).device(CudaDevice)
}


@Platform(include = Array("torch/all.h"))
@Namespace("at") object kInt extends TensorOptions() {
}

@Platform(include = Array("/home/nazar/CLionProjects/torch_app/helper.h",
                          "torch/all.h"))
@Namespace("at")
@NoOffset object Functions extends NativeLoader {


  @Opaque class Type() extends Pointer(null.asInstanceOf[Pointer]) {
    allocate()
    @native def allocate(): Unit
  }

  @native @ByVal def create_options[T, TT <: TensorType[T]](dtype: Int): TensorOptions[TT]

  @native @ByVal def make_ones[T](dtype: Int, @StdVector data: LongPointer): Tensor[T, CPUTensorType[T]]

  @native @ByVal def ones[T, TT <: TensorType[T]](@ByVal size: IntList, @Const @ByRef options: TensorOptions[TT]): Tensor[T, TT]
  @native @ByVal def ones[T](@ByVal size: IntList): Tensor[T, CPUTensorType[T]]

  @native def int_list(@Cast(Array("size_t")) size: Int, data: Array[Int]): IntList

  @native @ByVal def arange[T, TT <: TensorType[T]](@ByVal start: Scalar[Long], @ByVal end: Scalar[Long], @ByRef options: TensorOptions[TT]): Tensor[T, TT]
  @native @ByVal def arange[T, TT <: TensorType[T]](@ByVal start: Scalar[Long], @ByVal end: Scalar[Long], @ByVal step: Scalar[Long], @Const @ByRef options: TensorOptions[TT]): Tensor[T, TT]
  @native @ByVal def arange(@ByVal start: Scalar[Long], @ByVal end: Scalar[Long]): Tensor[Float, CPUTensorType[Float]]
  @native @ByVal def arange(@ByVal start: Scalar[Long], @ByVal end: Scalar[Long], @ByVal step: Scalar[Long]): Tensor[Float, CPUTensorType[Float]]

  @native @ByVal def tensor[T, TT <: TensorType[T]](@ByVal values: IntList)(implicit @Const @ByRef options: TensorOptions[TT]): Tensor[T, TT]
  @native @ByVal def tensor[T, TT <: TensorType[T]](@ByVal values: FloatList)(implicit @Const @ByRef options: TensorOptions[TT]): Tensor[T, TT]


  class Deallocator_Pointer(p: Pointer) extends FunctionPointer(p) {
    @Name(Array("deleter")) def call(data: Pointer): Unit = {
      //p.deallocate()
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
