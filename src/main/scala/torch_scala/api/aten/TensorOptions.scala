package torch_scala.api.aten

import org.bytedeco.javacpp.{FloatPointer, Pointer}
import org.bytedeco.javacpp.annotation._

import scala.reflect.runtime.universe.TypeTag

@Platform(include = Array("torch/all.h"))
@Namespace("at") class TensorOptions[T: TypeTag]() extends Pointer(null.asInstanceOf[Pointer]) {
  allocate()
  @native def allocate(): Unit
  @native @ByVal def device(@ByRef d: Device): TensorOptions[T]
  @native @ByVal def device_index(@Cast(Array("int16_t")) device_index: Short): TensorOptions[T]
}


@Platform(include = Array("/home/nazar/CLionProjects/torch_app/helper.h"))
@Namespace("at")
@NoOffset object TensorOptions {

  @native @ByVal def create_options[T, TT <: TensorType[T]](dtype: Int): TensorOptions[TT]

  implicit val intTensorOptions: TensorOptions[CPUTensorType[Int]] = create_options[Int, CPUTensorType[Int]](0)
  implicit val floatTensorOptions: TensorOptions[CPUTensorType[Float]] = create_options[Float, CPUTensorType[Float]](1)
  implicit val longTensorOptions: TensorOptions[CPUTensorType[Long]] = create_options[Long, CPUTensorType[Long]](2)
  implicit val doubleTensorOptions: TensorOptions[CPUTensorType[Double]] = create_options[Double, CPUTensorType[Double]](3)
  implicit val complexTensorOptions: TensorOptions[CPUTensorType[Complex]] = create_options[Complex, CPUTensorType[Complex]](4)

  implicit val cudaIntTensorOptions: TensorOptions[CudaTensorType[Int]] = create_options[Int, CudaTensorType[Int]](0).device(CudaDevice)
  implicit val cudaFloatTensorOptions: TensorOptions[CudaTensorType[Float]] = create_options[Float, CudaTensorType[Float]](1).device(CudaDevice)
  implicit val cudaLongTensorOptions: TensorOptions[CudaTensorType[Long]] = create_options[Long, CudaTensorType[Long]](2).device(CudaDevice)
  implicit val cudaDoubleTensorOptions: TensorOptions[CudaTensorType[Double]] = create_options[Double, CudaTensorType[Double]](3).device(CudaDevice)
}