package torch_scala.api.aten

import org.bytedeco.javacpp._
import org.bytedeco.javacpp.annotation.{Cast, Name, Namespace, Platform}
import torch_scala.NativeLoader

abstract class ArrayRef[T, P](final val dtypeName: String) extends Pointer(null.asInstanceOf[Pointer]) {
  def toArray: Array[T]
}


@Platform(include = Array("c10/util/ArrayRef.h"))
@Namespace("c10") @Name(Array("ArrayRef<float>")) class ArrayRefFloat(list_data: Array[Float]) extends  ArrayRef[Float, FloatPointer]("float") with NativeLoader {

  val size: Int = list_data.length

  @native def allocate(@Cast(Array("float*")) d: FloatPointer, @Cast(Array("size_t")) length: Int): Unit
  allocate(new FloatPointer(list_data:_*), list_data.length)

  @native @Cast(Array("float*")) def data(): FloatPointer

  override def toArray: Array[Float] = {
    val d = data()
    Array.range(0, size).map(d.get(_))
  }
}


@Platform(include = Array("c10/util/ArrayRef.h"))
@Namespace("c10") @Name(Array("ArrayRef<int>")) class ArrayRefInt(list_data: Array[Int]) extends  ArrayRef[Int, IntPointer]("int") with NativeLoader {

  val size: Int = list_data.length

  @native def allocate(@Cast(Array("int*")) d: IntPointer, @Cast(Array("size_t")) length: Int): Unit
  allocate(new IntPointer(list_data:_*), list_data.length)

  @native @Cast(Array("int*")) def data(): IntPointer

  override def toArray: Array[Int] = {
    val d = data()
    Array.range(0, size).map(d.get(_))
  }
}


@Platform(include = Array("c10/util/ArrayRef.h"))
@Namespace("c10") @Name(Array("ArrayRef<long>")) class ArrayRefLong(list_data: Array[Long]) extends  ArrayRef[Long, LongPointer]("long") with NativeLoader {

  val size: Int = list_data.length

  @native def allocate(@Cast(Array("long*")) d: LongPointer, @Cast(Array("size_t")) length: Int): Unit
  allocate(new LongPointer(list_data:_*), list_data.length)

  @native @Cast(Array("long*")) def data(): LongPointer

  override def toArray: Array[Long] = {
    val d = data()
    Array.range(0, size).map(d.get(_))
  }
}


@Platform(include = Array("c10/util/ArrayRef.h"))
@Namespace("c10") @Name(Array("ArrayRef<double>")) class ArrayRefDouble(list_data: Array[Double]) extends  ArrayRef[Double, DoublePointer]("double") with NativeLoader {

  val size: Int = list_data.length

  @native def allocate(@Cast(Array("double*")) d: DoublePointer, @Cast(Array("size_t")) length: Int): Unit
  allocate(new DoublePointer(list_data:_*), list_data.length)

  @native @Cast(Array("double*")) def data(): DoublePointer

  override def toArray: Array[Double] = {
    val d = data()
    Array.range(0, size).map(d.get(_))
  }
}

@Platform(include = Array("torch/all.h", "<complex>"))
@Namespace("at") @Name(Array("ArrayRef<std::complex<float> >")) class ArrayRefComplex(list_data: Array[Complex]) extends  ArrayRef[Complex, PointerPointer[Complex]]("complex") with NativeLoader {

  val size: Int = list_data.length

  @native def allocate(@Cast(Array("std::complex<float>*")) d: PointerPointer[Complex], @Cast(Array("size_t")) length: Int): Unit

  allocate(new PointerPointer(list_data:_*), list_data.length)

  @native @Cast(Array("std::complex<float>*")) def data(): PointerPointer[FloatPointer]

  override def toArray: Array[Complex] = {
    val d = data()
    Array.range(0, size).map(d.get(_)).map(new FloatPointer(_)).map(Complex.apply)
  }
}


object ArrayRef {
  def apply[T](data: Array[T])= data.head match {
    case x: Int => new ArrayRefInt(data.asInstanceOf[Array[Int]])
    case x: Float => new ArrayRefFloat(data.asInstanceOf[Array[Float]])
    case x: Long => new ArrayRefLong(data.asInstanceOf[Array[Long]])
    case x: Double => new ArrayRefDouble(data.asInstanceOf[Array[Double]])
  }
}

@Platform(include = Array("c10/util/ArrayRef.h"))
@Namespace("c10") @Name(Array("ArrayRef<int64_t>")) class IntList(list_data: Array[Long]) extends Pointer(null.asInstanceOf[Pointer]) with NativeLoader {
  @native def allocate(@Cast(Array("long*")) d: LongPointer, @Cast(Array("size_t")) length: Int): Unit
  allocate(new LongPointer(list_data:_*), list_data.length)

  @native @Cast(Array("long*")) def data(): LongPointer
}

object IntList {
  def apply(list_data: Array[Long]): IntList = new IntList(list_data)
  def apply(list_data: Array[Int]): IntList = new IntList(list_data.map(_.toLong))
}

@Platform(include = Array("torch/all.h"))
@Namespace("at") @Name(Array("ArrayRef<at::Tensor>")) class TensorList[T, TT <: TensorType[T]](list_data: Array[Tensor[T, TT]]) extends PointerPointer[Tensor[T, TT]] with NativeLoader {
  @native def allocate(@Cast(Array("at::Tensor*")) d: PointerPointer[Tensor[T, TT]], @Cast(Array("size_t")) length: Int): Unit
  allocate(new PointerPointer[Tensor[T, TT]](list_data:_*), list_data.length)

  @native @Cast(Array("at::Tensor*")) def data(): PointerPointer[Tensor[T, TT]]
}