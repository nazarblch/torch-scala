package torch_scala.api.aten

import javax.activation.UnsupportedDataTypeException
import org.bytedeco.javacpp.{FloatPointer, Pointer}
import org.bytedeco.javacpp.annotation._
import torch_scala.NativeLoader

@Platform(include = Array("torch/all.h"))
@Namespace("at") @NoOffset @Name(Array("Scalar")) class Scalar[T](value: T) extends Pointer with NativeLoader {
  @native def allocate(v: Int): Unit
  @native def allocate(v: Float): Unit
  @native def allocate(@Cast(Array("long")) v: Long): Unit
  @native def allocate(v: Double): Unit

  value match {
    case v: Int => allocate(v)
    case v: Float => allocate(v)
    case v: Long => allocate(v)
    case v: Double => allocate(v)
    case _ => throw new UnsupportedDataTypeException()
  }

  @native @Name(Array("to<int>")) def toInt(): Int
  @native @Name(Array("to<float>")) def toFloat(): Float
  @native @Cast(Array("long")) @Name(Array("to<long>")) def toLong(): Long
  @native @Name(Array("to<double>")) def toDouble(): Double
}




class Complex(a: Float, b: Float) extends FloatPointer(a, b) {
  def Re: Float = super.get(0)
  def Im: Float = super.get(1)
}

object Complex {
  def apply(a: Float, b: Float): Complex = new Complex(a, b)
  def apply(p: FloatPointer): Complex = new Complex(p.get(0), p.get(1))
}
