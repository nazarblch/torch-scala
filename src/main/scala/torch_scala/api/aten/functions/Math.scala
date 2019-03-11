package torch_scala.api.aten.functions

import org.bytedeco.javacpp.annotation._
import torch_scala.NativeLoader
import torch_scala.api.aten.{Tensor, TensorType}
import torch_scala.api.types.{FloatOrDouble, IsFloatOrDouble}


@Platform(include = Array("torch/all.h"))
@Namespace("at") @NoOffset object Math extends NativeLoader {

  @native @ByVal def cos[T, TT <: TensorType[T]](@ByRef self: Tensor[T, TT]): Tensor[T, TT]
  @native @ByVal def sin[T, TT <: TensorType[T]](@ByRef self: Tensor[T, TT]): Tensor[T, TT]
  @native @ByVal def exp[T, TT <: TensorType[T]](@ByRef self: Tensor[T, TT]): Tensor[T, TT]

  @native @ByVal def matmul[T, TT <: TensorType[T]](@ByRef self: Tensor[T, TT], @ByRef other: Tensor[T, TT]): Tensor[T, TT]

  @native @ByVal def dot[T, TT <: TensorType[T]](@ByRef self: Tensor[T, TT], @ByRef other: Tensor[T, TT]): Tensor[T, TT]

  @native @ByVal def add[T, TT <: TensorType[T]](@ByRef self: Tensor[T, TT], @ByRef other: Tensor[T, TT]): Tensor[T, TT]

  implicit class MathTensor[T, TT <: TensorType[T]](self: Tensor[T, TT]) {
    def cos(): Tensor[T, TT] = Math.cos(self)
    def sin(): Tensor[T, TT] = Math.sin(self)
    def exp(): Tensor[T, TT] = Math.exp(self)

    def matmul(other: Tensor[T, TT]) = new Tensor[T, TT]( Math.matmul(self, other) )
    def dot(other: Tensor[T, TT]) = new Tensor( Math.dot(self, other) )
    def add(other: Tensor[T, TT]) = new Tensor( Math.add(self, other) )
  }

}
