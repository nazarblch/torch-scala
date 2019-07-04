package torch_scala.api.aten

import org.bytedeco.javacpp.Pointer
import org.bytedeco.javacpp.annotation._
import torch_scala.NativeLoader

import scala.reflect.ClassTag



@Platform(include = Array("ATen/ATen.h", "<vector>"))
@NoOffset @Name(Array("std::vector<at::Tensor>")) class TensorVector[T: ClassTag, TT <: TensorType] extends Pointer {

  allocate()

  @native private def allocate(): Unit

  def empty: Boolean = size() == 0

  @native @Cast(Array("long")) def size(): Long

  @native def clear(): Unit = {
    resize(0)
  }

  @native def resize(@Cast(Array("long")) n: Long): Unit

  @native @ByRef private def at(@Cast(Array("long")) i: Long): Tensor[T, TT]

  @native def push_back(@ByRef value: Tensor[T, TT]): Unit

  def data(): Array[Tensor[T, TT]] = {
    Array.range(0, size().toInt).map(i => apply(i))
  }

  def apply(i: Int): Tensor[T, TT] = new Tensor[T, TT](at(i))



}



@Platform(include = Array("ATen/ATen.h", "<tuple>"))
@NoOffset @Name(Array("std::tuple<at::Tensor, at::Tensor>")) class TensorTuple[T1: ClassTag, T2: ClassTag, TT <: TensorType] extends Pointer {

  allocate()

  @native private def allocate(): Unit

}

@Platform(include = Array("ATen/ATen.h", "<tuple>"))
@NoOffset @Name(Array("std::tuple<at::Tensor &, at::Tensor & >")) class TensorRefTuple[T1: ClassTag, T2: ClassTag, TT <: TensorType] extends Pointer {

  allocate()

  @native private def allocate(): Unit

}

@Platform(include = Array("ATen/ATen.h", "<tuple>"))
@NoOffset @Name(Array("std::tuple<at::Tensor,at::Tensor,at::Tensor>")) class TensorTriple[T1: ClassTag, T2: ClassTag, T3: ClassTag, TT <: TensorType] extends Pointer {

  allocate()

  @native private def allocate(): Unit

}

@Platform(include = Array("ATen/ATen.h", "<tuple>", "<vector>"))
@NoOffset @Name(Array("std::tuple<at::Tensor,at::Tensor,at::Tensor,std::vector<at::Tensor> >")) class TensorTripleAndVector[T: ClassTag, TT <: TensorType] extends Pointer {

  allocate()

  @native private def allocate(): Unit

}

@Platform(include = Array("ATen/ATen.h", "<tuple>"))
@NoOffset @Name(Array("std::tuple<at::Tensor &,at::Tensor &,at::Tensor &>")) class TensorRefTriple[T1: ClassTag, T2: ClassTag, T3: ClassTag, TT <: TensorType] extends Pointer {

  allocate()

  @native private def allocate(): Unit

}

@Platform(include = Array("ATen/ATen.h", "<tuple>"))
@NoOffset @Name(Array("std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>")) class TensorTuple4[T: ClassTag, TT <: TensorType] extends Pointer {

  allocate()
  @native private def allocate(): Unit

}

@Platform(include = Array("ATen/ATen.h", "<tuple>"))
@NoOffset @Name(Array("std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>")) class TensorTuple5[T: ClassTag, TT <: TensorType] extends Pointer {

  allocate()
  @native private def allocate(): Unit

}