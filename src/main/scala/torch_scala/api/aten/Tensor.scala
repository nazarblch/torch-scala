package torch_scala.api.aten

import org.bytedeco.javacpp.{Pointer, _}
import org.bytedeco.javacpp.annotation._
import torch_scala.{NativeLoader, Torch}
import torch_scala.api.aten
import torch_scala.api.aten.functions.Functions.Deallocator_Pointer

import scala.reflect.ClassTag
import torch_scala.api._
import torch_scala.api.exception.InvalidDeviceException
import PrimitivePointer._
import torch_scala.api.aten.functions.Functions
import torch_scala.api.types.DataType


@Platform(include = Array("ATen/ATen.h"))
@Namespace("at") @NoOffset class Tensor[T, TT <: TensorType[T]](nt: Tensor[T, TT]) extends Pointer with NativeLoader {
  allocate(nt)

  def tensorType: TT = is_cuda() match {
    case true => new CudaTensorType[T](if(device().has_index()) device().index() else 0, scalar_type()).asInstanceOf[TT]
    case false => new CPUTensorType[T](scalar_type()).asInstanceOf[TT]
  }

  /** Data type of this tensor. */
  def dataType: DataType[T] = {
    DataType.fromCValue[T](scalar_type())
  }

  def shape = Shape.apply(sizesIterator.toArray)

  @native private def allocate(@ByRef other: Tensor[T, TT]): Unit

  @native @Cast(Array("long")) def dim: Long

  @native @Cast(Array("const char *")) override def toString: String

  @native @Cast(Array("long")) def storage_offset: Long

  @native def defined: Boolean

  @native def reset(): Unit

  @native def is_same(@ByRef tensor: Tensor[T, TT]): Boolean

  @native @Cast(Array("size_t")) def use_count: Long

  @native @Cast(Array("size_t")) def weak_use_count: Long

  @native def print(): Unit

  @native @ByVal private def sizes: IntList
  private def sizesIterator: Iterator[Int] = {
    val ss = sizes.data()
    (0 until dim.toInt).map(ss.get(_).toInt).toIterator
  }

  def num_elements: Long = {
    val ss = sizes.data()
    (0 until dim.toInt).map(ss.get(_)).product
  }

  @native @ByVal def strides: IntList

  @native @ByVal private def reshape(@ByVal shape: IntList): Tensor[T, TT]
  def reshape(shape: Shape): Tensor[T, TT] = {
    val t = reshape(new IntList(shape.asArray.map(_.toLong)))
    new Tensor(t)
  }

  @native @Cast(Array("long")) def ndimension: Long

  @native def is_contiguous: Boolean

  @native @ByRef def `type`: Functions.Type

  @native @Cast(Array("int8_t")) def scalar_type(): Short


  @native @Name(Array("cpu")) @ByVal private def to_cpu(): Tensor[T, CPUTensorType[T]]
  @native @Name(Array("cuda")) @ByVal private def to_cuda(): Tensor[T, CudaTensorType[T]]
  @native @ByVal private def to(@ByVal d: Device, @Cast(Array("int8_t")) dtype: Short): Tensor[T, CudaTensorType[T]]
  def cpu(): Tensor[T, CPUTensorType[T]] = new Tensor(to_cpu())
  def cuda(): Tensor[T, CudaTensorType[T]] = {
    if(!Torch.is_available()) {
      InvalidDeviceException(s"cuda is not available")
    }
    new Tensor(to_cuda())
  }
  def cuda(index: Short): Tensor[T, CudaTensorType[T]] = {
    if(index >= Torch.device_count()) {
      InvalidDeviceException(s"cuda index $index >= available devise count")
    }
    new Tensor( to(CudaDevice(index), scalar_type()) )
  }

  @native @Name(Array("data<int>")) private def data_int(): IntPointer
  @native @Name(Array("data<float>")) private def data_float(): FloatPointer
  @native @Name(Array("data<long long int>")) private def data_long(): LongPointer
  @native @Name(Array("data<double>")) private def data_double(): DoublePointer
  def data(): Array[T] = scalar_type() match {
    case 3 =>
      val dd = data_int()
      Array.range(0, num_elements.toInt).map(i => dd.get(i.toLong)).asInstanceOf[Array[T]]
    case 6 =>
      val dd = data_float()
      Array.range(0, num_elements.toInt).map(i => dd.get(i.toLong)).asInstanceOf[Array[T]]
    case 4 =>
      val dd = data_long()
      Array.range(0, num_elements.toInt).map(i => dd.get(i.toLong)).asInstanceOf[Array[T]]
    case 7 =>
      val dd = data_double()
      Array.range(0, num_elements.toInt).map(i => dd.get(i.toLong)).asInstanceOf[Array[T]]
  }

  @native @Name(Array("item<float>")) private def item_float(): Float
  @native @Name(Array("item<int>")) private def item_int(): Int
  @native @Name(Array("item<long>")) private def item_long(): Long
  @native @Name(Array("item<double>")) private def item_double(): Double

  def item(): T = scalar_type() match {
    case 3 => item_int().asInstanceOf[T]
    case 4 => item_long().asInstanceOf[T]
    case 6 => item_float().asInstanceOf[T]
    case 7 => item_double().asInstanceOf[T]
  }

  def scalar(): Scalar[T] = new Scalar[T](item())

  @native @ByVal @Name(Array("operator[]")) private def apply(@ByVal index: Tensor[Long, CPUTensorType[Long]]): Tensor[T, TT]
  @native @ByVal @Name(Array("operator[]")) private def apply(@Cast(Array("long")) index: Long): Tensor[T, TT]
  @native @ByVal private def index_select(@Cast(Array("long")) dim: Long,  @ByRef index: Tensor[Long, CPUTensorType[Long]]): Tensor[T, TT]

  def select_in_dim(dim: Long, index: Array[Long]): Tensor[T, TT] = new Tensor[T, TT]( index_select(
    dim,
    Tensor.apply[Long](index, Shape(index.length))
  ))

  def select(index: Array[Long], otherIndexes: Array[Long]*): Tensor[T, TT] = {
    val selectors: Array[Array[Long]] = Array(index) ++ otherIndexes
    var res = this

    for (i <- selectors.indices) {
      res = res.select_in_dim(i, selectors(i))
    }

    new Tensor[T, TT](res)
  }

  @native def is_cuda(): Boolean

  @native @ByVal def device(): Device

  @native @ByVal private def slice(@Cast(Array("long")) dim: Long, @Cast(Array("long")) start: Long = 0, @Cast(Array("long")) end: Long = 9223372036854775807l, @Cast(Array("long")) step: Long = 1): Tensor[T, TT]

  def apply(
             firstIndexer: Indexer,
             otherIndexers: Indexer*
           ): Tensor[T, TT] = {
    val stridedSlice = Indexer.toStridedSlice(firstIndexer, otherIndexers: _*)
    val beginTensor: Array[Int] = stridedSlice._1
    val endTensor: Array[Int] = stridedSlice._2
    val stridesTensor: Array[Int] = stridedSlice._3

    val beginMask: Long = stridedSlice._4
    var endMask: Long = stridedSlice._5
    var ellipsisMask: Long = stridedSlice._6
    var newAxisMask: Long = stridedSlice._7
    var shrinkAxisMask: Long = stridedSlice._8

    var res = this

    for (i <- beginTensor.indices) {
      val e = if(endTensor(i) == -1) shape.apply(i) else endTensor(i)
      res = res.slice(i, beginTensor(i), e, stridesTensor(i))
    }

    new Tensor[T, TT](res)
  }


  @native @ByRef private def put_(@ByRef indices: Tensor[T, TT], @ByRef values: Tensor[T, TT], accumulate: Boolean): Tensor[T, TT]
  def put(indices: Tensor[T, TT], values: Tensor[T, TT]): Tensor[T, TT] = new Tensor(put_(indices, values, false))


  @native @Name(Array("operator+=")) @ByRef private def addeq(@ByRef other: Tensor[T, TT]): Tensor[T, TT]
  def += (other: Tensor[T, TT]): Tensor[T, TT] = new Tensor(addeq(other))
  @native @Name(Array("operator+=")) @ByRef private def addeq(@ByVal other: Scalar[T]): Tensor[T, TT]
  def += (other: T): Tensor[T, TT] = new Tensor(addeq(other))
  @native @Name(Array("operator-")) @ByVal private def minus(): Tensor[T, TT]
  def - (): Tensor[T, TT] = new Tensor(minus())
  def unary_- : Tensor[T, TT] = new Tensor(minus())
  @native @Name(Array("operator-=")) @ByRef private def subeq(@ByRef other: Tensor[T, TT]): Tensor[T, TT]
  def -= (other: Tensor[T, TT]): Tensor[T, TT] = new Tensor(subeq(other))
  @native @Name(Array("operator-=")) @ByRef private def subeq(@ByVal other: Scalar[T]): Tensor[T, TT]
  def -= (other: T): Tensor[T, TT] = new Tensor(subeq(other))
  @native @Name(Array("operator*=")) @ByRef private def muleq(@ByRef other: Tensor[T, TT]): Tensor[T, TT]
  def *= (other: Tensor[T, TT]): Tensor[T, TT] = new Tensor(muleq(other))
  @native @Name(Array("operator*=")) @ByRef private def muleq(@ByVal other: Scalar[T]): Tensor[T, TT]
  def *= (other: T): Tensor[T, TT] = new Tensor(muleq(other))
  @native @Name(Array("operator/=")) @ByRef private def releq(@ByRef other: Tensor[T, TT]): Tensor[T, TT]
  def /= (other: Tensor[T, TT]): Tensor[T, TT] = new Tensor(releq(other))
  @native @Name(Array("operator/=")) @ByRef private def releq(@ByVal other: Scalar[T]): Tensor[T, TT]
  def /= (other: T): Tensor[T, TT] = new Tensor(releq(other))

  @native @Name(Array("operator+")) @ByVal private def add(@ByRef other: Tensor[T, TT]): Tensor[T, TT]
  def + (other: Tensor[T, TT]): Tensor[T, TT] = new Tensor(add(other))
  @native @Name(Array("operator+")) @ByVal private def add(@ByVal other: Scalar[T]): Tensor[T, TT]
  def + (other: Scalar[T]): Tensor[T, TT] = new Tensor(add(other))
  @native @Name(Array("operator-")) @ByVal private def sub(@ByRef other: Tensor[T, TT]): Tensor[T, TT]
  def - (other: Tensor[T, TT]): Tensor[T, TT] = new Tensor(sub(other))
  @native @Name(Array("operator-")) @ByVal private def sub(@ByVal other: Scalar[T]): Tensor[T, TT]
  def - (other: Scalar[T]): Tensor[T, TT] = new Tensor(sub(other))
  @native @Name(Array("operator*")) @ByVal private def mul(@ByRef other: Tensor[T, TT]): Tensor[T, TT]
  def * (other: Tensor[T, TT]): Tensor[T, TT] = new Tensor(mul(other))
  @native @Name(Array("operator*")) @ByVal private def mul(@ByVal other: Scalar[T]): Tensor[T, TT]
  def * (other: Scalar[T]): Tensor[T, TT] = new Tensor(mul(other))
  @native @Name(Array("operator/")) @ByVal private def rel(@ByRef other: Tensor[T, TT]): Tensor[T, TT]
  def / (other: Tensor[T, TT]): Tensor[T, TT] = new Tensor(rel(other))
  @native @Name(Array("operator/")) @ByVal private def rel(@ByVal other: Scalar[T]): Tensor[T, TT]
  def / (other: Scalar[T]): Tensor[T, TT] = new Tensor(rel(other))

  @native @ByVal private def sum(@ByVal dims: IntList, keepdim: Boolean): Tensor[T, TT]
  def sum(dims: Array[Int], keepdim: Boolean = false): Tensor[T, TT] = {
    val t = sum(new IntList(dims.map(_.toLong)), keepdim)
    new Tensor(t)
  }

  def sum(dims: Int*): Tensor[T, TT] = sum(dims.toArray, false)

  @native @ByVal private def pow(@ByRef other: Tensor[T, TT]): Tensor[T, TT]
  def ** (other: Tensor[T, TT]): Tensor[T, TT] = new Tensor(pow(other))
  @native @ByVal private def pow(@ByVal other: Scalar[T]): Tensor[T, TT]
  def ** (other: Scalar[T]): Tensor[T, TT] = new Tensor(pow(other))


  @native @Name(Array("sqrt")) @ByVal private def sqrt_op(): Tensor[T, TT]
  def sqrt(): Tensor[T, TT] = {
    new Tensor(sqrt_op())
  }

  @native @Name(Array("abs")) @ByVal private def abs_op(): Tensor[T, TT]
  def abs(): Tensor[T, TT] = {
    new Tensor(abs_op())
  }

  @native @ByVal private def t(): Tensor[T, TT]
  def T: Tensor[T, TT] = {
    new Tensor(t())
  }

  @native @Name(Array("max")) @ByVal private def max_op(): Tensor[T, TT]
  def max: Tensor[T, TT] = {
    new Tensor(max_op())
  }

  @native @Name(Array("min")) @ByVal private def min_op(): Tensor[T, TT]
  def min: Tensor[T, TT] = {
    new Tensor(min_op())
  }





}


object Tensor {

  def summarize[T, TT <: TensorType[T]](tensor: Tensor[T, TT], maxEntries: Int): String = {
    def summarize[T, TT <: TensorType[T]](tensor: Tensor[T, TT], maxEntries: Int, level: Int): String = tensor.dim match {
      case 0 => tensor.item().toString
      case 1 =>
        val n = tensor.num_elements.toInt
        val slice =
          if (tensor.num_elements <= math.max(maxEntries, 6))
            tensor.data()
          else
            (tensor(0 :: maxEntries / 2).data() :+ "...") ++ tensor(n - maxEntries / 2 :: n).data()
        slice.mkString("[", "\t", "]")
      case _ =>
        val innerSummary = {
          def summarizeSlice(index: Int) = {
            summarize(tensor(index).reshape(tensor.shape(1 ::)), maxEntries, level + 1)
          }

          if (tensor.shape(0) <= math.max(maxEntries, 6))
            for (i <- 0 until tensor.shape(0)) yield summarizeSlice(i)
          else {
            val start = for (i <- 0 until maxEntries / 2) yield summarizeSlice(i)
            val end = for (i <- tensor.shape(0) - maxEntries / 2 until tensor.shape(0)) yield summarizeSlice(i)
            (start :+ "...") ++ end
          }
        }
        val padding = " " * (level + 1)
        val extraLine = if (tensor.dim >= 3) "\n" else ""
        innerSummary.mkString("[", "\n" + extraLine + padding, "]")
    }

    tensor.toString + tensor.shape.toString + "\n"  + summarize(if (tensor.is_cuda()) tensor.cpu() else tensor, maxEntries, 0) + "\n"
  }

  def apply[T:ClassTag](data: Array[T], shape: Shape)(implicit options: TensorOptions[CPUTensorType[T]]): Tensor[T, CPUTensorType[T]] = {
    val pt = data.head match {
      case h: Float => new FloatPointer(data.asInstanceOf[Array[Float]]:_*)
      case h: Int => new IntPointer(data.asInstanceOf[Array[Int]]:_*)
      case h: Long => new LongPointer(data.asInstanceOf[Array[Long]]:_*)
      case h: Double => new DoublePointer(data.asInstanceOf[Array[Double]]:_*)
    }

    val nt = Functions.from_blob[T](pt, new IntList(shape.asArray.map(_.toLong)), new Deallocator_Pointer(null))

    new Tensor[T, CPUTensorType[T]](nt)

  }

  def apply[T: ClassTag, P, TT <: TensorType[T]](data: ArrayRef[T, P])(implicit options: TensorOptions[TT]): Tensor[T, TT] = {

    val nt = data match {
      case x: ArrayRefInt => Functions.tensor(x)(options.asInstanceOf[TensorOptions[TensorType[Int]]])
      case x: ArrayRefFloat => Functions.tensor(x)(options.asInstanceOf[TensorOptions[TensorType[Float]]])
      case x: ArrayRefLong => Functions.tensor(x)(options.asInstanceOf[TensorOptions[TensorType[Long]]])
      case x: ArrayRefDouble => Functions.tensor(x)(options.asInstanceOf[TensorOptions[TensorType[Double]]])
    }

    new Tensor[T, TT](nt.asInstanceOf[Tensor[T, TT]])

  }

  def apply[T: ClassTag, P, TT <: TensorType[T]](data: Array[T])(implicit options: TensorOptions[TT]): Tensor[T, TT] = {
    apply(ArrayRef(data).asInstanceOf[ArrayRef[T, P]])
  }

  def cpu[T: ClassTag](data: Array[T])(implicit options: TensorOptions[CPUTensorType[T]]): Tensor[T, CPUTensorType[T]] = apply[T, Any, CPUTensorType[T]](data)

  def cuda[T: ClassTag](data: Array[T])(implicit options: TensorOptions[CudaTensorType[T]]): Tensor[T, CudaTensorType[T]] = apply[T, Any, CudaTensorType[T]](data)

  def ones[T: ClassTag, TT <: TensorType[T]](shape: Shape)(implicit options: TensorOptions[TT]): Tensor[T, TT] = {
    new Tensor[T, TT](Functions.zeros(new IntList(shape.asArray.map(_.toLong))))
  }

  def ones_like[T, TT <: TensorType[T]](self: Tensor[T, TT]): Tensor[T, TT] = {
    new Tensor[T, TT](Functions.zeros_like(self))
  }

  def zeros[T: ClassTag, TT <: TensorType[T]](shape: Shape)(implicit options: TensorOptions[TT]): Tensor[T, TT] = {
    new Tensor[T, TT](Functions.zeros(new IntList(shape.asArray.map(_.toLong))))
  }

  def zeros_like[T, TT <: TensorType[T]](self: Tensor[T, TT]): Tensor[T, TT] = {
    new Tensor[T, TT](Functions.zeros_like(self))
  }
}
