package torch_scala.autograd

import org.scalatest.{FlatSpec, Matchers}
import torch_scala.api.aten.{CPU, Shape, Tensor}
import MathVariable._
import torch_scala.api.aten.functions.Math._
import torch_scala.nn.Linear

class FunctionSpec extends FlatSpec with Matchers {

  "Function" should "transpose" in {
    val v = Variable(Tensor.randn[Double, CPU](Shape(4, 6)))
    val w = Variable(Tensor.randn[Double, CPU](Shape(6, 4)))

    val vt = v.T
    vt.shape shouldBe Shape(6, 4)

    vt.backward(w)
    v.grad.shape shouldBe Shape(4, 6)
  }

  it should "do a dot product" in {
    val x = Variable(Tensor.arange[Double, CPU](0, 12).reshape(Shape(3, 4)))
    val y = Variable(Tensor.arange[Double, CPU](0, 8).reshape(Shape(4, 2)))
    val z: Variable[Double, CPU] = Matmul(x, y).forward()
    z.shape shouldBe Shape(3, 2)

    println(z)

    val g = Variable(Tensor.arange[Double, CPU](0, 6).reshape(z.shape))
    z.backward(g)

    println(x.grad.data)
    println(y.grad.data)

    x.grad.data.data() shouldBe Array( //
        1, 3, 5, //
        7, 3, 13, //
        23, 33, 5, //
        23, 41, 59)
    x.grad.shape shouldBe Shape(3, 4)

    y.grad.data.data shouldBe Array( //
        40, 52, //
        46, 61, //
        52, 70, //
        58, 79)
    y.grad.shape shouldBe Shape(4, 2)
  }


  it should "do an affine operation" in {

    val inputFeatures = 4
    val outputFeatures = 3
    val numSamples = 16

    def makeVariable(shape: Int*): Variable[Double, CPU] = {
      Variable(Tensor.arange[Double, CPU](0, shape.product.toLong).reshape(Shape.fromSeq(shape)))
    }

    val w = makeVariable(outputFeatures, inputFeatures)
    val b = makeVariable(1, outputFeatures)

    val x = makeVariable(numSamples, inputFeatures)

    val d = x mm w.T
    d.shape shouldBe Shape(numSamples, outputFeatures)

    val y = d + b
    y.shape shouldBe d.shape

    val dy = makeVariable(y.shape.asArray: _*)
    y.backward(dy)

    println(w.grad.shape)
    println(b.grad)

    println(b.grad.data.sum(0))

  }


  it should "linear" in {

    val x = Tensor.arange[Double, CPU](0,12).reshape(Shape(4, 3))
    val w = Tensor.arange[Double, CPU](0,15).reshape(Shape(5, 3)).T
    val b = Tensor.arange[Double, CPU](0,5)

    val weights = Variable(w, name = Some("weights"))
    val bias = Variable(b, name = Some("bias"))
    val input = Variable(x, name = Some("x"))

    val l = Linear(weights, bias)

    val output: Variable[Double, CPU] = l.forward(input)

    println(output)

    val dy = Variable(Tensor.arange[Double, CPU](0, 20).reshape(Shape(4, 5)), name = Some("dy"))

    output.backward(dy)

    println(weights.grad)
    println(bias.grad)

    output.data shouldBe Tensor.apply[Double, CPU](
      5.0, 15.0, 25.0, 35.0, 45.0,
      14.0, 51.0, 88.0, 125.0, 162.0,
      23.0, 87.0, 151.0, 215.0, 279.0,
      32.0, 123.0, 214.0, 305.0, 396.0).reshape(Shape(4, 5))

    weights.grad.data shouldBe x.T.matmul(dy.data)

    bias.grad.data shouldBe dy.data.sum(0)



  }

  it should "handle 2 layers" in {

    val x = Variable(Tensor.arange[Double, CPU](0,12).reshape(Shape(4, 3)), name = Some("x"))

    val w1 = Variable(Tensor.arange[Double, CPU](0,15).reshape(Shape(5, 3)).T, name = Some("w1"))
    val b1 = Variable(Tensor.arange[Double, CPU](0,5).reshape(Shape(1, 5)), name = Some("b1"))
    val l1 = Linear(w1, b1)

    val w2 = Variable(Tensor.arange[Double, CPU](0,30).reshape(Shape(6, 5)).T, name = Some("w2"))
    val b2 = Variable(Tensor.arange[Double, CPU](0,6).reshape(Shape(1, 6)), name = Some("b2"))
    val l2 = Linear(w2, b2)

    val out = x ~> l1 ~> l2
    val dOut = Variable(Tensor.arange[Double, CPU](0,24).reshape(Shape(4, 6)))
    out.backward(dOut)

    println(out)
    println(w2.grad)
    println(b2.grad)
    println(w1.grad)
    println(b1.grad)

    out.data shouldBe Tensor[Double, CPU](350.0, 976.0, 1602.0, 2228.0, 2854.0, 3480.0, 1250.0, 3451.0,
      5652.0, 7853.0, 10054.0, 12255.0, 2150.0, 5926.0, 9702.0, 13478.0,
      17254.0, 21030.0, 3050.0, 8401.0, 13752.0, 19103.0, 24454.0,
      29805.0).reshape(Shape(4, 6))

    w2.grad.data shouldBe Tensor[Double, CPU](936.0, 3564.0, 6192.0, 8820.0, 11448.0, 1010.0, 3840.0, 6670.0,
      9500.0, 12330.0, 1084.0, 4116.0, 7148.0, 10180.0, 13212.0, 1158.0,
      4392.0, 7626.0, 10860.0, 14094.0, 1232.0, 4668.0, 8104.0, 11540.0,
      14976.0, 1306.0, 4944.0, 8582.0, 12220.0,
      15858.0).reshape(Shape(6, 5)).T

    b2.grad.data shouldBe Tensor[Double, CPU] (36.0, 40.0, 44.0, 48.0, 52.0, 56.0).reshape(Shape(1, 6))

    w1.grad.data shouldBe Tensor[Double, CPU] (23850.0, 27650.0, 31450.0, 25632.0, 29708.0, 33784.0, 27414.0,
          31766.0, 36118.0, 29196.0, 33824.0, 38452.0, 30978.0, 35882.0,
          40786.0)
          .reshape(Shape(5, 3)).T


    b1.grad.data shouldBe Tensor[Double, CPU](3800.0, 4076.0, 4352.0, 4628.0, 4904.0).reshape(Shape(1, 5))

  }


}
