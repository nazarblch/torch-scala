package torch_scala.api.std

import org.bytedeco.javacpp.Pointer
import org.bytedeco.javacpp.annotation.{Name, Namespace, NoOffset, Platform}

@Platform(include = Array("<array>"))
@NoOffset @Name(Array("std::array<bool, 2>")) class ArrayBool2 extends Pointer(null.asInstanceOf[Pointer]) {
  @native def allocate(): Unit
}

@Platform(include = Array("<array>"))
@NoOffset @Name(Array("std::array<bool, 3>")) class ArrayBool3 extends Pointer(null.asInstanceOf[Pointer]) {
  @native def allocate(): Unit
}

@Platform(include = Array("<array>"))
@NoOffset @Name(Array("std::array<bool, 4>")) class ArrayBool4 extends Pointer(null.asInstanceOf[Pointer]) {
  @native def allocate(): Unit
}
