package torch_scala

import java.io.File
import java.net.URL
import java.nio.file.{Files, StandardCopyOption}

import org.bytedeco.javacpp.annotation.{Cast, Namespace, Platform}
import org.bytedeco.javacpp.tools.{InfoMap, InfoMapper}
import org.bytedeco.javacpp._
import org.bytedeco.javacpp.annotation._
import org.bytedeco.javacpp.tools._


@Properties(target = "torch_native_lib1234",
            value = Array(new Platform(include = Array("torch/all.h")))
)
class NativeLibraryConfig extends InfoMapper {
  def map(infoMap: InfoMap): Unit = {
    //infoMap.put(new Info("data<long>").javaNames("data_int"))
  }
}

object OS {
  val osName: String = System.getProperty("os.name", "generic")
  val tmpDir = System.getProperty("java.io.tmpdir")
  val JNILIBS = "jnilibs"
}


object NativeLoader {
  //val workingDir = System.getProperty("user.dir")
  System.load("/home/nazar/IdeaProjects/torch_scala/src/native/libjava_torch_lib0.so")

  //loadLibraryFromJar("java_torch_lib0")

  def getLibraryUrl(libraryName: String): URL = {
    var url: URL = null
    if (OS.osName.startsWith("Windows"))
      url = this.getClass.getResource("/" + libraryName + ".dll")
    else if (OS.osName.startsWith("Mac")) {
      url = this.getClass.getResource("/lib" + libraryName + ".dylib")
      if (url == null)
        url = this.getClass.getResource("/lib" + libraryName + ".so")
      if (url == null)
        url = this.getClass.getResource("/lib" + libraryName + ".bundle")
    }
    else if (OS.osName.startsWith("Linux"))
      url = this.getClass.getResource("/lib" + libraryName + ".so")

    if (url == null)
      throw new UnsupportedOperationException("Library " + libraryName + " not found.")
    else url
  }

  def createTempDir(): File = {
    val f = new File(OS.tmpDir + "/" + OS.JNILIBS)
    f.mkdir()
    f
  }

  def loadLibraryFromJar(libraryName: String): Unit = {
    val tempDir = createTempDir
    val url = getLibraryUrl(libraryName)
    val fileName = new File(url.getPath).getName
    val lib = new File(tempDir, fileName)
    try {
      val is = getLibraryUrl(libraryName).openStream
      try
        Files.copy(is, lib.toPath, StandardCopyOption.REPLACE_EXISTING)
      finally if (is != null) is.close()
    }
    try
      System.load(lib.getAbsolutePath) // JVM requires absolute path

    catch {
      case e: Exception =>
        e.printStackTrace()
        throw e
    }
  }

}


@Platform(include = Array("torch/all.h"))
@Namespace("torch::cuda") object Torch {

  val loader = NativeLoader

  @native @Cast(Array("size_t")) def device_count(): Int

  /// Returns true if at least one CUDA device is available.
  @native def is_available(): Boolean

  /// Returns true if CUDA is available, and CuDNN is available.
  @native def cudnn_is_available(): Boolean

}
