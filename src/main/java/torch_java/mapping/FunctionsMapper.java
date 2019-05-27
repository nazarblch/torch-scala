package torch_java.mapping;

import generate.Info;
import generate.InfoMap;
import generate.InfoMapper;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;


@Properties(
        value = @Platform(
                includepath = {"/home/nazar/libtorch/include"},
                //preloadpath = {""},
                //linkpath = {"/home/nazar/libtorch/lib"},
                include = {"ATen/Functions.h"}
                //preload = {"libtorch"},
                //link = {"NativeLibrary"}
        ),
        target = "NativeFunctions"
)
public class FunctionsMapper implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(
                new Info("std::tuple<Tensor,Tensor>")
                        .pointerTypes("TensorTuple<T1, T2, TT>")
        );
        infoMap.put(
                new Info("std::tuple<Tensor&,Tensor&>")
                        .pointerTypes("TensorRefTuple<T1, T2, TT>")
        );
        infoMap.put(
                new Info("Tensor")
                        .pointerTypes("Tensor<T, TT>")
        );



    }


}