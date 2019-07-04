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
        target = "torch_scala.api.aten.functions.NativeFunctions"
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
        infoMap.put(new Info("Tensor").pointerTypes("Tensor<T, TT>"));
        infoMap.put(new Info("TensorOptions").pointerTypes("TensorOptions<T, TT>"));
        infoMap.put(new Info("TensorList").pointerTypes("TensorList<T, TT>"));
        infoMap.put(new Info("Scalar").pointerTypes("Scalar<T>"));
        infoMap.put(new Info("ScalarType").pointerTypes("Short"));
        infoMap.put(new Info("std::tuple<Tensor,Tensor,Tensor>").pointerTypes("TensorTriple<T1,T2,T3,TT>"));
        infoMap.put(new Info("std::tuple<Tensor,Tensor,Tensor,Tensor>").pointerTypes("TensorTuple4<T,TT>"));
        infoMap.put(new Info("std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>").pointerTypes("TensorTuple5<T,TT>"));
        infoMap.put(new Info("std::tuple<Tensor&,Tensor&,Tensor&>").pointerTypes("TensorRefTriple<T1,T2,T3,TT>"));
        infoMap.put(new Info("std::tuple<Tensor,Tensor,Tensor,std::vector<Tensor> >").pointerTypes("TensorTripleAndVector<T,TT>"));

        infoMap.put(new Info("std::array").skip().annotations("@StdArray"));
        infoMap.put(new Info("c10::optional").skip().annotations("@C10Optional"));
        infoMap.put(new Info("ScalarType").skip().annotations("@Cast(\"c10::ScalarType\")"));

    }


}