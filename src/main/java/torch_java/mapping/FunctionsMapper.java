package torch_java.mapping;

import generate.Info;
import generate.InfoMap;
import generate.InfoMapper;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

@Properties(
        value = @Platform(
                include = {"ATen/Functions.h", "adapters/OptionalAdapter.h", "adapters/StdArrayAdapter.h"}
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
        infoMap.put(new Info("std::tuple<Tensor,Tensor,Tensor>").pointerTypes("TensorTriple<T1,T2,T3,TT>"));
        infoMap.put(new Info("std::tuple<Tensor,Tensor,Tensor,Tensor>").pointerTypes("TensorTuple4<T,TT>"));
        infoMap.put(new Info("std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>").pointerTypes("TensorTuple5<T,TT>"));
        infoMap.put(new Info("std::tuple<Tensor&,Tensor&,Tensor&>").pointerTypes("TensorRefTriple<T1,T2,T3,TT>"));
        infoMap.put(new Info("std::tuple<Tensor,Tensor,Tensor,std::vector<Tensor> >").pointerTypes("TensorTripleAndVector<T,TT>"));
        infoMap.put(new Info("std::tuple<Tensor,Tensor,Tensor,int64_t>").pointerTypes("TensorTripleAndLong<T,TT>"));
        infoMap.put(new Info("std::tuple<Tensor,Tensor,double,int64_t>").pointerTypes("TensorTupleAndDoubleLong<T,TT>"));

        infoMap.put(new Info("std::array<bool,3>").pointerTypes("ArrayBool3"));
        infoMap.put(new Info("std::array<bool,4>").pointerTypes("ArrayBool4"));
        infoMap.put(new Info("std::array<bool,2>").pointerTypes("ArrayBool2"));
        infoMap.put(new Info("c10::optional").skip().annotations("@C10Optional"));
        infoMap.put(new Info("std::function<void(void*)>").annotations("@Cast(\"std::function<void(void*)>\")").pointerTypes("FunctionPointer"));
        infoMap.put(new Info("const std::function<void(void*)>").annotations("@Cast(\"const std::function<void(void*)>\")").pointerTypes("FunctionPointer"));

    }


}