package mapping;

import javacpp_tools.Info;
import javacpp_tools.InfoMap;
import javacpp_tools.InfoMapper;
import org.bytedeco.javacpp.annotation.*;

@Properties(
        value = @Platform(
                include = {"ATen/NativeFunctions.h", "adapters/OptionalAdapter.h", "adapters/StdArrayAdapter.h"}
        ),
        target = "torch_scala.api.aten.functions.NativeFunctions"
)
public class FunctionsMapper implements InfoMapper {

    public void map(InfoMap infoMap) {

        infoMap.put(new Info().linePatterns("#define TENSOR(T, S)", "#undef TENSOR").skip());
        infoMap.put(new Info("AT_FORALL_COMPLEX_TYPES").skip());
        infoMap.put(new Info("AT_FORALL_SCALAR_TYPES_AND3").skip());

        infoMap.put(new Info("std::tuple<Tensor,Tensor>").pointerTypes("TensorTuple<T1, T2, TT>"));
        infoMap.put(new Info("std::tuple<at::Tensor,at::Tensor>").pointerTypes("TensorTuple<T1, T2, TT>"));
        infoMap.put(new Info("std::tuple<Tensor&,Tensor&>").pointerTypes("TensorRefTuple<T1, T2, TT>"));
        infoMap.put(new Info("std::tuple<at::Tensor&,at::Tensor&>").pointerTypes("TensorRefTuple<T1, T2, TT>"));
        infoMap.put(new Info("Tensor").pointerTypes("Tensor<T, TT>"));
        infoMap.put(new Info("at::Tensor").pointerTypes("Tensor<T, TT>"));
        infoMap.put(new Info("TensorOptions").pointerTypes("TensorOptions<T, TT>"));
        infoMap.put(new Info("TensorList").pointerTypes("TensorList<T, TT>"));
        infoMap.put(new Info("Scalar").pointerTypes("Scalar<T>"));
        infoMap.put(new Info("Device").pointerTypes("Device<TT>"));
        infoMap.put(new Info("std::tuple<Tensor,Tensor,Tensor>").pointerTypes("TensorTriple<T1,T2,T3,TT>"));
        infoMap.put(new Info("std::tuple<at::Tensor,at::Tensor,at::Tensor>").pointerTypes("TensorTriple<T1,T2,T3,TT>"));
        infoMap.put(new Info("std::tuple<Tensor,Tensor,Tensor,Tensor>").pointerTypes("TensorTuple4<T,TT>"));
        infoMap.put(new Info("std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor>").pointerTypes("TensorTuple4<T,TT>"));
        infoMap.put(new Info("std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>").pointerTypes("TensorTuple5<T,TT>"));
        infoMap.put(new Info("std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>").pointerTypes("TensorTuple5<T,TT>"));
        infoMap.put(new Info("std::tuple<Tensor&,Tensor&,Tensor&>").pointerTypes("TensorRefTriple<T1,T2,T3,TT>"));
        infoMap.put(new Info("std::tuple<at::Tensor&,at::Tensor&,at::Tensor&>").pointerTypes("TensorRefTriple<T1,T2,T3,TT>"));
        infoMap.put(new Info("std::tuple<Tensor,Tensor,Tensor,std::vector<Tensor> >").pointerTypes("TensorTripleAndVector<T,TT>"));
        infoMap.put(new Info("std::tuple<at::Tensor,at::Tensor,at::Tensor,std::vector<at::Tensor> >").pointerTypes("TensorTripleAndVector<T,TT>"));
        infoMap.put(new Info("std::tuple<Tensor,Tensor,Tensor,int64_t>").pointerTypes("TensorTripleAndLong<T,TT>"));
        infoMap.put(new Info("std::tuple<at::Tensor,at::Tensor,at::Tensor,int64_t>").pointerTypes("TensorTripleAndLong<T,TT>"));
        infoMap.put(new Info("std::tuple<Tensor,Tensor,double,int64_t>").pointerTypes("TensorTupleAndDoubleLong<T,TT>"));
        infoMap.put(new Info("std::tuple<at::Tensor,at::Tensor,double,int64_t>").pointerTypes("TensorTupleAndDoubleLong<T,TT>"));
        infoMap.put(new Info("std::tuple<double,int64_t>").pointerTypes("DoubleLong"));
        infoMap.put(new Info("std::tuple<double,double>").pointerTypes("DoubleDouble"));
        infoMap.put(new Info("std::tuple<Tensor,Tensor,Tensor,Tensor,int64_t>").pointerTypes("TensorTuple4AndLong<T,TT>"));
        infoMap.put(new Info("std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t>").pointerTypes("TensorTuple4AndLong<T,TT>"));

        infoMap.put(new Info("std::array<bool,3>").pointerTypes("ArrayBool3"));
        infoMap.put(new Info("std::array<bool,4>").pointerTypes("ArrayBool4"));
        infoMap.put(new Info("std::array<bool,2>").pointerTypes("ArrayBool2"));
        infoMap.put(new Info("c10::optional").skip().annotations("@C10Optional"));
        infoMap.put(new Info("c10::List").skip().annotations("@C10List"));
        infoMap.put(new Info("TORCH_API").skip().annotations(""));
        infoMap.put(new Info("CAFFE2_API").skip().annotations(""));
        infoMap.put(new Info("std::function<void(void*)>").annotations("@Cast(\"std::function<void(void*)>\")").pointerTypes("FunctionPointer"));
        infoMap.put(new Info("const std::function<void(void*)>").annotations("@Cast(\"const std::function<void(void*)>\")").pointerTypes("FunctionPointer"));

    }


}