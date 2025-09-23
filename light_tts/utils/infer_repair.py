from torch.nn.modules.module import Module
from torch.nn.utils.weight_norm import WeightNorm
from torch.nn.parameter import Parameter, UninitializedParameter
from torch import norm_except_dim


def new_apply(module, name: str, dim: int) -> 'WeightNorm':
    """
    该函数用于替换原始的apply 函数，避免每一次前向推理都需要重新计算 weight
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, WeightNorm) and hook.name == name:
            raise RuntimeError("Cannot register two weight_norm hooks on "
                                "the same parameter {}".format(name))
        
    if dim is None:
        dim = -1

    fn = WeightNorm(name, dim)

    weight = getattr(module, name)
    if isinstance(weight, UninitializedParameter):
        raise ValueError(
            'The module passed to `WeightNorm` can\'t have uninitialized parameters. '
            'Make sure to run the dummy forward before applying weight normalization')
    # remove w from parameter list
    # del module._parameters[name]

    # add g and v as new parameters and express w as g/||v|| * v
    module.register_parameter(name + '_g', Parameter(norm_except_dim(weight, 2, dim).data))
    module.register_parameter(name + '_v', Parameter(weight.data))
    
    del module._parameters[name]
    setattr(module, name, fn.compute_weight(module))

    def repair(self):
        # del module._parameters[name]
        setattr(self, name, fn.compute_weight(self))
        # module._parameters[name] = getattr(module, name)

    module.re_init_weight_norm_params = repair

    return fn

# 修改默认的apply 行为后才能使用 torch.jit.Script.避免重复计算
WeightNorm.apply = staticmethod(new_apply)


def repair_weight_norm_weights(module: Module):
    if hasattr(module, "re_init_weight_norm_params"):
        module.re_init_weight_norm_params(module)
        del module.re_init_weight_norm_params
    
    for name, sub_module in module.named_modules():
        if hasattr(sub_module, "re_init_weight_norm_params"):
            sub_module.re_init_weight_norm_params(sub_module)
            del sub_module.re_init_weight_norm_params
    return 

Module.repair_weight_norm_weights = repair_weight_norm_weights