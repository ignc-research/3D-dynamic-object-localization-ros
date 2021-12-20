import torch
from torch.fx.graph import (
    Node,
)
import torch.nn.quantized as nnq
import torch.nn.quantized.dynamic as nnqd

from ..quantization_mappings import (
    get_static_quant_module_class,
    get_quantized_operator,
)
from ..custom_module_class_mappings import (
    get_quantized_custom_module_class,
)
from .pattern_utils import (
    register_quant_pattern,
)
from .utils import (
    _parent_name,
    quantize_node,
    get_per_tensor_qparams,
    activation_is_statically_quantized,
    weight_is_quantized,
    weight_dtype,
    get_linear_prepack_op_for_dtype,
)

from abc import ABC, abstractmethod
import operator

# -------------------------
# Pattern Registrations
# -------------------------

# 1. Post Training Static Quantization and Quantization Aware Training Patterns

# Base Pattern Handler
class QuantizeHandler(ABC):
    """ Base handler class for the quantizer patterns
    """
    def __init__(self, quantizer, node):
        """ Records pattern information in __init__, which will be used
        in convert
        """
        # this is an indicator of whether all the inputs are Node or not
        # since some op might be quantized differently depending on whether
        # all inputs are tensors or not, e.g. add/mul
        self.all_nodes = True

    @abstractmethod
    def convert(self, quantizer, node, load_arg, debug=False):
        """ Convert the given node to a quantized node and insert
        it to the quantized graph
        """
        return NotImplemented

@register_quant_pattern(operator.add)
@register_quant_pattern((torch.nn.ReLU, operator.add))
@register_quant_pattern((torch.nn.functional.relu, operator.add))
class Add(QuantizeHandler):
    def __init__(self, quantizer, node):
        super().__init__(quantizer, node)
        self.relu_node = None
        if (node.op == 'call_function' and node.target is torch.nn.functional.relu) or \
           (node.op == 'call_module' and isinstance(quantizer.modules[node.target], torch.nn.ReLU)):
            self.relu_node = node
            node = node.args[0]
        assert node.op == 'call_function' and node.target == operator.add
        self.add_node = node
        self.all_nodes = all([isinstance(a, Node) for a in self.add_node.args[:2]])

    def convert(self, quantizer, node, load_arg, debug=False):
        if not self.all_nodes:
            # add scalar
            if self.relu_node is not None:
                op = torch.ops.quantized.add_relu
            else:
                op = torch.ops.quantized.add
            return quantizer.quantized_graph.create_node(
                'call_function', op,
                load_arg(quantized=[0])(self.add_node.args), self.add_node.kwargs)
        else:
            activation_post_process = quantizer.activation_post_process_map[node.name]
            scale, zero_point = activation_post_process.calculate_qparams()
            scale = float(scale)
            zero_point = int(zero_point)
            if self.relu_node is not None:
                op = torch.ops.quantized.add_relu
            else:
                op = torch.ops.quantized.add
            kwargs = self.add_node.kwargs
            kwargs.update({'scale': scale, 'zero_point': zero_point})
            return quantizer.quantized_graph.create_node(
                'call_function', op, load_arg(quantized=True)(self.add_node.args), kwargs)

@register_quant_pattern(operator.mul)
@register_quant_pattern((torch.nn.ReLU, operator.mul))
@register_quant_pattern((torch.nn.functional.relu, operator.mul))
class Mul(QuantizeHandler):
    def __init__(self, quantizer, node):
        super().__init__(quantizer, node)
        self.relu_node = None
        if (node.op == 'call_function' and node.target is torch.nn.functional.relu) or \
           (node.op == 'call_module' and isinstance(quantizer.modules[node.target], torch.nn.ReLU)):
            self.relu_node = node
            node = node.args[0]
        assert node.op == 'call_function' and node.target == operator.mul
        self.mul_node = node
        self.all_nodes = all([isinstance(a, Node) for a in self.mul_node.args[:2]])

    def convert(self, quantizer, node, load_arg, debug=False):
        if not self.all_nodes:
            # mul scalar
            if self.relu_node is not None:
                op = torch.ops.quantized.mul_relu
            else:
                op = torch.ops.quantized.mul
            return quantizer.quantized_graph.create_node(
                'call_function', op, load_arg(quantized=[0])(self.mul_node.args), self.mul_node.kwargs)
        else:
            activation_post_process = quantizer.activation_post_process_map[node.name]
            scale, zero_point = activation_post_process.calculate_qparams()
            scale = float(scale)
            zero_point = int(zero_point)
            if self.relu_node is not None:
                op = torch.ops.quantized.mul_relu
            else:
                op = torch.ops.quantized.mul
            kwargs = self.mul_node.kwargs
            kwargs.update({'scale': scale, 'zero_point': zero_point})
            return quantizer.quantized_graph.create_node('call_function', op, load_arg(quantized=True)(self.mul_node.args), kwargs)

@register_quant_pattern(torch.cat)
class Cat(QuantizeHandler):
    def convert(self, quantizer, node, load_arg, debug=False):
        if not self.all_nodes:
            return NotImplemented
        activation_post_process = quantizer.activation_post_process_map[node.name]
        scale, zero_point = activation_post_process.calculate_qparams()
        scale = float(scale)
        zero_point = int(zero_point)
        kwargs = load_arg(quantized=False)(node.kwargs)
        kwargs.update({'scale': scale, 'zero_point': zero_point})
        return quantizer.quantized_graph.create_node(
            'call_function', torch.ops.quantized.cat, load_arg(quantized=[0])(node.args), kwargs)

# handle conv, maybe followed by relu
# NB: matching order is reversed, that is we match from the bottom of this list to the beginning
@register_quant_pattern(torch.nn.Conv1d)
@register_quant_pattern(torch.nn.Conv2d)
@register_quant_pattern(torch.nn.Conv3d)
@register_quant_pattern(torch.nn.functional.conv2d)
@register_quant_pattern(torch.nn.qat.Conv2d)
@register_quant_pattern(torch.nn.intrinsic.ConvReLU1d)
@register_quant_pattern(torch.nn.intrinsic.ConvReLU2d)
@register_quant_pattern(torch.nn.intrinsic.ConvReLU3d)
@register_quant_pattern(torch.nn.intrinsic.qat.ConvBn2d)
@register_quant_pattern(torch.nn.intrinsic.qat.ConvBnReLU2d)
@register_quant_pattern(torch.nn.intrinsic.qat.ConvReLU2d)
@register_quant_pattern((torch.nn.functional.relu, torch.nn.functional.conv2d))
@register_quant_pattern((torch.nn.ReLU, torch.nn.functional.conv2d))
# just for error checks
@register_quant_pattern((torch.nn.ReLU, torch.nn.Conv2d))
@register_quant_pattern((torch.nn.functional.relu, torch.nn.Conv2d))
class ConvRelu(QuantizeHandler):
    def __init__(self, quantizer, node):
        super().__init__(quantizer, node)
        self.relu_node = None
        if (node.op == 'call_function' and node.target is torch.nn.functional.relu) or \
           (node.op == 'call_module' and isinstance(quantizer.modules[node.target], torch.nn.ReLU)):
            self.relu_node = node
            node = node.args[0]
        self.conv_node = node
        if node.op == 'call_module':
            self.conv = quantizer.modules[self.conv_node.target]

    def convert(self, quantizer, node, load_arg, debug=False):
        # TODO: debug option for conv module
        if self.conv_node.op == 'call_module':
            # note that relu should already be fused into conv module in the fusion step
            assert self.relu_node is None, 'conv module and relu fusion is not executed, ' \
                'please make sure to run fusion before prepare'
            # 1. attach activation post process to module
            if type(self.conv) in [
                    torch.nn.intrinsic.ConvReLU1d,
                    torch.nn.intrinsic.ConvReLU2d,
                    torch.nn.intrinsic.ConvReLU3d
            ]:
                self.conv[1].activation_post_process = quantizer.activation_post_process_map[node.name]
            else:
                self.conv.activation_post_process = quantizer.activation_post_process_map[node.name]
            # 2. select quantized class
            qconv_cls = get_static_quant_module_class(type(self.conv))
            quantized = qconv_cls.from_float(self.conv)
            parent_name, name = _parent_name(self.conv_node.target)
            setattr(quantizer.modules[parent_name], name, quantized)
            return quantizer.quantized_graph.create_node(
                'call_module',
                self.conv_node.target,
                (load_arg(quantized=True)(self.conv_node.args[0]),),
                {})
        elif self.conv_node.op == 'call_function':
            if self.relu_node is not None:
                raise Exception("functional conv + relu is not supported yet")
            if debug:
                args = load_arg(quantized=[0, 1])(self.conv_node.args)
                args = load_arg(quantized=False)(self.conv_node.args)
                kwargs = load_arg(quantized=False)(self.conv_node.kwargs)
                conv_out = quantizer.quantized_graph.create_node(
                    'call_function', torch.nn.functional.conv2d, args, kwargs)
                root_module = quantizer.modules['']
                return quantize_node(
                    root_module, quantizer.quantized_graph, conv_out, quantizer.activation_post_process_map[self.conv_node.name])
            else:
                assert len(self.conv_node.args) == 7, \
                    'only conv2d calls with all arguments specified is support right now in debug=False option'
                args = load_arg(quantized=[0, 1])(self.conv_node.args)
                # pack weight
                weight = load_arg(quantized=True)(self.conv_node.args[1])
                other_args = load_arg(quantized=False)(self.conv_node.args[2:])
                prepack_args = tuple([weight] + list(other_args))
                packed_weight = quantizer.quantized_graph.create_node(
                    'call_function', torch.ops.quantized.conv2d_prepack, prepack_args, {})
                # construct conv input
                conv_input = load_arg(quantized=True)(self.conv_node.args[0])
                activation_post_process = quantizer.activation_post_process_map[self.conv_node.name]
                scale, zero_point, _ = get_per_tensor_qparams(activation_post_process)
                qconv_args = (conv_input, packed_weight, scale, zero_point)
                kwargs = load_arg(quantized=False)(self.conv_node.kwargs)
                return quantizer.quantized_graph.create_node(
                    'call_function', torch.ops.quantized.conv2d, qconv_args, kwargs)

# handle linear, maybe followed by relu
@register_quant_pattern(torch.nn.Linear)
@register_quant_pattern(torch.nn.functional.linear)
@register_quant_pattern(torch.nn.qat.Linear)
@register_quant_pattern(torch.nn.intrinsic.LinearReLU)
@register_quant_pattern(torch.nn.intrinsic.qat.LinearReLU)
@register_quant_pattern((torch.nn.functional.relu, torch.nn.functional.linear))
@register_quant_pattern((torch.nn.ReLU, torch.nn.functional.linear))
# for error checks
@register_quant_pattern((torch.nn.ReLU, torch.nn.Linear))
@register_quant_pattern((torch.nn.functional.relu, torch.nn.Linear))
class LinearReLUQuantizeHandler(QuantizeHandler):
    def __init__(self, quantizer, node):
        super().__init__(quantizer, node)
        self.relu_node = None
        if (node.op == 'call_function' and node.target is torch.nn.functional.relu) or \
           (node.op == 'call_module' and isinstance(quantizer.modules[node.target], torch.nn.ReLU)):
            self.relu_node = node
            node = node.args[0]
        self.linear_node = node
        if node.op == 'call_module':
            self.linear = quantizer.modules[self.linear_node.target]

    def convert(self, quantizer, node, load_arg, debug=False):
        qconfig = quantizer.qconfig_map[node.name]
        activation_statically_quantized = activation_is_statically_quantized(qconfig)
        # TODO: debug option for linear module
        if self.linear_node.op == 'call_module':
            # note that relu should already be fused into conv module in the fusion step
            assert self.relu_node is None, 'linear module and relu fusion is not executed, ' \
                'please make sure to run fusion before prepare'
            # 1. attach output activation post process to linear module
            if node.name in quantizer.activation_post_process_map:
                # this is the static quantization case
                output_activation_post_process = quantizer.activation_post_process_map[node.name]
            else:
                output_activation_post_process = None

            if output_activation_post_process:
                if type(self.linear) == torch.nn.intrinsic.LinearReLU:
                    float_linear_module = self.linear[1]
                else:
                    float_linear_module = self.linear
                float_linear_module.activation_post_process = output_activation_post_process

            # 2. select corresponding quantized linear class for the float linear class
            if type(self.linear) in [torch.nn.Linear, torch.nn.qat.Linear]:
                qlinear = nnq.Linear if activation_statically_quantized else nnqd.Linear
            elif type(self.linear) in [torch.nn.intrinsic.LinearReLU, torch.nn.intrinsic.qat.LinearReLU]:
                assert activation_statically_quantized, \
                    'Only static quantization is supported for LinearReLU'
                qlinear = torch.nn.intrinsic.quantized.LinearReLU
            else:
                raise Exception("unhandled linear type:", type(self.linear))
            quantized = qlinear.from_float(self.linear)
            parent_name, name = _parent_name(self.linear_node.target)
            setattr(quantizer.modules[parent_name], name, quantized)
            # activation needs to be quantized for static quantization
            return quantizer.quantized_graph.create_node(
                'call_module',
                self.linear_node.target,
                (load_arg(quantized=activation_statically_quantized)(self.linear_node.args[0]),), {})
        elif self.linear_node.op == 'call_function':
            if debug:
                quantized_input_idxs = []
                if activation_statically_quantized:
                    quantized_input_idxs.append(0)
                if weight_is_quantized(qconfig):
                    quantized_input_idxs.append(1)
                args = load_arg(quantized=quantized_input_idxs)(self.linear_node.args)
                args = load_arg(quantized=False)(self.linear_node.args)
                kwargs = load_arg(quantized=False)(self.linear_node.kwargs)
                linear_out = quantizer.quantized_graph.create_node(
                    'call_function', torch.nn.functional.linear, args, kwargs)
                if activation_statically_quantized:
                    # quantize output for statically quantized linear op
                    root_module = quantizer.modules['']
                    return quantize_node(
                        root_module,
                        quantizer.quantized_graph,
                        linear_out,
                        quantizer.activation_post_process_map[self.linear_node.name])
                else:
                    # output for dynamically quantized linear op is not quantized
                    return linear_out
            else:  # non-debug option
                # linear args
                # (x, weight, bias, ...)
                weight_quantized = weight_is_quantized(qconfig)
                linear_weight = load_arg(quantized=weight_quantized)(self.linear_node.args[1])

                # get other arguments
                kwargs = load_arg(quantized=False)(self.linear_node.kwargs)
                # pack weight
                bias = None
                # all args after bias, including bias
                other_args = load_arg(quantized=False)(self.linear_node.args[2:])
                if len(self.linear_node.args) > 2:
                    bias = load_arg(quantized=False)(self.linear_node.args[2])
                    other_args = other_args[1:]  # remove the bias argument
                else:
                    assert 'bias' in kwargs, \
                        'expect bias provided as a keyword argument when it is not a positional argument'
                    bias = kwargs['bias']
                    kwargs.pop('bias')
                prepack_args = (linear_weight, bias)
                prepack_op = get_linear_prepack_op_for_dtype(weight_dtype(qconfig))
                packed_weight = quantizer.quantized_graph.create_node(
                    'call_function', prepack_op, prepack_args, {})
                # construct linear input
                if activation_statically_quantized:
                    linear_input = load_arg(quantized=True)(self.linear_node.args[0])
                    activation_post_process = \
                        quantizer.activation_post_process_map[self.linear_node.name]
                    scale, zero_point, _ = get_per_tensor_qparams(activation_post_process)
                    qlinear_args = (linear_input, packed_weight, scale, zero_point)
                    return quantizer.quantized_graph.create_node(
                        'call_function', torch.ops.quantized.linear, qlinear_args, kwargs)
                else:
                    linear_input = load_arg(quantized=False)(self.linear_node.args[0])
                    qlinear_args = (linear_input, packed_weight)
                    return quantizer.quantized_graph.create_node(
                        'call_function', torch.ops.quantized.linear_dynamic, qlinear_args, kwargs)

@register_quant_pattern(torch.nn.BatchNorm2d)
@register_quant_pattern(torch.nn.BatchNorm3d)
@register_quant_pattern(torch.nn.intrinsic.BNReLU2d)
@register_quant_pattern(torch.nn.intrinsic.BNReLU3d)
class BatchNorm(QuantizeHandler):
    def __init__(self, quantizer, node):
        super().__init__(quantizer, node)
        assert node.op == 'call_module'
        self.bn_node = node
        self.bn = quantizer.modules[self.bn_node.target]

    def convert(self, quantizer, node, load_arg, debug=False):
        # 1. attach activation post process to module
        activation_post_process = quantizer.activation_post_process_map[node.name]
        if type(self.bn) in \
            [torch.nn.intrinsic.BNReLU2d,
             torch.nn.intrinsic.BNReLU3d]:
            self.bn[1].activation_post_process = activation_post_process
        else:
            self.bn.activation_post_process = activation_post_process
        qbn_cls = get_static_quant_module_class(type(self.bn))
        quantized = qbn_cls.from_float(self.bn)
        parent_name, name = _parent_name(self.bn_node.target)
        setattr(quantizer.modules[parent_name], name, quantized)
        return quantizer.quantized_graph.create_node(
            'call_module',
            self.bn_node.target,
            load_arg(quantized=[0])(self.bn_node.args),
            load_arg(quantized=False)(self.bn_node.kwargs))

ARGS_TO_SKIP = {
    torch._ops.ops.quantized.hardswish: ['inplace'],
    torch._ops.ops.quantized.instance_norm:
    ['running_mean', 'running_var', 'use_input_stats', 'momentum'],
}
@register_quant_pattern(torch.nn.ELU)
@register_quant_pattern(torch.nn.Hardswish)
@register_quant_pattern(torch.nn.InstanceNorm1d)
@register_quant_pattern(torch.nn.InstanceNorm2d)
@register_quant_pattern(torch.nn.InstanceNorm3d)
@register_quant_pattern(torch.nn.LayerNorm)
@register_quant_pattern(torch.nn.functional.hardswish)
@register_quant_pattern(torch.nn.functional.instance_norm)
@register_quant_pattern(torch.nn.functional.layer_norm)
class DefaultNode(QuantizeHandler):
    ''' Common quantized op, first input and first output will be quantized
    '''
    def convert(self, quantizer, node, load_arg, debug=False):
        if not self.all_nodes:
            return NotImplemented
        assert node.op in ['call_module', 'call_function'], 'Only call_module and ' + \
            'call_function are handled in DefaultNode'
        activation_post_process = quantizer.activation_post_process_map[node.name]
        if node.op == 'call_module':
            module = quantizer.modules[node.target]
            module.activation_post_process = activation_post_process
            quantized_module_cls = get_static_quant_module_class(type(module))
            quantized_module = quantized_module_cls.from_float(module)
            parent_name, name = _parent_name(node.target)
            setattr(quantizer.modules[parent_name], name, quantized_module)
            return quantizer.quantized_graph.create_node(
                'call_module',
                node.target,
                load_arg(quantized=[0])(node.args),
                load_arg(quantized=False)(node.kwargs))
        else:
            # call_function
            scale, zero_point = activation_post_process.calculate_qparams()
            scale = float(scale)
            zero_point = int(zero_point)

            quantized_op = get_quantized_operator(node.target)
            args = load_arg(quantized=[0])(node.args)
            kwargs = load_arg(quantized=False)(node.kwargs)
            kwargs.update({'output_scale': scale, 'output_zero_point': zero_point})
            if quantized_op in ARGS_TO_SKIP:
                args_to_skip = ARGS_TO_SKIP[quantized_op]
                for arg in args_to_skip:
                    if arg in kwargs:
                        kwargs.pop(arg)
            return quantizer.quantized_graph.create_node(
                'call_function', quantized_op, args, kwargs)

# TODO: elu is using scale/zero_point instead of output_scale, output_zero_point
@register_quant_pattern(torch.nn.functional.elu)
class ELU(QuantizeHandler):
    def convert(self, quantizer, node, load_arg, debug=False):
        activation_post_process = quantizer.activation_post_process_map[node.name]
        scale, zero_point = activation_post_process.calculate_qparams()
        scale = float(scale)
        zero_point = int(zero_point)
        quantized_op = get_quantized_operator(node.target)
        args = load_arg(quantized=[0])(node.args)
        kwargs = load_arg(quantized=False)(node.kwargs)
        kwargs.update({'output_scale': scale, 'output_zero_point': zero_point})
        kwargs.pop('inplace')
        return quantizer.quantized_graph.create_node(
            'call_function', quantized_op, args, kwargs)

# these ops have quantized equivalents that do not need any extra information
@register_quant_pattern(torch.nn.AdaptiveAvgPool1d)
@register_quant_pattern(torch.nn.AdaptiveAvgPool2d)
@register_quant_pattern(torch.nn.AdaptiveAvgPool3d)
@register_quant_pattern(torch.nn.AvgPool1d)
@register_quant_pattern(torch.nn.AvgPool2d)
@register_quant_pattern(torch.nn.AvgPool3d)
@register_quant_pattern(torch.nn.Dropout)
@register_quant_pattern(torch.nn.Hardsigmoid)
@register_quant_pattern(torch.nn.Hardtanh)
@register_quant_pattern(torch.nn.LeakyReLU)
@register_quant_pattern(torch.nn.MaxPool1d)
@register_quant_pattern(torch.nn.MaxPool2d)
@register_quant_pattern(torch.nn.MaxPool3d)
@register_quant_pattern(torch.nn.ReLU)
@register_quant_pattern(torch.nn.ReLU6)
@register_quant_pattern(torch.nn.Sigmoid)
@register_quant_pattern(torch.nn.Tanh)
@register_quant_pattern(torch.adaptive_avg_pool1d)
@register_quant_pattern(torch.nn.functional.adaptive_avg_pool2d)
@register_quant_pattern(torch.nn.functional.adaptive_avg_pool3d)
@register_quant_pattern(torch.nn.functional.dropout)
@register_quant_pattern(torch.nn.functional.hardsigmoid)
@register_quant_pattern(torch.nn.functional.hardtanh)
@register_quant_pattern(torch.nn.functional.hardtanh_)
@register_quant_pattern(torch.nn.functional.interpolate)
@register_quant_pattern(torch.nn.functional.leaky_relu)
@register_quant_pattern(torch.nn.functional.max_pool1d)
@register_quant_pattern(torch.nn.functional.max_pool2d)
@register_quant_pattern(torch.nn.functional.max_pool3d)
@register_quant_pattern(torch.nn.functional.relu)
@register_quant_pattern(torch.nn.functional.relu6)
@register_quant_pattern(torch.avg_pool1d)
@register_quant_pattern(torch._C._nn.avg_pool2d)
@register_quant_pattern(torch._C._nn.avg_pool3d)
@register_quant_pattern(torch.chunk)
@register_quant_pattern(torch.clamp)
@register_quant_pattern(torch.flatten)
@register_quant_pattern(torch.transpose)
@register_quant_pattern(torch.max)
@register_quant_pattern(torch.mean)
@register_quant_pattern(torch.min)
@register_quant_pattern(torch.repeat_interleave)
@register_quant_pattern(torch.sigmoid)
@register_quant_pattern(torch.sort)
@register_quant_pattern(torch.squeeze)
@register_quant_pattern(torch.stack)
@register_quant_pattern(torch.tanh)
@register_quant_pattern(torch.unsqueeze)
@register_quant_pattern(operator.getitem)
@register_quant_pattern(operator.floordiv)
@register_quant_pattern('chunk')
@register_quant_pattern('clamp')
@register_quant_pattern('contiguous')
@register_quant_pattern('detach')
@register_quant_pattern('detach_')
@register_quant_pattern('hardsigmoid')
@register_quant_pattern('hardsigmoid_')
@register_quant_pattern('leaky_relu')
@register_quant_pattern('leaky_relu_')
@register_quant_pattern('mean')
@register_quant_pattern('numel')
@register_quant_pattern('permute')
@register_quant_pattern('relu')
@register_quant_pattern('relu_')
@register_quant_pattern('repeat')
@register_quant_pattern('repeat_interleave')
@register_quant_pattern('reshape')
@register_quant_pattern('resize_')
@register_quant_pattern('shape')
@register_quant_pattern('sigmoid')
@register_quant_pattern('sigmoid_')
@register_quant_pattern('size')
@register_quant_pattern('squeeze')
@register_quant_pattern('squeeze_')
@register_quant_pattern('tanh')
@register_quant_pattern('tanh_')
@register_quant_pattern('transpose')
@register_quant_pattern('unsqueeze')
@register_quant_pattern('unsqueeze_')
@register_quant_pattern('view')
class CopyNode(QuantizeHandler):
    def convert(self, quantizer, node, load_arg, debug=False):
        return quantizer.quantized_graph.node_copy(node, load_arg(quantized=None))

# Default quantization handler, used for quantization of input and output
# of quantizable objects (e.g. modules and functionals)
class DefaultQuant(QuantizeHandler):
    def convert(self, quantizer, node):
        assert self.all_nodes
        root_module = quantizer.modules['']
        return quantize_node(
            root_module,
            quantizer.quantized_graph,
            node, quantizer.activation_post_process_map[node.name])

class CustomModuleQuantizeHandler(QuantizeHandler):
    def convert(self, quantizer, node, load_arg, debug=False):
        """ Convert a float custom module to quantized custom module
        """
        assert node.op == 'call_module'
        observed_custom_module = quantizer.modules[node.target]
        if node.name in quantizer.activation_post_process_map:
            observed_custom_module.activation_post_process = \
                quantizer.activation_post_process_map[node.name]
        quantized_custom_module_class = \
            get_quantized_custom_module_class(observed_custom_module._FLOAT_MODULE)
        quantized_custom_module = \
            quantized_custom_module_class.from_observed(observed_custom_module)
        parent_name, name = _parent_name(node.target)
        setattr(quantizer.modules[parent_name], name, quantized_custom_module)
        # hardcoded the qunatized input to be None (take whatever is in the environemnt),
        # we can extend this
        # if there is a need, e.g. get the indexes of quantized inputs from some
        # module attribute like module._QUANTIZED_INPUT_INDEXES
        return quantizer.quantized_graph.node_copy(node, load_arg(quantized=None))

class StandaloneModuleQuantizeHandler(QuantizeHandler):
    """ Converts an observed standalone module to quantized standalone module
    by calling convert_fx on the observed standalone module.
    """
    def convert(self, quantizer, node, load_arg, debug=False):
        assert node.op == 'call_module'
        qconfig = quantizer.qconfig_map[node.name]
        convert = torch.quantization.quantize_fx._convert_standalone_module_fx
        observed_standalone_module = quantizer.modules[node.target]
        quantized_standalone_module = convert(observed_standalone_module, debug=debug)
        parent_name, name = _parent_name(node.target)
        # update the modules dict
        setattr(quantizer.modules[parent_name], name, quantized_standalone_module)
        quantizer.modules[node.target] = quantized_standalone_module
        return quantizer.quantized_graph.node_copy(node, load_arg(quantized=None))
