import warnings
from typing import Tuple, Dict, List

import numpy as np
import onnx
from onnx import numpy_helper

from tf_verify.config import config


def _parse_onnxshape_to_inttuple(onnx_shape: onnx.TensorShapeProto) -> Tuple[int, ...]:
    shape = tuple(map(lambda dim: 1 if dim.dim_value is None else int(dim.dim_value), onnx_shape.dim))
    # no shape means a single value; convert NCHW to NHWC
    return (shape if len(shape) != 4 else (shape[0], shape[2], shape[3], shape[1])) if shape else (1,)


def _parse_shape_nchw_to_nhwc(array):
    assert len(array) == 4, "Unexpected shape size"
    return [array[0], array[2], array[3], array[1]]


def _parse_nparray_nchw_to_nhwc(array: np.ndarray) -> np.ndarray:
    return array if array.ndim != 4 else array.transpose((0, 2, 3, 1))


def _parse_index_nchw_to_nhwc(index: int) -> int:
    assert 0 <= index <= 3, f"index out of range: {index}"
    return (0 if index == 0 else 3) if index == 1 else index - 1


def _reshape_nhwc(shape_in: Tuple[int, ...], shape_out: Tuple[int, ...]) -> np.ndarray:
    ndim_in, ndim_out = len(shape_in), len(shape_out)
    total_in, total_out = np.prod(shape_in[1:ndim_in]), np.prod(shape_out[1:ndim_out])
    assert total_in == total_out, "Reshape doesn't have same number of neurons before and after"
    array = np.asarray(range(total_in)).reshape(shape_in[1:ndim_in])
    if array.ndim == 3:
        array = array.transpose((2, 0, 1))
    array = array.reshape(shape_out[1:ndim_out])
    return array.transpose((1, 2, 0)) if array.ndim == 3 else array


def prepare_model(model: onnx.ModelProto) -> Tuple[
    Dict[str, Tuple[int, ...]], Dict[str, np.ndarray], Dict[str, onnx.ValueInfoProto], Dict[str, onnx.ValueInfoProto],
    List[str]]:
    shapes_map: Dict[str, Tuple[int, ...]] = {}
    constants_map: Dict[str, np.ndarray] = {}
    input_nodes_map: Dict[str, onnx.ValueInfoProto] = {}
    output_nodes_map: Dict[str, onnx.ValueInfoProto] = {}
    placeholder_names: List[str] = []
    NP_OPERATIONS = {"Add": np.add, "Sub": np.subtract, "Mul": np.multiply, "Div": np.divide}
    # handle parameters
    for initializer in model.graph.initializer:
        constants = _parse_nparray_nchw_to_nhwc(numpy_helper.to_array(initializer))  # weights or bias of network
        constants_map[initializer.name], shapes_map[initializer.name] = constants, constants.shape

    for input_name in model.graph.input:
        placeholder_names.append(input_name.name)
        if input_name.name not in shapes_map:
            shapes_map[input_name.name] = _parse_onnxshape_to_inttuple(input_name.type.tensor_type.shape)
            input_nodes_map[input_name.name] = input_name
    for node in model.graph.node:
        op_type = node.op_type

        input_name, output_name = node.input[0] if node.input else [], node.output[0]

        output_nodes_map[output_name] = node
        for ipt_name in node.input:
            input_nodes_map[ipt_name] = node

        if op_type == "Constant":
            constants = _parse_nparray_nchw_to_nhwc(numpy_helper.to_array(node.attribute[0].t)).copy()
            constants_map[output_name], shapes_map[output_name] = constants, constants.shape

        elif op_type in ["Add", "Sub", "Mul", "Div"]:
            shapes_map[output_name] = shapes_map[input_name]
            input_name2 = node.input[1]
            if not (input_name in constants_map and input_name in constants_map):
                continue
            constants_map[output_name] = NP_OPERATIONS[op_type](constants_map[input_name], constants_map[input_name2])

        elif op_type == "Flatten":
            shapes_map[output_name] = (1, int(np.prod(shapes_map[input_name][1:])))

        elif op_type in ["MatMul", "Gemm"]:
            transA, transB = 0, 0
            for attribute in node.attribute:
                if attribute.name == "transA":
                    transA = attribute.i
                elif attribute.name == "transB":
                    transB = attribute.i
            # shape of inputs and weights
            input_shape_A = ([1] if len(shapes_map[input_name]) == 1 else []) + list(shapes_map[input_name])
            input_shape_B = list(shapes_map[node.input[1]]) + ([1] if len(shapes_map[node.input[1]]) == 1 else [])
            shapes_map[output_name] = (input_shape_A[transA], input_shape_B[1 - transB])

        elif op_type in ["Conv", "MaxPool", "AveragePool"]:
            input_shape, output_shape = shapes_map[input_name], []

            require_kernel_shape = (node.op_type in ["MaxPool", "AveragePool"])
            if not require_kernel_shape:
                filter_shape = shapes_map[node.input[1]]
                kernel_shape = filter_shape[1:-1]

            strides = [1] * 2
            padding = [0] * 0
            dilations = [1] * 2
            ceil_mode = 0
            auto_pad = 'NOTSET'
            group = 1

            for attribute in node.attribute:
                if attribute.name == 'strides':
                    strides = attribute.ints
                elif attribute.name == 'pads':
                    padding = attribute.ints
                elif attribute.name == 'auto_pad':
                    auto_pad = attribute.s
                elif attribute.name == 'kernel_shape':
                    kernel_shape = attribute.ints
                elif attribute.name == 'dilations':
                    dilations = attribute.ints
                elif attribute.name == 'group':
                    group = attribute.i
                elif attribute.name == 'ceil_mode':
                    ceil_mode = attribute.i
            effective_kernel_shape = [(kernel_shape[i] - 1) * dilations[i] + 1 for i in range(len(kernel_shape))]
            output_shape.append(input_shape[0])

            for i in range(len(kernel_shape)):
                effective_input_size = input_shape[1 + i] + padding[i] + padding[i + len(kernel_shape)]
                if ceil_mode == 1:
                    strided_kernel_positions = int(
                        np.ceil((effective_input_size - effective_kernel_shape[i]) / float(strides[i])))
                else:
                    strided_kernel_positions = int(
                        np.floor((effective_input_size - effective_kernel_shape[i]) / strides[i]))
                output_shape.append(1 + strided_kernel_positions)
            output_shape.append(input_shape[3] if require_kernel_shape else filter_shape[0])
            shapes_map[output_name] = tuple(output_shape)
        elif op_type in ["Relu", "Sigmoid", "Tanh", "Softmax", "BatchNormalization", "LeakyRelu"]:
            shapes_map[output_name] = shapes_map[input_name]

        # Gather is for the moment solely for shapes
        elif node.op_type == "Gather":
            axis = 0
            for attribute in node.attribute:
                axis = attribute.i
            if input_name in constants_map and node.input[1] in constants_map:
                data = constants_map[input_name]
                indexes = constants_map[node.input[1]]
                constants_map[output_name] = np.take(data, indexes, axis)

            if input_name in shapes_map and node.input[1] in shapes_map:
                r = len(shapes_map[input_name])
                q = len(shapes_map[node.input[1]])
                out_rank = q + r - 1
                if out_rank == 0:
                    shapes_map[output_name] = shapes_map[node.input[1]]
                else:
                    output_shape = []
                    for i in range(out_rank):
                        if i < axis:
                            output_shape.append(shapes_map[input_name][i])  # i < axis < r
                        elif i >= axis and i < axis + q:
                            output_shape.append(shapes_map[input_name][i - axis])  # i - axis < q
                        else:
                            output_shape.append(shapes_map[input_name][i - q + 1])  # i < out_rank < q + r - 1
                    shapes_map[output_name] = output_shape
        elif node.op_type == "Shape":
            if input_name in shapes_map:
                constants_map[output_name] = shapes_map[input_name]
                shapes_map[output_name] = [len(shapes_map[input_name])]

        # elif node.op_type == "Cast":
        # shape_map[output_name] = shape_map[input_name]
        # print("CASTING ", input_name, shape_map[input_name], shape_map[output_name])

        elif node.op_type == "Reshape":
            # print("RESHAPE ", node.input, node.output)
            if node.input[1] in constants_map:
                total = 1
                replace_index = -1
                for index in range(len(constants_map[node.input[1]])):
                    if constants_map[node.input[1]][index] == -1:
                        replace_index = index
                    else:
                        total *= constants_map[node.input[1]][index]

                if replace_index != -1:
                    constants_map[node.input[1]][replace_index] = np.prod(shapes_map[input_name]) / total

                if len(constants_map[node.input[1]]) == 4:
                    shapes_map[output_name] = [constants_map[node.input[1]][0], constants_map[node.input[1]][2],
                                               constants_map[node.input[1]][3], constants_map[node.input[1]][1]]
                else:
                    shapes_map[output_name] = constants_map[node.input[1]]

        elif node.op_type == "Unsqueeze":
            if input_name in shapes_map:
                axis = node.attribute[0].ints
                output_shape = list(shapes_map[input_name])
                if input_name in constants_map:
                    constants_map[output_name] = constants_map[input_name]
                for i in axis:
                    output_shape.insert(i, 1)
                    if input_name in constants_map:
                        constants_map[output_name] = np.expand_dims(constants_map[output_name], axis=i)
                shapes_map[output_name] = output_shape

        elif node.op_type == "Concat":
            all_constant = True
            n_dim = len(shapes_map[input_name])
            if n_dim > 2:
                axis = _parse_index_nchw_to_nhwc(node.attribute[0].i)
            else:
                axis = node.attribute[0].i
            for input_name in node.input:
                if not input_name in constants_map:
                    all_constant = False
                    break
            if all_constant:
                constants_map[output_name] = np.concatenate([constants_map[input] for input in node.input],
                                                            axis=axis)
            all_shape_known = True
            for input_name in node.input:
                if not input_name in shapes_map:
                    all_shape_known = False
                    break
            assert all_shape_known, "Unknown shape for at least one node input!"
            new_axis_size = 0
            for input_name in node.input:
                new_axis_size += shapes_map[input_name][axis]
            shapes_map[output_name] = [shapes_map[input_name][i] if i != axis else new_axis_size for i in
                                       range(len(shapes_map[input_name]))]
            if not all_constant:
                assert axis == n_dim - 1, "ELINA currently only supports concatenation on the channel dimension"

        elif node.op_type == "Tile":
            repeats = _parse_shape_nchw_to_nhwc(constants_map[node.input[1]])
            input_shape = list(shapes_map[input_name])
            assert len(repeats) == len(input_shape), "Expecting one repeat factor per dimension"
            output_shape = [factor * size for factor, size in zip(repeats, input_shape)]
            shapes_map[output_name] = output_shape

            repeat_index = np.where(np.array(repeats) != 1)[0]
            assert len(repeat_index) == 1, "ELINA backend currently only supports repeats for one dimension"
            repeat_index = repeat_index.item()
            assert repeat_index == 1, "ELINA backend currently only supports repeats for the first dimension"
            assert input_shape[0] == 1, "ELINA backend currently only supports repeats for dimensions of size 1"

        elif node.op_type == "Expand":
            if node.input[1] in constants_map:
                if len(constants_map[node.input[1]]) == 4:
                    shapes_map[output_name] = [constants_map[node.input[1]][0], constants_map[node.input[1]][2],
                                               constants_map[node.input[1]][3], constants_map[node.input[1]][1]]
                else:
                    shapes_map[output_name] = constants_map[node.input[1]]

                result = np.zeros(shapes_map[output_name]) + constants_map[input_name]
                constants_map[output_name] = result
        elif node.op_type == "Pad":
            input_shape = np.array(shapes_map[input_name])
            for attribute in node.attribute:
                if attribute.name == "pads":
                    padding = np.array(attribute.ints)
                if attribute.name == "mode":
                    assert attribute.s == bytes(b'constant'), "only zero padding supported"
                if attribute.name == "value":
                    assert attribute.f == 0, "only zero padding supported"
            output_shape = np.copy(input_shape)
            input_dim = len(input_shape)
            assert len(padding) == 2 * input_dim
            for i in range(2, input_dim):  # only pad spatial dimensions
                output_shape[i - 1] += padding[i] + padding[i + input_dim]
            shapes_map[output_name] = list(output_shape)
        else:
            assert 0, f"Operations of type {node.op_type} are not yet supported."
    return shapes_map, constants_map, output_nodes_map, input_nodes_map, placeholder_names


class ONNXTranslator:
    def __init__(self, model: onnx.ModelProto, is_gpupoly: bool):
        onnx.checker.check_model(model)
        self.model = model
        self.is_gpupoly = is_gpupoly
        self.nodes = self.model.graph.node
        self.shapes_map, self.constants_map, self.output_nodes_map, self.input_nodes_map, self.placeholder_names = \
            prepare_model(model)

    def translate(self) -> Tuple[List[str], List[Dict[str, Tuple]]]:
        OPERATIONS_IGNORED = ["Pack", "Shape", "StridedSlice", "Prod", "Unsqueeze", "Softmax", "Concat", "Flatten",
                              "BatchNormalization"]
        # Check if there Are Add/Sub and Div/Mul layers that can be interpreted as normalization layer
        LAYERS_STOP_NORMALIZATION = ["MatMul", "Gemm", "Conv", "MaxPool", "Relu", "Sigmoid", "Tanh", "LeakyRelu"]

        operation_types: List[str] = ["Placeholder"]
        operation_resources: List[Dict[str, Tuple]] = []
        # placeholder = self.model.graph.input[0]
        placeholder = self._find_input()
        placeholder_shape = self._clean_shape(_parse_onnxshape_to_inttuple(placeholder.type.tensor_type.shape))
        in_out_placeholder = ([], placeholder.name, placeholder_shape)
        operation_resources.append({"deepzono": in_out_placeholder, "deeppoly": in_out_placeholder})

        stop_norm_layer = len(self.nodes)
        extract_mean, extract_std = False, False
        for i, node in enumerate(self.nodes):
            op_type = node.op_type
            if op_type in LAYERS_STOP_NORMALIZATION or (extract_mean and extract_std):
                stop_norm_layer = i
                break
            if op_type in ["Add", "Sub"]:
                extract_mean = True
            if op_type in ["Div", "Mul"]:
                extract_std = True
        extract_norm = extract_std and extract_mean

        reshape_map: Dict[str, str] = {}
        for i, node in enumerate(self.nodes):
            op_type = node.op_type
            # print(f"{op_type}".center(100, "-"))
            if op_type == "Constant":
                continue

            if op_type in OPERATIONS_IGNORED:
                input_name, output_name = node.input[0], node.output[0]
                reshape_map[output_name] = reshape_map[input_name] if input_name in reshape_map else input_name
                continue

            operation_types.append(op_type)
            # take means and stds out of the network
            if (extract_norm and i <= stop_norm_layer and len(operation_types) == 2
                    and op_type in ["Add", "Sub", "Mul", "Div"] and node.output[0] not in self.constants_map):
                constant = self._get_onnx_constants(node)[0].reshape(-1)
                if op_type == "Add":
                    config.mean = np.multiply(constant, -1)
                    print(f"Mean of {config.mean} extracted from network")
                elif op_type == "Sub":
                    config.mean = constant
                    print(f"Mean of {config.mean} extracted from network")
                elif op_type == "Mul":
                    config.std = np.divide(1, constant)
                    print(f"Std of {config.std} extracted from network")
                elif op_type == "Div":
                    config.std = constant
                    print(f"Std of {config.std} extracted from network")
                self._ignore_node(node, operation_types, reshape_map)
                continue
            input_onnx_names = []
            for name in node.input:
                kind = self._get_kind(name)
                if name in reshape_map:
                    name = reshape_map[name]
                if kind == 'Constant':
                    continue
                input_onnx_names.append(name)

            shape = self._clean_shape(self._get_shape(node.output[0]))
            in_out_info = (input_onnx_names, node.output[0], shape)

            padding_merger_dict = {}
            if op_type == "MatMul":
                deepzono_res = deeppoly_res = self.matmul_resources(node) + in_out_info
                operation_resources.append({'deepzono': deepzono_res, 'deeppoly': deeppoly_res})
            elif node.op_type == "Gemm":
                deepzono_res = deeppoly_res = self.gemm_resources(node) + in_out_info
                operation_resources.append({'deepzono': deepzono_res, 'deeppoly': deeppoly_res})
            elif node.op_type in ["Add", "Mul"]:
                left_type, right_type = self._get_kind(node.input[0]), self._get_kind(node.input[1])
                if left_type == 'Constant' and right_type == 'Constant':
                    operation_types.pop()
                elif left_type == 'Constant' or right_type == 'Constant':
                    deepzono_res = deeppoly_res = self._get_onnx_constants(node) + in_out_info
                    operation_resources.append({'deepzono': deepzono_res, 'deeppoly': deeppoly_res})
                else:
                    if node.op_type != "Add":
                        assert 0, "we don't support residual operations other then add"
                    operation_types[-1] = "Resadd"
                    operation_resources.append({'deepzono': in_out_info, 'deeppoly': in_out_info})

            elif node.op_type == "Sub":
                left_type, right_type = self._get_kind(node.input[0]), self._get_kind(node.input[1])
                if left_type == 'Constant' and right_type == 'Constant':
                    assert 0, "we don't support the subraction of two constants yet"
                elif left_type == 'Constant' or right_type == 'Constant':
                    deepzono_res = deeppoly_res = self.sub_resources(node) + in_out_info
                    operation_resources.append({'deepzono': deepzono_res, 'deeppoly': deeppoly_res})
                else:
                    assert 0, "we don't support the ressub yet"
                    operation_types[-1] = "Ressub"
                    operation_resources.append({'deepzono': in_out_info, 'deeppoly': in_out_info})
            elif node.op_type == "Conv":
                filters, bias, image_shape, strides, pad_top, pad_left, pad_bottom, pad_right, kernel_shape = \
                    self.conv_resources(node)
                if node.name in padding_merger_dict:
                    image_shape, in_out_info, pad_top, pad_left, pad_bottom, pad_right = \
                        self.merge_padding(node, padding_merger_dict, in_out_info, pad_top, pad_left, pad_bottom,
                                           pad_right)
                deeppoly_res = (filters, bias, image_shape, strides, pad_top, pad_left, pad_bottom,
                                pad_right) + in_out_info
                deepzono_res = deeppoly_res
                operation_resources.append({'deepzono': deepzono_res, 'deeppoly': deeppoly_res})
            elif node.op_type == "Pad":
                image_shape, pad_top, pad_left, pad_bottom, pad_right = self.pad_resources(node)
                deeppoly_res = (image_shape, pad_top, pad_left, pad_bottom, pad_right) + in_out_info
                deepzono_res = deeppoly_res
                consequent_nodes = [node_i for node_i in self.nodes if output_name in node_i.input]
                can_be_merged = all([node_i.op_type in ["Conv"] for node_i in consequent_nodes])
                if can_be_merged:
                    padding_merger_dict.update({node_i.name: deeppoly_res for node_i in consequent_nodes})
                    self._ignore_node(node, operation_types, reshape_map)
                else:
                    operation_resources.append({'deepzono': deepzono_res, 'deeppoly': deeppoly_res})
            elif node.op_type == "MaxPool" or node.op_type == "AveragePool":
                image_shape, kernel_shape, strides, padding, dilations, pad_top, pad_left, pad_bottom, pad_right, \
                    ceil_mode, storage_order = self.pool_resources(node)
                if node.name in padding_merger_dict:
                    image_shape, in_out_info, pad_top, pad_left, pad_bottom, pad_right = \
                        self.merge_padding(node, padding_merger_dict, in_out_info, pad_top, pad_left, pad_bottom,
                                           pad_right)
                deeppoly_res = (image_shape, kernel_shape, strides, pad_top, pad_left, pad_bottom,
                                pad_right) + in_out_info
                # TODO padding is expected to be string in tf. dilations, auto_pad, ceil_mode, storage_order are unused at the moment
                deepzono_res = deeppoly_res
                operation_resources.append({'deepzono': deepzono_res, 'deeppoly': deeppoly_res})
            elif node.op_type == "Placeholder":
                assert 0, "Placeholder is not in the ONNX graph"
            elif node.op_type in ["Relu", "Sigmoid", "Tanh", "LeakyRelu"]:
                deeppoly_res = self.nonlinearity_resources(node) + in_out_info
                deepzono_res = deeppoly_res
                operation_resources.append({'deepzono': deepzono_res, 'deeppoly': deeppoly_res})

            # Gather is for the moment solely for shapes
            elif node.op_type == "Gather":
                only_shape, image_shape, indexes, axis = self.gather_resources(node)
                if only_shape:
                    self._ignore_node(node, operation_types, reshape_map)
                else:
                    deepzono_res = deeppoly_res + (image_shape, indexes, axis) + in_out_info
                    operation_resources.append({'deepzono': deepzono_res, 'deeppoly': deeppoly_res})

            elif node.op_type == "Expand":
                only_shape, image_shape, to_expand = self.expand_resources(node)
                if only_shape:
                    operation_types.pop()
                else:
                    deeppoly_res = (image_shape, indexes, axis) + in_out_info
                    deepzono_res = deeppoly_res
                    operation_resources.append({'deepzono': deepzono_res, 'deeppoly': deeppoly_res})

            elif node.op_type == "Reshape":
                if node.output[0] in self.input_nodes_map \
                        and self.input_nodes_map[node.output[0]].op_type in ["MatMul", "Gemm"]:

                    self._ignore_node(node, operation_types, reshape_map)

                elif node.output[0] in self.input_nodes_map \
                        and self.input_nodes_map[node.output[0]].op_type in ["Relu", "Sigmoid", "Tanh", "LeakyRelu"] \
                        and self.input_nodes_map[self.input_nodes_map[node.output[0]].output[0]].op_type == "Reshape":
                    # ignore this reshape even in the shape_map
                    self.shapes_map[node.output[0]] = self.shapes_map[node.input[0]]
                    self.shapes_map[self.input_nodes_map[node.output[0]].output[0]] = self.shapes_map[node.input[0]]
                    self._ignore_node(node, operation_types, reshape_map)
                else:
                    shape_in = self._get_shape(node.input[0])
                    shape_out = self._get_shape(node.output[0])
                    if len(shape_in) == 2 and len(shape_out) == 2:
                        self._ignore_node(node, operation_types, reshape_map)
                    else:
                        indexes = _reshape_nhwc(shape_in, shape_out)
                        deeppoly_res = (indexes,) + in_out_info
                        deepzono_res = deeppoly_res
                        operation_resources.append({'deepzono': deepzono_res, 'deeppoly': deeppoly_res})

            elif node.op_type == "Concat":
                n_dim = len(self.shapes_map[node.input[0]])
                if n_dim > 2:
                    axis = _parse_index_nchw_to_nhwc(node.attribute[0].i)
                else:
                    axis = node.attribute[0].i
                assert axis == n_dim - 1, "ELINA backend currently only supports concatenation on the channel dimension"
                channels = []
                for input_node in node.input:
                    channels.append(self._get_shape(input_node)[axis])
                # width = shape[1]
                # height = shape[2]
                operation_resources.append({'deeppoly': (channels,) + in_out_info})

            elif node.op_type == "Tile":
                repeats = _parse_shape_nchw_to_nhwc(self.constants_map[node.input[1]])
                repeat_factor = repeats[repeats != 1].item()
                operation_resources.append({'deeppoly': (repeat_factor,) + in_out_info})

            else:
                assert 0, "Operations of type " + node.op_type + " are not yet supported."

            assert all([0 not in y[-1] for x in operation_resources for y in
                        x.values()]), "Ensure inputs and outpus include no dimensions of size 0"

        return operation_types, operation_resources

    def _find_input(self) -> onnx.ValueInfoProto:
        inputs_dir = {x.name: x for x in self.model.graph.input}
        all_inputs = [x for y in self.nodes for x in y.input]
        [all_inputs.remove(x) for y in self.nodes for x in y.output if x in all_inputs]
        [all_inputs.remove(x.name) for x in self.model.graph.initializer if x.name in all_inputs]
        assert all_inputs[0] in inputs_dir
        return inputs_dir[all_inputs[0]]

    @staticmethod
    def _clean_shape(shape_raw: Tuple[int, ...]):
        shape_cleaned = [1 if x == 0 else x for x in shape_raw]
        if 0 in shape_raw:
            warnings.warn(f"0-sized dimension encountered: {shape_raw} and changed to: {shape_cleaned}", RuntimeWarning)
        return shape_cleaned

    def _ignore_node(self, node: onnx.NodeProto, operation_types: List, reshape_map: Dict):
        operation_types.pop()
        input_name, output_name = node.input[0], node.output[0]
        reshape_map[output_name] = reshape_map[input_name] if input_name in reshape_map else input_name

    def merge_padding(self, node, padding_merger_dict, in_out_info, pad_top, pad_left, pad_bottom, pad_right):
        image_shape, m_pad_top, m_pad_left, m_pad_bottom, m_pad_right, input_node, _, _ = padding_merger_dict[node.name]
        in_out_info = (input_node, in_out_info[1], in_out_info[2])
        pad_top += m_pad_top
        pad_left += m_pad_left
        pad_bottom += m_pad_bottom
        pad_right += m_pad_right
        return image_shape, in_out_info, pad_top, pad_left, pad_bottom, pad_right

    def _get_kind(self, name: str) -> str:
        if name in self.constants_map:
            kind = 'Constant'
        elif name in self.placeholder_names:
            kind = 'Placeholder'
        else:
            kind = self.output_nodes_map[name].op_type
        return kind

    def _get_shape(self, name: str) -> Tuple[int, ...]:
        if name in self.shapes_map:
            return self.shapes_map[name]

    def matmul_resources(self, node: onnx.NodeProto) -> Tuple[np.ndarray]:
        inputs = node.input
        left, right = inputs[0], inputs[1]
        if left in self.constants_map:
            matrix = self.reshape_adjust(right, self.constants_map[left], True)
        else:
            matrix = self.reshape_adjust(left, self.constants_map[right].transpose())
        return matrix,

    def reshape_adjust(self, element: str, matrix: np.ndarray, is_right: bool = False):
        if self._get_kind(element) in ['Reshape', 'Flatten'] and not self.is_gpupoly:
            # TODO check whether it should be triggered for Flatten layers to
            shape_in = self._get_shape(self.output_nodes_map[element].input[0])
            # print(f'shape_in: {shape_in}')
            shape_out = self._get_shape(self.output_nodes_map[element].output[0])
            # print(f'shape_out: {shape_out}')
            indexes = _reshape_nhwc(shape_in, shape_out)
            # indexes = indexes[0]
            inverse_perm = np.arange(len(indexes))[np.argsort(indexes)]
            matrix = matrix[inverse_perm, :] if is_right else matrix[:, inverse_perm]
        return matrix

    def gemm_resources(self, node):
        inputs = node.input
        left = inputs[0]
        right = inputs[1]
        bias = self.constants_map[inputs[2]]

        transA = False
        transB = False
        alpha = 1.0
        beta = 1.0
        for att in node.attribute:
            if 'transA' == att.name:
                transA = att.i == 1
            elif 'transB' == att.name:
                transB = att.i == 1
            elif 'alpha' == att.name:
                alpha = att.f
            elif 'beta' == att.name:
                beta = att.f
            else:
                assert 0, "Unkown attribute " + att.name + " for operation type " + node.op_type

        if left in self.constants_map:
            matrix = self.constants_map[left] if not transA else self.constants_map[left].transpose()
            matrix = self.reshape_adjust(right, matrix, True)

        else:
            matrix = self.constants_map[right].transpose() if not transB else self.constants_map[right]
            matrix = self.reshape_adjust(left, matrix)

        return matrix * alpha, bias * beta

    def _get_onnx_constants(self, node: onnx.NodeProto) -> Tuple[np.ndarray]:
        inputs = node.input
        left, right = inputs[0], inputs[1]
        addend = self.constants_map[left] if left in self.constants_map else self.constants_map[right]
        return addend,

    def sub_resources(self, node):
        inputs = node.input
        left = inputs[0]
        right = inputs[1]

        if left in self.constants_map:
            addend = self.constants_map[left]
            is_minuend = True
        else:
            addend = self.constants_map[right]
            is_minuend = False
        return addend, is_minuend

    def conv_resources(self, node):
        inputs = node.input
        image = inputs[0]
        filters = self.constants_map[node.input[1]].transpose(1, 2, 3, 0)
        if len(node.input) == 3:
            bias = self.constants_map[node.input[2]]
        else:
            bias = np.zeros(filters.shape[3])
        image_shape = self._get_shape(image)[1:]
        pads = [0, 0, 0, 0]
        for attribute in node.attribute:
            if attribute.name == 'strides':
                strides = attribute.ints
            elif attribute.name == 'pads':
                pads = attribute.ints
            elif attribute.name == 'kernel_shape':
                kernel_shape = attribute.ints

        pad_top = pads[0]
        pad_left = pads[1]
        pad_bottom = pads[2]
        pad_right = pads[3]
        # assert pad_top == pad_bottom, 'different padding for top and bottom is not supported in ERAN'
        # assert pad_left == pad_right, 'different padding for left and right is not supported in ERAN'
        return filters, bias, image_shape, strides, pad_top, pad_left, pad_bottom, pad_right, kernel_shape

    def pad_resources(self, node):
        inputs = node.input
        image = inputs[0]
        image_shape = self._get_shape(image)[1:]

        pads = [0, 0, 0, 0]
        for attribute in node.attribute:
            if attribute.name == 'pads':
                pads = attribute.ints

        pad_top = pads[2]
        pad_left = pads[3]
        pad_bottom = pads[6]
        pad_right = pads[7]
        return image_shape, pad_top, pad_left, pad_bottom, pad_right

    def pool_resources(self, node):
        image = node.input[0]

        image_shape = self._get_shape(image)[1:]

        padding = 'NOTSET'
        ceil_mode = 0
        storage_order = 0
        pads = [0, 0, 0, 0]
        dilations = None

        for attribute in node.attribute:
            if attribute.name == 'kernel_shape':
                kernel_shape = attribute.ints
            if attribute.name == 'strides':
                strides = attribute.ints
            elif attribute.name == 'pads':
                pads = attribute.ints
            elif attribute.name == 'dilations':
                dilations = attribute.ints
            elif attribute.name == 'auto_pad':
                padding = attribute.s
            elif attribute.name == 'ceil_mode':
                ceil_mode = attribute.i
            elif attribute.name == 'storage_order':
                storage_order = attribute.i
        pad_top = pads[0]
        pad_left = pads[1]
        pad_bottom = pads[2]
        pad_right = pads[3]
        assert pad_top == pad_bottom, 'different padding for top and bottom is not supported in ERAN'
        assert pad_left == pad_right, 'different padding for left and right is not supported in ERAN'
        return image_shape, kernel_shape, strides, padding, dilations, pad_top, pad_left, pad_bottom, pad_right, ceil_mode, storage_order

    def nonlinearity_resources(self, op):
        return ()

    def gather_resources(self, node):
        inputs = node.input
        image = inputs[0]
        if node.output[0] in self.constants_map:
            only_shape = True
            image_shape, indexes, axis = None, None, None
        else:
            only_shape = False
            image_shape = self._get_shape(image)[1:]
            indexes = self.constants_map[node.input[1]]
            axis = node.attribute[0].i
        return only_shape, image_shape, indexes, axis

    def expand_resources(self, node):
        if node.output[0] in self.constants_map:
            only_shape = True
            image_shape, to_expand = None, None
        else:
            assert 0, "Implementation for 'Expand' is missing."
        return only_shape, image_shape, to_expand
