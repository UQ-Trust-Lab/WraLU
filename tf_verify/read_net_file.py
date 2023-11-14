import re
from typing import List, Optional

import numpy as np
import onnx
from numpy import ndarray


def read_tensorflow_net(net_file_path: str, input_length: int, is_trained_with_pytorch: bool, is_gpupoly: bool):
    import tensorflow as tf
    tf = tf.compat.v1 if tf.__version__[0] == '2' else tf
    tf.InteractiveSession().as_default()
    tf.disable_eager_execution()

    def myConst(vec):
        return tf.constant(vec.tolist(), dtype=tf.float64)

    def parseVec(net):
        return np.array(eval(net.readline()[:-1]))

    def numel(x):
        def product(iterable: List):
            product = 1
            for x in iterable:
                product *= x
            return product

        return product([int(i) for i in x.shape])

    def change_format(arg, repl):
        for a in repl:
            arg = arg.replace(a + "=", "'" + a + "':")
        return eval("{" + arg + "}")

    def extract_numbers(text: str, name: str) -> Optional[ndarray]:
        text = re.search(f"{name}=\[(.+?)\]", text)
        if text is None:
            return None
        text = text.group(1).split(',')
        return np.asarray([np.float64(word) for word in text])

    def permutation(W, h, w, c):
        m = np.zeros((h * w * c, h * w * c))
        column = 0
        for i in range(h * w):
            for j in range(c):
                m[i + j * h * w, column] = 1
                column += 1

        return np.matmul(W, m)

    mean, std = .0, .0
    net_file = open(net_file_path, 'r')
    x = tf.placeholder(tf.float64, [input_length], name="x")
    y, z1, z2 = None, None, None
    last_layer = None
    h, w, c = None, None, None
    is_conv = False

    while True:
        current_line = net_file.readline()[:-1]

        if "Normalize" in current_line:
            mean = extract_numbers(current_line, "mean")
            std = extract_numbers(current_line, "std")
        elif "ParSum1" in current_line:
            z1 = x
        elif "ParSum2" in current_line:
            z2, x = x, z1
        elif "ParSumComplete" in current_line:
            x = tf.add(z2, x)
        elif "ParSumReLU" in current_line:
            x = tf.nn.relu(tf.add(z2, x))
        elif "SkipNet1" in current_line:
            y = x
        elif "SkipNet2" in current_line:
            x, y = y, x
        elif "SkipCat" in current_line:
            x = tf.concat([y, x], 1)
        elif current_line in ["ReLU", "Sigmoid", "Tanh", "Sign", "Affine", "LeakyRelu"]:
            if last_layer in ["Conv2D", "ParSumComplete", "ParSumReLU"] and is_trained_with_pytorch and not is_gpupoly:
                W = myConst(permutation(parseVec(net_file), h, w, c).transpose())
            else:
                W = myConst(parseVec(net_file).transpose())
            b = parseVec(net_file)
            b = myConst(b)
            x = tf.nn.bias_add(tf.matmul(tf.reshape(x, [1, numel(x)]), W), b)
            if current_line == "ReLU":
                x = tf.nn.relu(x)
            elif current_line == "Sigmoid":
                x = tf.nn.sigmoid(x)
            elif current_line == "LeakyRelu":
                x = tf.nn.leaky_relu(x)
            elif current_line == "Sign":
                x = tf.math.sign(x)
            elif current_line == "Tanh":
                x = tf.nn.tanh(x)
            else:
                raise Exception(f"Unsupported activation: {current_line}")
        elif current_line == "MaxPooling2D":
            maxpool_line = net_file.readline()[:-1]
            if "stride" in maxpool_line:
                args = change_format(maxpool_line, ["input_shape", "pool_size", "stride", "padding"])
                stride = [1] + args['stride'] + [1]
            else:
                args = change_format(maxpool_line, ["input_shape", "pool_size"])
                stride = [1] + args['pool_size'] + [1]
            if "padding" in maxpool_line and args["padding"] == 1:
                padding_arg = "SAME"
            else:
                padding_arg = "VALID"
            ksize = [1] + args['pool_size'] + [1]
            x = tf.nn.max_pool(tf.reshape(x, [1] + args["input_shape"]), padding=padding_arg, strides=stride,
                               ksize=ksize)
        elif current_line == "Conv2D":
            is_conv = True
            line = net_file.readline()
            start = 0
            if "ReLU" in line:
                start = 5
            elif "Sigmoid" in line:
                start = 8
            elif "Tanh" in line:
                start = 5
            elif "Sign" in line:
                start = 5
            elif "Affine" in line:
                start = 7

            if 'padding' in line:
                args = change_format(line[start:-1], ["filters", "input_shape", "kernel_size", "stride", "padding"])
            else:
                args = change_format(line[start:-1], ["filters", "input_shape", "kernel_size"])

            W = myConst(parseVec(net_file))

            if "padding" in line and args["padding"] >= 1:
                padding_arg = "SAME"
            else:
                padding_arg = "VALID"

            if "stride" in line:
                stride_arg = [1] + args["stride"] + [1]
            else:
                stride_arg = [1, 1, 1, 1]

            x = tf.nn.conv2d(tf.reshape(x, [1] + args["input_shape"]), filter=W, strides=stride_arg,
                             padding=padding_arg)

            b = myConst(parseVec(net_file))
            h, w, c = [int(i) for i in x.shape][1:]
            x = tf.nn.bias_add(x, b)
            if "ReLU" in line:
                x = tf.nn.relu(x)
            elif "Sigmoid" in line:
                x = tf.nn.sigmoid(x)
            elif "Tanh" in line:
                x = tf.nn.tanh(x)
            elif "Sign" in line:
                x = tf.math.sign(x)
            else:
                raise Exception("Unsupported activation: ", {current_line})
        elif current_line == "":
            break
        else:
            raise Exception("Unsupported Operation: ", current_line)
        last_layer = current_line

    model = x
    return model, is_conv, mean, std


def read_onnx_net(net_file):
    onnx_model = onnx.load(net_file)

    onnx.checker.check_model(onnx_model)

    is_conv = False

    for node in onnx_model.graph.node:
        if node.op_type == 'Conv':
            is_conv = True
            break

    return onnx_model, is_conv
