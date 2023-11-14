import csv
import itertools
import re
from typing import List

import numpy as np

from experiments.arguments import ERANArgs, ERANParser
from tf_verify.analyzer import layers
from tf_verify.config import config
from tf_verify.eran import ERAN
from tf_verify.onnx_translator import ONNXTranslator
from tf_verify.optimizer import Optimizer
from tf_verify.read_net_file import read_onnx_net


def initialise_args():
    args = ERANArgs()
    parser = ERANParser(args)
    parser.set_args(args)
    args.prepare_and_check()
    args.print_args()
    args.synchronize_eran_config()
    return args


def initialise_eran_model(net_file_path):
    model, is_conv = read_onnx_net(net_file_path)
    eran = ERAN(model, is_onnx=True)

    return eran, is_conv


def initialise_gpupoly_model(net_file_path):
    model, is_conv = read_onnx_net(net_file_path)
    translator = ONNXTranslator(model, True)
    operations, resources = translator.translate()
    optimizer = Optimizer(operations, resources)
    nn = layers()
    network, relu_layer_ids, gpu_layers_num = optimizer.get_gpupoly(nn)

    return network, nn, relu_layer_ids, gpu_layers_num, is_conv


def initialise_samples_data(dataset):
    means, stds = get_means_stds(dataset)
    samples = get_samples(dataset)
    return samples, means, stds


def parse_net_name(net_file_path):
    net_file_path = net_file_path.split('/')[-1].split('.')
    if len(net_file_path) > 2:
        net_file_path = ".".join(net_file_path[i] for i in range(len(net_file_path) - 1))
    else:
        net_file_path = net_file_path[0]
    return net_file_path


def skip_sample(i, ignored_samples):
    return i in ignored_samples or (config.from_test is not None and i < config.from_test)
    # return (config.domain == "refinepoly" and i in ignored_samples) \
    #        or (config.domain == "refinegpupoly" and i in ignored_samples) \
    #        or (config.from_test is not None and i < config.from_test)


def stop_experiment(num):
    return config.num_tests is not None and num >= config.num_tests


def get_inputs(test, means, stds, dataset, domain, is_conv):
    image = np.float64(test[1:]) / np.float64(255)
    specLB, specUB = np.copy(image), np.copy(image)
    normalize(specLB, means, stds, dataset, domain, is_conv)
    normalize(specUB, means, stds, dataset, domain, is_conv)
    label = int(test[0])
    return image, specLB, specUB, label


def get_means_stds(dataset):
    if dataset in {"mnist", "fashion_mnist", "emnist"}:
        return [0.], [1.]
    elif dataset == "cifar10":
        return [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
    elif dataset == "acasxu":
        return [1.9791091e+04, 0.0, 0.0, 650.0, 600.0], [60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0]
    else:
        raise NotImplementedError(f"Dataset {dataset} not supported.")


def get_samples(dataset):
    if dataset == "cifar10":
        csvfile = open(f'../data/{dataset}_test_5000.csv', 'r')
        print("[INFO] CIFAR10: use the first 5000 examples.")
        return csv.reader(csvfile, delimiter=',')
    elif dataset == "mnist":
        csvfile = open(f'../data/{dataset}_test_full.csv', 'r')
        print("[INFO] MNIST: use full examples.")
        return csv.reader(csvfile, delimiter=',')
    elif dataset == "fashion_mnist":
        csvfile = open(f'../data/{dataset}_test_2000.csv', 'r')
        print("[INFO] FashionMNIST: use the first 2000 examples.")
        return csv.reader(csvfile, delimiter=',')
    elif dataset == "emnist":
        csvfile = open(f'../data/{dataset}_test_2000.csv', 'r')
        print("[INFO] EMNIST: use the first 2000 examples.")
        return csv.reader(csvfile, delimiter=',')
    elif dataset == "acasxu":
        print("[INFO] ACASXU: set input box and output constraints manually.")
        return None
    else:
        raise NotImplementedError(f"Dataset {dataset} not supported.")




def normalize(image, means: List[float], stds: List[float], dataset: str, domain: str, is_conv: bool = False):
    # normalization taken out of the network
    if dataset == 'acasxu':
        for i in range(len(image)):
            image[i] -= means[i]
            if stds is not None:
                image[i] /= stds[i]

    elif dataset in {'mnist', 'fashion_mnist', 'emnist'}:
        for i in range(len(image)):
            image[i] = (image[i] - means[0]) / stds[0]

    elif dataset == 'cifar10':
        tmp = np.zeros(3072)
        count = 0
        for i in range(1024):
            tmp[count] = (image[count] - means[0]) / stds[0]
            count = count + 1
            tmp[count] = (image[count] - means[1]) / stds[1]
            count = count + 1
            tmp[count] = (image[count] - means[2]) / stds[2]
            count = count + 1

        is_gpupoly = (domain == 'gpupoly' or domain == 'refinegpupoly')
        if not is_gpupoly:
            for i in range(3072):
                image[i] = tmp[i]
            # for i in range(1024):
            #    image[i*3] = tmp[i]
            #    image[i*3+1] = tmp[i+1024]
            #    image[i*3+2] = tmp[i+2048]
        else:
            count = 0
            for i in range(1024):
                image[i] = tmp[count]
                count = count + 1
                image[i + 1024] = tmp[count]
                count = count + 1
                image[i + 2048] = tmp[count]
                count = count + 1

    else:
        raise NotImplementedError(f"Dataset {dataset} not supported.")
    return image

def denormalize(image, means, stds, dataset):
    if dataset in ['mnist', 'fashion']:
        for i in range(len(image)):
            image[i] = image[i] * stds[0] + means[0]
    elif dataset == 'cifar10':
        count = 0
        tmp = np.zeros(3072)
        for _ in range(1024):
            tmp[count] = image[count] * stds[0] + means[0]
            count += 1
            tmp[count] = image[count] * stds[1] + means[1]
            count += 1
            tmp[count] = image[count] * stds[2] + means[2]
            count += 1

        for i in range(3072):
            image[i] = tmp[i]


def config_input_box_and_output_constraints(specnumber):
    config.input_box = '../data/acasxu/specs/acasxu_prop_' + str(specnumber) + '_input_prenormalized.txt'
    config.output_constraints = '../data/acasxu/specs/acasxu_prop_' + str(specnumber) + '_constraints.txt'


def get_output_constraints(file_path: str) -> List:
    with open(file_path, 'r') as f:
        lines = f.readlines()  # AND

    num_labels = int(lines[0])  # The first line is a number of labels
    and_list = []
    for line in lines[1:]:
        property = re.split(' +', line)

        labels = []  # OR
        for i, item in enumerate(property):
            if not item.startswith('y'):
                break
            # Remove the first character 'y' and '\n', and convert to int
            idx = int(item.replace('\n', '').replace('y', ''))
            labels.append(idx)

        operator = property[i].replace('\n', '')
        i += 1
        if operator in {'min', 'max'}:
            for other_label in range(num_labels):
                if other_label not in labels:
                    if operator == 'min':
                        and_list.append([(other_label, label, 0) for label in labels])
                    else:
                        and_list.append([(label, other_label, 0) for label in labels])
        elif operator in {'notmin', 'notmax'}:
            assert len(labels) == 1, f"notmin/notmax makes only sense with one label, but got {labels}"
            others = filter(lambda x: x not in labels, range(num_labels))
            label = labels[0]
            if operator == 'notmin':
                and_list.append([(label, other, 0) for other in others])
            else:
                and_list.append([(other, label, 0) for other in others])
        elif operator in {'<', '>'}:
            label2 = int(property[i].replace('\n', '').replace('y', ''))
            if operator == '<':
                and_list.append([(label2, label, 0) for label in labels])
            else:
                and_list.append([(label, label2, 0) for label in labels])
        elif operator == '<=':
            try:
                number = float(property[i].replace('\n', ''))
            except Exception as e:
                print(e)
                raise Exception(f"<= makes only sense with a float, but got {property[i]}")
            and_list.append([(label, -1, number) for label in labels])
        else:
            raise NotImplementedError(f"Operator {operator} not implemented")

    return and_list


def get_input_boxes(file_path: str) -> List:
    with open(file_path, 'r') as f:
        lines = f.read()
    lines = lines.split('\n')
    intervals_list = []
    for line in lines:
        if line != "":
            interval_strings = re.findall("\[-?\d*\.?\d+, *-?\d*\.?\d+\]", line)
            intervals = []
            for interval in interval_strings:
                interval = interval.replace('[', '').replace(']', '')
                [lb, ub] = interval.split(",")
                intervals.append((float(lb), float(ub)))
            intervals_list.append(intervals)

    boxes = itertools.product(*intervals_list)  # return every combination
    return list(boxes)


def init_domain(domain:str):
    if domain == 'refinepoly':
        return 'deeppoly'
    elif domain == 'refinegpupoly':
        return 'gpupoly'
    elif domain == 'refinezono':
        return 'deepzono'
    elif domain in {'deeppoly', 'deepzono', 'gpupoly'}:
        return domain
    else:
        raise NotImplementedError(f"Domain {domain} not implemented")