#!/bin/bash
cd ../

pthn=experiment_normal_cpu.py
run_cmd="python3 $pthn"

domains=("deepzono")
net_files=(
  "../nets/onnx/mnist/mnist_relu_3_50.onnx"
  "../nets/onnx/cifar10/cifar_relu_4_100.onnx"
  "../nets/onnx/mnist/mnist_relu_6_100.onnx"
  "../nets/onnx/cifar10/cifar_relu_6_100.onnx"
  "../nets/onnx/mnist/mnist_convSmallRELU__Point.onnx"
  "../nets/onnx/cifar10/cifar_convSmallRELU__Point.onnx"
  "../nets/onnx/mnist/mnist_relu_9_200.onnx"
  "../nets/onnx/cifar10/cifar_relu_9_200.onnx"
  "../nets/onnx/mnist/mnist_convBigRELU__DiffAI.onnx"
  "../nets/onnx/cifar10/cifar_convBigRELU__DiffAI.onnx"
)
datasets=(
  "mnist"
  "cifar10"
  "mnist"
  "cifar10"
  "mnist"
  "cifar10"
  "mnist"
  "cifar10"
  "mnist"
  "cifar10"
)
epsilons=(
  "0.03"
  "0.001"
  "0.019"
  "0.0007"
  "0.1"
  "0.004"
  "0.012"
  "0.0008"
  "0.305"
  "0.007"
)

for ((k = 0; k < ${#domains[@]}; k++)); do
  domain=${domains[k]}
  for ((i = 0; i < ${#datasets[@]}; i++)); do
    dataset=${datasets[i]}
    net_file=${net_files[i]}
    epsilon=${epsilons[i]}

    $run_cmd --domain $domain --epsilon $epsilon --dataset $dataset --net_file $net_file
  done
done


cd ./run_cpu || exit
