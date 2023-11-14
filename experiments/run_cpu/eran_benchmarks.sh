#!/bin/bash
cd ../

file=experiment_normal_cpu.py
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
cutoffs=(
        "0.0"
        "0.0"
        "0.15"
        "0.0"
        "0.05"
        "0.05"
        "0.2"
        "0.0"
        "0.5"
        "0.2"
        )

for i in "${!net_files[@]}"; do
  domain=deeppoly
  run_cmd="python3 $file --domain $domain --dataset ${datasets[$i]} --net_file ${net_files[$i]} --epsilon ${epsilons[$i]}"
  $run_cmd

  domain=refinepoly
  convex_method=triangle
  $run_cmd --convex_method $convex_method --domain $domain

  methods=("fast" "sci" "sciplus")
  ns=(20 100)
  ks=(3 4)
  s=1
  for method in "${methods[@]}"; do
    for n in "${ns[@]}"; do
      for k in "${ks[@]}"; do
        if [[ "$n" -eq 20 && "$k" -eq 4 ]] || [[ "$n" -eq 100 && "$k" -eq 3 ]]; then
          continue
        fi
        $run_cmd --ns "$n" --k "$k" --s "$s" --convex_method "$method" --domain "$domain" --use_cutoff_of "${cutoffs[$i]}"
      done
    done
  done
done

cd ./run_cpu || exit