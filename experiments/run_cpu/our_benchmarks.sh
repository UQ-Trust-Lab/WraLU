#!/bin/bash
cd ../

file=experiment_normal_cpu.py
net_files=(
  "../nets/onnx/mnist/mnist_relu_2_1024_96.onnx"
  "../nets/onnx/fmnist/fashion_mnist_relu_2_1024_84.onnx"
  "../nets/onnx/emnist/emnist_relu_2_1024_86.onnx"
  "..nets/onnx/cifar10/cifar10_relu_2_1024_55.onnx"
  "../nets/onnx/mnist/mnist_relu_4_1024_96.onnx"
  "..nets/onnx/fmnist/fashion_mnist_relu_4_1024_84.onnx"
  "..nets/onnx/emnist/emnist_relu_4_1024_88.onnx"
  "..nets/onnx/cifar10/cifar10_relu_4_1024_54.onnx"
  "..nets/onnx/mnist/mnist_convBigRELU__Point_98.onnx"
  "..nets/onnx/fmnist/fashion_mnist_convBigRELU__Point_91.onnx"
  "..nets/onnx/emnist/emnist_convBigRELU__Point_93.onnx"
  "..nets/onnx/cifar10/cifar10_convBigRELU__Point_68.onnx"
)
datasets=(
  "mnist"
  "fashion_mnist"
  "emnist"
  "cifar10"
  "mnist"
  "fashion_mnist"
  "emnist"
  "cifar10"
  "mnist"
  "fashion_mnist"
  "emnist"
  "cifar10"
)
epsilons=(
  "0.025"
  "0.025"
  "0.025"
  "0.002"
  "0.015"
  "0.015"
  "0.015"
  "0.0015"
  "0.025"
  "0.007"
  "0.014"
  "0.003"
)
cutoffs=(
        "0.03"
        "0.03"
        "0.05"
        "0.05"
        "0.05"
        "0.05"
        "0.05"
        "0.05"
        "0.05"
        "0.05"
        "0.05"
        "0.05"
        )

for i in "${!net_files[@]}"; do
  domain=deeppoly
  run_cmd="python3 $file --domain $domain --dataset ${datasets[$i]} --net_file ${net_files[$i]} --epsilon ${epsilons[$i]}"
#  $run_cmd

  domain=refinepoly
  convex_method=triangle
#  $run_cmd --convex_method $convex_method --domain $domain

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