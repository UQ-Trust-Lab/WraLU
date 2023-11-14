#!/bin/bash
cd ../

pthn=experiment_normal_cpu.py
net_file=../nets/onnx/mnist/mnist_relu_3_50.onnx
epsilon=0.03
dataset=mnist
domain=deeppoly
run_cmd="python3 $pthn --domain $domain --dataset $dataset --net_file $net_file --epsilon $epsilon"

domain=refinepoly

methods=("fast")

for method in "${methods[@]}"; do
  $run_cmd --ns 20 --k 3 --s 1 --convex_method "$method" --domain "$domain" --use_cutoff_of 0.00
  $run_cmd --ns 20 --k 3 --s 2 --convex_method "$method" --domain "$domain" --use_cutoff_of 0.00

  $run_cmd --ns 20 --k 4 --s 1 --convex_method "$method" --domain "$domain" --use_cutoff_of 0.00
  $run_cmd --ns 20 --k 4 --s 2 --convex_method "$method" --domain "$domain" --use_cutoff_of 0.00

  $run_cmd --ns 50 --k 3 --s 1 --convex_method "$method" --domain "$domain" --use_cutoff_of 0.00
  $run_cmd --ns 50 --k 3 --s 2 --convex_method "$method" --domain "$domain" --use_cutoff_of 0.05

  $run_cmd --ns 50 --k 4 --s 1 --convex_method "$method" --domain "$domain" --use_cutoff_of 0.00
  $run_cmd --ns 50 --k 4 --s 2 --convex_method "$method" --domain "$domain" --use_cutoff_of 0.05
done

methods=("sci" "sciplus")

for method in "${methods[@]}"; do
  $run_cmd --ns 20 --k 3 --s 1 --convex_method "$method" --domain "$domain" --use_cutoff_of 0.00
  $run_cmd --ns 20 --k 3 --s 2 --convex_method "$method" --domain "$domain" --use_cutoff_of 0.00

  $run_cmd --ns 20 --k 4 --s 1 --convex_method "$method" --domain "$domain" --use_cutoff_of 0.00
  $run_cmd --ns 20 --k 4 --s 2 --convex_method "$method" --domain "$domain" --use_cutoff_of 0.00

  $run_cmd --ns 20 --k 5 --s 1 --convex_method "$method" --domain "$domain" --use_cutoff_of 0.00
  $run_cmd --ns 20 --k 5 --s 2 --convex_method "$method" --domain "$domain" --use_cutoff_of 0.00
  $run_cmd --ns 20 --k 5 --s 3 --convex_method "$method" --domain "$domain" --use_cutoff_of 0.05

  $run_cmd --ns 50 --k 3 --s 1 --convex_method "$method" --domain "$domain" --use_cutoff_of 0.00
  $run_cmd --ns 50 --k 3 --s 2 --convex_method "$method" --domain "$domain" --use_cutoff_of 0.05

  $run_cmd --ns 50 --k 4 --s 1 --convex_method "$method" --domain "$domain" --use_cutoff_of 0.00
  $run_cmd --ns 50 --k 4 --s 2 --convex_method "$method" --domain "$domain" --use_cutoff_of 0.05

  $run_cmd --ns 50 --k 5 --s 1 --convex_method "$method" --domain "$domain" --use_cutoff_of 0.00
  $run_cmd --ns 50 --k 5 --s 2 --convex_method "$method" --domain "$domain" --use_cutoff_of 0.05
  $run_cmd --ns 50 --k 5 --s 3 --convex_method "$method" --domain "$domain" --use_cutoff_of 0.05
done

cd ./run_cpu || exit