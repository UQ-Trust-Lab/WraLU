#!/bin/bash
cd ../

pthn=experiment_normal_cpu.py
net_file=../nets/onnx/mnist/mnist_relu_3_50.onnx
epsilon=0.03
dataset=mnist
domain=deeppoly
run_cmd="python3 $pthn --domain $domain --dataset $dataset --net_file $net_file --epsilon $epsilon"

# Run first command
#$run_cmd

# Run second command
domain=refinepoly
convex_method=triangle
#$run_cmd --convex_method $convex_method --domain $domain

# Run third command
methods=("sci")
ns=(20)
ks=(3)
s=1
cutoff=0.00
for method in "${methods[@]}"; do
  for n in "${ns[@]}"; do
    for k in "${ks[@]}"; do
      if [[ "$n" -eq 20 && "$k" -eq 4 ]] || [[ "$n" -eq 100 && "$k" -eq 3 ]]; then
        continue
      fi
      $run_cmd --ns "$n" --k "$k" --s "$s" --convex_method "$method" --domain "$domain" --use_cutoff_of "$cutoff" --numerical_focus 2
    done
  done
done

cd ./run_cpu || exit