# WraLU: An Approach to Approximate ReLU Hull

This is about to how to use **WraLU** to calculate the **ReLU hull** for neural network verification.

Our paper *ReLU Hull Approximation* has been accepted by 24'POPL. Also you can learn this approach by [a quick introduction](https://zhongkuima.github.io/24popl_relu_hull.html).

We have integrate WraLU to PRIMA (ERAN project, https://github.com/eth-sri/eran), so the process of installation is similar to ERAN. We recommend that install it according to our guide, because they have different installation paths.

## Installation Guide

### Install WraLU

First, cd to `WraLU` directory and bash `install.sh` or manually install the necessary libraries (refer to https://github.com/eth-sri/eran).

Our core algorithm is in `WraLU/kact/krelu/sci.py`

You need to download network files from ERAN and put them in `nets/onnx`.

### Run Experiments of ReLU hull approximation

We compare different methods, including one exact method and one approximate method from the state-of-the-art technologies with our two methods.

First, you need to generate some polytope samples.

```bash
cd experiments_hull
cd polytope_samples
python3 sample_generator.py
cd ..
```

Next, calculate the bounds of variables of the polytopes samples.

```bash
cd polytope_bounds
python3 cal_polytope_bounds.py
cd ..
```

Then, we run different methods to calculate the ReLU hull.

```bash
cd output_constraints
python3 cal_constraints.py
cd ..
```

Finally, we calculate the volumes of the resulting polytopes by different methods to compare their precision.

```bash
cd volumes
python3 cal_volume.py
cd ../..
```

### Run Experiments of Neural Network Verification

In this section, we integrate our methods in PRIMA and assess its performance on neural network verification.

```bash
cd experiments
cd run_cpu
```

If you want to run a small example to test our methods.

```bash
bash example.sh
```

If you want to compare the different grouping strategies on a small network (3*50), run the following code.

```bash
bash test_hyperparameters.sh
```

If you want to run the verification on ERAN benchmarks, run the following code.

```bash
bash eran_benchmarks.sh
```

If you want to run the verification on our benchmarks, run the following code.

```bash
bash our_benchmarks.sh
```

If you want to compare with DeepZono, run the following code.

```bash
bash eran_benchmarks_deepzono.sh
```

