import os
import sys
from datetime import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
cpu_affinity = os.sched_getaffinity(0)
os.sched_setaffinity(0, cpu_affinity)
sys.path.insert(0, '../')
sys.path.insert(0, '../tf_verify/')
sys.path.insert(0, '../../ELINA/python_interface/')

import time

import numpy as np
from tf_verify.config import config
from tf_verify.read_zonotope_file import read_zonotope
from experiments.utils import init_domain, initialise_args, parse_net_name, initialise_eran_model, \
    initialise_samples_data, skip_sample, get_inputs, normalize
from logger.logger import Logger

RECORD = True
VALID_DOMAINS = {'deepzono', 'refinezono', 'deeppoly', 'refinepoly'}


def run_experiment():
    args = initialise_args()
    domain = args.domain
    assert domain in VALID_DOMAINS, f'Domain must be in {VALID_DOMAINS}'
    dataset = args.dataset
    net_file_path = args.net_file
    net_name = parse_net_name(net_file_path)
    epsilon = args.epsilon
    krelu_method, ns, k, s, cutoff = args.convex_method, args.ns, args.k, args.s, args.use_cutoff_of

    eran, is_conv = initialise_eran_model(net_file_path)
    samples, means, stds = initialise_samples_data(dataset)

    ignored_samples = set()
    print(f'[INFO] Net name: {net_name}')
    if os.path.exists(f'incorrect_classified_samples/{net_name}.txt'):
        with open(f'incorrect_classified_samples/{net_name}.txt', 'r') as f:
            lines = f.readlines()
            ignored_samples = set(eval(lines[0]))
    print(f'[INFO] There has been {len(ignored_samples)} ignored samples')

    logger = Logger(dataset, net_name, domain, epsilon, krelu_method, ns, k, s, cutoff)
    assert Logger.current_logger is not None

    samples_ic, samples_c, samples_dp, samples_rp, samples_if = [], [], [], [], []
    total_time = 0.0
    for i, test in enumerate(samples):
        if skip_sample(i, ignored_samples):
            continue

        print(f"SAMPLE-{i}, {domain}, {net_name}".center(100, "=") +
              f"\n[SETTINGS] epsilon={epsilon}, ns={ns}, k={k}, s={s}, krelu_method={krelu_method}\n" +
              "Check sample is correctly classified".center(100, "-"))
        if config.mean is None and config.std is None:
            config.mean, config.std = means, stds
        print(f'[INFO] Mean: {config.mean}, std: {config.std}')
        image, specLB, specUB, label = get_inputs(test, config.mean, config.std, dataset, domain, is_conv)

        label, nn, nlb, nub, _, _, _ = eran.analyze_box(specLB, specUB, init_domain(domain), label=label)
        # print(f'[INFO] Final bounds: \n{nlb[-1]} \n{nub[-1]}')
        # raise NotImplementedError

        if label != int(test[0]):
            ignored_samples.add(i)
            samples_ic.append(i)
            print(f"[RESULT] Incorrectly classified, {label}!={int(test[0])}")
            continue
        samples_c.append(i)
        print("[RESULT] Correctly classified")

        print(f"Verify sample by {init_domain(domain)}".center(100, "-"))
        start = time.time()

        specLB = np.clip(image.copy() - epsilon, 0, 1)
        specUB = np.clip(image.copy() + epsilon, 0, 1)
        normalize(specLB, config.mean, config.std, dataset, domain, is_conv=is_conv)
        normalize(specUB, config.mean, config.std, dataset, domain, is_conv=is_conv)

        # print(specLB[:100])
        # print(specUB[:100])

        results = eran.analyze_box(specLB, specUB, init_domain(domain), label=label)
        perturbed_label, _, nlb, nub, failed_labels, x, _ = results
        # print(f'[INFO] Final bounds: \n{nlb[-1]} \n{nub[-1]}')

        verified_by_deeppoly = False
        if perturbed_label == label:
            verified_by_deeppoly = True
            samples_dp.append(i)
            print(f"[RESULT] Verified by {init_domain(domain)}")
        else:
            print(f"[RESULT] NOT verified by {init_domain(domain)}, failed labels: {failed_labels}")

        verified_by_refinepoly = False
        infeasible_model = None
        if not verified_by_deeppoly and domain == "refinepoly":
            print(f"Verify sample by {domain}".center(100, "-"))
            results = eran.analyze_box(specLB, specUB, domain, label=label, prop=failed_labels)
            perturbed_label, _, nlb, nub, failed_labels, x, infeasible_model = results

            if perturbed_label == label:
                print(f"[RESULT] Verified by {domain}")
                verified_by_refinepoly = True
                samples_rp.append(i)
            else:
                print(f"[RESULT] NOT verified by {domain}")

        used_time = time.time() - start
        total_time += used_time

        if infeasible_model is not None:
            # Save the infeasible model for debugging
            samples_if.append(i)
        #     domain, ns, k, s, eps = config.domain, config.sparse_n, config.k, config.s, config.epsilon
        #     current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        #     file_name = f"infeasible_models/{net_name}_{i}_{domain}_{krelu_method}_{ns}_{k}_{s}_{eps}_{cutoff}_{current_time}"
        #     if not os.path.exists("infeasible_models"):
        #         os.makedirs("infeasible_models")
        #     infeasible_model.write(file_name + ".mps")
        #     infeasible_model.write(file_name + ".lp")
        #     print(f'[INFO] Infeasible model saved to {file_name}.lp/.mps')

        if RECORD:
            logger.record_general(i, True, verified_by_deeppoly, verified_by_refinepoly, used_time)
            logger.log()

        print("RESULTS".center(100, "-") +
              f"\nepsilon={epsilon}, krelu_method={krelu_method}, ns={ns}, k={k}, s={s}\n"
              f"[STATS]\n"
              f"[VERIFIED (BP+Refine)/NUM]: ({len(samples_dp)}+{len(samples_rp)})/{len(samples_c)}  "
              f"[CURRENT/AVERAGE/TOTAL TIME]: {used_time:.2f}s/"
              f"{0. if len(samples_c) == 0. else total_time / len(samples_c):.2f}s/"
              f"{total_time:.2f}s\n"
              f"[INCORRECTED]:  {len(samples_ic)}\n"
              f"".center(100, "=") +
              f"\n")

        if len(samples_c) >= 100:
            break
        if args.samples_num is not None and len(samples_c) >= args.samples_num:
            break

    logger.log_summary(samples_ic, samples_if)

    # if domain == 'deeppoly' and len(ignored_samples) > 0:
    #     if not os.path.exists('incorrect_classified_samples'):
    #         os.makedirs('incorrect_classified_samples')
    #     with open(f'incorrect_classified_samples/{net_name}.txt', 'w') as f:
    #         f.write(str(set(ignored_samples)))


run_experiment()
