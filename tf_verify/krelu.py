import itertools
import multiprocessing
import time

import numpy as np
from elina_abstract0 import *
from elina_dimension import *
from elina_linexpr0 import *
from elina_scalar import *
from fconv import *
from fppoly import *
import deepzono_nodes as dn

from kact import ktanh_with_sci, ksigm_with_sci, krelu_with_sci, krelu_with_sciplus, generate_sparse_cover00, \
    krelu_with_sci_redundant
from logger.logger import Logger
from tf_verify.config import config

"""
For representing the constraints CDD format is used
http://web.mit.edu/sage/export/cddlib-094b.dfsg/doc/cddlibman.ps:
each row represents b + Ax >= 0
example: 2*x_1 - 3*x_2 >= 1 translates to [-1, 2, -3]
"""


class KAct:
    type = None
    methods = {
        "ReLU": {"fast": fkrelu, "cdd": krelu_with_cdd, "sci": krelu_with_sci, "sciplus": krelu_with_sciplus,
                 "sci2": krelu_with_sci_redundant},
        "Tanh": {"fast": ftanh_orthant, "cdd": ktanh_with_cdd, "sci": ktanh_with_sci},
        "Sigmoid": {"fast": fsigm_orthant, "cdd": ksigm_with_cdd, "sci": ksigm_with_sci},
    }

    def __init__(self, input_hrep, lbi, ubi, method):
        assert KAct.type is not None
        assert all(lb <= ub for lb, ub in zip(lbi, ubi))
        self.k, self.input_hrep = len(input_hrep[0]) - 1, np.array(input_hrep)
        if method in ["fast", "cdd"]:
            self.cons = KAct.methods[KAct.type][method](self.input_hrep)
        else:
            try:
                # Sometimes the pycddlib library crashes, so we need to catch the exception
                self.cons = KAct.methods[KAct.type][method](self.input_hrep, lbi, ubi,
                                                            check=config.check_vertices)
            except Exception as e:
                print(e)
                self.cons = None


def make_kactivation_obj(input_hrep, lbi, ubi, method):
    return KAct(input_hrep, lbi, ubi, method)


def encode_kactivation_cons(nn, man, element, offset, layerno, length, lbi, ubi, constraint_groups, need_pop, domain,
                            activation_type):
    if config.approx_k == 'triangle':
        print(f'[KACT] Using triangle relaxation')
        constraint_groups.append([])  # Handle triangle approximation in ai_milp.py directly
        return

    sparse_n, K, s, cutoff, approx = config.sparse_n, config.k, config.s, config.cutoff, config.approx_k
    print(f'[INFO] Method={config.approx_k} ns={config.sparse_n} k={config.k} s={config.s} cutoff={config.cutoff:.2f}')
    assert sparse_n > 0 and K > 2 and s > 0, "Invalid parameters for sparse encoding"

    if need_pop:
        constraint_groups.pop()

    lbi, ubi = np.asarray(lbi, dtype=np.double), np.asarray(ubi, dtype=np.double)

    if activation_type == "ReLU":
        kact_args = sparse_heuristic_with_cutoff(length, lbi, ubi, sparse_n, K, s, cutoff)
    else:
        kact_args = sparse_heuristic_curve(length, lbi, ubi, sparse_n, K, s, cutoff, activation_type == "Sigmoid")

    if len(kact_args) == 0:
        constraint_groups.append([])
        return

    kact_cons = []
    tdim = ElinaDim(offset + length)
    if domain == 'refinezono':
        element = dn.add_dimensions(man, element, offset + length, 1)

    KAct.man = man
    KAct.element = element
    KAct.tdim = tdim
    KAct.length = length
    KAct.layerno = layerno
    KAct.offset = offset
    KAct.domain = domain
    KAct.type = activation_type

    start = time.time()
    if domain == 'refinezono':
        with multiprocessing.Pool(config.numproc) as pool:
            input_hrep_array = pool.map(get_ineqs_zono, kact_args)
        with multiprocessing.Pool(config.numproc) as pool:
            # kact_results = pool.starmap(make_kactivation_obj, zip(input_hrep_array, len(input_hrep_array) * [approx]))
            kact_results = list(pool.starmap(make_kactivation_obj,
                                             zip(input_hrep_array, [approx] * len(input_hrep_array))))
    else:
        total_size = 0
        for var_ids in kact_args:
            size = 3 ** len(var_ids) - 1
            total_size = total_size + size

        linexpr0 = elina_linexpr0_array_alloc(total_size)
        i = 0
        for var_ids in kact_args:
            for coeffs in itertools.product([-1, 0, 1], repeat=len(var_ids)):
                if all(c == 0 for c in coeffs):
                    continue
                linexpr0[i] = generate_linexpr0(offset, var_ids, coeffs)
                i += 1
        upper_bounds = get_upper_bound_for_linexpr0(man, element, linexpr0, total_size, layerno)

        i = 0
        input_hrep_array, lb_array, ub_array = [], [], []
        for var_ids in kact_args:
            input_hrep = []
            for coeffs in itertools.product([-1, 0, 1], repeat=len(var_ids)):
                if all(c == 0 for c in coeffs):
                    continue
                input_hrep.append([upper_bounds[i]] + [-c for c in coeffs])
                i += 1
            input_hrep_array.append(input_hrep)
            lb_array.append([lbi[varid] for varid in var_ids])
            ub_array.append([ubi[varid] for varid in var_ids])
        # print(f"\tGet input constrs... {time.time() - start:.2f}s")

        start = time.time()
        with multiprocessing.Pool(config.numproc) as pool:
            kact_results = pool.starmap(make_kactivation_obj, zip(input_hrep_array, lb_array, ub_array,
                                                                  [approx] * len(input_hrep_array)))
            kact_results = list(kact_results)

        time_convex_hull = time.time() - start

    groups_num = constrs_num = coeffs_num_total = coeffs_num_zero = 0
    for gid, inst in enumerate(kact_results):
        if inst.cons is None:
            continue
        inst.varsid = kact_args[gid]
        kact_cons.append(inst)

        if inst.cons.shape[0] == 3:
            continue
        groups_num += 1
        constrs_num += inst.cons.shape[0]
        coeffs_num = inst.cons.shape[0] * (inst.cons.shape[1] - 1)  # Count the non zero coefficients in inst.cons
        coeffs_num_total += coeffs_num
        coeffs_num_zero += coeffs_num - np.count_nonzero(inst.cons[:, :-1])

    if domain == 'refinezono':
        element = dn.remove_dimensions(man, element, offset + length, 1)

    logger = Logger.current_logger
    if logger is not None:
        logger.record_kact(time_convex_hull, constrs_num, coeffs_num_total, coeffs_num_zero, groups_num)

    # print(f"\tCal convex hull/approx...{time_convex_hull:.4f}s, "
    #       f"{constrs_num} constraints, "
    #       f"{coeffs_num_zero}/{coeffs_num_total}="
    #       f"{int(coeffs_num_zero / float(coeffs_num_total) * 100) if float(coeffs_num_total) > 0 else 0}%"
    #       f" zero coeffs")

    constraint_groups.append(kact_cons)


def sparse_heuristic_with_cutoff(length, lb, ub, sparse_n, k, s, cutoff):
    assert length == len(lb) == len(ub), "length of lb, ub should be the same"

    all_vars = [i for i in range(length) if lb[i] < 0 < ub[i]]
    # Order the variables by the product of their bounds
    # all_vars.sort(key=lambda i: -lb[i] * ub[i])
    vars_above_cutoff = [i for i in all_vars if -lb[i] * ub[i] > cutoff]
    # vars_above_cutoff = [i for i in vars_above_cutoff if lb[i] < -cutoff and ub[i] > cutoff]

    num_vars_above_cutoff = len(vars_above_cutoff)
    kact_args = []

    while vars_above_cutoff and sparse_n >= k:
        grouplen = min(sparse_n, len(vars_above_cutoff))
        group = vars_above_cutoff[:grouplen]
        vars_above_cutoff = vars_above_cutoff[grouplen:]
        if grouplen == 1:
            continue
        elif grouplen <= k:
            kact_args.append(group)
            continue
        if 2 < k < 5 and s > 0:
            sparsed_combs = generate_sparse_cover(grouplen, k, s=s)
        else:
            sparsed_combs = generate_sparse_cover00(grouplen, k, s)
        kact_args.extend(tuple(group[i] for i in comb) for comb in sparsed_combs)

    print(f"\t{length} neurons -> {len(all_vars)} split zero -> {num_vars_above_cutoff} after cutoff "
          f"=> {len(kact_args)} Groups")

    # kact_args.extend((var,) for var in all_vars)  # Also just apply 1-relu for every var for triangle relaxation.
    # kact_args = [(var,) for var in all_vars]

    return kact_args


def sparse_heuristic_curve(length, lb, ub, sparse_n, k, s, cutoff, is_sigm):
    assert length == len(lb) == len(ub)
    all_vars = list(range(length))
    limit = 4 if is_sigm else 3
    vars_above_cutoff = [i for i in all_vars if ub[i] - lb[i] >= cutoff and lb[i] <= limit and ub[i] >= -limit]
    num_vars_above_cutoff = len(vars_above_cutoff)
    kact_args = []
    while vars_above_cutoff and sparse_n >= k:
        grouplen = min(sparse_n, len(vars_above_cutoff))
        group = vars_above_cutoff[:grouplen]
        vars_above_cutoff = vars_above_cutoff[grouplen:]
        if grouplen == 1:
            continue
        elif grouplen <= k:
            kact_args.append(group)
            continue
        sparsed_combs = generate_sparse_cover(grouplen, k, s=s)
        kact_args.extend(tuple(group[i] for i in comb) for comb in sparsed_combs)

    # print(f"\t{length} neurons -> {len(all_vars)} split zero -> {num_vars_above_cutoff} after cutoff "
    #       f"=> {len(kact_args)} Groups")

    # kact_args.extend((var,) for var in vars_above_cutoff)  # Also just apply 1-relu for every var.

    return kact_args


def generate_linexpr0(offset, varids, coeffs):
    # returns ELINA expression, equivalent to sum_i(varids[i]*coeffs[i])
    assert len(varids) == len(coeffs)
    n = len(varids)

    linexpr0 = elina_linexpr0_alloc(ElinaLinexprDiscr.ELINA_LINEXPR_SPARSE, n)
    cst = pointer(linexpr0.contents.cst)
    elina_scalar_set_double(cst.contents.val.scalar, 0)

    for i, (x, coeffx) in enumerate(zip(varids, coeffs)):
        linterm = pointer(linexpr0.contents.p.linterm[i])
        linterm.contents.dim = ElinaDim(offset + x)
        coeff = pointer(linterm.contents.coeff)
        elina_scalar_set_double(coeff.contents.val.scalar, coeffx)

    return linexpr0


def get_ineqs_zono(varsid):
    input_hrep = []

    # Get bounds on linear expressions over variables before relu
    # Order of coefficients determined by logic here
    for coeffs in itertools.product([-1, 0, 1], repeat=len(varsid)):
        if all(c == 0 for c in coeffs):
            continue

        linexpr0 = generate_linexpr0(KAct.offset, varsid, coeffs)
        element = elina_abstract0_assign_linexpr_array(KAct.man, True, KAct.element, KAct.tdim, linexpr0, 1, None)
        bound_linexpr = elina_abstract0_bound_dimension(KAct.man, KAct.element, KAct.offset + KAct.length)
        upper_bound = bound_linexpr.contents.sup.contents.val.dbl
        input_hrep.append([upper_bound] + [-c for c in coeffs])
    return input_hrep
