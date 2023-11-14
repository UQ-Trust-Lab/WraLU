from konvex.krelu_methods_time import krelu_with_mlf, krelu_with_mlfs
from konvexappro.group import generate_sparse_cover0, generate_sparse_cover1
from konvexappro.krelu_methods import krelu_with_cdd, krelu_with_pycdd, krelu_with_triangle, fkrelu, krelu_with_mlfss
from konvexappro.krelu_methods_beta_bigmatrix import krelu_with_big, krelu_with_big2
from konvexappro.krelu_methods_beta_ordering import krelu_with_mlf_order, krelu_with_mlf_order2
from konvexappro.krelu_methods_beta_refine import krelu_with_mlfs_beta_refine, krelu_with_mlfs_beta_refine2, \
    krelu_with_mlf_beta_y
from konvexappro.krelu_methods_old_lp import krelu_with_lf, krelu_with_lfs, krelu_with_lfss, krelu_with_binlfb, \
    krelu_with_binlfbs, krelu_with_lfb, krelu_with_lfbs, krelu_with_lfc, krelu_with_lfcs, krelu_with_lfcss
from konvexappro.ksigmtanh_methods import *

KRELU_METHODS = {"cdd": krelu_with_cdd,
                 "pycdd": krelu_with_pycdd,
                 "triangle": krelu_with_triangle,
                 "fast": fkrelu,
                 "mlf": krelu_with_mlf,
                 "mlfs": krelu_with_mlfs,
                 "lf": krelu_with_lf,
                 "lfs": krelu_with_lfs,
                 "mlfss": krelu_with_mlfss,
                 "lfss": krelu_with_lfss,
                 "binlfb": krelu_with_binlfb,
                 "binlfbs": krelu_with_binlfbs,
                 "lfb": krelu_with_lfb,
                 "lfbs": krelu_with_lfbs,
                 "lfc": krelu_with_lfc,
                 "lfcs": krelu_with_lfcs,
                 "lfcss": krelu_with_lfcss,
                 "mlfsbeta": krelu_with_mlfs_beta_refine,
                 "mlfsbeta2": krelu_with_mlfs_beta_refine2,
                 "mlfbetay": krelu_with_mlf_beta_y,
                 "mlforder": krelu_with_mlf_order,
                 "mlforder2": krelu_with_mlf_order2,
                 "mlfbig": krelu_with_big,
                 "mlfbig2": krelu_with_big2,
                 }

KSIGMOID_METHODS = {"cdd": ksigm_with_cdd,
                    "fast": fksigm,
                    "orthant": fsigm_orthant,
                    "sci": ksigm_with_sci,
                    "sciplus": ksigm_with_sciplus,
                    }

KTANH_METHODS = {"cdd": ktanh_with_cdd,
                 "fast": fktanh,
                 "orthant": ftanh_orthant,
                 "sci": ktanh_with_sci,
                 "sciplus": ktanh_with_sciplus,
                 }

METHODS_DIC = {"relu": KRELU_METHODS,
               "sigmoid": KSIGMOID_METHODS,
               "tanh": KTANH_METHODS}

ACT_FUNCTIONS = {"relu": None,
                 "sigmoid": sigmoid,
                 "tanh": tanh}

KGROUP_METHODS = {0: generate_sparse_cover0,
                  1: generate_sparse_cover1}

CONSTRAINTS_SAMPLE1 = [[1, 1],
                       [1, -1]]
CONSTRAINTS_SAMPLE2 = [[1., 0., -1.],
                       [2., 0., 1.],
                       [1., -1., -1.],
                       [3., 1., 1.],
                       [3., -1., 1.]]
CONSTRAINTS_SAMPLE3 = [[2, 1, 1],
                       [2, 1, -1],
                       [1.2, -1, 0],
                       [2, -1, 1],
                       [2, -1, -1]]
