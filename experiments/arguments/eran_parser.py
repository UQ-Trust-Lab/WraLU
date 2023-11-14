import argparse
import os

from .eran_args import ERANArgs

SUPPORTED_DOMAINS = ("deeppoly", "refinepoly", "gpupoly", "refinegpupoly", "deepzono")
SUPPORTED_FILE_EXTENSIONS = (".onnx",)
SUPPORTED_DATASETS = ("mnist", "cifar10", "acasxu", "fashion_mnist", "emnist")

SUPPORTED_CONVEX_METHODS = ("cdd", "fast", "sci", "sciplus", "triangle")


class ERANParser(argparse.ArgumentParser):
    def __init__(self, args: ERANArgs):
        argparse.ArgumentParser.__init__(self, description="ERAN Example",
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # About basic information
        self.add_argument("--net_file", type=is_network_file, default=args.net_file,
                          help="The network file name/path (.onnx)")
        self.add_argument("--dataset", type=is_dataset, default=args.dataset,
                          help="The dataset (mnist, cifar10)")
        self.add_argument("--domain", type=is_domain, default=args.domain,
                          help="The domain name (deeppoly, refinepoly, gpupoly, gpurefinepoly)")
        self.add_argument("--epsilon", type=float, default=args.epsilon,
                          help="The Epsilon for L_infinity perturbation")
        # About samples number
        self.add_argument("--samples_num", type=is_positive_int, default=args.samples_num,
                          help="The number of samples to test")
        self.add_argument("--samples_start", type=int, default=args.samples_start,
                          help="The first id of samples to test")
        # About complete verifier
        self.add_argument("--is_complete", action="store_true", default=args.is_complete,
                          help="Whether to use complete verification or not")
        # About milp
        self.add_argument("--use_milp", action="store_true", default=args.use_milp,
                          help="Whether to use milp or not")
        self.add_argument("--milp_neurons_num", type=int, default=args.milp_neurons_num,
                          help="The maximum number of neurons of one layer for MILP encoding")
        self.add_argument("--milp_layers_num", type=int, default=args.milp_layers_num,
                          help="The number of layers forMILP encoding.")

        # About k-activation refinement
        self.add_argument("--ns", type=is_positive_int, default=args.ns,
                          help="The number of variables to group by k-activation")
        self.add_argument("--k", type=is_positive_int, default=args.k,
                          help='The group size of k-activation')
        self.add_argument("--s", type=is_positive_int, default=args.s,
                          help='The overlap size between two k-activation group')
        self.add_argument("--use_default_heuristic", action="store_true", default=args.use_heuristic,
                          help="Whether to use the area heuristic for the k-activation approximation "
                               "or to always create new noise symbols per relu for the DeepZono ReLU approximation")
        self.add_argument("--convex_method", type=is_convex_method, default=args.convex_method,
                          help="The method to calculate k-activation")
        # self.add_argument("--group_method", type=is_group_method, default=args.group_method,
        #                   help="The method to group neurons when using k-activation")
        # self.add_argument("--group_constrs_method", type=is_group_constr_method, default=args.group_constr_method,
        #                   help="The method to construct input constraints when using k-activation")
        self.add_argument("--use_skipgroup", action="store_true", default=args.use_skipgroup,
                          help="Whether to use k skip group when using k convex method.")
        self.add_argument("--use_cutoff_of", type=float, default=args.use_cutoff_of,
                          help="Used to ignore some groups when encoding k activation; otherwise, the final LP"
                               "verifying problem sometimes maybe infeasible.")
        # About bound refinement
        self.add_argument("--use_bound_refinement", action="store_true", default=args.use_bound_refinement,
                          help="Whether to refine the bounds of intermediate neurons")
        # About timeout
        self.add_argument("--timeout_lp", type=is_positive_float, default=args.timeout_lp,
                          help="The timeout for the LP solver to refine")
        self.add_argument("--timeout_final_lp", type=is_positive_float, default=args.timeout_final_lp,
                          help="The timeout for the final LP solver to final verify")
        self.add_argument("--timeout_milp", type=is_positive_float, default=args.timeout_milp,
                          help="The timeout for the MILP solver to refine")
        self.add_argument("--timeout_final_milp", type=is_positive_float, default=args.timeout_final_lp,
                          help="The timeout for the final MILP solve to final verify")
        # About general settings
        self.add_argument("--processes_num", type=int, default=args.processes_num,
                          help="The number of processes for MILP/LP/k-activation")
        self.add_argument("--debug", action="store_true", default=args.debug,
                          help="Whether to display debug info")

        # ACAS Xu
        self.add_argument("--specnumber", type=int, default=args.specnumber,
                          help="The number of the specification to test")
        self.add_argument("--numerical_focus", type=int, default=args.numerical_focus,
                            help="Numerical focus of Gurobi")
        self.add_argument("--check_vertices", action="store_true", default=args.check_vertices,
                            help="Whether to check the vertices in sci, sciplus")

    def set_args(self, args: ERANArgs):
        # console.log("Set arguments...", style="info")
        arguments = self.parse_args()
        for k, v in vars(arguments).items():
            setattr(args, k, v)
        args.json = vars(arguments)


def is_domain(domain: str) -> str:
    if domain not in SUPPORTED_DOMAINS:
        raise argparse.ArgumentTypeError(f"{domain} is not supported. Only support {SUPPORTED_DOMAINS}.")
    return domain


def is_dataset(dataset: str) -> str:
    if dataset not in SUPPORTED_DATASETS:
        raise argparse.ArgumentTypeError(f"{dataset} is not supported. Only support {SUPPORTED_DATASETS}.")
    return dataset


def is_network_file(net_file: str) -> str:
    if not os.path.isfile(net_file):
        raise argparse.ArgumentTypeError(f'The net file "{net_file}" is not found.')
    _, extension = os.path.splitext(net_file)
    if extension not in SUPPORTED_FILE_EXTENSIONS:
        raise argparse.ArgumentTypeError(f"{extension} is not supported. Only support {SUPPORTED_FILE_EXTENSIONS}.")
    return net_file


def is_convex_method(method: str) -> str:
    if method not in SUPPORTED_CONVEX_METHODS:
        raise argparse.ArgumentTypeError(f"{method} is not supported. Only support {SUPPORTED_CONVEX_METHODS}.")
    return method


def is_group_method(method: str) -> str:
    if method not in SUPPORTED_GROUP_METHODS:
        raise argparse.ArgumentTypeError(f"{method} is not supported. Only support {SUPPORTED_GROUP_METHODS}.")
    return method


def is_group_constr_method(method: str) -> str:
    if method not in SUPPORTED_GROUP_CONSTRS_METHODS:
        raise argparse.ArgumentTypeError(f"{method} is not supported. Only support {SUPPORTED_GROUP_CONSTRS_METHODS}.")
    return method


def is_positive_int(number: str) -> int:
    try:
        number = int(number)
    except:
        raise argparse.ArgumentTypeError("This argument should be an integer.")
    if number <= 0:
        raise argparse.ArgumentTypeError("This argument should be positive integer.")
    return number


def is_positive_float(number: str) -> float:
    try:
        number = float(number)
    except:
        raise argparse.ArgumentTypeError("This argument should be a float number.")
    if number <= 0:
        raise argparse.ArgumentTypeError("This argument should be positive float number.")
    return number
