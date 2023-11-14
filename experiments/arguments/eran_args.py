import multiprocessing
from enum import Enum

from tf_verify.config import config


class Device(Enum):
    CPU = 0
    CUDA = 1


class ERANArgs:
    def __init__(self, print_args=True):
        # About basic information
        self.net_file = ""
        self.dataset = ""
        self.domain = ""
        self.epsilon = 0.0

        # About samples number
        self.samples_num = None
        self.samples_start = 0

        # About complete verifier
        self.is_complete = False

        # About MILP
        self.use_milp = False
        self.milp_neurons_num = 0
        self.milp_layers_num = 0

        # About k-activation refinement
        self.ns = 100
        self.k = 3
        self.s = self.k - 1
        self.use_heuristic = True
        self.convex_method = "fast"
        # self.group_method = "prima"
        # self.group_constr_method = "prima"
        self.use_skipgroup = False
        self.use_cutoff_of = 0.3

        # About bound refinement
        self.use_bound_refinement = False

        self.timeout_lp = 10
        self.timeout_milp = 10
        self.timeout_final_lp = 1000
        self.timeout_final_milp = 100

        self.processes_num = multiprocessing.cpu_count()
        self.debug = False

        # GPU options
        self.device = Device.CPU  # Which device Deeppoly should run_cpu on

        self.json = None  # Store all the information of arguments

        self.means = None
        self.stds = None

        #ACAS Xu
        self.specnumber = None

        self.numerical_focus = 2
        self.check_vertices = False
    def prepare_and_check(self):
        # console.log("Prepare and check arguments...", style="info")
        if self.dataset == "cifar10":
            self.means, self.stds = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
        else:
            self.means, self.stds = [0.0], [1.0]

    def print_args(self):
        print("Model Arguments".center(100,"*"))
        print('name'.ljust(50), 'value'.rjust(50))
        print('net_file'.ljust(50), self.net_file.rjust(50))
        print('dataset'.ljust(50), self.dataset.rjust(50))
        print('domain'.ljust(50), self.domain.rjust(50))
        print('epsilon'.ljust(50), str(self.epsilon).rjust(50))
        print('samples_start'.ljust(50), str(self.samples_start).rjust(50))
        print('samples_num'.ljust(50), str(self.samples_num).rjust(50))
        print('convex_method'.ljust(50), self.convex_method.rjust(50))
        print('ns'.ljust(50), str(self.ns).rjust(50))
        print('k'.ljust(50), str(self.k).rjust(50))
        print('s'.ljust(50), str(self.s).rjust(50))
        print('use_bound_refinement'.ljust(50), str(self.use_bound_refinement).rjust(50))
        print()


    def synchronize_eran_config(self):
        config.domain = self.domain
        config.dataset = self.dataset
        config.netname = self.net_file
        config.approx_k = self.convex_method
        config.epsilon = self.epsilon
        config.sparse_n = self.ns
        config.k = self.k
        config.s = self.s
        config.cutoff = self.use_cutoff_of
        config.from_test = self.samples_start
        config.num_tests = self.samples_num
        config.timeout_final_lp = self.timeout_final_lp

        #ACAS Xu
        config.specnumber = self.specnumber

        config.numerical_focus = self.numerical_focus
        config.check_vertices = self.check_vertices