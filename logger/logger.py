import os
import re
from datetime import datetime


class Logger:
    current_logger = None

    def __init__(self, dataset: str,
                 net_name: str,
                 domain: str,
                 epsilon: float,
                 krelu_method: str,
                 ns: int,
                 k: int,
                 s: int,
                 cutoff: float):
        Logger.current_logger = self
        self.dataset = dataset
        self.net_name = net_name
        self.domain = domain
        self.epsilon = epsilon
        self.ns = ns
        self.k = k
        self.s = s
        self.cutoff = cutoff
        self.krelu_method = krelu_method

        self.created_time = str(re.sub("[:\- .]", "_", str(datetime.now())))

        self.record = {}

        self.record['general'] = {}
        self.record['general']['id'] = None
        self.record['general']['classified'] = False
        self.record['general']['verified_dp'] = False
        self.record['general']['verified_rp'] = False
        self.record['general']['time'] = 0

        self.record['kact'] = {}
        self.record['kact']['time'] = 0
        self.record['kact']['n_constrs'] = 0
        self.record['kact']['n_coeffs'] = 0
        self.record['kact']['n_zero_coeffs'] = 0
        self.record['kact']['n_groups'] = 0

        self.record['lp'] = {}
        self.record['lp']['time'] = 0
        self.record['lp']['n_constrs'] = 0
        self.record['lp']['n_solves'] = 0
        self.record['lp']['status'] = []
        self.record['lp']['objvals'] = []

        # Record all the record
        self.records = []

        self.summary = {}
        self.summary['cc_idx'] = []
        self.summary['vdp_idx'] = []
        self.summary['vrp_idx'] = []
        self.summary['general'] = {}
        self.summary['general']['num'] = 0
        self.summary['general']['classified'] = 0
        self.summary['general']['verified_dp'] = 0
        self.summary['general']['verified_rp'] = 0
        self.summary['general']['time'] = 0
        self.summary['kact'] = {}
        self.summary['kact']['time'] = 0
        self.summary['kact']['n_constrs'] = 0
        self.summary['kact']['n_coeffs'] = 0
        self.summary['kact']['n_zero_coeffs'] = 0
        self.summary['kact']['n_groups'] = 0
        self.summary['lp'] = {}
        self.summary['lp']['time'] = 0
        self.summary['lp']['n_constrs'] = 0
        self.summary['lp']['n_solves'] = 0
        self.summary['lp']['status'] = {}
        self.summary['lp']['status']['success'] = {}
        self.summary['lp']['status']['success']['2'] = 0
        self.summary['lp']['status']['success']['6'] = 0
        self.summary['lp']['status']['fail'] = {}
        self.summary['lp']['status']['fail']['2'] = 0
        self.summary['lp']['status']['fail']['3'] = 0
        self.summary['lp']['status']['fail']['4'] = 0
        self.summary['lp']['status']['fail']['9'] = 0
        self.summary['lp']['status']['fail']['11'] = 0
        self.summary['lp']['objvals'] = {}
        self.summary['lp']['objvals']['success'] = {}
        self.summary['lp']['objvals']['success']['2'] = 0
        self.summary['lp']['objvals']['success']['6'] = 0
        self.summary['lp']['objvals']['fail'] = {}
        self.summary['lp']['objvals']['fail']['2'] = 0
        self.summary['lp']['objvals']['fail']['3'] = 0
        self.summary['lp']['objvals']['fail']['4'] = 0
        self.summary['lp']['objvals']['fail']['9'] = 0
        self.summary['lp']['objvals']['fail']['11'] = 0

        # If the document does not exist, create it
        if domain in {'deepzono', 'deeppoly'}:
            self.file_path = f"logs/{dataset}/{net_name}" \
                             f"/{domain}_{epsilon}_{self.created_time}.csv"
        elif krelu_method == 'triangle':
            self.file_path = f"logs/{dataset}/{net_name}" \
                             f"/{domain}_{krelu_method}_{epsilon}_{self.created_time}.csv"
        else:
            self.file_path = f"logs/{dataset}/{net_name}" \
                             f"/{domain}_{krelu_method}_{ns}_{k}_{s}_{epsilon}_{cutoff}_{self.created_time}.csv"

        if not os.path.exists(os.path.dirname(self.file_path)):
            os.makedirs(os.path.dirname(self.file_path))

        # Create a csv file with tab as the separator
        with open(self.file_path, "w") as f:
            # Write the header
            f.write('general')
            f.write('\t' * (len(self.record['general'])))
            f.write('kact')
            f.write('\t' * (len(self.record['kact'])))
            f.write('lp')
            f.write('\t' * (len(self.record['lp'])))
            f.write('\n')
            f.write('id\t'
                    'time\t'
                    'classified\t'
                    'verified_dp\t'
                    'verified_rp\t'
                    'time\t'
                    'n_constrs\t'
                    'n_coeffs\t'
                    'n_zero_coeffs\t'
                    'n_groups\t'
                    'time\t'
                    'n_constrs\t'
                    'n_solves\t'
                    'status\t'
                    'objvals\t')

    def record_general(self, id, classified, verified_dp, verified_rp, time):
        self.record['general']['id'] = id
        self.record['general']['classified'] = classified
        self.record['general']['verified_dp'] = verified_dp
        self.record['general']['verified_rp'] = verified_rp
        self.record['general']['time'] = time

    def record_kact(self, time, n_constrs, n_coeffs, n_zero_coeffs, n_groups):
        self.record['kact']['time'] += time
        self.record['kact']['n_constrs'] += n_constrs
        self.record['kact']['n_coeffs'] += n_coeffs
        self.record['kact']['n_zero_coeffs'] += n_zero_coeffs
        self.record['kact']['n_groups'] += n_groups

    def record_lp(self, time, n_constrs, status, objvals):
        self.record['lp']['time'] += time
        self.record['lp']['n_constrs'] = n_constrs
        self.record['lp']['n_solves'] += 1
        self.record['lp']['status'].append(status)
        self.record['lp']['objvals'].append(objvals)

    def log(self):
        self.records.append(self.record)

        with open(self.file_path, "a") as f:
            f.write(f"\n{self.record['general']['id']}\t"
                    f"{self.record['general']['time']}\t"
                    f"{self.record['general']['classified']}\t"
                    f"{self.record['general']['verified_dp']}\t"
                    f"{self.record['general']['verified_rp']}\t"
                    f"{self.record['kact']['time']}\t"
                    f"{self.record['kact']['n_constrs']}\t"
                    f"{self.record['kact']['n_coeffs']}\t"
                    f"{self.record['kact']['n_zero_coeffs']}\t"
                    f"{self.record['kact']['n_groups']}\t"
                    f"{self.record['lp']['time']}\t"
                    f"{self.record['lp']['n_constrs']}\t"
                    f"{self.record['lp']['n_solves']}\t"
                    f"{self.record['lp']['status']}\t"
                    f"{self.record['lp']['objvals']}\t")

        # Record the summary
        self.summary['cc_idx'].append(self.record['general']['id'])
        self.summary['general']['classified'] += 1
        if self.record['general']['verified_dp']:
            self.summary['general']['verified_dp'] += 1
            self.summary['vdp_idx'].append(self.record['general']['id'])
        if self.record['general']['verified_rp']:
            self.summary['general']['verified_rp'] += 1
            self.summary['vrp_idx'].append(self.record['general']['id'])
        self.summary['general']['num'] += 1
        self.summary['general']['time'] += self.record['general']['time']

        self.summary['kact']['time'] += self.record['kact']['time']
        self.summary['kact']['n_constrs'] += self.record['kact']['n_constrs']
        self.summary['kact']['n_coeffs'] += self.record['kact']['n_coeffs']
        self.summary['kact']['n_zero_coeffs'] += self.record['kact']['n_zero_coeffs']
        self.summary['kact']['n_groups'] += self.record['kact']['n_groups']
        self.summary['lp']['time'] += self.record['lp']['time']
        self.summary['lp']['n_constrs'] += self.record['lp']['n_constrs']
        self.summary['lp']['n_solves'] += self.record['lp']['n_solves']

        for status, objval in zip(self.record['lp']['status'], self.record['lp']['objvals']):
            if status == 2:
                if objval >= 0:
                    self.summary['lp']['status']['success']['2'] += 1
                    self.summary['lp']['objvals']['success']['2'] += objval
                else:
                    self.summary['lp']['status']['fail']['2'] += 1
                    self.summary['lp']['objvals']['fail']['2'] += objval
            elif status == 6:
                self.summary['lp']['status']['success']['6'] += 1
                try:
                    self.summary['lp']['objvals']['success']['6'] += objval
                except Exception:
                    pass
            elif status == 3:
                self.summary['lp']['status']['fail']['3'] += 1
                try:
                    self.summary['lp']['objvals']['fail']['3'] += objval
                except Exception:
                    pass
            elif status == 4:
                self.summary['lp']['status']['fail']['4'] += 1
                try:
                    self.summary['lp']['objvals']['fail']['4'] += objval
                except Exception:
                    pass
            elif status == 9:
                self.summary['lp']['status']['fail']['9'] += 1
                try:
                    self.summary['lp']['objvals']['fail']['9'] += objval
                except Exception:
                    pass
            elif status == 11:
                self.summary['lp']['status']['fail']['11'] += 1
                try:
                    self.summary['lp']['objvals']['fail']['11'] += objval
                except Exception:
                    pass

        # Reset the record
        self.record['general']['id'] = None
        self.record['general']['time'] = 0
        self.record['kact']['time'] = 0
        self.record['kact']['n_constrs'] = 0
        self.record['kact']['n_coeffs'] = 0
        self.record['kact']['n_zero_coeffs'] = 0
        self.record['kact']['n_groups'] = 0
        self.record['lp']['time'] = 0
        self.record['lp']['n_constrs'] = 0
        self.record['lp']['n_solves'] = 0
        self.record['lp']['status'] = []
        self.record['lp']['objvals'] = []

    @staticmethod
    def _cal_mean(values):
        return sum(values) / len(values) if len(values) > 0 else 0

    def log_summary(self, samples_ic, samples_if):
        # Write the summary
        with open(self.file_path, "a") as f:
            f.write('\nSummary\n')
            f.write(f'Incorrectly classified\t{samples_ic}\n')
            f.write(f'Infeasible samples\t{samples_if}\n')
            f.write(f'Correctly classified\t{self.summary["cc_idx"]}\n')
            f.write(f'Verified by DP\t{self.summary["vdp_idx"]}\n')
            f.write(f'Verified by RP\t{self.summary["vrp_idx"]}\n')
            f.write(f'Number of samples\t{self.summary["general"]["num"]}\n')
            f.write(f'Time\t{self.summary["general"]["time"]}\n')
            f.write(f'Number of correctly classified samples\t{self.summary["general"]["classified"]}\n')
            f.write(f'Number of verified by DP samples\t{self.summary["general"]["verified_dp"]}\n')
            f.write(f'Number of verified by RP samples\t{self.summary["general"]["verified_rp"]}\n')
            f.write(f'KACT time\t{self.summary["kact"]["time"]}\n')
            f.write(f'KACT number of constraints\t{self.summary["kact"]["n_constrs"]}\n')
            f.write(f'KACT number of coefficients\t{self.summary["kact"]["n_coeffs"]}\n')
            f.write(f'KACT number of zero coefficients\t{self.summary["kact"]["n_zero_coeffs"]}\n')
            f.write(f'KACT number of groups\t{self.summary["kact"]["n_groups"]}\n')
            f.write(f'LP time\t{self.summary["lp"]["time"]}\n')
            f.write(f'LP number of constraints\t{self.summary["lp"]["n_constrs"]}\n')
            f.write(f'LP number of solves\t{self.summary["lp"]["n_solves"]}\n')
            f.write(f'Successful solves\n')
            f.write('Status\t')
            for status in self.summary['lp']['status']['success'].keys():
                f.write(f'{status}\t')
            f.write('\n')
            f.write('Number\t')
            for n_solve in self.summary['lp']['status']['success'].values():
                f.write(f'{n_solve}\t')
            f.write('\n')
            f.write('Objective values\t')
            for objvals in self.summary['lp']['objvals']['success'].values():
                f.write(f'{objvals}\t')
            f.write('\n')
            f.write('Failed solves\n')
            f.write('Status\t')
            for status in self.summary['lp']['status']['fail'].keys():
                f.write(f'{status}\t')
            f.write('\n')
            f.write('Number\t')
            for n_solve in self.summary['lp']['status']['fail'].values():
                f.write(f'{n_solve}\t')
            f.write('\n')
            f.write('Objective values\t')
            for objvals in self.summary['lp']['objvals']['fail'].values():
                f.write(f'{objvals}\t')
            f.write('\n')

        Logger.current_logger = None
