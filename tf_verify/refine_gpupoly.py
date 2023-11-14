from krelu import *
from optimizer import *


class GPUPolyAnalyzer():
    def __init__(self, nn, network, gpu_relu_layers):
        self.nn = nn
        self.network = network
        self.gpu_relu_layers = gpu_relu_layers
        self.nlb = None
        self.nub = None
        self.relu_layer_idxes = None
        self.layer_idx_second_fc = None

    def get_second_fc_layer_idx(self):
        counter = 0
        for i in range(self.nn.numlayer):
            if self.nn.layertypes[i] == 'FC':
                counter += 1
                if counter == 2:
                    self.layer_idx_second_fc = i
                    return i
        assert counter == 2, "The network should have at least two FC layers"

    def get_all_neuron_bounds(self):
        nlb, nub, relu_layer_idxes = [], [], []
        layer_idx = 2
        for i in range(self.nn.numlayer):
            if layer_idx not in self.gpu_relu_layers:
                bounds = self.network.evalAffineExpr(layer=layer_idx)
                lbi, ubi = bounds[:, 0], bounds[:, 1]
                layer_idx += 1
            else:
                lbi, ubi = np.maximum(nlb[-1], 0), np.maximum(nub[-1], 0)
                relu_layer_idxes.append(len(nlb))
                layer_idx += 2
            nlb.append(lbi)
            nub.append(ubi)
        self.nlb, self.nub, self.relu_layer_idxes = nlb, nub, relu_layer_idxes

        return nlb, nub, relu_layer_idxes

    def refine_neuron_bounds(self, relu_groups, layer_idx, layer_idx_second_fc):
        warnings.warn("Refinement of bounds is not implemented yet")
        pre_idx = self.nn.predecessors[layer_idx + 1][0] - 1
        lbi, ubi = self.nlb[pre_idx], self.nub[pre_idx]
        neurons_num = len(lbi)

        candidate_vars = [i for i in range(neurons_num) if lbi[i] < 0 < ubi[i] or lbi[i] > 0]

        use_milp_temp = config.use_milp if pre_idx == layer_idx_second_fc else False
        timeout = config.timeout_milp if pre_idx == layer_idx_second_fc else config.timeout_lp

        start = time.time()
        resl, resu, indices = get_bounds_for_layer_with_milp(self.nn, self.nn.specLB, self.nn.specUB, pre_idx,
                                                             pre_idx, neurons_num, self.nlb, self.nub, relu_groups,
                                                             use_milp_temp, candidate_vars, timeout)
        end = time.time()
        print(f"Refinement of bounds time: {end - start:.3f}. MILP used: {use_milp_temp}")
        self.nlb[pre_idx], self.nub[pre_idx] = resl, resu

        return self.nlb, self.nub

    @staticmethod
    def _init_input_constrs(kact_args, num_neurons, candidate_coeffs):

        total_size = sum(len(candidate_coeffs) ** len(varsid) - 1 for varsid in kact_args)
        A = np.zeros((total_size, num_neurons), dtype=np.double)
        i = 0
        for varsid in kact_args:
            for coeffs in itertools.product(candidate_coeffs, repeat=len(varsid)):
                if all(c == 0 for c in coeffs):
                    continue
                A[i, varsid] = np.asarray(coeffs)
                i += 1
        return A

    @staticmethod
    def _cal_input_constrs_bias(input_constrs, network, gpu_layer_idx):
        max_eqn_per_call = 1024
        bounds = np.zeros(shape=(0, 2))
        for i_a in range((int)(np.ceil(input_constrs.shape[0] / max_eqn_per_call))):
            temp = input_constrs[i_a * max_eqn_per_call:(i_a + 1) * max_eqn_per_call]
            bounds_temp = network.evalAffineExpr(temp, layer=gpu_layer_idx,
                                                 back_substitute=network.FULL_BACKSUBSTITUTION,
                                                 dtype=np.double)
            bounds = np.concatenate([bounds, bounds_temp], axis=0)
        upper_bounds = bounds[:, 1]

        return upper_bounds

    @staticmethod
    def _get_input_contrs(neuron_idxes_grouped, candidate_coeffs, bias, lb, ub):

        input_constrs_grouped, lbs_grouped, ubs_grouped = [], [], []
        i = 0
        for neuron_idxes in neuron_idxes_grouped:
            input_constrs = []
            for coeffs in itertools.product(candidate_coeffs, repeat=len(neuron_idxes)):
                if all(c == 0 for c in coeffs):
                    continue
                input_constrs.append([bias[i]] + [-c for c in coeffs])
                i += 1
            input_constrs_grouped.append(input_constrs)
            lbs_grouped.append([lb[idx] for idx in neuron_idxes])
            ubs_grouped.append([ub[idx] for idx in neuron_idxes])

        return input_constrs_grouped, lbs_grouped, ubs_grouped

    def encode_kact_constraints(self, neurons_grouped, gpu_layer, lb, ub):
        kact_objs = []

        # Initialize input constraints
        coeffs = [-1, 0, 1]
        input_constrs_A = self._init_input_constrs(neurons_grouped, len(lb), coeffs)
        input_constrs_b = self._cal_input_constrs_bias(input_constrs_A, self.network, gpu_layer)

        # Construct input constraints
        input_hrep_grouped, lbs_grouped, ubs_grouped = self._get_input_contrs(neurons_grouped, coeffs, input_constrs_b,
                                                                              lb, ub)

        KAct.type = "ReLU"

        start = time.time()
        with multiprocessing.Pool(config.numproc) as pool:
            kact_constrs_grouped = list(pool.starmap(make_kactivation_obj,
                                                     zip(input_hrep_grouped,
                                                         lbs_grouped,
                                                         ubs_grouped,
                                                         [config.approx_k] * len(input_hrep_grouped))))
        time_convex_hull = time.time() - start

        groups_num = constrs_num = coeffs_num_total = coeffs_num_zero = 0
        for gid, inst in enumerate(kact_constrs_grouped):
            if inst.cons is None:
                continue

            inst.varsid = neurons_grouped[gid]
            kact_objs.append(inst)

            if inst.cons.shape[0] == 3:
                continue
            groups_num += 1
            constrs_num += inst.cons.shape[0]
            coeffs_num = inst.cons.shape[0] * (inst.cons.shape[1] - 1)  # Count the non-zero coefficients in inst.cons
            coeffs_num_total += coeffs_num
            coeffs_num_zero += coeffs_num - np.count_nonzero(inst.cons[:, :-1])

        # Record
        logger = Logger.current_logger
        if logger is not None:
            logger.record_kact(time_convex_hull, constrs_num, coeffs_num_total, coeffs_num_zero, groups_num)

        print(f"\tCal convex hull/approx...{time_convex_hull:.4f}s ")
        return kact_objs

    def create_gurobi_model(self, relu_groups):
        nn, nlb, nub = self.nn, self.nlb, self.nub
        # if config.complete:
        #     counter, var_list, model = create_model(nn, nn.specLB, nn.specUB, nlb, nub, relu_groups, nn.numlayer,
        #                                             use_milp=True, is_nchw=True, partial_milp=-1, max_milp_neurons=1e6)
        #     # model.setParam(GRB.Param.TimeLimit, timeout_final_milp) #set later
        # else:
        grb_var_counter, grb_vars, model, kact_constrs = create_model(nn, nn.specLB, nn.specUB, nlb, nub, relu_groups,
                                                                      nn.numlayer, use_milp=False, is_nchw=True)
        model.setParam(GRB.Param.TimeLimit, 1000)
        model.setParam(GRB.Param.Cutoff, 1e-6)
        model.setParam(GRB.Param.OptimalityTol, 1e-3)
        model.setParam(GRB.Param.FeasibilityTol, 1e-3)
        model.setParam(GRB.Param.NumericFocus, 2)
        # model.setParam(GRB.Param.MarkowitzTol, 0.9)
        model.setParam(GRB.Param.Method, 3)
        print(f'[INFO] Setting timeout to 1000s')
        print(f'[INFO] Setting cutoff to 1e-6')
        print(f'[INFO] Setting optimality tolerance to 1e-3')
        print(f'[INFO] Setting feasibility tolerance to 1e-3')
        print(f'[INFO] Setting numeric focus to 2')
        # print(f'[INFO] Setting Markowitz tolerance to 0.1')
        print(f'[INFO] Setting method to 3 (Concurrent)')
        model.update()

        return grb_var_counter, grb_vars, model, kact_constrs


def refine_gpupoly_results(nn, network, layers_num, relu_layer_idxes, label, adv_labels):
    analyzer = GPUPolyAnalyzer(nn, network, relu_layer_idxes)

    nlb, nub, relu_layers_indxes_gpu = analyzer.get_all_neuron_bounds()

    layer_idx_second_fc = analyzer.get_second_fc_layer_idx()

    index = 0
    relu_groups = []
    for l in relu_layer_idxes:
        gpu_layer = l - 1
        layer_idx = relu_layers_indxes_gpu[index]

        if config.refine_neurons:
            nlb, nub = analyzer.refine_neuron_bounds(relu_groups, layer_idx, layer_idx_second_fc)

        lbi, ubi = nlb[layer_idx - 1], nub[layer_idx - 1]
        num_neurons = len(lbi)

        if config.approx_k == 'triangle':
            relu_groups.append([])
            continue
        print(f'[INFO] Method={config.approx_k} ns={config.sparse_n} k={config.k} s={config.s} '
              f'cutoff={config.cutoff:.2f}')
        kact_args = sparse_heuristic_with_cutoff(num_neurons, lbi, ubi, config.sparse_n, config.k, config.s,
                                                 config.cutoff)

        if len(kact_args) == 0:
            relu_groups.append([])
            continue

        relu_groups.append(analyzer.encode_kact_constraints(kact_args, gpu_layer, lbi, ubi))
        index += 1

    print(f'[INFO] Labels {adv_labels} to be verified', end=' ')

    start = time.time()
    counter, var_list, model, kact_constrs = analyzer.create_gurobi_model(relu_groups)

    infeasible_model = None
    flag = True
    x = None

    for adv_label in adv_labels:
        print(f'{label}vs{adv_label}', end='')

        model.setObjective(LinExpr(var_list[counter + label] - var_list[counter + adv_label]), GRB.MINIMIZE)
        model.optimize(lp_callback)
        print(f"({model.Runtime:.2f}s)", end=" ")

        logger = Logger.current_logger
        if logger is not None:
            try:
                objval = model.objval
            except:
                objval = 0.0
            logger.record_lp(model.Runtime, len(model.getConstrs()), model.Status, objval)

        # if model.Status in [3, 4]:
        #     time_start = time.time()
        #     model.computeIIS()
        #     print(f'Compute IIS ({time.time() - time_start:.2f}s)')
        #     # Remove all constraints that are in the IIS
        #     for constr in model.getConstrs():
        #         if constr.IISConstr and (
        #                 constr.ConstrName.startswith('relu_triangle') or constr.ConstrName.startswith('kact')):
        #             model.remove(constr)
        #     model.update()
        #     print(f"[INFO] Remain {len(model.getConstrs())} constraints from the model")
        #     # model.update()
        #
        #     print(f'LP model', end=' ')
        #     model.optimize(lp_callback)

        print(f"{model.Status}", end=' ')
        if model.Status == 6:
            pass
        elif model.Status == 2:
            print(f"(objval={model.objval:.4f})", end=' ')
            if model.objval < 0 and model.objval != math.inf:
                flag = False
                x = model.x[0:len(nn.specLB)]
        else:
            flag = False
            if model.Status in [3, 4]:
                infeasible_model = model

        if not flag:
            break
    print()
    return flag, x, infeasible_model
