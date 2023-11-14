from datetime import datetime

import gurobipy
from elina_abstract0 import *
from elina_manager import *

from logger.logger import Logger
from deeppoly_nodes import *
from deepzono_nodes import *


class layers:
    def __init__(self):
        self.layertypes = []
        self.weights = []
        self.biases = []
        self.filters = []
        self.numfilters = []
        self.filter_size = []
        self.input_shape = []
        self.strides = []
        self.padding = []
        self.out_shapes = []
        self.pool_size = []
        self.numlayer = 0
        self.ffn_counter = 0
        self.conv_counter = 0
        self.residual_counter = 0
        self.pad_counter = 0
        self.pool_counter = 0
        self.concat_counter = 0
        self.tile_counter = 0
        self.activation_counter = 0
        self.specLB = []
        self.specUB = []
        self.original = []
        self.zonotope = []
        self.predecessors = []
        self.lastlayer = None
        self.last_weights = None
        self.label = -1
        self.prop = -1

    def calc_layerno(self):
        return self.ffn_counter + self.conv_counter + self.residual_counter + self.pool_counter + self.activation_counter + self.concat_counter + self.tile_counter + self.pad_counter

    def is_ffn(self):
        return not any(x in ['Conv2D', 'Conv2DNoReLU', 'Resadd', 'Resaddnorelu'] for x in self.layertypes)

    def set_last_weights(self, constraints):
        length = 0.0
        last_weights = [0 for weights in self.weights[-1][0]]
        for or_list in constraints:
            for (i, j, cons) in or_list:
                if j == -1:
                    last_weights = [l + w_i + float(cons) for l, w_i in zip(last_weights, self.weights[-1][i])]
                else:
                    last_weights = [l + w_i + w_j + float(cons) for l, w_i, w_j in
                                    zip(last_weights, self.weights[-1][i], self.weights[-1][j])]
                length += 1
        self.last_weights = [w / length for w in last_weights]

    def back_propagate_gradient(self, nlb, nub):
        # assert self.is_ffn(), 'only supported for FFN'

        grad_lower = self.last_weights.copy()
        grad_upper = self.last_weights.copy()
        last_layer_size = len(grad_lower)
        for layer in range(len(self.weights) - 2, -1, -1):
            weights = self.weights[layer]
            lb = nlb[layer]
            ub = nub[layer]
            layer_size = len(weights[0])
            grad_l = [0] * layer_size
            grad_u = [0] * layer_size

            for j in range(last_layer_size):

                if ub[j] <= 0:
                    grad_lower[j], grad_upper[j] = 0, 0

                elif lb[j] <= 0:
                    grad_upper[j] = grad_upper[j] if grad_upper[j] > 0 else 0
                    grad_lower[j] = grad_lower[j] if grad_lower[j] < 0 else 0

                for i in range(layer_size):
                    if weights[j][i] >= 0:
                        grad_l[i] += weights[j][i] * grad_lower[j]
                        grad_u[i] += weights[j][i] * grad_upper[j]
                    else:
                        grad_l[i] += weights[j][i] * grad_upper[j]
                        grad_u[i] += weights[j][i] * grad_lower[j]
            last_layer_size = layer_size
            grad_lower = grad_l
            grad_upper = grad_u
        return grad_lower, grad_upper


class Analyzer:
    def __init__(self, ir_list, nn, domain, output_constraints, label, prop,
                 K=config.k, s=config.s, approx_k=config.approx_k,
                 complete=config.complete, use_milp=config.use_milp,
                 partial_milp=config.partial_milp, max_milp_neurons=config.max_milp_neurons,
                 timeout_lp=config.timeout_lp, timeout_milp=config.timeout_milp,
                 timeout_final_lp=config.timeout_final_lp, timeout_final_milp=config.timeout_final_milp,
                 use_default_heuristic=config.use_default_heuristic, testing=False):
        self.ir_list = ir_list
        self.nn = nn

        if domain in {'deeppoly', 'refinepoly'}:
            print(f'[INFO] Using {domain} manager')
            self.man = fppoly_manager_alloc()
            self.is_greater = is_greater
        elif domain in {'deepzono', 'refinezono'}:
            print(f'[INFO] Using {domain} manager')
            self.man = zonoml_manager_alloc()
            self.is_greater = is_greater_zono
        else:
            raise NotImplementedError(f'Domain {domain} not supported')

        self.domain = domain
        self.refine = domain in {'refinezono', 'refinepoly'}
        self.complete = complete
        self.use_default_heuristic = use_default_heuristic

        self.label = label
        self.prop = prop
        self.output_constraints = output_constraints

        self.relu_groups = []
        self.approx_k = approx_k
        self.K = K
        self.s = s

        self.timeout_lp = timeout_lp
        self.timeout_final_lp = timeout_final_lp
        self.use_milp = use_milp
        self.partial_milp = partial_milp
        self.max_milp_neurons = max_milp_neurons
        self.timeout_milp = timeout_milp
        self.timeout_final_milp = timeout_final_milp

        self.testing = testing

    def __del__(self):
        elina_manager_free(self.man)

    def get_abstract0(self):
        element = self.ir_list[0].transformer(self.man)
        nlb, nub = [], []
        for i in range(1, len(self.ir_list)):
            node = self.ir_list[i]
            # print(str(node.__class__.__name__).center(80, '-'))
            if type(node) in [DeeppolyReluNode, DeeppolySigmoidNode, DeeppolyTanhNode,
                              DeepzonoRelu, DeepzonoSigmoid, DeepzonoTanh]:
                element = self.ir_list[i].transformer(self.nn, self.man, element, nlb, nub,
                                                      self.relu_groups, self.refine,
                                                      self.timeout_lp, self.timeout_milp,
                                                      self.use_default_heuristic, self.testing,
                                                      use_milp=self.use_milp, approx=self.approx_k, K=self.K, s=self.s)
            else:
                element = self.ir_list[i].transformer(self.nn, self.man, element, nlb, nub,
                                                      self.relu_groups, self.refine,
                                                      self.timeout_lp, self.timeout_milp,
                                                      self.use_default_heuristic, self.testing)
            element = element
        if self.refine:
            gc.collect()

        return element, nlb, nub

    def analyze_acasxu(self, element, nlb, nub, model, vars_list, counter):
        print(f'[INFO] We have output constraints. Verifying properties...')
        start = time.time()
        success = True
        for or_list in self.output_constraints:  # AND
            print(f'[INFO] Verifying {or_list}...')
            or_result = False
            for constr_tuple in or_list:  # OR
                print(f'[INFO] Verifying {constr_tuple}...', end=' ')
                if constr_tuple[1] == -1:  # the case of a variable <= a constant, the second element is -1
                    or_result = (nub[-1][constr_tuple[0]] <= constr_tuple[2])
                else:
                    if 'zono' in self.domain:
                        or_result = self.is_greater(self.man, element, constr_tuple[0], constr_tuple[1])
                    else:
                        or_result = self.is_greater(self.man, element, constr_tuple[0], constr_tuple[1],
                                                    self.use_default_heuristic)
                print(f'{or_result}')
                if or_result:
                    break

                if not self.refine:
                    continue

                print(f'[INFO] Verifying {constr_tuple} by LP Model...', end=' ')
                if constr_tuple[1] == -1:  # the case of a variable <= a constant, the second element is -1
                    obj = LinExpr() + constr_tuple[2] - vars_list[counter + constr_tuple[0]]
                else:
                    obj = LinExpr() + vars_list[counter + constr_tuple[0]] - vars_list[counter + constr_tuple[1]]
                model.setObjective(obj, GRB.MINIMIZE)
                model.optimize()
                if model.Status == 6:
                    or_result = True
                elif model.Status == 2 and model.ObjVal >= 0.0:
                    or_result = True
                else:
                    or_result = False
                try:
                    print(f'{or_result} {model.Status}({model.ObjVal})')
                except:
                    print(f'{or_result} {model.Status}')
                if or_result:
                    break

            if not or_result:
                success = False
                break
        print(f'[INFO] Result of verification: {success} {time.time() - start:.2f}s')
        return success, nlb, nub, None, None

    def init_grb_model(self, nlb, nub):
        print(f'[INFO] Building Main LP/MILP model...', end=' ')
        time_start = time.time()
        self.nn.ffn_counter = 0
        self.nn.conv_counter = 0
        self.nn.pool_counter = 0
        self.nn.pad_counter = 0
        self.nn.concat_counter = 0
        self.nn.tile_counter = 0
        self.nn.residual_counter = 0
        self.nn.activation_counter = 0
        counter, vars_list, model, kact_constrs = create_model(self.nn, self.nn.specLB, self.nn.specUB, nlb, nub,
                                                               self.relu_groups, self.nn.numlayer, self.complete)
        print(f'{time.time() - time_start:.2f}s')
        timeout = self.timeout_final_milp if self.use_milp else self.timeout_final_lp
        model.setParam(GRB.Param.TimeLimit, timeout)
        model.setParam(GRB.Param.Cutoff, 1e-6)
        # model.setParam(GRB.Param.OptimalityTol, 1e-3)
        # model.setParam(GRB.Param.FeasibilityTol, 1e-3)
        model.setParam(GRB.Param.NumericFocus, config.numerical_focus)
        # model.setParam(GRB.Param.MarkowitzTol, 0.1)
        model.setParam(GRB.Param.Method, 3)
        print(f'[INFO] Setting timeout to {timeout}s')
        print(f'[INFO] Setting cutoff to 1e-6')
        # print(f'[INFO] Setting optimality tolerance to 1e-3')
        # print(f'[INFO] Setting feasibility tolerance to 1e-3')
        print(f'[INFO] Setting numeric focus to {config.numerical_focus}')
        # print(f'[INFO] Setting Markowitz tolerance to 0.1')
        print(f'[INFO] Setting method to 3 (Concurrent)')
        model.update()

        vars_num = len(vars_list)
        print(f'[INFO] Model has {vars_num} variables and {len(model.getConstrs())} constraints')

        if self.partial_milp == 0:
            return model, vars_list, counter, None, None, None

        print(f'[INFO] Building Partial MILP model...', end=' ')
        time_start = time.time()
        self.nn.ffn_counter = 0
        self.nn.conv_counter = 0
        self.nn.pool_counter = 0
        self.nn.pad_counter = 0
        self.nn.concat_counter = 0
        self.nn.tile_counter = 0
        self.nn.residual_counter = 0
        self.nn.activation_counter = 0
        counter_partial_milp, vars_list_partial_milp, model_partial_milp = \
            create_model(self.nn, self.nn.specLB, self.nn.specUB, nlb, nub, self.relu_groups, self.nn.numlayer,
                         self.complete, partial_milp=self.partial_milp, max_milp_neurons=self.max_milp_neurons)
        print(f'{time.time() - time_start:.2f}s')

        model_partial_milp.setParam(GRB.Param.TimeLimit, self.timeout_final_milp)
        print(f'[INFO] Setting timeout to {self.timeout_final_milp}s')
        model_partial_milp.update()
        print(f'[INFO] Model has {len(vars_list_partial_milp)} variables '
              f'and {len(model_partial_milp.getConstrs())} constraints')
        return model, vars_list, counter, model_partial_milp, vars_list_partial_milp, counter_partial_milp

    def analyze(self, terminate_on_failure: bool = True):
        print(f'[INFO] {self.domain} analyzing {self.label}...')
        element, nlb, nub = self.get_abstract0()
        output_size = self.ir_list[-1].output_length

        infeasible_model = None
        model = vars_list = counter = model_partial_milp = vars_list_partial_milp = counter_partial_milp = None
        if self.domain == 'refinepoly':
            result = self.init_grb_model(nlb, nub)
            model, vars_list, counter, model_partial_milp, vars_list_partial_milp, counter_partial_milp = result

        # ACAS XU
        if self.output_constraints is not None:
            return self.analyze_acasxu(element, nlb, nub, model, vars_list, counter)

        dominant_class = -1
        label_failed = []
        x = None
        candidate_labels = [i for i in range(output_size)] if self.label == -1 else [self.label]
        adv_labels = [i for i in range(output_size)] if self.prop == -1 else self.prop

        print(f'[INFO] Verify {candidate_labels} vs {adv_labels}...')
        for label in candidate_labels:
            domaint = True
            for adv_label in adv_labels:
                if label == adv_label:
                    continue
                # print(f'[INFO] Verifying {label} >= {adv_label} by', end=' ')
                time_start = time.time()
                if 'zono' in self.domain:
                    # print(f'DeepZono', end=' ')
                    result = self.is_greater(self.man, element, label, adv_label)
                else:
                    # print(f'DeepPoly', end=' ')
                    result = self.is_greater(self.man, element, label, adv_label, self.use_default_heuristic)
                # print(f'{result} {time.time() - time_start:.2f}s')

                if not self.refine:
                    if self.label != -1 and not result:
                        label_failed.append(adv_label)
                        domaint = False
                    continue

                if result:
                    continue

                domaint = False
                if self.refine:
                    print(f'[INFO] Verifying {label} >= {adv_label} by {self.domain}', end=' ')
                    obj = LinExpr() + vars_list[counter + label] - vars_list[counter + adv_label]
                    model.setObjective(obj, GRB.MINIMIZE)

                    # if self.complete:
                    #     print(f'Complete MILP model', end=' ')
                    #     model.optimize(milp_callback)
                    #     print(f'({model.Runtime:.2f}s)')
                    #     logger = Logger.current_logger
                    #     logger.record_lp(model.Runtime, len(model.getConstrs()), model.Status, 0.0)
                    #     if not hasattr(model, "objbound") or model.objbound <= 0:
                    #         if self.label != -1:
                    #             label_failed.append(adv_label)
                    #         # if model.solcount > 0:
                    #         #     x = model.x[:len(self.nn.specLB)]
                    #         domaint = False
                    #     elif model.Status in [3,4]:
                    #         warnings.warn(f"Model status: {model.Status} (INFEASIBLE or INF_OR_UNB)")
                    #         domaint = False
                    #     else:
                    #         continue
                    #     if terminate_on_failure:
                    #         return dominant_class, nlb, nub, label_failed, x

                    print(f'LP model', end=' ')
                    model.optimize(lp_callback)
                    # model.optimize()
                    print(f"{model.Runtime:.2f}s", end=' ')

                    logger = Logger.current_logger
                    if logger is not None:
                        try:
                            objval = model.objval
                        except Exception:
                            objval = 0.0
                        logger.record_lp(model.Runtime, len(model.getConstrs()), model.Status, objval)

                    if model.Status == 6:
                        print(f"Status: 6 (CUTOFF)")
                        domaint = True
                    elif model.Status == 2:
                        print(f"Status: 2 (OPTIMAL), objval={model.objval}")
                        domaint = True
                        if model.objval < 0 and model.objval != math.inf:
                            x = model.x[0:len(self.nn.specLB)]
                            domaint = False
                    elif model.Status == 9:
                        print(f"Status: 9 (TIME LIMIT)")
                        domaint = False
                    elif model.Status == 11:
                        print(f"Status: 11 (INTERRUPTED)")
                        domaint = False
                    elif model.Status == 4:
                        print("Status: 4 (INF_OR_UNBD)")
                        domaint = False
                        infeasible_model = model
                        # domaint = self.solve_by_relaxed_model(model, counter + label, counter + adv_label)
                    elif model.Status == 3:
                        print("Status: 3 (INFEASIBLE)")
                        domaint = False
                        infeasible_model = model
                        # domaint = self.solve_by_relaxed_model(model, counter + label, counter + adv_label)
                    else:
                        print(f"Status: {model.Status}")
                        domaint = False

                    if self.partial_milp == 0:
                        if not domaint and terminate_on_failure:
                            if self.label != -1:
                                label_failed.append(adv_label)
                            break
                        continue

                    # print("Partial MILP model", end=" ")
                    # obj = LinExpr()
                    # obj += 1 * vars_list_partial_milp[counter_partial_milp + label]
                    # obj += -1 * vars_list_partial_milp[counter_partial_milp + adv_label]
                    # model_partial_milp.setObjective(obj, GRB.MINIMIZE)
                    # model_partial_milp.optimize(milp_callback)
                    # print(f"Run time: {model.Runtime:.2f}s")
                    # if model_partial_milp.Status in [2, 9, 11] and model_partial_milp.ObjBound > 0:
                    #     print(f"Model status: {model_partial_milp.Status}, ObjBound={model_partial_milp.ObjBound}")
                    # elif model_partial_milp.Status not in [2, 9, 11]:
                    #     print("Partial milp model was not successful status is", model_partial_milp.Status)
                    #     domaint = False
                    #     # model_partial_milp.write("final.mps")
                    # if not domaint and terminate_on_failure:
                    #     if self.label != -1:
                    #         label_failed.append(adv_label)
                    #     break
            if domaint:
                print(f'[INFO] {label} is the dominant class')
                dominant_class = label
                break

        elina_abstract0_free(self.man, element)

        return dominant_class, nlb, nub, label_failed, x, infeasible_model

    # def solve_by_relaxed_model(self, model: gurobipy.Model, var_label: int, var_adv_label: int):
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
    #     # model.optimize()
    #     print(f"{model.Runtime:.2f}s", end=' ')
    #
    #     logger = Logger.current_logger
    #     if logger is not None:
    #         try:
    #             objval = model.objval
    #         except Exception:
    #             objval = 0.0
    #         logger.record_lp(model.Runtime, len(model.getConstrs()), model.Status, objval)
    #
    #     if model.Status == 6:
    #         print(f"Status: 6 (CUTOFF)")
    #         domaint = True
    #     elif model.Status == 2:
    #         print(f"Status: 2 (OPTIMAL), objval={model.objval}")
    #         domaint = True
    #         if model.objval < 0 and model.objval != math.inf:
    #             x = model.x[0:len(self.nn.specLB)]
    #             domaint = False
    #     elif model.Status == 9:
    #         print(f"Status: 9 (TIME LIMIT)")
    #         domaint = False
    #     elif model.Status == 11:
    #         print(f"Status: 11 (INTERRUPTED)")
    #         domaint = False
    #     elif model.Status == 4:
    #         print("Status: 4 (INF_OR_UNBD)")
    #         domaint = False
    #     elif model.Status == 3:
    #         print("Status: 3 (INFEASIBLE)")
    #         domaint = False
    #     else:
    #         print(f"Status: {model.Status}")
    #         domaint = False
    #
    #     return domaint

        # relaxed_model = model.copy()
        # relaxed_model.setParam('OutputFlag', 0)
        # timeout = self.timeout_final_milp if self.use_milp else self.timeout_final_lp
        # relaxed_model.setParam(GRB.Param.TimeLimit, timeout)
        # relaxed_model.setParam(GRB.Param.Cutoff, 0.00001)
        # relaxed_model.setParam(GRB.Param.FeasibilityTol, 2e-5)
        # relaxed_model.setParam(GRB.Param.NumericFocus, 2)
        # relaxed_model.setParam(GRB.Param.Method, 3)
        #
        # kact_constrs = [constr for constr in relaxed_model.getConstrs()
        #                 if constr.ConstrName.startswith('kact') or constr.ConstrName.startswith('relu_triangle')]
        # print(f'[INFO] Relaxing {len(kact_constrs)} kact constraints', end=' ')
        # relaxed_model.feasRelax(2, True, None, None, None, kact_constrs, [1.0] * len(kact_constrs))
        #
        # relaxed_model.setObjective(relaxed_model.getVars()[var_label] - relaxed_model.getVars()[var_adv_label],
        #                            GRB.MINIMIZE)
        # relaxed_model.optimize()
        # status = relaxed_model.Status
        #
        # logger = Logger.current_logger
        # if logger is not None:
        #     try:
        #         obj_val = relaxed_model.getObjective().getValue()
        #     except Exception:
        #         obj_val = 0.0
        #     logger.record_lp(relaxed_model.Runtime, len(relaxed_model.getConstrs()), 2, obj_val)
        #
        # print(f"Status {status}", end=' ')
        # if status == 2:
        #     obj_val = relaxed_model.getObjective().getValue()
        #     print(f"obj ({relaxed_model.getObjective()})>={obj_val:.4f} ({model.Runtime:.2f}s)")
        #     if obj_val > 0:
        #         return True
        # elif status == 6:
        #     print(f'({model.Runtime:.2f}s)')
        #     return True
        # else:
        #     print(f'({model.Runtime:.2f}s)')
        #     return False
