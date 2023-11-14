from analyzer import *
from onnx_translator import *
from optimizer import *
from tensorflow_translator import *

VALID_DOMAINS = ['deepzono', 'refinezono', 'deeppoly', 'refinepoly']


class ERAN:
    def __init__(self, model, session=None, is_onnx=False):
        translator = ONNXTranslator(model, False) if is_onnx else TFTranslator(model, session)
        operations, resources = translator.translate()
        self.input_shape = resources[0]["deeppoly"][2]
        self.optimizer = Optimizer(operations, resources)
        print(f'[INFO] This network has {str(self.optimizer.get_neuron_count())} neurons.')

    def analyze_box(self, specLB: np.ndarray, specUB: np.ndarray, domain: str,
                    label: int = -1, prop=-1,
                    output_constraints=None,
                    lexpr_weights: np.ndarray = None, lexpr_cst: np.ndarray = None, lexpr_dim: np.ndarray = None,
                    uexpr_weights: np.ndarray = None, uexpr_cst: np.ndarray = None, uexpr_dim: np.ndarray = None,
                    expr_size: int = 0,
                    spatial_constraints=None,
                    terminate_on_failure=True):

        assert domain in VALID_DOMAINS, f"The domain is not valid, it should be one of {VALID_DOMAINS}."
        specLB = np.reshape(specLB.copy(), (-1,))
        specUB = np.reshape(specUB.copy(), (-1,))
        nn = layers()
        nn.specLB = specLB
        nn.specUB = specUB

        if domain in {"deepzono", "refinezono"}:
            execute_list, output_info = self.optimizer.get_deepzono(nn, specLB, specUB)
        elif domain in {"deeppoly", "refinepoly"}:
            execute_list, output_info = self.optimizer.get_deeppoly(nn, specLB, specUB,
                                                                    lexpr_weights, lexpr_cst, lexpr_dim,
                                                                    uexpr_weights, uexpr_cst, uexpr_dim,
                                                                    expr_size, spatial_constraints)
        else:
            raise NotImplementedError(f"Domain {domain} is not implemented.")

        analyzer = Analyzer(execute_list, nn, domain, output_constraints, label, prop)

        dominant_class, nlb, nub, failed_labels, x, infeasible_model = analyzer.analyze(
            terminate_on_failure=terminate_on_failure)

        if failed_labels is not None and len(failed_labels) == 0:
            failed_labels = None  # rather return nothing than an incomplete list

        return dominant_class, nn, nlb, nub, failed_labels, x, infeasible_model

    def analyze_zonotope(self, zonotope, domain, timeout_lp, timeout_milp, use_default_heuristic,
                         output_constraints=None, testing=False, prop=-1):
        """
        This function runs the analysis with the provided model and session from the constructor, the box specified by specLB and specUB is used as input. Currently we have three domains, 'deepzono',      		'refinezono' and 'deeppoly'.

        Arguments
        ---------
        original : numpy.ndarray
            ndarray with the original input
        zonotope : numpy.ndarray
            ndarray with the zonotope
        domain : str
            either 'deepzono', 'refinezono', 'deeppoly' or 'refinepoly', decides which set of abstract transformers is used.

        Return
        ------
        dominant_class : int
            if the analysis is succesfull (it could prove robustness for this box) then the index of the class that dominates is returned
            if the analysis couldn't prove robustness then -1 is returned
        """
        assert domain in ['deepzono', 'refinezono'], "domain isn't valid, must be 'deepzono' or 'refinezono'"
        nn = layers()
        nn.zonotope = zonotope
        if domain == 'deepzono' or domain == 'refinezono':
            execute_list, output_info = self.optimizer.get_deepzono(nn, zonotope)
            analyzer = Analyzer(execute_list, nn, domain, output_constraints, label=-1, prop=prop)
        elif domain == 'deeppoly' or domain == 'refinepoly':
            assert 0
            # execute_list   = self.optimizer.get_deeppoly(original, zonotope, True)
            # analyzer       = Analyzer(execute_list, nn, domain, timeout_lp, timeout_milp, specnumber, use_default_heuristic)
        dominant_class, nlb, nub, failed_labels, x = analyzer.analyze()
        return dominant_class, nn, nlb, nub, failed_labels, x
