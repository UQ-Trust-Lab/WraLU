from typing import List

from kact.constants import KRELU_METHODS, KSIGMOID_METHODS, KTANH_METHODS


def check_methods(methods: List[str]) -> List[str]:
    for method in methods:
        if method not in KRELU_METHODS.keys() \
                and method not in KSIGMOID_METHODS.keys() \
                and method not in KTANH_METHODS.keys():
            raise ValueError(f"Unknown method: {method}")
    return _reorder_methods(methods)

def _reorder_methods(methods: List[str]) -> List[str]:
    if "triangle" in methods:
        methods.remove("triangle")
        methods.insert(0, "triangle")

    if "pycdd" in methods:
        methods.remove("pycdd")
        methods.insert(0, "pycdd")

    if "cdd" in methods:
        methods.remove("cdd")
        methods.insert(0, "cdd")

    return methods