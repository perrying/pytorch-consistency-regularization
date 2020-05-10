from .cross_entropy import CrossEntropy
from .mean_squared import MeanSquared


def gen_consistency(type, cfg):
    if type == "ce":
        return CrossEntropy()
    elif type == "ms":
        return MeanSquared()
    else:
        return None