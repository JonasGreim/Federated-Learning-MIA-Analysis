import math


def map_regularization(weight_decay):
    return math.isclose(weight_decay, 0.0005, rel_tol=1e-9)


def map_distribution(alpha):
    if alpha == 0.0:
        return "IID"
    elif alpha == 0.5:
        return "semi-non-IID"
    elif alpha == 0.1:
        return "non-IID"
    else:
        return f"unknown ({alpha})"


def map_model_name(model):
    if model == "simple_model":
        return "Shokri-CNN"
    elif model == "simple_model_with_dropout":
        return "Shokri-CNN"
    elif model == "complex_model":
        return "ResNet-18"
    elif model == "complex_model_with_dropout":
        return "ResNet-18"
    else:
        return f"unknown ({model})"

def map_regularization_diagram(regularization):
    if regularization:
        return "Ja"
    else:
        return "Nein"
    return regularization
