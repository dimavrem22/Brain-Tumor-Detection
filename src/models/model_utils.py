import torch.nn as nn
from typing import Tuple, Optional
from enum import auto, StrEnum

activations = {
    "ReLU": nn.ReLU(),
    "Sigmoid": nn.Sigmoid(),
    "Tanh": nn.Tanh(),
}

def make_mlp_layers(dims, activation="ReLU", last_layer_activation=False):

    layers = []

    for idx, dim in enumerate(dims[:-1]):
        layers.append(nn.Linear(dim, dims[idx + 1]))
        layers.append(activations[activation])

    if last_layer_activation:
        return layers
    
    return layers[:-1]