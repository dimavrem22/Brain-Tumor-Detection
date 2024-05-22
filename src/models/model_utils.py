
import torch
import torch.nn as nn

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

def save_checkpoint(model, file_path):
    checkpoint = {
        "model_state_dict": model.state_dict(),
    }
    torch.save(checkpoint, file_path)
    print(f"Checkpoint saved to {file_path}")


def load_checkpoint(model, file_path):
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Checkpoint loaded from {file_path}")