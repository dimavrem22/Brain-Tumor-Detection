from torch import nn
from enum import Enum
from torchvision.models import (
    efficientnet_b0,
    efficientnet_b1,
    efficientnet_b2,
    efficientnet_b3,
    efficientnet_b4,
    efficientnet_b5,
    efficientnet_b6,
    efficientnet_b7,
)

from src.models.model_utils import make_mlp_layers


class MODEL_MODES(Enum):
    TRAINING = 1
    INFERENCE = 2
    FINE_TUNE = 3


def get_efficientnet(version="b0", pretrained=True):
    # Mapping of version string to the corresponding EfficientNet function
    efficientnet_versions = {
        "b0": efficientnet_b0,
        "b1": efficientnet_b1,
        "b2": efficientnet_b2,
        "b3": efficientnet_b3,
        "b4": efficientnet_b4,
        "b5": efficientnet_b5,
        "b6": efficientnet_b6,
        "b7": efficientnet_b7,
    }
    return efficientnet_versions[version](pretrained)


class EfficientNet(nn.Module):
    def __init__(
        self,
        efficient_net_v="b0",
        pretrained=True,
        predictor_hidden_dims=[],
        output_dim=4,
    ):
        super(EfficientNet, self).__init__()
        self.pretrained = pretrained
        self.mode = MODEL_MODES.TRAINING

        # EfficientNet
        self.efficient_net = get_efficientnet(efficient_net_v, pretrained)

        # Modifying regressor
        mlp_dims = [1000] + predictor_hidden_dims + [output_dim]
        self.efficient_net.classifier = nn.Sequential(
            *self.efficient_net.classifier,  # Unpack the original classifier layers
            *make_mlp_layers(dims=mlp_dims, activation="ReLU", last_layer_activation=False)
        )

    def forward(self, x):
        return self.efficient_net(x)

    def set_mode(self, mode: MODEL_MODES):

        self.mode = mode

        if mode == MODEL_MODES.TRAINING:
            for param in self.efficient_net.parameters():
                param.requires_grad = True

        elif mode == MODEL_MODES.FINE_TUNE or mode == MODEL_MODES.INFERENCE:
            for param in self.efficient_net.parameters():
                param.requires_grad = False

        if mode == MODEL_MODES.FINE_TUNE:
            for param in self.efficient_net.classifier.parameters():
                param.requires_grad = True

