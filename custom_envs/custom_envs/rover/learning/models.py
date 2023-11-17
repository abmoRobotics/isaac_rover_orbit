

from skrl.models.torch.gaussian import GaussianMixin
from skrl.models.torch.deterministic import DeterministicMixin
from skrl.models.torch.base import Model as BaseModel

import torch.nn as nn
import torch


def get_activation(activation_name):
    activation_fns = {
        "leaky_relu": nn.LeakyReLU(),
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "elu": nn.ELU(),
        "relu6": nn.ReLU6(),
        "selu": nn.SELU(),
    }
    if activation_name not in activation_fns:
        raise ValueError(f"Activation function {activation_name} not supported.")
    return activation_fns[activation_name]

class HeightmapEncoder(nn.Module):
    def __init__(self, in_channels, encoder_features=[80, 60], encoder_activation="leaky_relu"):
        super().__init__()
        self.encoder_layers = nn.ModuleList()
        for feature in encoder_features:
            self.encoder_layers.append(nn.Linear(in_channels, feature))
            self.encoder_layers.append(get_activation(encoder_activation))
            in_channels = feature

    def forward(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
        return x
    
class GaussianNeuralNetwork(GaussianMixin, BaseModel):
    """Gaussian neural network model."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        encoder_features=[80, 60],
        encoder_activation="leaky_relu",
        **kwargs,
    ):
        """Initialize the Gaussian neural network model.

        Args:
            observation_space (gym.spaces.Space): The observation space of the environment.
            action_space (gym.spaces.Space): The action space of the environment.
            device (torch.device): The device to use for computation.
            encoder_features (list): The number of features for each encoder layer.
            encoder_activation (str): The activation function to use for each encoder layer.
        """
        BaseModel.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions=False, clip_log_std=True, min_log_std=-20.0, max_log_std=2.0, reduction="sum")

        self.proprioception_channels = 4
        self.dense_channels = 634
        self.sparse_channels = 1112

        self.dense_encoder = HeightmapEncoder(self.dense_channels, encoder_features, encoder_activation)
        self.sparse_encoder = HeightmapEncoder(self.sparse_channels, encoder_features, encoder_activation)

        self.mlp = nn.ModuleList()


        in_channels = self.proprioception_channels + encoder_features[-1] + encoder_features[-1]
        action_space = action_space.shape[0]
        mlp_features = [256, 160, 128]
        for feature in mlp_features:
            self.mlp.append(nn.Linear(in_channels, feature))
            self.mlp.append(get_activation(encoder_activation))
            in_channels = feature
        
        self.mlp.append(nn.Linear(in_channels, action_space))
        self.mlp.append(nn.Tanh())
        self.log_std_parameter = nn.Parameter(torch.zeros(action_space))


    def compute(self, states, role="actor"):
        dense_start = self.proprioception_channels
        dense_end = dense_start + self.dense_channels
        sparse_end = dense_end + self.sparse_channels
        x = states["states"]
        x0 = x[:, 0:4]
        x1 = self.dense_encoder(x[:, dense_start:dense_end])
        x2 = self.sparse_encoder(x[:, dense_end:sparse_end])
        
        x = torch.cat([x0, x1, x2], dim=1)
        for layer in self.mlp:
            x = layer(x)
        
        return x, self.log_std_parameter, {}
    
class DeterministicNeuralNetwork(DeterministicMixin, BaseModel):
    """Gaussian neural network model."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        encoder_features=[80, 60],
        encoder_activation="leaky_relu",
        **kwargs,
    ):
        """Initialize the Gaussian neural network model.

        Args:
            observation_space (gym.spaces.Space): The observation space of the environment.
            action_space (gym.spaces.Space): The action space of the environment.
            device (torch.device): The device to use for computation.
            encoder_features (list): The number of features for each encoder layer.
            encoder_activation (str): The activation function to use for each encoder layer.
        """
        BaseModel.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        self.proprioception_channels = 4
        self.dense_channels = 634
        self.sparse_channels = 1112

        self.dense_encoder = HeightmapEncoder(self.dense_channels, encoder_features, encoder_activation)
        self.sparse_encoder = HeightmapEncoder(self.sparse_channels, encoder_features, encoder_activation)

        self.mlp = nn.ModuleList()


        in_channels = self.proprioception_channels + encoder_features[-1] + encoder_features[-1]
        action_space = action_space.shape[0]
        mlp_features = [256, 160, 128]
        for feature in mlp_features:
            self.mlp.append(nn.Linear(in_channels, feature))
            self.mlp.append(get_activation(encoder_activation))
            in_channels = feature
        
        self.mlp.append(nn.Linear(in_channels, 1))

    def compute(self, states, role="actor"):
        dense_start = self.proprioception_channels
        dense_end = dense_start + self.dense_channels
        sparse_end = dense_end + self.sparse_channels
        x = states["states"]
        x0 = x[:, 0:dense_start]
        x1 = self.dense_encoder(x[:, dense_start:dense_end])
        x2 = self.sparse_encoder(x[:, dense_end:sparse_end])
        
        x = torch.cat([x0, x1, x2], dim=1)
        for layer in self.mlp:
            x = layer(x)
        
        return x, {}


if __name__ == "__main__":
    pass
    # import gym
    # import numpy as np
    # action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    # observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(28,), dtype=np.float32)
    # heightmap_encoder = GaussianNeuralNetwork(observation_space, action_space, "cpu")
    # torch.manual_seed(41)
    # states = torch.rand(10, 28)
    # #print(states)
    # for idx, param in enumerate(heightmap_encoder.parameters()):
    #     print(idx, param.shape)
    
    # torch.onnx.export(heightmap_encoder,
    #                   states, 
    #                   "heightmap_encoder.onnx",
    #                   export_params=True)