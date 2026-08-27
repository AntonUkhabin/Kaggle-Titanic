import torch.nn as nn


class DNN(nn.Module):
    '''Fully connected neural network for binary tabular classification.'''

    def __init__(self, input_size: int):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(16, 1),
        )

    def forward(self, features):
        return self.mlp(features).squeeze(-1)


def build_torch_model(model_name: str, input_size: int) -> nn.Module:
    '''Build a PyTorch model by name.'''

    if model_name == 'dnn':
        return DNN(input_size=input_size)

    raise ValueError(f'Unknown PyTorch model: {model_name}')