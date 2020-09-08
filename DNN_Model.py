import torch.nn as nn


# fully connected layer with 1 hidden layer.
# PostDNN determines the each time steps of LSTM's output.
class PostDNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=1):
        super(PostDNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.pDNN = nn.Sequential(
            nn.Linear(in_features=self.input_size, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.num_classes)
        )

        self.sig = nn.Sigmoid()

    def forward(self, x):
        linear_out = self.pDNN(x)
        linear_out = linear_out.squeeze(-1)
        sigmoid_out = self.sig(linear_out)

        return linear_out, sigmoid_out
