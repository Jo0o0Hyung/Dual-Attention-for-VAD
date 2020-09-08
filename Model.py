import torch.nn as nn

from RNN_Model import BasicRNN, AttentionRNN
from DNN_Model import PostDNN


class Model(nn.Module):
    def __init__(self, rnn_model, input_size, rnn_hidden_size, num_layers, dnn_hidden_size, seq_len,
                 attention_type, num_classes=1):
        super(Model, self).__init__()
        self.rnn_model = rnn_model
        self.input_size = input_size
        self.rnn_hidden_size = rnn_hidden_size
        self.num_layers = num_layers
        self.dnn_hidden_size = dnn_hidden_size
        self.seq_len = seq_len
        self.attention_type = attention_type
        self.num_classes = num_classes

        if rnn_model == 'BasicRNN':
            self.rnn = BasicRNN(input_size=self.input_size, hidden_size=self.rnn_hidden_size, num_layers=self.num_layers)
        elif rnn_model == 'AttentionRNN':
            self.rnn = AttentionRNN(input_size=self.input_size, hidden_size=self.rnn_hidden_size, seq_len=self.seq_len,
                                    num_layers=self.num_layers, attention_type=self.attention_type)

        self.dnn = PostDNN(input_size=self.rnn_hidden_size, hidden_size=self.dnn_hidden_size,
                           num_classes=self.num_classes)

    def forward(self, x):
        if self.rnn_model == 'BasicRNN':
            output, hidden = self.rnn(x)
            linear_out, sig_out = self.dnn(output)

            return linear_out, sig_out

        elif self.rnn_model == 'AttentionRNN':
            output, hidden = self.rnn(x)
            linear_out, sig_out = self.dnn(output)

            return linear_out, sig_out
