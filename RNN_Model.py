import torch.nn as nn
from Attention_Module import TemporalAttention, FrequentialAttention, Dual_Attention_1


class BasicRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(BasicRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                           batch_first=True)

    def forward(self, x):
        output, hidden = self.rnn(x)

        return output, hidden


class AttentionRNN(nn.Module):
    def __init__(self, input_size, hidden_size, seq_len, num_layers, attention_type):
        super(AttentionRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.attention_type = attention_type

        self.AtRNN = nn.ModuleDict()

        for i in range(self.num_layers):
            if i == 0:
                self.AtRNN.update({'RNN_%d' % (i+1): nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                                                             num_layers=1, batch_first=True)})
            else:
                self.AtRNN.update({'RNN_%d' % (i+1): nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size,
                                                             num_layers=1, batch_first=True)})
        if self.attention_type == 'TA':
            self.TA = TemporalAttention()
        elif self.attention_type == 'FA':
            self.FA = FrequentialAttention(sequential_length=self.seq_len)
        elif self.attention_type == 'DA1':
            self.DA = Dual_Attention_1()
        elif self.attention_type == 'DA2':
            self.TA = TemporalAttention()
            self.FA = FrequentialAttention(sequential_length=self.seq_len)
        self.sig = nn.Sigmoid()

    def forward(self, x):

        for i in range(self.num_layers):
            if i == 0:
                output, hidden = self.AtRNN['RNN_%d' % (i+1)](x)
            else:
                output, hidden = self.AtRNN['RNN_%d' % (i+1)](output)

        if self.attention_type == 'TA':
            ta = self.TA(output)
            output = output + self.sig(ta)

        elif self.attention_type == 'FA':
            fa = self.TA(output)
            output = output + self.sig(fa)

        elif self.attention_type == 'DA1':
            da = self.DA(output)
            output = output + self.sig(da)

        elif self.attention_type == 'DA2':
            ta = self.TA(output)
            fa = self.FA(output)
            output = output + self.sig(ta + fa)

        return output, hidden
