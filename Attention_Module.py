import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    def __init__(self, kernel_size=11):
        super(TemporalAttention, self).__init__()
        self.kernel_size = kernel_size

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=5, kernel_size=self.kernel_size, padding=5),
            nn.BatchNorm1d(5),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=5, kernel_size=self.kernel_size, padding=5),
            nn.BatchNorm1d(5),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=5, kernel_size=self.kernel_size, padding=5),
            nn.BatchNorm1d(5),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=1, kernel_size=self.kernel_size, padding=5),
            nn.BatchNorm1d(1)
        )

    def forward(self, x):
        x_avg = torch.mean(x, dim=-1).unsqueeze(dim=1)
        x_std = torch.std(x, dim=-1).unsqueeze(dim=1)
        x_max, _ = torch.max(x, dim=-1)
        x_max = x_max.unsqueeze(dim=1)

        x_total = torch.cat((x_avg, x_std, x_max), dim=1)

        output = self.conv1(x_total)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = torch.transpose(output, 2, 1)
        output = output.expand_as(x)

        return output


class FrequentialAttention(nn.Module):
    def __init__(self, sequential_length, kernel_size=21):
        super(FrequentialAttention, self).__init__()
        self.sequential_length = sequential_length
        self.kernel_size = kernel_size

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=5, kernel_size=self.kernel_size, padding=10),
            nn.BatchNorm1d(5),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=5, kernel_size=self.kernel_size, padding=10),
            nn.BatchNorm1d(5),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=5, kernel_size=self.kernel_size, padding=10),
            nn.BatchNorm1d(5),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=1, kernel_size=self.kernel_size, padding=10),
            nn.BatchNorm1d(1)
        )

    def forward(self, x):
        # Training
        x_sub = x[:, :self.sequential_length, :]
        x_sub_avg = torch.mean(x_sub, dim=1).unsqueeze(dim=1)
        x_sub_std = torch.std(x_sub, dim=1).unsqueeze(dim=1)
        x_sub_max, _ = torch.max(x_sub, dim=1)
        x_sub_max = x_sub_max.unsqueeze(dim=1)

        x_sub_total = torch.cat((x_sub_avg, x_sub_std, x_sub_max), dim=1)

        output = self.conv1(x_sub_total)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)

        output = output.expand_as(x_sub)

        # Validation or Test
        if x.size()[1] > self.sequential_length:
            iter_idx = (x.size()[1] // self.sequential_length) + 1
            for i in range(1, iter_idx):
                if i == iter_idx-1:
                    x_sub = x[:, i * self.sequential_length:, :]
                    if x_sub.size()[1] == 0:
                        break
                    x_sub_avg = torch.mean(x_sub, dim=1).unsqueeze(dim=1)
                    x_sub_std = torch.std(x_sub, dim=1).unsqueeze(dim=1)
                    x_sub_max, _ = torch.max(x_sub, dim=1)
                    x_sub_max = x_sub_max.unsqueeze(dim=1)

                    x_sub_total = torch.cat((x_sub_avg, x_sub_std, x_sub_max), dim=1)

                    output_sub = self.conv1(x_sub_total)
                    output_sub = self.conv2(output_sub)
                    output_sub = self.conv3(output_sub)
                    output_sub = self.conv4(output_sub)
                    output_sub = output_sub.expand_as(x_sub)

                    output = torch.cat((output, output_sub), dim=1)

                else:
                    x_sub = x[:, i*self.sequential_length:(i+1)*self.sequential_length, :]

                    x_sub_avg = torch.mean(x_sub, dim=1).unsqueeze(dim=1)
                    x_sub_std = torch.std(x_sub, dim=1).unsqueeze(dim=1)
                    x_sub_max, _ = torch.max(x_sub, dim=1)
                    x_sub_max = x_sub_max.unsqueeze(dim=1)

                    x_sub_total = torch.cat((x_sub_avg, x_sub_std, x_sub_max), dim=1)

                    output_sub = self.conv1(x_sub_total)
                    output_sub = self.conv2(output_sub)
                    output_sub = self.conv3(output_sub)
                    output_sub = self.conv4(output_sub)
                    output_sub = output_sub.expand_as(x_sub)

                    output = torch.cat((output, output_sub), dim=1)

        return output


class Dual_Attention_1(nn.Module):
    def __init__(self):
        super(Dual_Attention_1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=7, padding=3),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=7, padding=3),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=7, padding=3),
            nn.BatchNorm2d(1)
        )

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        output = self.conv3(x)
        output = output.squeeze(dim=1)

        return output
