import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate, dropout_rate):
        super(Bottleneck, self).__init__()
        self.dropout_rate = dropout_rate
        interChannels = 4 * growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        if self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
        out = torch.cat((x, out), 1)
        return out


class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate, dropout_rate):
        super(SingleLayer, self).__init__()
        self.dropout_rate = dropout_rate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        if self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
        out = torch.cat((x, out), 1)
        return out


class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


class Model(nn.Module):
    def __init__(self, nClasses=256, growthRate=12, depth=40, bottleneck=False,
                 dropout_rate=0.0, n_layers=3, **kwargs):
        super().__init__()

        self.n_layers = n_layers
        print("n_layers", n_layers)

        # dense blocks per layer
        nDenseBlocks = (depth - 4) // n_layers
        if bottleneck:
            nDenseBlocks //= 2

        if bottleneck:
            reduction = 0.5
        else:
            reduction = 1.0

        # initial convolution
        nChannels = 2 * growthRate
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1,
                               bias=False)

        for layer_n in range(1, n_layers + 1):
            dense_layer = self._make_dense(
                nChannels, growthRate, nDenseBlocks, bottleneck, dropout_rate)
            setattr(self, f'dense{layer_n}', dense_layer)
            nChannels += nDenseBlocks * growthRate
            if layer_n < n_layers:
                nOutChannels = int(math.floor(nChannels * reduction))
                trainsition_layer = Transition(nChannels, nOutChannels)
                setattr(self, f'trans{layer_n}', trainsition_layer)
                nChannels = nOutChannels

        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc = nn.Linear(nChannels, nClasses)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck, dropout_rate):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate, dropout_rate))
            else:
                layers.append(SingleLayer(nChannels, growthRate, dropout_rate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        for i in range(1, self.n_layers):
            dense_layer = getattr(self, f'dense{i}')
            trans_layer = getattr(self, f'trans{i}')
            out = trans_layer(dense_layer(out))
        last_dense_layer = getattr(self, f'dense{self.n_layers}')
        out = last_dense_layer(out)

        out = F.avg_pool2d(F.relu(self.bn1(out)), out.size()[-1])
        out = torch.squeeze(torch.squeeze(out, 2), 2)
        out = F.log_softmax(self.fc(out))
        return out

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
