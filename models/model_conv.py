import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm import tqdm


class Model(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        # define required layers
        fet_out_1 = 12
        fet_out_2 = 36
        fet_out_3 = 108
        fet_out_4 = 324
        n_classes = 256
        self.conv1 = nn.Conv2d(3, fet_out_1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(fet_out_1)
        self.conv2 = nn.Conv2d(fet_out_1, fet_out_2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(fet_out_2)
        self.conv3 = nn.Conv2d(fet_out_2, fet_out_3, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(fet_out_3)
        self.conv4 = nn.Conv2d(fet_out_3, fet_out_4, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(fet_out_4)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.main_avg_pool = nn.AvgPool2d(2)
        self.conv_to_class = nn.Conv2d(fet_out_4, n_classes, kernel_size=1)
        self.activation = F.relu

    def forward(self, x):
        # reshape input first with batch size tracked
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.main_avg_pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.main_avg_pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.main_avg_pool(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.activation(x)
        x = F.avg_pool2d(x, kernel_size=x.size()[-1])

        x = self.conv_to_class(x)
        x = x.view(x.size(0), -1)
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
