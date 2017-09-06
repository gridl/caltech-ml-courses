import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm import tqdm


# model = torch.nn.Sequential(
#     torch.nn.Linear(64 * 64 * 3, 2048),
#     torch.nn.Sigmoid(),
#     torch.nn.Linear(2048, 1024),
#     torch.nn.Sigmoid(),
#     torch.nn.Linear(1024, 512),
#     torch.nn.Sigmoid(),
#     torch.nn.Linear(512, 257)
# )


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # define required layers
        self.fc1 = nn.Linear(64 * 64 * 3, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 257)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        # reshape input first with batch size tracked
        x = x.view(x.size(0), -1)
        # use required layers
        x = self.dropout(F.sigmoid(self.fc1(x)))
        x = self.dropout(F.sigmoid(self.fc2(x)))
        x = F.sigmoid(self.fc3(x))
        x = self.fc4(x)
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
