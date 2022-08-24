import torch
import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self, fc_out):
        super(MyNet, self).__init__()

        # Convolution block
        self.conv_layer = nn.Sequential(
            nn.Conv1d(1, 4, 8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv1d(4, 8, 8, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2, padding=0)
        )

        # Fully connected block
        self.fc_layer = nn.Sequential(
            nn.Linear(8 * 55, 300),
            #nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(300, fc_out)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(-1, 8 * 55)
        x = self.fc_layer(x)
        return x