import torch
import matplotlib.pyplot as plt
import numpy  as np
import torch.nn as nn

class CnnDisease(nn.Module):
    def __init__(self):
        super(CnnDisease, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=5)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, 2)
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self,x):
        x = x.unsqueeze(2)  
        x = torch.relu(self.conv1(x))
        x = self.pool(x).squeeze(2)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        
        return x                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        