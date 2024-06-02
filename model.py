import torch
import matplotlib.pyplot as plt
import numpy  as np
import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(15,128) 
        self.fc2 = nn.Linear(128,32)
        self.fc3 = nn.Linear(32,2)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        
    def forward(self,x):
        # print(x.shape)
        x = self.linear(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        # print(x.shape)
        
        return x