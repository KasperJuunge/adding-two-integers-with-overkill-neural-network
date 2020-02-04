import torch
import torch.nn as nn
import torch.nn.functional as F


class AddNet(nn.Module):

    def __init__(self):
        super(AddNet, self).__init__()

        self.fc1 = nn.Linear(2,256) 
        self.fc2 = nn.Linear(256,512)
        self.fc3 = nn.Linear(512,1024)
        self.fc4 = nn.Linear(1024,512)
        self.fc5 = nn.Linear(512,256) 
        self.fc6 = nn.Linear(256, 1)

        self.relu = nn.ReLU()

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.xavier_uniform_(self.fc5.weight)
        nn.init.xavier_uniform_(self.fc6.weight)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        out = self.relu(self.fc6(x))
        return out


