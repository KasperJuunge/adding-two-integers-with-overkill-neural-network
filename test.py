from model import AddNet
import torch
import numpy as np

# Instatiate model
model = AddNet()

# Load trained model
saved_training = torch.load('C:\\Users\\kaspe\\Code Projects\\Add_DNN\\best_model.torch')
state_dict = saved_training['model_state_dict']
model.load_state_dict(state_dict)
model.eval()

# Promt user for input
print('Enter an integer: ')
x = int(input())
print('Enter another integer: ')
y = int(input())

# Pass to model
inp = torch.tensor([[x, y]], dtype=torch.float32)
print('The result is: ')
print(model(inp).item())