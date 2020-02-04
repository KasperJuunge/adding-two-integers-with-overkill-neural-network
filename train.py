import torch
import numpy as np
import pickle
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import AddNet
from sklearn.model_selection import train_test_split
from earlystopper import EarlyStopper
from tqdm import tqdm
from serialization import deserialize

print(torch.__version__)

## Control
batch_size = 128
nr_epochs = 1000
learning_rate = 1e-5
device = 'cuda'
maxPatience = 30
train_loss = []
test_loss = []


# Instatiation
model = AddNet()
model.to(device)
cost_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
earlyStopper = EarlyStopper(maxPatience=maxPatience)
tensorboard = SummaryWriter()

# Load Dataset
inputs_train, inputs_val, inputs_test = deserialize('C:\\Users\\kaspe\\Code Projects\\Add_DNN\\inputs.pickle')

# Visualize
print(inputs_train.shape, inputs_val.shape, inputs_test.shape)

# Create tensor dataset
dataset_train = TensorDataset(inputs_train)
dataset_val = TensorDataset(inputs_val)
dataset_test = TensorDataset(inputs_test)
# Create dataloaders
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)



train_loss = []
val_loss = []
i = 0

# Train loop
for epoch in tqdm(range(nr_epochs), position=0, leave=True):
    
    # Train
    ep_losses = []
    model.train()
    for batch in dataloader_train:
        inputs = batch[0]
        labels = inputs[:,0] + inputs[:,1]
        labels.unsqueeze_(-1)
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Compute loss
        optimizer.zero_grad()
        out = model(inputs)
        loss = cost_func(out, labels)

        # Optimize
        loss.backward()
        optimizer.step()
        ep_losses.append(loss.item())

    mean_loss = np.mean(ep_losses)
    tensorboard.add_scalar('Loss Train', mean_loss, epoch)
    train_loss.append(mean_loss)
    # print(mean_loss)    

    # Validation
    ep_losses = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader_val:
            inputs = batch[0]
            labels = inputs[:,0] + inputs[:,1]
            labels.unsqueeze_(-1)
            inputs = inputs.to(device)
            labels = labels.to(device)
        

            # Compute loss
            out = model(inputs)
            loss = cost_func(out, labels)
            ep_losses.append(loss.item())
    mean_loss = np.mean(ep_losses)
    val_loss.append(mean_loss)

    # Update Tensorboard
    tensorboard.add_scalar('Loss Test', mean_loss, epoch)
    
    # Check Earlystop
    stop_check = earlyStopper.check_early_stop(mean_loss,model)
    i+=1
    if stop_check:
        print('Earlystop has stopped the training!')
        break

    torch.save({
                'model_state_dict': earlyStopper.bestModel.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, 'best_model.torch')