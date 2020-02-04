import torch
import pickle
import numpy as np
from torch.utils.data import TensorDataset,DataLoader
from sklearn.model_selection import train_test_split
from os import path
from serialization import deserialize, serialize


# Check if dataset already exists?
if path.isfile('C:\\Users\\kaspe\\Code Projects\\Add_DNN\\inputs.pickle'):
    print('Data does already exist..')

    # Load Data
    inputs = deserialize('C:\\Users\\kaspe\\Code Projects\\Add_DNN\\inputs.pickle')

    # Visualize
    print(inputs[0].shape, inputs[1].shape, inputs[2].shape)


else:
    print('Data does not already exist..')
    
    # Create Data
    inputs = np.random.randint(low=0, high=10, size=[10000,2])
    
    # Convert to torch tensors
    inputs = torch.from_numpy(inputs)

    # Convert to float
    inputs = inputs.float()
    
    # Train-test split
    inputs_train, inputs_test = train_test_split(inputs,train_size=0.7,test_size=0.3)
    inputs_test, inputs_val = train_test_split(inputs_test,train_size=0.5,test_size=0.5)
    
    # Convert to torch tensors
    inputs = [inputs_train, inputs_val, inputs_test]
    
    # Visualize
    print(inputs[0].shape, inputs[1].shape, inputs[2].shape)
    
    # Save
    serialize(inputs, 'C:\\Users\\kaspe\\Code Projects\\Add_DNN\\inputs.pickle')
    