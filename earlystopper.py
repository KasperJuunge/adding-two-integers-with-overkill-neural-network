import torch
import numpy as np
import copy
import torch
from torch import nn
from torch.nn import functional as F

class EarlyStopper():
    def __init__(self,maxPatience=10,saveName = 'EarlyStoppingCheckpoint.pth'):
        self.best_loss = 1e9
        self.maxPatience = maxPatience
        self.bestModel = None
        self.patience = 0
        self.saveName = saveName

    def check_early_stop(self, loss, model, optimizer = None):
        if loss<self.best_loss:
            self.bestModel = copy.deepcopy(model)
            self.patience = 0
            self.best_loss = loss
            self.saveModel(model,optimizer)
        else:
            self.patience += 1
        return self.patience>self.maxPatience

    def saveModel(self,model,optimizer):
        if optimizer is None:
                torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': None
                }, self.saveName)
        else:
                torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, self.saveName)
