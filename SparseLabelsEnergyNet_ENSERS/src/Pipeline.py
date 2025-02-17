"""
class for model training and testing 
"""

import pdb
import h5py
import numpy as np
import torch as T
from torch.utils.data import DataLoader
# from torch.optim.lr_scheduler import ExponentialLR
# from torch.optim.lr_scheduler import MultiplicativeLR
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
import os
from os.path import dirname, realpath, join


class ModelPipeline():

    def __init__(self, Model, hyperParams, experPaths, rawData, dataset, utils, args):
        """
        ARGS:
            n_sensor (int): number of sensors
        """
        self.args = args
        self.info = args.logger.info

        self.hp = hyperParams
        self.rawData = rawData
        self.dataset = dataset
        self.path = experPaths

        self.lossFn = utils
        self.model = Model(hyperParams, self.lossFn, rawData.imDim, args).to(args.device)

        self.info(self)
        self.info(self.model)

    
    def saveModel(self, epoch, optimizer, scheduler):
        PATH = join(self.path.weights, f'weights_epoch{epoch}.tar')
        state = {'epoch': epoch,
                 'model_state_dict': self.model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'scheduler_state_dict': scheduler.state_dict()}
        T.save(state, PATH)
        print(f'model saved at epoch {epoch}')


    def loadModel(self, epoch, optimizer=None, scheduler=None):
        """Loads pre-trained network from file"""
        try:
            PATH = join(self.path.weights, f'weights_epoch{epoch}.tar')
            checkpoint = T.load(PATH, map_location=T.device(self.args.device))
            checkpoint_epoch = checkpoint['epoch']
            print(f'Found model at epoch: {checkpoint_epoch}')
        except FileNotFoundError:
            if epoch>0: raise FileNotFoundError(f'Error: Could not find PyTorch network at epoch: {epoch}')
            return

        # Load optimizer/scheduler
        if (not optimizer is None):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if (not scheduler is None):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return optimizer, scheduler


    def __repr__(self):
        # numParameters = count_parameters(self.model, self.args.logger)
        description = f'\n\n\
            {self.hp}'
        return description

    
    def trainingEpoch(self, optimizer, train_loader, epoch):
        self.model.train()
        running_loss = 0

        for batchIdx, data in enumerate(train_loader):
            optimizer.zero_grad()

            loss = self.lossFn(self.model, data, epoch, batchIdx)     
                                              
            loss.backward()
            optimizer.step()
            running_loss += loss.data
        return running_loss


    def train(self):
        hp = self.hp

        train_dataset = self.dataset(self.rawData, 'train', self.path, hp, device=self.args.device, info=self.info)
        train_loader = DataLoader(train_dataset, batch_size=hp.batchSizeTrain, shuffle=True)  # todo:change

        optimizer = T.optim.Adam(self.model.parameters(), lr=hp.lr)
        lr_lambda = lambda epoch: 1 ** epoch
        scheduler = T.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        # ------------------- load model from checkpoint -----------------------
        if hp.epochStartTrain:
            optimizer, scheduler = self.loadModel(hp.epochStartTrain, optimizer, scheduler)

        # List to store loss values
        loss_history = []

        for epoch in range(hp.epochStartTrain+1, hp.numIters):         
            epochLr = optimizer.param_groups[0]['lr']  
            if epoch % hp.logInterval == 0: 
                self.info(f'\n \n({epoch:02.0f}), lr: {epochLr:.6f}')

            # Training epoch
            loss = self.trainingEpoch(optimizer, train_loader, epoch)

            # Store loss as a float to avoid CUDA issues
            loss_history.append(loss.item() if T.is_tensor(loss) else loss)

            # Adjust learning rate
            scheduler.step()

            # Save model periodically
            if (epoch % hp.checkpointInterval == 0) and (epoch > 0):
                self.saveModel(epoch, optimizer, scheduler)

            # Log progress
            if epoch % hp.logInterval == 0: 
                self.info(f'   ({epoch}) Training loss: {loss:.8f}')
                
        return loss_history

        
#         hp = self.hp

#         train_dataset = self.dataset(self.rawData, 'train', self.path, hp, device=self.args.device, info=self.info)
#         train_loader = DataLoader(train_dataset, batch_size=hp.batchSizeTrain, shuffle=True)  # todo:change

#         optimizer = T.optim.RMSprop(self.model.parameters(), lr=hp.lr)
#         lr_lambda = lambda epoch: 1 ** epoch
#         scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

#         # ------------------- load model from checkpoint -----------------------
#         if hp.epochStartTrain:
#             optimizer, scheduler = self.loadModel(hp.epochStartTrain, optimizer, scheduler)

#         for epoch in range(hp.epochStartTrain+1, hp.numIters):         

#             epochLr = optimizer.param_groups[0]['lr']  
#             if epoch % hp.logInterval == 0: self.info(f'\n \n({epoch:02.0f}), lr: {epochLr:.6f}')
            
#             loss = self.trainingEpoch(optimizer, train_loader, epoch)

#             # -------------------------- adjusted lr ---------------------------
#             scheduler.step()

#             # ------------------- Save model periodically ----------------------
#             if (epoch % hp.checkpointInterval == 0) and (epoch > 0):
#                 self.saveModel(epoch, optimizer, scheduler)

#             # ------------------------ print progress --------------------------
#             if epoch % hp.logInterval == 0: self.info(f'   ({epoch}) Training loss: {loss:.8f}')

    
    def savePredictions(self, predData, epoch, trainBool):
        info = self.hp.predData_Info if hasattr(self.hp, 'predData_Info') else ''
        predData_Path = join(self.path.run, f'predDataTest_epoch{epoch}{info}.hdf5')

        predArray, targetArray = predData

        with h5py.File(predData_Path, 'w') as f:
            f.create_dataset('pred', data=predArray.detach().cpu().numpy())
            f.create_dataset('target', data=targetArray.detach().cpu().numpy())
        print(f'pred data saved at {predData_Path}')

    
    def testingEpoch(self, test_loader):
        predLs = []; dataLs = []

        for batchIdx, data in enumerate(test_loader):

            statePred, stateData = self.lossFn(self.model, data, 0, batchIdx) 

            predLs.append(statePred)
            dataLs.append(stateData)
        
        predData = T.cat(predLs, dim=0), T.cat(dataLs, dim=0)  # (numSampTest, timeStepModel, M, numNodes)
        return predData
    
    
    def test(self):
        hp = self.hp
        
        test_dataset = self.dataset(self.rawData, 'test', self.path, hp, device=self.args.device, info=self.info)
        test_loader = DataLoader(test_dataset, batch_size=hp.batchSizeTest, shuffle=False)

        # ------------------------ Load saved weights --------------------------
        epoch = hp.loadWeightsEpoch
        self.loadModel(epoch)
        self.model.eval()

        
        if hasattr(hp, 'PINN'):
            predData = self.testingEpochPINN(test_loader)
        else:
            predData = self.testingEpoch(test_loader)
        self.savePredictions(predData, epoch, False) 
