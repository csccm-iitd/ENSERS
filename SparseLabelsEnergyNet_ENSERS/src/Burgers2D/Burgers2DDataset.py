"""
Dataset
"""

import pdb
import logging
from numpy import random
from torch.utils.data.dataset import Dataset
import numpy as np
from numpy.random import choice
import torch as T

import os
from os.path import dirname, realpath, join, exists
import sys
filePath = realpath(__file__)
srcDir = dirname(dirname(filePath))
sys.path.append(srcDir)

  


class DatasetClass(Dataset):

    def __init__(self, rawData, use, experPaths, hyperParams, device='cpu', info=print):
        """
        Vars:
            self.rawData.data (Tensor): (timeStep, 2, numNodes)
            rawDataNodesLoc (Tensor): (rawDataLen, numDataNodes)
            rawsensorNodesLoc (Tensor): (rawDataLen, numSensorTrain)
        """
        self.use = use
        self.device = device
        self.info = info

        self.rawData = rawData
        self.ep = experPaths
        self.hp = hyperParams

        self.dataIdxTrainLs = np.arange(0, len(self.rawData.data) - self.hp.timeStepModel, self.hp.dataSkipSteps)[0:self.hp.numSampTrain]
        self.selectedDataLen = self.dataIdxTrainLs[-1]
        self.dataIdxTestLs =  np.arange(0, self.selectedDataLen, int((self.selectedDataLen-1e-1)//self.hp.numSampTest))[1:self.hp.numSampTest+1]
        self.getTrainData() if use == 'train' else self.getTestData()
               
            
    def getTrainData(self):

        rawDataLen = len(self.rawData.data)
        assert rawDataLen - self.hp.timeStepModel >= self.hp.numSampTrain, 'not enough data'

        dataIdxTrainLs = self.dataIdxTrainLs
        self.info(f'\nTrain data Idx List: {dataIdxTrainLs}')

        # ---------------------- state Data nodes & values -------------------------                                                                
        rawDataNodesLoc = self.Loc(rawDataLen, self.hp.numDataNodes)  # (rawDataLen, 2, numDataNodes)
        self.dataNodesLocTrain = self.catTimeStep(rawDataNodesLoc, dataIdxTrainLs)  # (numSampTrain, timeStepModel, 2, numDataNodes) 

        rawDataNodesValue = self.Value(rawDataNodesLoc)  # (rawDataLen, 2, numDataNodes)              
        self.stateDataTrain = self.catTimeStep(rawDataNodesValue, dataIdxTrainLs)  # (numSampTrain, timeStepModel, 2, numDataNodes)

        # ---------------------- sensor Data nodes & values ------------------------
        rawSensorLocTrain = self.Loc(rawDataLen, self.hp.numSensorTrain)  # (rawDataLen, 2, numSensorTrain)
        self.sensorLocTrain = self.catTimeStep(rawSensorLocTrain, dataIdxTrainLs)  # (numSampTrain, timeStepModel, 2, numSensorTrain)

        rawSensorValueTrain = self.Value(rawSensorLocTrain)  # (rawDataLen, 2, numSensorTrain)        
        self.sensorDataTrain = self.catTimeStep(rawSensorValueTrain, dataIdxTrainLs)  # (numSampTrain, timeStepModel, 2, numSensorTrain)


    def getTestData(self):

        rawDataLen = len(self.rawData.data)
        assert rawDataLen - self.hp.timeStepModel >= self.hp.numSampTest, 'not enough data'
 
        dataIdxTestLs = self.dataIdxTestLs
        self.info(f'\nTest data Idx List: {dataIdxTestLs}')

        # ---------------------- state Data nodes & values -------------------------
        self.stateDataTest = self.catTimeStep(self.rawData.data, dataIdxTestLs)  # (numSampTest, timeStepModel, 2, numNodes)

        # ---------------------- sensor Data nodes & values ------------------------
        rawSensorLocTest = self.Loc(rawDataLen, self.hp.numSensorTest)  # (rawDataLen, 2, numSensorTest)   
        self.sensorLocTest = self.catTimeStep(rawSensorLocTest, dataIdxTestLs)  # (numSampTest, timeStepModel, 2, numSensorTest)

        rawSensorValueTest = self.Value(rawSensorLocTest, noise=self.hp.noise)  # (rawDataLen, 2, numSensorTest)            
        self.sensorDataTest = self.catTimeStep(rawSensorValueTest, dataIdxTestLs)  # (numSampTest, timeStepModel, 2, numSensorTest)


    def Loc(self, numSamp, numDataLoc):
        """ 
        Randomly select data location in image.

        Args:
            numSamp (int): number of samples
            numDataLoc (int): number of data location
        Returns:
            dataLoc (Tensor): (rawDataLen, 3, numDataNodes)
        """
        dataLoc = T.zeros((numSamp, numDataLoc), dtype=T.int64)

        for i in range(numSamp):
            np.random.seed(i)
            dataLoc[i] = T.from_numpy(np.random.choice(range(self.rawData.numNodes), size=numDataLoc, replace=False)).to(T.int64)
        return T.cat([dataLoc[:, None]]*2, dim=1)

    
    def catTimeStep(self, rawNodes, idxLs):
        """
        Concatenate consecutive time step.

        Args:
            rawNodes (Tensor): (rawDataLen, 2, numDataNodes)
            idxLs (Tensor): (numSamp)
        Returns:
            catValue (Tensor): (numSamp, timeStepModel, 2, numDataNodes)
        """
        catValue = []
        for i in range(self.hp.timeStepModel):
            nodesValue = rawNodes[idxLs + i]  # (numSamp, 2, numDataNodes)
            catValue.append(nodesValue[:, None])  # (numSamp, 1, 2, numDataNodes)

        return T.cat(catValue, dim=1)  # (numSamp, timeStepModel, 2, numDataNodes)


    def Value(self, NodesLoc, noise=None):
        """
        Give value at data/sensor node locations.
        Adds noise of specified noise level.

        Args:
            NodesLoc (Tensor): (rawDataLen, 2, numDataNodes).
            noise (float, optional): noise level in SNRdb (default: `None`).
        Vars:
            self.rawData.data (Tensor): (timeStep, 2, numNodes)
        Returns:
            Value (Tensor): (rawDataLen, 2, numDataNodes)
        """
        data = self.rawData.data
        if noise:
            data = data + self.rawData.Noise[self.rawData.SNRdbLs.index(noise)]
            self.info(f'{noise} SNRdb Noise added to sensors value')
        return data[:NodesLoc.shape[0]].gather(2, NodesLoc)


    def __len__(self):
        len = self.hp.numSampTrain if self.use == 'train' else self.hp.numSampTest
        return len


    def __getitem__(self, idx):
        d = self.device

        if self.use == 'train':
            return self.sensorLocTrain[idx].to(d), self.sensorDataTrain[idx].to(d), self.dataNodesLocTrain[idx].to(d), self.stateDataTrain[idx].to(d)
        else:
            return self.sensorLocTest[idx].to(d), self.sensorDataTest[idx].to(d), self.stateDataTest[idx].to(d)

