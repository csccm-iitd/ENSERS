"""
Loads data
"""

import pdb
import json
from scipy.io import loadmat 
import numpy as np
import torch as T
from torch import Tensor

import sys
from os.path import dirname, realpath, join, exists

filePath = realpath(__file__)
projectDir = dirname(dirname(dirname(filePath)))
sys.path.append(projectDir)

from src.Utils import Dict2Class, awgn


class LoadData:
    """ Loads data present in /data folder and saves in self.data. 
    Var: 
        loadRun: runs to be loaded
    Returns:
        self.data (Tensor): (timeStep, 2, numNodes)
    """

    def __init__(self, args, experPaths):

        self.SNRdbLs = [10, 20, 40, 60, 80]
        self.info = args.logger.info if hasattr(args, 'logger') else print
        
        self.dataDir = experPaths.data
        dataParams = self.loadDataParams()

        self.timeStep = dataParams.numtimeStep
        self.dataTimeGrid = dataParams.timeGrid
        self.dt = dataParams.save_dt
        self.dx = dataParams.dx
        self.imDim = dataParams.imDim
        self.nu = dataParams.nu

        self.dataDir = experPaths.data
        
        self.loadVertexValues()
        self.loadBoundaryVertices()
        self.loadNoise()
        self.info(f'data loaded \ndata shape: {self.data.shape}\n')

    
    def loadDataParams(self):
        path = join(self.dataDir, f'dataParams.json')
        with open(path, 'r') as file:  
            dict = json.load(file)
        return Dict2Class(dict)


    def loadVertexValues(self):
        """ 
        Vars:
            self.data (Tensor): (timeStep, 3, numNodes)
                timeStep: num steps
        """

        mat = loadmat(join(self.dataDir, 'flowData.mat'))
        print(mat.keys())
        data = mat['flowData']
        
        self.numNodes = data.shape[2]
        self.data = T.tensor(data, dtype=T.float32)

    
    def loadBoundaryVertices(self):
        """
        Returns: boundaryVertices (np.array[int]): boundary nodes index, shape: [numBoundaryNodes]
        """
        # path = join(self.dataDir, f'run{self.loadRun}', f'BoundaryVertices.npy')
        # self.boundaryVertices = T.tensor(np.load(path).astype(np.int64))
        self.boundaryVertices = T.tensor([0], dtype=T.int64)

    def calcNoise(self):
        '''calculate and store noise'''
        Data = self.data.numpy()
        
        dataNoise = np.zeros((len(self.SNRdbLs),) + Data.shape)
        for i, SNRdB in enumerate(self.SNRdbLs):
            for j in range(Data.shape[0]):
                for k in range(Data.shape[1]):
                    dataNoise[i, j, k,:] = awgn(Data[j, k], SNRdB)        

        path = join(self.dataDir, f'dataNoise.npy')
        np.save(path, dataNoise)

    
    def loadNoise(self):
        Path = join(self.dataDir, 'dataNoise.npy')
        if not exists(Path): self.calcNoise()
        self.Noise = T.tensor(np.load(Path), dtype=T.float32)



if __name__ == '__main__':
    
    from src.Utils import Parser
    from src.Paths import Paths
    from FlowPastCylinderPlots import Plots

    # ---------------------------- Argument Parser -----------------------------
    args = Parser().parse()
    pathDict = {'data': 'solver/openFoam_flowPastCylinder'}
    experDir = dirname(realpath(__file__))
    experPaths = Paths(experDir, args.os, pathDict=pathDict)

    # -------------------------- load and plot data ----------------------------
    rawData = LoadData(args, experPaths)

    # plotData = rawData.data.numpy()
    # savePath = join(experPaths.experDir, 'openFoamSimulation')
    # plotParams = ([1, 5, 10, 15, 20], rawData.imDim)
    # Plots().femSimulation(plotData, plotParams, savePath)

    # -------------------------- noisy data plots ------------------------------
    SNRdbLs = [10, 20, 60, None]
    data = rawData.data.numpy()
    plotData = np.zeros((len(SNRdbLs), rawData.numNodes))
    for i, noise in enumerate(SNRdbLs):
        _data = data
        if noise: _data = _data + rawData.Noise[rawData.SNRdbLs.index(noise)].numpy()
        plotData[i] = _data[10, 0]
        
    savePath = join(experPaths.experDir, 'FlowPastCylinderNoisyData')
    plotParams = {
        'imDim': rawData.imDim,
        'titleLs': [f'SNRdB {i}' for i in SNRdbLs]
    }
    Plots().noisePlots(plotData, Dict2Class(plotParams), savePath)
