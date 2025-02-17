"""
Loads data
"""

import pdb
import json
import numpy as np
import torch as T
from torch import Tensor
import matplotlib.pyplot as plt

from os.path import dirname, realpath, join, exists
import sys
filePath = realpath(__file__)
srcDir = dirname(dirname(filePath))
sys.path.append(srcDir)

from Utils import Dict2Class, awgn


class LoadData:
    """ Loads data present in /data folder and saves in self.data. 
    Var: 
        loadRun: runs to be loaded
    Returns:
        self.data (Tensor): (timeStep, 2, numNodes)
    """

    def __init__(self, args, experPaths):

        self.info = args.logger.info if hasattr(args, 'logger') else print
        
        nx = 63
        self.loadRun = 14
        self.SNRdbLs = [10, 20, 60, 80]
        self.dataDir = experPaths.data

        dataParams = self.loadDataParams()
        self.timeStep = dataParams.numtimeStep
        self.dataTimeGrid = dataParams.timeGrid
        self.dt = dataParams.save_dt
        self.dx = 1/nx
        self.imDim = nx, nx
        self.nu = dataParams.nu

        self.dataDir = experPaths.data
        self.runPath = lambda run: join(self.dataDir, f'run{run}', f'raw{run}.npy')
        
        self.loadVertexValues()
        self.loadBoundaryVertices()
        self.loadNoise()
        self.info(f'data loaded \ndata shape: {self.data.shape}\n')

    
    def loadDataParams(self):
        path = join(self.dataDir, f'run{self.loadRun}', f'dataParams.json')
        with open(path, 'r') as file:  
            dict = json.load(file)
        return Dict2Class(dict)


    def loadVertexValues(self):
        """ 
        Vars:
            self.data (Tensor): (timeStep, 2, numNodes)
                timeStep: num steps
        """
        
        data = np.load(self.runPath(self.loadRun))
        self.numNodes = data.shape[2]
        self.data = T.tensor(data, dtype=T.float32)


    def loadVertexCoords(self, meshRun):
        """Vars: self.vertexCoords (Tensor): [numNodes, 1]"""
        
        dir = join(self.dataDir, f'run{meshRun}')
        meshPath = join(dir, 'meshVertices.npy')
        self.vertexCoords = T.tensor(np.load(meshPath), dtype=T.float32)

    
    def loadBoundaryVertices(self):
        """
        Returns: boundaryVertices (np.array[int]): boundary nodes index, shape: [numBoundaryNodes]
        """
        path = join(self.dataDir, f'run{self.loadRun}', f'BoundaryVertices.npy')
        self.boundaryVertices = T.tensor(np.load(path).astype(np.int64))


    def calcNoise(self):
        '''calculate and store noise'''
        Data = self.data.numpy()
        
        dataNoise = np.zeros((len(self.SNRdbLs),) + Data.shape)
        for i, SNRdB in enumerate(self.SNRdbLs):
            for j in range(Data.shape[0]):
                for k in range(Data.shape[1]):
                    dataNoise[i, j, k,:] = awgn(Data[j, k], SNRdB)        

        path = join(self.dataDir, f'run{self.loadRun}', f'dataNoise.npy')
        np.save(path, dataNoise)

    
    def loadNoise(self):
        Path = join(self.dataDir, f'run{self.loadRun}', 'dataNoise.npy')
        if not exists(Path): self.calcNoise()
        self.Noise = T.tensor(np.load(Path), dtype=T.float32)



if __name__ == '__main__':
    
    from Utils import Parser
    from Paths import Paths
    from Burgers2DPlots import Plots

    # ---------------------------- Argument Parser -----------------------------
    args = Parser().parse()
    pathDict = {'data': 'solver/fenics_data_periodic', 'paperfig': 'paperFig'}
    experDir = dirname(realpath(__file__))
    experPaths = Paths(experDir, args.os, pathDict=pathDict)

    # -------------------------- load and plot data ----------------------------
    rawData = LoadData(args, experPaths)

    # plotData = rawData.data.numpy()
    # savePath = join(experPaths.experDir, 'femSimulation')
    # plotParams = ([1, 15, 30, 45], rawData.imDim)
    # Plots().femSimulation(plotData, plotParams, savePath)

    # # ------------------------------ paper figs --------------------------------
    # Plots().paperFig1(plotData, (rawData.imDim, [2, 3, 4]), experPaths.paperfig)

    # -------------------------- noisy data plots ------------------------------
    SNRdbLs = [10, 20, 60, None]
    data = rawData.data.numpy()
    plotData = np.zeros((len(SNRdbLs), rawData.numNodes))
    for i, noise in enumerate(SNRdbLs):
        _data = data
        if noise: _data = _data + rawData.Noise[rawData.SNRdbLs.index(noise)].numpy()
        plotData[i] = _data[10, 0]
        
    savePath = join(experPaths.experDir, 'Burgers2DNoisyData')
    plotParams = {
        'imDim': rawData.imDim,
        'titleLs': [f'SNRdB {i}' for i in SNRdbLs]
    }
    Plots().noisePlots(plotData, Dict2Class(plotParams), savePath)