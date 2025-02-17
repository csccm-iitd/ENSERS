"""
Main script for experiment 'Basic'
for setting params and training/testing
"""

import pdb
import logging
import torch as T

import sys
from os.path import dirname, realpath, join

filePath = realpath(__file__)
experDir = dirname(realpath(__file__))
projectDir = dirname(dirname(dirname(filePath)))
sys.path.append(projectDir)

from src.Utils import Parser, startSavingLogs, save_args
from src.Pipeline import ModelPipeline
from src.Paths import Paths

from src.Burgers2D.Burgers2DDataset import DatasetClass
from src.Burgers2D.Burgers2DEnsersModel import Model
from src.Burgers2D.Burgers2DPhysicsLoss import Loss
from src.Burgers2D.Burgers2DLoadData import LoadData
import matplotlib.pyplot as plt


def setHyperParams(hp):
    # model 
    hp.numEmbed = 8
    hp.timeStepModel = 5
    
    # training
    hp.numIters = 2001
    hp.lr = 0.0002
    hp.innerLrTrain0 = 0.1
    hp.innerLrTrainRate = 0.006
    hp.numInnerItersTrain = 4
    hp.batchSizeTrain = 11
    hp.numSensorTrain = 24
    hp.epochStartTrain = 00
    hp.zetaRate = 0.0001
    hp.zeta0 = 0.005
    
    # testing
    hp.innerLrTest = 5
    hp.batchSizeTest = 6
    hp.numInnerItersTest = 100
    hp.numSensorTestLs = [4, 16]
    hp.loadWeightsEpoch = 2000
    
    # data
    hp.numSampTrain = 22
    hp.numSampTest = 12
    hp.numDataNodes = 800
    hp.dataSkipSteps = 2
    hp.noiseLs = [10, 20, 60, None]  # SNRdb
    
    # logging
    hp.save = 1
    hp.show = 0
    hp.saveLogs = 1
    hp.saveInterval = 20
    hp.logInterval = 100
    hp.checkpointInterval = 200

def addPaths(ep, runName):
    ep.weights = f'{runName}/checkpoints'
    ep.data = f'../../src/Burgers2D/solver/fenics_data_periodic'
    ep.code = f'../../src/Burgers2D'
    ep.run = f'{runName}'



if __name__ == '__main__':

    args = Parser().parse()
    logger = logging.getLogger('my_module')
    logger.setLevel(logging.DEBUG)

    # set useful paths to instance `experPaths`
    runName = 'firstTry'
    experPaths = Paths(experDir, args.os)
    addPaths(experPaths, runName)

    # set hyper params for run
    class HyperParams: pass
    hp = HyperParams()
    setHyperParams(hp)
    
    # load saved hyper params for testing from old runs
    # hp = loadRunArgs(experPaths.run)
    
    startSavingLogs(args, experPaths.run, logger)
    rawData = LoadData(args, experPaths)

    lossfn = Loss(rawData, hp, experPaths, grad_kernels=[5, 5], device=args.device, info=args.logger.info)
    modelPipeline = ModelPipeline(Model, hp, experPaths, rawData, DatasetClass, lossfn, args)
    loss_history = modelPipeline.train()
    # Plot and save the loss vs epoch graph
    plt.figure(figsize=(8, 6))
    plt.plot(range(hp.epochStartTrain+1, hp.numIters), loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_vs_epoch.pdf",bbox_inches='tight')  #   bbox_inches='tight')Save the plot
    plt.show()  # Display the plot


    for hp.numSensorTest in hp.numSensorTestLs:
        for hp.noise in hp.noiseLs:
            hp.predData_Info = f'trn_sensor{ hp.numSensorTrain}_tst_Sensors{hp.numSensorTest}_SNRdb{hp.noise}'
            modelPipeline.test()

    # save hyper params for the run
    sv_args = hp
    save_args(sv_args, experPaths.run)

    