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

from src.FlowPastCylinder.FlowPastCylinderDataset import DatasetClass
from src.FlowPastCylinder.FlowPastCylinderEnsersModel import Model
from src.FlowPastCylinder.FlowPastCylinderPhysicsLoss import Loss
from src.FlowPastCylinder.FlowPastCylinderLoadData import LoadData



def setHyperParams(hp):
    # model 
    hp.numEmbed = 8
    hp.timeStepModel = 5

    # training
    hp.numIters = 3001
    hp.lr = 0.0003
    hp.innerLrTrain0 = 0.1
    hp.innerLrTrainRate = 0.002
    hp.numInnerItersTrain = 5
    hp.batchSizeTrain = 9
    hp.numSensorTrain = 24
    hp.epochStartTrain = 00
    hp.zetaRate = 0.0006
    hp.zeta0 = 0.01

    # testing
    hp.innerLrTest = 5
    hp.batchSizeTest = 6
    hp.numInnerItersTest = 100
    hp.numSensorTestLs = [4, 16]
    hp.loadWeightsEpoch = 3000

    # data
    hp.numSampTrain = 18
    hp.numSampTest = 12
    hp.numDataNodes = 500
    hp.dataSkipSteps = 1
    hp.noiseLs = [10, 20, 60, None]  # SNRdb
    
    # logging
    hp.save = 1
    hp.show = 0
    hp.saveLogs = 1
    hp.saveInterval = 20
    hp.logInterval = 100
    hp.checkpointInterval = 1000

def addPaths(ep, runName):
    ep.weights = f'{runName}/checkpoints'
    ep.data = f'../../src/FlowPastCylinder/solver/openFoam_flowPastCylinder'
    ep.code = f'../../src/FlowPastCylinder'
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
    modelPipeline.train()

    for hp.numSensorTest in hp.numSensorTestLs:
        for hp.noise in hp.noiseLs:
            hp.predData_Info = f'_train_sensors{hp.numSensorTrain}_test_Sensors{hp.numSensorTest}_SNRdb{hp.noise}'
            modelPipeline.test()

    # save hyper params for the run
    sv_args = hp
    save_args(sv_args, experPaths.run)  

    