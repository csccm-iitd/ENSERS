
""" 
Plot predictions and reults.
"""


#%%
import pdb
import h5py
import torch as T
import numpy as np
from numpy.random import choice
from numpy.linalg import norm
import pickle
from os.path import dirname, realpath, join
import sys

filePath = realpath(__file__)
experDir = dirname(realpath(__file__))
projectDir = dirname(dirname(dirname(filePath)))
sys.path.append(projectDir)

from src.Utils import Arguments, loadRunArgs, Dict2Class
from src.Paths import Paths
from src.FlowPastCylinder.FlowPastCylinderPlots import Plots
from src.FlowPastCylinder.FlowPastCylinderLoadData import LoadData

trn_sensors = 24
args = Arguments()
pathDict = {'run': 'firstTry', 'data': f'../../src/FlowPastCylinder/solver/openFoam_flowPastCylinder'}
experPaths = Paths(experDir, args.os, pathDict)
hp = loadRunArgs(experPaths.run)

rawData = LoadData(args, experPaths)
SensorsLs = hp.numSensorTestLs
SNRdbLs = hp.noiseLs


#%% ------------ load saved predictions during testing ---------------------

try:
    predData = {}
    for j, Sensors in enumerate(SensorsLs):
        for i, SNRdb in enumerate(SNRdbLs):
            name = f'predDataTest_epoch{hp.loadWeightsEpoch}_train_sensors{trn_sensors}_test_Sensors{Sensors}_SNRdb{SNRdb}.hdf5'
            predData[f'_train_sensors{trn_sensors}_test_Sensors{Sensors}_SNRdb{SNRdb}'] = h5py.File(join(experPaths.run, name), 'r')
    print(f'loaded pred data')
except:
    print(f'{join(experPaths.run, name)}')
    raise Exception(FileNotFoundError)


#%% -------------------- plot Pred with no noise ---------------------------

savePath = join(experPaths.run, f'FlowPastCylinderpredPlot{0}_train_sensors{trn_sensors}_epoch{hp.loadWeightsEpoch}')
#choice([1, 2, 3], size=hp.numSampTest, replace=True)
plotParams = {'tStepModelPlot':[2]*hp.numSampTest, 'imDim': rawData.imDim, 'tStepPlot':slice(0, hp.numSampTest, 2)}
plotData = predData[f'_train_sensors{trn_sensors}_test_Sensors{16}_SNRdb{None}']
Plots().plotPred(plotData, Dict2Class(plotParams), savePath)


#%% --------------------- calculate L2 Error -------------------------------

idx = 2; M=3
plotData = np.zeros((len(SensorsLs), M, hp.numSampTest, len(SNRdbLs)))

for s, Sensors in enumerate(SensorsLs):
    for i, SNRdb in enumerate(SNRdbLs):
        
        pred = predData[f'_train_sensors{trn_sensors}_test_Sensors{Sensors}_SNRdb{SNRdb}']['pred'][:, idx]  # (numSampTest, M, numNodes)
        target = predData[f'_train_sensors{trn_sensors}_test_Sensors{Sensors}_SNRdb{SNRdb}']['target'][:, idx]

        l2Error = np.zeros(pred.shape[:-1])
        for j in range(hp.numSampTest):
            for k in range(pred.shape[1]):
                l2Error[j, k] = norm(target[j, k] - pred[j, k]) / norm(target[j, k])
    
        plotData[s, 0, :, i] = l2Error[:, 0].reshape((-1))
        plotData[s, 1, :, i] = l2Error[:, 1].reshape((-1))
        plotData[s, 2, :, i] = l2Error[:, 2].reshape((-1))
    
# Save plotData to a file
file_name = "plotData_fc.pkl"
with open(file_name, 'wb') as f:
    pickle.dump(plotData, f)
#%% --------------------- violin plot of L2 error --------------------------

for s, Sensors in enumerate(SensorsLs):
    savePath = join(experPaths.run, f'FlowPastCylinderViolinPlot{0}_train_sensors{trn_sensors}_test_Sensors{Sensors}_epoch{hp.loadWeightsEpoch}') 
    plotParams = {
        'xticks': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
        'xticklabels': [10, 10, 10, 20, 20, 20, 60, 60, 60, 'None', 'None', 'None'],
        'xticksPlot': [[10, 40, 70, 100], [20, 50, 80, 110], [30, 60, 90, 120]],
        'ylabel': 'Error',
        'xlabel': 'SNRdb',
        'title': f'Sensors: {Sensors}',
        'label': ['U', 'V', 'P'],
        'facecolor': ['green', '#D43F3A', 'pink']
    }
    Plots().violinplot(plotData[s], Dict2Class(plotParams), savePath)


# %%
