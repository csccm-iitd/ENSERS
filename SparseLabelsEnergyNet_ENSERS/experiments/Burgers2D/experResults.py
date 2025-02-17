
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
from src.Burgers2D.Burgers2DPlots import Plots
from src.Burgers2D.Burgers2DLoadData import LoadData


args = Arguments()
pathDict = {'run': 'firstTry', 'data': f'../../src/Burgers2D/solver/fenics_data_periodic'}
experPaths = Paths(experDir, args.os, pathDict)
hp = loadRunArgs(experPaths.run)

rawData = LoadData(args, experPaths)
SensorsLs = hp.numSensorTestLs
SNRdbLs = hp.noiseLs

trn_sensors =24
#%% load saved predictions during testing
try:
    predData = {}
    for j, Sensors in enumerate(SensorsLs):
        for i, SNRdb in enumerate(SNRdbLs):
            name = f'predDataTest_epoch{hp.loadWeightsEpoch}trn_sensor{trn_sensors}_tst_Sensors{Sensors}_SNRdb{SNRdb}.hdf5'
            predData[f'trn_sensor{trn_sensors}_tst_Sensors{Sensors}_SNRdb{SNRdb}'] = h5py.File(join(experPaths.run, name), 'r')
    print(f'loaded pred data')
except:
    print(f'{join(experPaths.run, name)}')
    raise Exception(FileNotFoundError)


#%% line plot
savePath = join(experPaths.run, f'Burgers2DpredPlot{0}_trn_sensor{trn_sensors}_epoch{hp.loadWeightsEpoch}')
#choice([1, 2, 3], size=hp.numSampTest, replace=True)
plotParams = {'tStepModelPlot':[2]*hp.numSampTest, 'imDim': rawData.imDim, 'tStepPlot':slice(0, hp.numSampTest, 2)}
plotData = predData[f'trn_sensor{trn_sensors}_tst_Sensors{16}_SNRdb{None}']
Plots().plotPred(plotData, Dict2Class(plotParams), savePath)


#%% calculate L2 Error
idx = 2; M=2
plotData = np.zeros((len(SensorsLs), M, hp.numSampTest, len(SNRdbLs)))

for s, Sensors in enumerate(SensorsLs):
    for i, SNRdb in enumerate(SNRdbLs):
        
        pred = predData[f'trn_sensor{trn_sensors}_tst_Sensors{Sensors}_SNRdb{SNRdb}']['pred'][:, idx]  # (numSampTest, 2, numNodes)
        target = predData[f'trn_sensor{trn_sensors}_tst_Sensors{Sensors}_SNRdb{SNRdb}']['target'][:, idx]

        l2Error = np.zeros(pred.shape[:-1])
        for j in range(hp.numSampTest):
            for k in range(pred.shape[1]):
                l2Error[j, k] = norm(target[j, k] - pred[j, k]) / norm(target[j, k])
    
        plotData[s, 0, :, i] = l2Error[:, 0].reshape((-1))
        plotData[s, 1, :, i] = l2Error[:, 1].reshape((-1))
    
# Save plotData to a file
file_name = "plotData.pkl"
with open(file_name, 'wb') as f:
    pickle.dump(plotData, f)

print(f"plotData saved to {file_name}")
#%% violin plot
for s, Sensors in enumerate(SensorsLs):
    savePath = join(experPaths.run, f'Burgers2DViolinPlot{0}trn_sensor{trn_sensors}_tst_Sensors{Sensors}_epoch{hp.loadWeightsEpoch}') 
    plotParams = {
        'xticks': [10, 20, 30, 40, 50, 60, 70, 80],
        'xticklabels': [10, 10, 20, 20, 60, 60, 'None', 'None'],
        'xticksPlot': [[20, 40, 60, 80], [10, 30, 50, 70]],
        'ylabel': 'L2 Error',
        'xlabel': 'SNRdb',
        'title': f'Sensors: {Sensors}',
        'label': ['Ux', 'Uy'],
        'facecolor': ['green', '#D43F3A']
    }
    Plots().violinplot(plotData[s], Dict2Class(plotParams), savePath)


# %%
