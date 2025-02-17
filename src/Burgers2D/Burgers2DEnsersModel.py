"""
Network Architecture
"""

from math import pi

import sympy as sym
from sympy.polys.monomials import itermonomials
from sympy.polys.orderings import monomial_key

from torch import linalg as LA
from typing import Union, Tuple, Optional, List

import higher
import torch as T
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear

import pdb
import os.path as osp


class Model(nn.Module):
    """
    Model predict state based on given sensor values and their respective postions.
    It partially solves minimization problem    
        \hat embed = argmin_{embed} E_\theta(sensor, embed)

    using few(4-6) steps of Adam algorithm during training and many(150-200) 
    steps during testing to obtain embeddings vector.
    """

    def __init__(self, hyper_Params, lossFn, imDim, args):
        super(Model, self).__init__()

        self.info = args.logger.info
        self.device = args.device
        self.args = args
        self.imDim = imDim

        self.hp = hyper_Params
        self.decoderNet()
        self.reset_parameters()

    
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

    
    def decoderNet(self):
        H, W = self.imDim
        # self.lin = nn.Sequential(   nn.Linear(self.hp.numEmbed, 64), nn.Softplus(), 
        #                             nn.Linear(64, H*W*2*self.hp.timeStepModel)  )
        self.lin = nn.Sequential(   nn.Linear(self.hp.numEmbed, 64), nn.Softplus(), 
                                    nn.Linear(64, 64), nn.Softplus(), 
                                    nn.Linear(64, H*W*2*self.hp.timeStepModel)  )


    def predictStates(self, embed):
        """
        Args:
            embed (Tensor): (currentBatchSize, numEmbed)
        Vars:
            out (Tensor): (currentBatchSize, numNodes*2*timeStepModel)
        """
        H, W = self.imDim
        out = self.lin(embed)
        return out.reshape(self.currentBatchSize, self.hp.timeStepModel, 2, H*W)  # (currentBatchSize, timeStepModel, 2, numNodes)

        # _statePred = T.cat(T.split(out[:, None], out.shape[1]//self.hp.timeStepModel, dim=2), dim=1)[:, :, None]
        # statePred = T.cat(T.split(_statePred, _statePred.shape[3]//2, dim=3), dim=2)


    def calc_energy(self, statePred, sensorData, sensorLoc):
        """
        Args:
            sensorLoc (Tensor): (currentBatchSize, timeStepModel, 2, numSensor)
            sensorData (Tensor): (currentBatchSize, timeStepModel, 2, numSensor)
        Vars:
            statePred (Tensor): (currentBatchSize, timeStepModel, 2, numNodes)
            sensorPred (Tensor): (currentBatchSize, timeStepModel, 2, numSensor)
        """        
        sensorPred = statePred.gather(3, sensorLoc)  # (currentBatchSize, timeStepModel, 2, numSensor)
        # energy = F.mse_loss(sensorPred, sensorData)
        energy = F.huber_loss(sensorPred, sensorData)
        return energy


    def forward(self, sensorData):
        """
        Args:
            sensorLoc (Tensor): (currentBatchSize, timeStepModel, 2, numSensor)
            sensorData (Tensor): (currentBatchSize, timeStepModel, 2, numSensor)  
        Vars:
        Returns:
            statePred (Tensor): (currentBatchSize, timeStepModel, 2, numNodes)
        """

        # lr = self.hp.innerLrTrain if self.training else self.hp.innerLrTest
        lr = self.hp.innerLrTrain0 + self.hp.innerLrTrainRate*self.epoch if self.training else self.hp.innerLrTest
        numEpoch = self.hp.numInnerItersTrain if self.training else self.hp.numInnerItersTest
        self.currentBatchSize = sensorData.shape[0]


        # Initial guess of the embed.
        embed = nn.init.xavier_uniform_(T.zeros((self.currentBatchSize, self.hp.numEmbed), device=self.device, requires_grad=True))

        # Differentiable optimizer to update the label with.
        inner_opt = higher.get_diff_optim(T.optim.Adam([embed], lr=lr), [embed], device=self.device)


        # Take a few gradient steps to find the labels that
        # optimize the energy function.
        for epoch in range(numEpoch):
            statePred = self.predictStates(embed)
            E = self.calc_energy(statePred, sensorData, self.sensorLoc)

            if (self.epoch%self.hp.logInterval == 0) and (epoch % 2 == 0): 
                self.info(f'        ({epoch:02.0f}) loss: {E.item():02.5f}, lr: {lr:.6f}')
            embed, = inner_opt.step(E, params=[embed])

        
        # predict state with calculated embedding
        return self.predictStates(embed)

