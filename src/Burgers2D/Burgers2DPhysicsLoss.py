from os import name
import pdb
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple
import numpy as np

import sys
from os.path import dirname, realpath, join

filePath = realpath(__file__)
srcDir = dirname(dirname(filePath))
sys.path.append(srcDir)


class LapLaceFilter2d(object):
    """
    Smoothed Laplacian 2D, assumes periodic boundary condition.
    https://en.wikipedia.org/wiki/Finite_difference_coefficient
    Args:
        dx (float): spatial discretization, assumes dx = dy
        kernel_size (int): choices=[3, 5]
        device (PyTorch device): active device
    """
    def __init__(self, dx, kernel_size=3, device='cpu'):
        super().__init__()
        self.dx = dx
        # no smoothing
        WEIGHT_3x3 = torch.FloatTensor([[[[0, 1, 0],
                                          [1, -4, 1],
                                          [0, 1, 0]]]]).to(device)
        # smoothed
        # WEIGHT_3x3 = torch.FloatTensor([[[[1, 2, 1],
        #                                   [-2, -4, -2],
        #                                   [1, 2, 1]]]]).to(device) / 4.

        # WEIGHT_3x3 = WEIGHT_3x3 + torch.transpose(WEIGHT_3x3, -2, -1)

        # print(WEIGHT_3x3)

        WEIGHT_5x5 = torch.FloatTensor([[[[0, 0, -1, 0, 0],
                                          [0, 0, 16, -0, 0],
                                          [-1, 16, -60, 16, -1],
                                          [0, 0, 16, 0, 0],
                                          [0, 0, -1, 0, 0]]]]).to(device) / 12.
        if kernel_size == 3:
            self.padding = _quadruple(1)
            self.weight = WEIGHT_3x3
        elif kernel_size == 5:
            self.padding = _quadruple(2)
            self.weight = WEIGHT_5x5

    def __call__(self, u):
        """
        Args:
            u (torch.Tensor): [B, C, H, W]
        Returns:
            div_u(torch.Tensor): [B, C, H, W]
        """
        u_shape = u.shape
        u = u.view(-1, 1, *u_shape[-2:])
        u = F.conv2d(F.pad(u, self.padding, mode='circular'), 
            self.weight, stride=1, padding=0, bias=None) / (self.dx**2)
        return u.view(u_shape)



class SobelFilter2d(object):
    """
    Sobel filter to estimate 1st-order gradient in horizontal & vertical 
    directions. Assumes periodic boundary condition.
    Args:
        dx (float): spatial discretization, assumes dx = dy
        kernel_size (int): choices=[3, 5]
        device (PyTorch device): active device
    """
    def __init__(self, dx, kernel_size=3, device='cpu'):
        self.dx = dx
        # smoothed central finite diff
        WEIGHT_H_3x3 = torch.FloatTensor([[[[-1, 0, 1],
                                            [-2, 0, 2],
                                            [-1, 0, 1]]]]).to(device) / 8.

        # larger kernel size tends to smooth things out
        WEIGHT_H_5x5 = torch.FloatTensor([[[[1, -8, 0, 8, -1],
                                            [2, -16, 0, 16, -2],
                                            [3, -24, 0, 24, -3],
                                            [2, -16, 0, 16, -2],
                                            [1, -8, 0, 8, -1]]]]).to(device) / (9*12.)
        if kernel_size == 3:
            self.weight_h = WEIGHT_H_3x3
            self.weight_v = WEIGHT_H_3x3.transpose(-1, -2)
            self.weight = torch.cat((self.weight_h, self.weight_v), 0)
            self.padding = _quadruple(1)
        elif kernel_size == 5:
            self.weight_h = WEIGHT_H_5x5
            self.weight_v = WEIGHT_H_5x5.transpose(-1, -2)
            self.weight = torch.cat((self.weight_h, self.weight_v), 0)
            self.padding = _quadruple(2)        

    def __call__(self, u):
        """
        Compute both hor and ver grads
        Args:
            u (torch.Tensor): (B, C, H, W)
        Returns:
            grad_u: (B, C, 2, H, W), grad_u[:, :, 0] --> grad_h
                                     grad_u[:, :, 1] --> grad_v
        """
        # (B*C, 1, H, W)
        u_shape = u.shape
        u = u.view(-1, 1, *u_shape[-2:])
        u = F.conv2d(F.pad(u, self.padding, mode='circular'), self.weight, 
                        stride=1, padding=0, bias=None) / (self.dx)
        return u.view(*u_shape[:2], *u.shape[-3:])

    def grad_h(self, u):
        """
        Get image gradient along horizontal direction, or x axis.
        Perioid padding before conv.
        Args:
            u (torch.Tensor): [B, C, H, W]
        Returns:
            ux (torch.Tensor): [B, C, H, W] calculated gradient
        """
        ux = F.conv2d(F.pad(u, self.padding, mode='circular'), self.weight_h, 
                        stride=1, padding=0, bias=None) / (self.dx)
        return ux
    
    def grad_v(self, u):
        """
        Get image gradient along vertical direction, or y axis.
        Perioid padding before conv.
        Args:
            u (torch.Tensor): [B, C, H, W]
        Returns:
            uy (torch.Tensor): [B, C, H, W] calculated gradient
        """
        uy = F.conv2d(F.pad(u, self.padding, mode='circular'), self.weight_v, 
                        stride=1, padding=0, bias=None) / (self.dx)
        return uy



class Loss:

    def __init__(self, rawData, hp, experPaths, grad_kernels=[3, 3], device='cpu', info=print):
        self.info = info
        self.hp = hp
        self.nu = rawData.nu
        self.dt = rawData.dt
        self.zeta = lambda epoch: hp.zetaRate*epoch + hp.zeta0
        self.timeStepModel = hp.timeStepModel
        self.q = self.timeStepModel - 2
        self.device = device
        self.imDim = rawData.imDim
        self.paths = experPaths
        self.loadIRKweights()

        # Create gradients
        self.grad1 = SobelFilter2d(rawData.dx, kernel_size=grad_kernels[0], device=device)
        self.grad2 = LapLaceFilter2d(rawData.dx, kernel_size=grad_kernels[1], device=device)


    def __call__(self, model, data, epoch, batchIdx):

        model.epoch = epoch

        if model.training:
            return self.trainingLoss(model, data, epoch, batchIdx)
        else:
            return self.testingPred(model, data, epoch, batchIdx)

    
    def testingPred(self, model, data, epoch, batchIdx):
        """
        Vars:
            sensorLoc (Tensor): (currentBatchSize, timeStepModel, 2, numSensor)
            sensorData (Tensor): (currentBatchSize, timeStepModel, 2, numSensor)
            stateData (Tensor): (currentBatchSize, timeStepModel, 2, numNodes)    
        Returns:
            statePred (Tensor): (currentBatchSize, timeStepModel, 2, numNodes)
        """
        sensorLoc, sensorData, stateData = data
        currentBatchSize = sensorLoc.shape[0]

        
        self.info(f'    batch: {batchIdx}, batch size: {currentBatchSize}')

        model.sensorLoc = sensorLoc
        return model(sensorData), stateData


    def trainingLoss(self, model, data, epoch, batchIdx):
        """
        Vars:
            sensorLoc (Tensor): (currentBatchSize, timeStepModel, 2, numSensor)
            sensorData (Tensor): (currentBatchSize, timeStepModel, 2, numSensor)

            dataNodesLoc (Tensor): (currentBatchSize, timeStepModel, 2, numDataNodes)
            stateData (Tensor): (currentBatchSize, timeStepModel, 2, numDataNodes)    
        """
        sensorLoc, sensorData, dataNodesLoc, stateData = data
        currentBatchSize = sensorLoc.shape[0]
        if epoch % self.hp.logInterval == 0: self.info(f'    batch: {batchIdx}, batch size: {currentBatchSize}')

        # predict
        model.sensorLoc = sensorLoc
        statePred = model(sensorData)   

        # data loss at data nodes location
        stateNodesPred = statePred.gather(3, dataNodesLoc)  # (currentBatchSize, timeStepModel, 2, numSensor) 
        dataloss = stateNodesPred - stateData  # (currentBatchSize, timeStepModel, 2, numSensor)

        physicsloss = self.IRKloss(statePred)*self.zeta(epoch)  # (currentBatchSize, 2, H, W, 5)

        totalloss = T.cat((physicsloss.reshape(-1), dataloss.reshape(-1)))
##################################################################################################################
#         totalloss = dataloss.reshape(-1) # Considering only data loss
##################################################################################################################
        return F.mse_loss(totalloss, T.zeros_like(totalloss))


    def ddt(self, uPred):
        """
        Args:
           uPred (Tensor): (currentBatchSize, 2, H, W) 
        """

        grad_ux = self.grad1.grad_h(0.5*uPred[:,:1,:,:]**2)  # (currentBatchSize, 1, H, W)
        grad_uy = self.grad1.grad_v(uPred[:,:1,:,:])

        grad_vx = self.grad1.grad_h(uPred[:,1:,:,:])
        grad_vy = self.grad1.grad_v(0.5*uPred[:,1:,:,:]**2)

        div_u = self.nu * self.grad2(uPred[:,:1,:,:])
        div_v = self.nu * self.grad2(uPred[:,1:,:,:])

        burger_rhs_u = -grad_ux - uPred[:,1:,:,:]*grad_uy + div_u  # (currentBatchSize, 1, H, W)
        burger_rhs_v = -uPred[:,:1,:,:]*grad_vx - grad_vy + div_v
        
        return T.cat((burger_rhs_u, burger_rhs_v), dim=1)  # (currentBatchSize, 2, H, W)


    def IRKloss(self, statePred, test=False):
        """
        Args:
            statePred (Tensor): (currentBatchSize, timeStepModel, 2, H*W)
        """
        H, W = self.imDim
        timeStepModel = statePred.shape[1]
        statePredexp = statePred.reshape((-1, timeStepModel, 2, H, W))
        U = statePredexp.reshape((-1, 2, H, W))  # (currentBatchSize*timeStepModel, 2, H, W)

        ddt_U = self.ddt(U)  # (currentBatchSize*timeStepModel, 2, H, W)        
        ddt_U = ddt_U.reshape((-1, timeStepModel, 2, H, W)).permute(0, 2, 3, 4, 1)  # (currentBatchSize, 2, H, W, timeStepModel)

        rhsPredIRK = ddt_U[:, :, :, :, 1:-1] @ self.IRK_weights  # (currentBatchSize, 2, H, W, q) @ (q, q+1) -> (currentBatchSize, 2, H, W, q+1)
        U0PredIRK = statePredexp[:, 1:].permute(0, 2, 3, 4, 1) - (self.q+1)*self.dt * rhsPredIRK  # (currentBatchSize, 2, H, W, q+1)

        U0TargetIRK = T.cat([statePredexp[:, 0][:, :, :, :, None]]*(self.q+1), dim=4)
        loss = U0TargetIRK - U0PredIRK

        if test: return U0TargetIRK, U0PredIRK
        return loss  # (currentBatchSize, 2, H, W, q+1)


    def loadIRKweights(self):
        q = self.q
        tmp = np.float32(np.loadtxt(join(self.paths.code, f'Butcher_IRK{q}.txt'), ndmin = 2))
        IRK_weights = np.reshape(tmp[0:q**2+q], (q+1,q))  # (q+1, q)
        self.IRK_weights = T.from_numpy(IRK_weights).to(T.float32).T.to(self.device)  # (q, q+1)



def testEqn(rawData, hp, experPaths, args):
    """
    plot ddtU vs rhs Eqn at each time.
    Test eqn and spacial gradients.
    Vars:
        rawData.data (Tensor): (timeStep, 2, numNodes)
    """
    H, W = rawData.imDim
    hp.timeStepModel = 5
    lf = Loss(rawData, hp, experPaths, grad_kernels=[5, 5], device=args.device)

    ddtU_target = np.gradient(rawData.data[:10], rawData.dt, axis=0)[3:3+5].reshape((-1, 2, H, W))
    U = rawData.data[3:8].reshape((-1, 2, H, W))
    ddtU_Pred = lf.ddt(U)
    ddtU_Pred = ddtU_Pred.numpy()

    savePath = join(experPaths.experDir, f'lossfnEqnTest')
    plotParams = [0]*4, 4, rawData.imDim
    plotData = {'pred': ddtU_Pred[:, None], 'target': ddtU_target[:, None]}
    Plots().plotPred(plotData, plotParams, savePath)


def test_TintegScheme(rawData, hp, experPaths, args):
    """
    plot U0 vs U0s pred by implicit Runge-Kutta with q stages.
    Vars:
        rawData.data (Tensor): (timeStep, 2, numNodes)
        q (int): timeStepModel-2
    """
    H, W = rawData.imDim
    hp.timeStepModel = 10
    lf = Loss(rawData, hp, experPaths, grad_kernels=[5, 5], device=args.device)

    statePred = rawData.data[0:10][None] #T.cat((data[0:10][None], data[30:40][None]), 0)  # (currentBatchSize, timeStepModel, 2, H*W)
    U0Target, U0Pred = lf.IRKloss(statePred, test=True)  # (currentBatchSize, 2, H, W, q+1)
    q = lf.q

    U0PredBatch1 = U0Pred[0:1].permute(4, 0, 1, 2, 3).reshape((q+1, 1, 2, -1))  # (q+1, 1, 2, numNodes)
    U0TargetBatch1 = U0Target[0:1].permute(4, 0, 1, 2, 3).reshape((q+1, 1, 2, -1))  # (q+1, 1, 2, numNodes)

    savePath = join(experPaths.experDir, f'lossfnIRKTest1')
    plotParams = [0]*(q+1), q+1, rawData.imDim
    plotData = {'pred': U0PredBatch1.numpy(), 'target': U0TargetBatch1.numpy()}
    Plots().plotPred(plotData, plotParams, savePath)


if __name__ == '__main__':
    """
    Test Loss on simulation data
    """

    from Utils import Parser
    from Paths import Paths
    from Burgers2DPlots import Plots
    from Burgers2DLoadData import LoadData

    args = Parser().parse()
    args.device = 'cpu'

    # set useful paths to instance `experPaths`
    runName = 'firstTry'
    experDir = dirname(realpath(__file__))
    pathDict = {'data': 'solver/fenics_data_periodic', 'code': './'}
    experPaths = Paths(experDir, args.os, pathDict=pathDict)

    # set hyper params for run
    class HyperParams: pass
    hp = HyperParams()
    hp.zeta = None

    # load data
    rawData = LoadData(args, experPaths)  # (timeStep, 2, numNodes)

    # test 1
    testEqn(rawData, hp, experPaths, args)

    # test 2
    test_TintegScheme(rawData, hp, experPaths, args)   
