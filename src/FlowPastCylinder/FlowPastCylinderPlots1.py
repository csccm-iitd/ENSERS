"""
Set of Plots for Burgers2D results 
"""
import pdb
from typing import Dict, Any
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
import numpy as np

from os.path import dirname, realpath, join
import sys
filePath = realpath(__file__)
srcDir = dirname(dirname(filePath))
sys.path.append(srcDir)

from Utils import Dict2Class


class Plots:

    def save_show(self, plot, save_Path, fig, format='pdf', bbox_inches=None, pad_inches=0.1):
        if save_Path:
            Path = save_Path+ f'.{format}'
            fig.savefig(Path, format=f'{format}', bbox_inches=bbox_inches, pad_inches=pad_inches) 
            print(f'saved plot: {Path}')
        fig.show() if plot else 0
        plt.close(fig)
        plt.close('all')


    def burger2D_imshow(self, imData, axes, imParams):
        # H, w = self.imDim
        # axes.imshow(imData.reshape(self.imDim)[2:-2, 2:-2], interpolation='nearest', 
        #             cmap=imParams.cmap, vmin=imParams.v_min, vmax=imParams.v_max)
        axes.imshow(imData.reshape(imParams.imDim), interpolation='nearest', 
                    cmap=imParams.cmap, vmin=imParams.v_min, vmax=imParams.v_max)


    def femSimulation(self, plotData, plotParams, savePath):
        """
        Vars:
            imData (ndarray): [maxtimeStep, 2, numNodes]
            idxLs (list[int]): idx of InitConds to plot; len=4
        """
        imData = plotData
        idxLs, self.imDim = plotParams

        mpl.rcParams['font.family'] = ['serif'] # default is sans-serif
        mpl.rc('text', usetex=False)
        cmap = mpl.cm.get_cmap('jet') 

        fig, ax = plt.subplots(2, len(idxLs))

        for i, idx in enumerate(idxLs):
            imParams = {'cmap':cmap, 'v_min':np.min(imData), 'v_max':np.max(imData)}

            self.burger2D_imshow(imData[idx, 0], ax[0, i], Dict2Class(imParams))
            self.burger2D_imshow(imData[idx, 1], ax[1, i], Dict2Class(imParams))
            
            ax[0, i].axis('off')
            ax[1, i].axis('off')

        self.save_show(1, savePath, fig)


    def plotPred(self, plotData: Dict, plotParams, savePath: str):
        """
        Args:
            plotData (Dict):
        Vars:
            pred (ndarray): (numSampTest, timeStepModel, 3, numNodes)
            target (ndarray): (numSampTest, timeStepModel, 3, numNodes)
        """

        pred = plotData['pred']
        target = plotData['target']
        idxLs, numPlots, imDim, imDimCrop = plotParams

        pred = pred.reshape((pred.shape[0], pred.shape[1], pred.shape[2])+ imDim)[:, :, :, 2:-2, 2:-2].reshape(-1)
        target = target.reshape((target.shape[0], target.shape[1], target.shape[2])+ imDim)[:, :, :, 2:-2, 2:-2].reshape(-1)

        pred = pred.reshape((pred.shape[0], pred.shape[1], pred.shape[2], np.prod(imDimCrop)))
        target = target.reshape((target.shape[0], target.shape[1], target.shape[2], np.prod(imDimCrop)))

        error = np.abs(pred[:] - target[:])

        ch = pred.shape[2]

        plt.close("all")
        mpl.rcParams['font.family'] = ['serif']  # default is sans-serif
        mpl.rc('text', usetex=False)
        mpl.rc('font', size=8)
        cmap = mpl.cm.get_cmap('jet')  
        cmap_error = mpl.cm.get_cmap('inferno') 
        
        
        fig, ax = plt.subplots(3*ch, numPlots, figsize=(numPlots*3, ch*6))

        for i in range(numPlots):
            for j in range(ch):
                t_ij = target[i, idxLs[i], j]
                p_ij = pred[i, idxLs[i], j]

                c_max = np.max(np.array([t_ij, p_ij]))
                c_min = np.min(np.array([t_ij, p_ij]))
                imParams = {'cmap':cmap, 'v_min':c_min, 'v_max':c_max}

                self.burger2D_imshow(target[i, idxLs[i], j], ax[3*j, i], Dict2Class(imParams))
                self.burger2D_imshow(pred[i, idxLs[i], j], ax[3*j+1, i], Dict2Class(imParams))

                c_max_error = np.max(np.abs(error[i, idxLs[i], j]))
                c_min_error = np.min(np.abs(error[i, idxLs[i], j]))
                imParams = {'cmap':cmap_error, 'v_min':c_min_error, 'v_max':c_max_error}

                self.burger2D_imshow(error[i, idxLs[i], j], ax[3*j+2, i], Dict2Class(imParams))
                

                p0 =ax[3*j, i].get_position().get_points().flatten()
                p1 = ax[3*j+1, i].get_position().get_points().flatten()
                ax_cbar = fig.add_axes([p1[2]+0.0075, p1[1], 0.005, p0[3]-p1[1]])
                ticks = np.linspace(0, 1, 5)
                tickLabels = np.linspace(c_min, c_max, 5)
                tickLabels = ["{:02.2f}".format(t0) for t0 in tickLabels]
                
                cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=plt.get_cmap(cmap), orientation='vertical', ticks=ticks)
                cbar.set_ticklabels(tickLabels)

                p0 =ax[3*j+2, i].get_position().get_points().flatten()
                ax_cbar = fig.add_axes([p0[2]+0.0075, p0[1], 0.005, p0[3]-p0[1]])
                ticks = np.linspace(0, 1, 5)
                tickLabels = np.linspace(c_min_error, c_max_error, 5)
                tickLabels = ["{:02.2f}".format(t0) for t0 in tickLabels]

                cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=plt.get_cmap(cmap_error), orientation='vertical', ticks=ticks)
                cbar.set_ticklabels(tickLabels)

                for ax0 in ax[:,i]:
                    ax0.set_xticklabels([])
                    ax0.set_yticklabels([])
                
                if i==0:
                    ax[3*j, i].set_ylabel('true', fontsize=14)
                    ax[3*j+1, i].set_ylabel('prediction', fontsize=14)
                    ax[3*j+2, i].set_ylabel('L1 error', fontsize=14)
        
        self.save_show(1, savePath, fig)


    def violinplot(self, l2Error, plotParams, savePath):
        """
        Args:
            plotParams (Dict):
        Vars:
            l2Error (ndarray): (numNoiseLevels, numSampTest*2*numNodes)
        """
        # xticks, xticklabels, SNRdbLs
        pp = plotParams

        fig, ax = plt.subplots()
        vp = ax.violinplot(l2Error, pp.xticks, widths=4, showmeans=True)#,showmeans=False, showmedians=False, showextrema=False)
        # styling:
        for body in vp['bodies']:
            body.set_alpha(0.2)
        ax.set(xlim=(0, 120), xticks=pp.xticks, xlabel=pp.xlabel, xticklabels=pp.xticklabels,
                ylim=(0, 0.5), yticks=np.arange(0, 0.5, 0.05), ylabel=pp.ylabel)
        self.save_show(1, savePath, fig)


    def noisePlots(self, plotData, plotParams, savePath):
        pp = plotParams
        self.imDim = pp.imDim

        mpl.rcParams['font.family'] = ['serif'] # default is sans-serif
        mpl.rc('text', usetex=False)
        mpl.rc('font', size=8)
        cmap = mpl.cm.get_cmap('jet') 

        fig, ax = plt.subplots(1, len(plotData))

        for i in range(len(plotData)):
            imParams = {'cmap':cmap, 'v_min':np.min(plotData[i]), 'v_max':np.max(plotData[i])}

            self.burger2D_imshow(plotData[i], ax[i], Dict2Class(imParams))
            ax[i].set(title=f'{pp.titleLs[i]}')
            ax[i].axis('off')

        self.save_show(1, savePath, fig, bbox_inches='tight', pad_inches=0.1)
    