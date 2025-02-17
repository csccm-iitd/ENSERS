"""
Set of Plots for Burgers2D results 
"""
import pdb
from typing import Dict, Any
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
import numpy as np

import os
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
        # fig.show() if plot else 0
        plt.close(fig)
        plt.close('all')


    def burger2D_imshow(self, imData, axes, imParams):
        H, w = self.imDim
        alpha = imParams.alpha if hasattr(imParams, 'alpha') else 1
        axes.imshow(imData.reshape((H, w)), interpolation='nearest', cmap=imParams.cmap, vmin=imParams.v_min, vmax=imParams.v_max, alpha=alpha)
        # axes.axis('off')


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

        self.save_show(1, savePath, fig)


    def plotPred(self, plotData: Dict, plotParams, savePath: str):
        """
        Args:
            plotData (Dict):
        Vars:
            pred (ndarray): (numSamp, timeStepModel, 2, numNodes)
            target (ndarray): (numSamp, timeStepModel, 2, numNodes)
        """
        
        pp = plotParams
        self.imDim = pp.imDim
        idxLs = pp.tStepModelPlot

        pred = plotData['pred'][pp.tStepPlot]
        target = plotData['target'][pp.tStepPlot]
        error = np.abs(pred[:] - target[:])

        numPlots = pred.shape[0]
        ch = pred.shape[2]

        plt.close("all")
        mpl.rcParams['font.family'] = ['serif']  # default is sans-serif
        mpl.rc('text', usetex=False)
        mpl.rc('font', size=8)
        cmap = mpl.cm.get_cmap('jet')  
        cmap_error = mpl.cm.get_cmap('inferno') 
        
        
        # fig, ax = plt.subplots(6, numPlots, figsize=(numPlots*3, 15))
        fig, ax = plt.subplots(3*ch, numPlots, figsize=(numPlots*3, ch*6))
        # fig.subplots_adjust(wspace=0.5)

        for i in range(numPlots):
            for j in range(ch):
                
                c_max = np.max(np.array([ target[i, idxLs[i], j], pred[i, idxLs[i], j] ]))
                c_min = np.min(np.array([ target[i, idxLs[i], j], pred[i, idxLs[i], j] ]))
                imParams = {'cmap':cmap, 'v_min':c_min, 'v_max':c_max}

                self.burger2D_imshow(target[i, idxLs[i], j], ax[3*j, i], Dict2Class(imParams))
                self.burger2D_imshow(pred[i, idxLs[i], j], ax[3*j+1, i], Dict2Class(imParams))

                c_max_error = np.max(error[i, idxLs[i], j])
                c_min_error = np.min(error[i, idxLs[i], j])
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
                    ax0.xaxis.set_ticks([])
                    ax0.yaxis.set_ticks([])

                # for ax0 in ax[:,i]:
                #     ax0.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
                #     ax0.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
                #     if(i > 0):
                #         ax0.set_yticklabels([])
                
                if i==0:
                    ax[3*j, i].set_ylabel('true', fontsize=14)
                    ax[3*j+1, i].set_ylabel('prediction', fontsize=14)
                    ax[3*j+2, i].set_ylabel('L1 error', fontsize=14)
                        
            # ax[0, i].set_title(f'idx={idxLs[i]}', fontsize=14)
        
        self.save_show(1, savePath, fig, bbox_inches='tight')


    def _violinplot(self, l2Error, plotParams, savePath):
        """
        Args:
            plotParams (Dict):
        Vars:
            l2Error (ndarray): (numNoiseLevels, numSampTest*2*numNodes)
        """
        # xticks, xticklabels, SNRdbLs
        pp = plotParams

        fig, ax = plt.subplots()
        vp = ax.violinplot(l2Error, pp.xticks, widths=2, showmeans=True)#,showmeans=False, showmedians=False, showextrema=False)
        # styling:
        for body in vp['bodies']:
            body.set_alpha(0.2)
        ax.set(xlim=(0, 120), xticks=pp.xticks, xlabel=pp.xlabel, xticklabels=pp.xticklabels,
                ylim=(0, 0.5), yticks=np.arange(0, 0.5, 0.05), ylabel=pp.ylabel)
        self.save_show(1, savePath, fig)


    def violinplot(self, l2Error, plotParams, savePath):
        """
        Args:
            plotParams (Dict):
        Vars:
            l2Error (ndarray): (M, numNoiseLevels, numSampTest*numNodes)
        """
        import matplotlib.patches as mpatches
        pp = plotParams

        labels = []
        def add_label(violin, label):
            color = violin["bodies"][0].get_facecolor().flatten()
            labels.append((mpatches.Patch(color=color), label))

        fig, ax = plt.subplots()

        for m in range(l2Error.shape[0]):
            vp = ax.violinplot(l2Error[m], pp.xticksPlot[m], widths=4, showmeans=True)#,showmeans=False, showmedians=False, showextrema=False)
            
            add_label(vp, pp.label[m])
            # styling:
            for body in vp['bodies']:
                body.set_facecolor(pp.facecolor[m])
                body.set_edgecolor('black')
                body.set_alpha(0.4)
        ax.set(xlim=(0, 90), xticks=pp.xticks, xlabel=pp.xlabel, xticklabels=pp.xticklabels,
                ylim=(0, 0.5), yticks=np.arange(0, 0.5, 0.05), ylabel=pp.ylabel, title=pp.title)
        
        plt.legend(*zip(*labels), loc=2)
        self.save_show(1, savePath, fig)


    def noisePlots(self, plotData, plotParams, savePath):
        pp = plotParams
        self.imDim = pp.imDim

        mpl.rcParams['font.family'] = ['serif'] # default is sans-serif
        mpl.rc('text', usetex=False)
        mpl.rc('font', size=4)
        cmap = mpl.cm.get_cmap('jet') 

        fig, ax = plt.subplots(1, len(plotData))

        for i in range(len(plotData)):
            imParams = {'cmap':cmap, 'v_min':np.min(plotData[i]), 'v_max':np.max(plotData[i])}

            self.burger2D_imshow(plotData[i], ax[i], Dict2Class(imParams))
            ax[i].set(title=f'{pp.titleLs[i]}')
            ax[i].axis('off')

        self.save_show(1, savePath, fig, bbox_inches='tight', pad_inches=0.1)

    
    def paperFig1(self, plotData: Dict, plotParams, savePath: str):
        """
        Args:
            data (ndarray): (numSamp, 2, numNodes)
        """
        def sensorCoords(num, imDim, numNodes, i):
            nx, ny = imDim
            np.random.seed(i)
            loc = np.random.choice(range(numNodes), size=num, replace=False)
            xx, yy = np.meshgrid(range(nx), range(ny))
            return xx.reshape(-1)[loc], yy.reshape(-1)[loc]

        data = plotData
        # pdb.set_trace()
        self.imDim, idxLs = plotParams
        numNodes = data.shape[2]
        alpha = [0.3, 1, 0.3]
        name = ['Sensor', 'Pred', 'Target']
        numSensor = [5, 0, 40]
        color = ['b', 'r', 'g']
        # sizes = (600, 600)#(256, 256)

        cmap = mpl.cm.get_cmap('jet')

        for i in range(3):
            for j in range(len(idxLs)):

                x, y = sensorCoords(numSensor[i], self.imDim, numNodes, 2**i*j)

                fig, ax = plt.subplots(1)
                fig.set_size_inches(2, 2, forward = False)

                imParams = {'cmap':cmap, 'v_min':np.min(data[idxLs[j], 0]), 'v_max':np.max(data[idxLs[i], 0]), 'alpha': alpha[i]}
                self.burger2D_imshow(data[idxLs[i], 0], ax, Dict2Class(imParams))
                
                ax.scatter(x, y, s=20, c=color[j])

                self.save_show(1, join(savePath, f'u{name[i]}{j}'), fig)

                # ---------------------------

                fig, ax = plt.subplots(1)
                fig.set_size_inches(2, 2, forward = False)

                imParams = {'cmap':cmap, 'v_min':np.min(data[idxLs[j], 1]), 'v_max':np.max(data[idxLs[i], 1]), 'alpha': alpha[i]}
                self.burger2D_imshow(data[idxLs[i], 1], ax, Dict2Class(imParams))
                ax.scatter(x, y, s=20, c=color[j])

                self.save_show(1, join(savePath, f'v{name[i]}{j}'), fig, format='png', bbox_inches='tight', pad_inches = 0)
        