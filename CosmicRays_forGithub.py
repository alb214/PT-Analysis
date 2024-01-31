# -*- coding: utf-8 -*-
"""
Created on Thu May 25 16:11:06 2023

@author: aboeh
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from basics_forGithub import NormyNormSpecial

def DeleteCosmicRays(X,Y,peaks_i,peaks_x,peaks_y):
    '''Delete one cosmic ray at at time.'''
    Ynew =  Y.copy()
    for i,j in enumerate(peaks_i):
     # j is index of ith cosmic ray
    
        for k in range(1,11): #11
            if (i-k) < 0:
                # if past first point in cosmic rays list
                # for ave, use 3 points to left of first point in costmic rays list
                ai = j - (k+4)
                bi = j - k
                break
            else:
                if (j-k) in peaks_i:
                    # if index (j-k) in list of indices of cosmic rays
                    pass
                else:
                    # if it's not in list, then also check point to the left 
                    if (j-(k+1)) in peaks_i:
                        # if that point is in list, then still same cosmic ray
                        pass
                    else:
                        # if both are not in list, then hopefully little region to the left is good for averaging
                        ai = j - (k+4)
                        bi = j - k
                        break
        # print('ai=',ai,'bi=',bi)#,'\n')
        if (bi is None) or (ai is None):
            # print(peaks_i)
            # maybe this isn't a cosmic ray?
            continue
                
        # print('j=',j)

        if ai < 0:
            ai = None
        elif bi > len(Y):
            bi = None
        else:
            pass
        if bi < 0:
            bi = 3
        # ave = np.mean([Y_more[ai],Y_more[bi]])
        ave = np.mean(Y[ai:bi])
        # print('ave',ave)
        ci = j-1
        di = j+1
        if ci < 0:
            ci = None
        elif di > len(Y):
            di = None
        else:
            pass
        if di < 2:
            # print('di=',di)
            di = 2
        # print('ci=',ci,'di=',di)
        Ynew[ci:di] = np.ones(2) * ave
    return Ynew         
        


def RemoveCosmicRays(X,Y,threshold = 0.05,plot=False,testing=False):
    '''
    '''
    if plot:
        fig,axes = plt.subplots(nrows=3, ncols=1,figsize=(5,5),sharex=True)
    region = [0,200]
    Ynorm = NormyNormSpecial(X,Y,region)
    deriv = savgol_filter(Ynorm, 7, polyorder = 5, deriv=1, mode='nearest')
    std = np.std(Ynorm[:10])
    indices = []
    for i,dx in enumerate(deriv):
        if (dx < -threshold) or (dx > threshold):
            indices.append(i)
    if len(indices) > 0:
        peaks_i,peaks_x,peaks_y = PeakIndices2PeakCoords(indices,X,Y)
        if plot:
            axes[1].plot(X,deriv)
            axes[0].plot(X,Y)
            axes[0].scatter(peaks_x,peaks_y,marker='x',color='r',s=50)
        Ynew = DeleteCosmicRays(X,Y,peaks_i,peaks_x,peaks_y)
        if plot:
            axes[2].plot(X,Ynew,color='k')
            axes[0].set(yscale='log',xlim=[0,500])
            axes[2].set(yscale='log',xlim=[0,500])
        return Ynew
    else:
        return Y
        
    


def PeakIndices2PeakCoords(peaks_i,X,Y):
    peaks_x = []
    peaks_y = []
    peaks_indices = []
    for j in range(len(peaks_i)):
        x = X[peaks_i[j]]
        y = Y[peaks_i[j]]
        if (x < -100 or x > 100):
            # skip center peaks
            peaks_x.append(x)
            peaks_y.append(y)
            peaks_indices.append(peaks_i[j])
    return peaks_indices,peaks_x,peaks_y