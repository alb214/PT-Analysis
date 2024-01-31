# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 14:45:35 2023

@author: aboeh


"""
import importlib,sys
importlib.reload(sys.modules['basics_forGithub'])
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                                AutoMinorLocator)

import pandas as pd
import numpy as np
import cv2 as cv
# from sklearn.covariance import MinCovDet 
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA as sk_pca  
# # from sklearn.preprocessing import StandardScaler  
# from sklearn import svm  
# from sklearn.cluster import KMeans
from sklearn.covariance import EmpiricalCovariance, MinCovDet 

from basics_forGithub import ReduceNoise,FindIndexX,CreateColorsList,MakeExponentialStr,ensure_dir, NormalizeTo1,NormalizeTo1Special,NormyNormSpecial, SortKeys, SortParticles
import Picocavities_forGithub as pico
from CosmicRays_forGithub import RemoveCosmicRays

# <> DEAL WITH THIS:
from lab02_instrument_response_function_v10 import IRF,CropLaserLine

version = 'v03'


cm = 1/2.54  # centimeters in inches

plt.rcParams["font.family"] = 'Times New Roman'#'Calibri'
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({
    "text.usetex": False,
})





# def MakeSample_dict(pth,folder,h5file,sample,readoutnoiselevel,X,power,show=True,save=False):
#     '''
#     h5file must be ..._combined.h5
#     X must be Gen_Wn.
#     Stores average of timescan in dictionary of particles.
#     Scaled offNPoM laser line is subtracted from each average.
#     All negative points in |X| < 20 cm-1 are set to 10**-4.
#     Spectrum is not cut.
#     Plot is on log scale.
#     '''
#     fullname = pth + folder + h5file
#     Title = sample + ' all particles - laser line subtracted'#h5file.split('.')[0] + ' ' + sample
#     sample_dict = {}
#     fig,ax = plt.subplots(nrows=1, ncols=1,figsize=(5,5))
#     ax.set_title(Title)
#     with h5py.File(fullname, 'r') as f:  
#         BG = (f['Reference and Background Spectra']['BG'][:]-readoutnoiselevel) / f['Reference and Background Spectra']['BG'].attrs['Exposure']
#         off = ((f['Reference and Background Spectra']['offNPoM'][:]-(readoutnoiselevel)) / f['Reference and Background Spectra']['offNPoM'].attrs['Exposure']) - BG
#         laser_line = CropLaserLine(off)  
#         group = f['ParticleScannerScan_0']
#         particles = [key for key in group.keys() if 'Particle' in key]
#         particles_sorted,timestamps = SortKeys(group,particles)
#     #     print(group['Particle_0'].keys()) #'CWL.thumb_image_0', 'SERS_0', 'z_scan_0'
#         for p,particle in enumerate(particles_sorted):
#             SERS = group[particle]['SERS_0'][:]
#             exposure = group[particle]['SERS_0'].attrs['Exposure']
#             if len(np.shape(SERS)) > 1:
#                 total = np.zeros(np.shape(SERS)[1])
#                 for row in SERS:
#                     Y = (((row-readoutnoiselevel)/exposure/power) - BG )
#                     total += Y
#                 average = (total / np.shape(SERS)[0])
#             else:
#                 average = (((SERS-readoutnoiselevel)/exposure/power) - BG )
#             ci = FindIndexX(0,X)
#             mx_ave = max(average[:ci])
#             laser_line_scaled = NormalizeTo1(laser_line) * mx_ave

#             Ynew = average - laser_line_scaled
#             Ynew[(Ynew <= 0.) & (np.abs(X) < 20.)] = 10**-4
#             sample_dict[particle] = Ynew#[ai:bi]
#             ax.plot(X,Ynew)      
#     ax.set(yscale='log')
#     if show:
#         fig.show()
#     if save:
#         figname = pth+folder+Title+'.png'
#         print(figname)
#         ensure_dir(figname)
#         fig.savefig(figname,format='png',bbox_inches='tight',transparent=False)
#     return sample_dict



def check_center(image,check=False,pixel_range=[40,60],rd_range=[0,16]):
    threshold=0.00001 
    img_0=np.asarray(image)
    img_cut=img_0[pixel_range[0]:pixel_range[1],pixel_range[0]:pixel_range[1],:]
    imgray = cv.cvtColor(img_cut, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgray,threshold, 255, cv.THRESH_TOZERO)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    result=False
    for i,j in enumerate (contours):
        (x, y), radius=cv.minEnclosingCircle(contours[i])
        min_0=(pixel_range[1]-pixel_range[0])/2 -5
        max_0=(pixel_range[1]-pixel_range[0])/2 +5
        if rd_range[1] > radius > rd_range[0]  and min_0 < x < max_0 and  min_0 < y < max_0:
            result=True
            break
    if check:
        plt.subplots()
        imgray_w = cv.cvtColor(img_0, cv.COLOR_BGR2GRAY)
        ret_w, thresh_w = cv.threshold(imgray_w, threshold, 255, cv.THRESH_BINARY)
        contours_w, hierarchy_w = cv.findContours(thresh_w, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        img_ct=cv.drawContours(img_0, contours_w, -1, (0,255,0), 1)
        plt.imshow(img_ct)
        plt.subplots()
        x, y, radius = np.int0((x,y,radius))
        img_c=cv.circle(img_cut, (x,y), radius, (0, 0, 255), 2)
        plt.imshow(img_c)
        print('Radius is:' +str(radius))
    return result


# ~a~a~a~a~a~ Using average spectrum for each particle ~a~a~a~a~a~
# Step (1)
def Centered(pth,folder,h5file,sample,readoutnoiselevel,X,power,threshold=0.05,mode='nano',skipsingles=True,show=True,save=False,testing=False):
    '''
    h5file must be ..._combined.h5
    X must be Gen_Wn.
    Stores average of timescan in dictionary of centered particles.
    Scaled offNPoM laser line is subtracted from each average.
    All negative points in |X| < 20 cm-1 are set to 10**-4.
    Spectrum is not cut.
    Plot is on linear scale.
    Modes: 'nano' or 'ave'
    '''
    fullname = pth + folder + h5file
    Title = sample + ' centered particles' 
    not_centered = {}
    centered = {}
    
    fig,axes = plt.subplots(nrows=2, ncols=1,figsize=(6,6),sharex=True)
    fig.suptitle(sample + ' rejecting not-centered particles removes all duds?')
    axes[0].set_title('centered')
    axes[1].set_title('not centered')
    with h5py.File(fullname, 'r') as f: 
        BG = (f['Reference and Background Spectra']['BG'][:]-readoutnoiselevel) / f['Reference and Background Spectra']['BG'].attrs['Exposure']
        group = f['ParticleScannerScan_0']
        particles = [key for key in group.keys() if 'Particle' in key]
        particles_sorted,timestamps = SortKeys(group,particles,meas='SERS_0')
        s=0
        n=0
        for p,particle in enumerate(particles_sorted):
            SERS = group[particle]['SERS_0'][:]
            exposure = group[particle]['SERS_0'].attrs['Exposure']
            if len(np.shape(SERS)) > 1:
                if mode == 'ave':
                    # Average SERS
                    if len(np.shape(SERS)) > 1:
                        average = AverageSERS(X,SERS,BG,readoutnoiselevel,exposure,power,threshold)
                    else:
                        Y = (((SERS-readoutnoiselevel)/exposure/power) - BG )
                        average = RemoveCosmicRays(X,Y,threshold=threshold,plot=False,testing=testing)
                elif mode == 'nano':
                    if np.shape(SERS)[0] > 2: #4:
                        
                        n+=1
                        # Nanocavity
                        multiplier = 1
                        deg = 1
                        num_stds = 2.5
                        im = []
                        for row in SERS:
                            Y = (((np.array(row)-readoutnoiselevel)/exposure/power) - BG )
                            Y_clnd = RemoveCosmicRays(X,Y,threshold=threshold,plot=False)
                            im.append(Y_clnd)
                        nanocavity_im = pico.MakeNanocavityIm(np.array(im),multiplier,deg,testing=testing)
                        if testing:
                            pico.PlotNanocavity(nanocavity_im,'Nanocavity '+particle)
                        average = pico.AverageOfRows(nanocavity_im)
                    else:
                        if testing:
                            print('Taking average of '+particle+' SERS')
                        average = AverageSERS(X,SERS,BG,readoutnoiselevel,exposure,power,threshold)
                else:
                    print('Mode must be nano or ave')
            else:
                s+=1
                if skipsingles:
                    continue
                else:
                    average = AverageSERS(X,SERS,BG,readoutnoiselevel,exposure,power,threshold) 
            Ynew = average
            img_0 = group[particle]['CWL.thumb_image_0']
            c_c=check_center(img_0)
            if c_c == True:
                centered[particle] = Ynew#[ai:bi]
                axes[0].plot(X,Ynew)
            else:
                not_centered[particle] = Ynew#[ai:bi]
                axes[1].plot(X,Ynew)
    for ax in axes:
        ax.set(yscale='log',xlim=[-1300,1300]) #ylim=[0,2500],
        ax.set_ylim(bottom = 10**-1)
    fig.tight_layout()
    if show:
        plt.show()
    if save:
        figname = pth+folder+Title+'.png'
        print(figname)
        ensure_dir(figname)
        fig.savefig(figname,format='png',bbox_inches='tight',transparent=False)
    print('Dataset contains {} single spectra (instead of timescans)'.format(s))
    # print('Particles w cosmic rays',lst)
    print('num in nano',n,'num particles',len(particles))
    return centered, not_centered

# SelectK
def CenteredSelectK(pth,folder,h5file,sample,selectK,readoutnoiselevel,X,power,threshold=0.05,mode='nano',skipsingles=True,show=True,save=False,testing=False):
    '''
    h5file must be ..._combined.h5
    X must be Gen_Wn.
    Stores average of timescan in dictionary of centered particles.
    All negative points in |X| < 20 cm-1 are set to 10**-4.
    Spectrum is not cut.
    Plot is on linear scale.
    Modes: 'nano' or 'ave'
    '''
    fullname = pth + folder + h5file
    Title = sample + ' centered particles' 
    not_centered = {}
    centered = {}
    fig,axes = plt.subplots(nrows=2, ncols=1,figsize=(6,6),sharex=True)
    fig.suptitle(sample + ' rejecting not-centered particles removes all duds?')
    axes[0].set_title('centered')
    axes[1].set_title('not centered')
    with h5py.File(fullname, 'r') as f: 
        BG = (f['Reference and Background Spectra']['BG'][:]-readoutnoiselevel) / f['Reference and Background Spectra']['BG'].attrs['Exposure']        
        group = f['ParticleScannerScan_0']
        particles = [key for key in group.keys() if 'Particle' in key]
        #particles_sorted,timestamps = SortKeys(group,particles)
        particles_sorted,timestamps = SortKeys(group,particles,meas='SERS_0')
        s=0
        n=0
        for p,particle in enumerate(particles_sorted):
            if particle in selectK:
                SERS = group[particle]['SERS_0'][:]
                exposure = group[particle]['SERS_0'].attrs['Exposure']
                if len(np.shape(SERS)) > 1:
                    if mode == 'ave':
                        # Average SERS
                        if len(np.shape(SERS)) > 1:
                            average = AverageSERS(X,SERS,BG,readoutnoiselevel,exposure,power,threshold)
                        else:
                            Y = (((SERS-readoutnoiselevel)/exposure) - BG ) / power
                            average = RemoveCosmicRays(X,Y,threshold=threshold,plot=False,testing=testing)
                    elif mode == 'nano':
                        if np.shape(SERS)[0] > 2: #4:
                            n+=1
                            # Nanocavity
                            multiplier = 1#3 #2
                            deg = 1
                            num_stds = 2.5#3
                            # if len(np.shape(SERS)) > 1:
                            im = []
                            for row in SERS:
                                Y = (((np.array(row)-readoutnoiselevel)/exposure) - BG ) /power #- off
                                Y_clnd = RemoveCosmicRays(X,Y,threshold=threshold,plot=False)
                                im.append(Y_clnd)
                            nanocavity_im = pico.MakeNanocavityIm(np.array(im),multiplier,deg,testing=testing)
                            if testing:
                                pico.PlotNanocavity(nanocavity_im,'Nanocavity '+particle)
                            average = pico.AverageOfRows(nanocavity_im)
                        else:
                            if testing:
                                print('Taking average of '+particle+' SERS')
                            average = AverageSERS(X,SERS,BG,readoutnoiselevel,exposure,power,threshold)
                    else:
                        print('Mode must be nano or ave')
                else:
                    s+=1
                    if skipsingles:
                        continue
                    else:
                        average = AverageSERS(X,SERS,BG,readoutnoiselevel,exposure,power,threshold)
                Y_irf = IRF(X,average,WLcenter)
                Ynew = average
                img_0 = group[particle]['CWL.thumb_image_0']
                c_c=check_center(img_0)
                if c_c == True:
                    centered[particle] = Ynew
                    axes[0].plot(X,Ynew)
                else:
                    not_centered[particle] = Ynew
                    axes[1].plot(X,Ynew)
    for ax in axes:
        ax.set(yscale='log',xlim=[-1300,1300]) #ylim=[0,2500],
        ax.set_ylim(bottom = 10**-1)
    fig.tight_layout()
    if show == True:
        plt.show()
    if save == True:
        figname = pth+folder+Title+'.png'
        print(figname)
        ensure_dir(figname)
        fig.savefig(figname,format='png',bbox_inches='tight',transparent=False)
    print('Dataset contains {} single spectra (instead of timescans)'.format(s))
    return centered, not_centered

def AverageSERS(X,SERS,BG,readoutnoiselevel,exposure,power,threshold):
    if len(np.shape(SERS)) > 1:
        total = np.zeros(np.shape(SERS)[1])
        for row in SERS:
            Y = (((row-readoutnoiselevel)/exposure/power) - BG ) #- off
            Y_clnd = RemoveCosmicRays(X,Y,threshold=threshold,plot=False)
            total += Y_clnd
        average = (total / np.shape(SERS)[0]) 
    else:
        Y = (((SERS-readoutnoiselevel)/exposure/power) - BG ) 
        average = RemoveCosmicRays(X,Y,threshold=threshold,plot=False)
    return average

# Step (2)
def HighI(X,sample_dict,percent):
    '''Input centered'''
    ai = FindIndexX(100,X)
    bi = FindIndexX(500,X)
    Averages = []
    K = []
    for particle in sample_dict.keys():
        Y = sample_dict[particle]
        Y_cut = Y[ai:bi]
        Averages.append(np.mean(Y_cut))
        K.append(particle)
    decoratedK = list(zip(Averages,K))
    sorteddecoratedK = sorted(decoratedK)
    sortedK = [tpl[1] for tpl in sorteddecoratedK]     
    percentile = int(percent * len(Averages))
    highIkeys = sortedK[percentile:]
    highI = {}
    for particle in sample_dict.keys():
        if particle in highIkeys:
            highI[particle] = sample_dict[particle]
    return highI

def LowI(X,sample_dict,percent):
    '''Input centered. retruns sample dict containing particles w intensities below percentile.'''
    ai = FindIndexX(100,X)
    bi = FindIndexX(500,X)
    Averages = []
    K = []
    for particle in sample_dict.keys():
        Y = sample_dict[particle]
        Y_cut = Y[ai:bi]
        Averages.append(np.mean(Y_cut))
        K.append(particle)
    decoratedK = list(zip(Averages,K))
    sorteddecoratedK = sorted(decoratedK)
    sortedK = [tpl[1] for tpl in sorteddecoratedK]    
    percentile = int(percent * len(Averages))
    lowIkeys = sortedK[:percentile]
    lowI = {}
    for particle in sample_dict.keys():
        if particle in lowIkeys:
            lowI[particle] = sample_dict[particle]
    return lowI


# Step (3)
def MakeMatrix(X,sample_dict,xlim=[-1300,1300],normalize=False,smooth=False,testing=False):
    '''
    * Don't use original sample_dict; use dict of centered particles. 
    Returns matrix with ordered rows of average SERS of each timescan, 
    cut to specified region,
    as well as cut X
    '''
    x1 = xlim[0]
    x2 = xlim[1]
    ai = FindIndexX(x1,X)
    bi = FindIndexX(x2,X)
    if testing:
        print('MakeMatrix {}={},{}={}'.format(x1,X[ai],x2,X[bi]))
    X_cut = X[ai:bi]
    if testing:
        print(len(X_cut))
    particles_sorted = SortParticles(list(sample_dict.keys()))
    M = []
    for p,particle in enumerate(particles_sorted):
        average = sample_dict[particle]
        if smooth == True:
            notchedge = 9.
            Y_smthd = ReduceNoise(X,average,notchedge,cutoff = 5000) #Smooth(Y[ai:bi],window_length,polyorder)
        else:
            Y_smthd = average
            
        if normalize:
            if x1 > -100:
                region = [x1,100]
            else:
                region=[-100,100]
            if testing:
                ci = FindIndexX(region[0],X[ai:bi])
                di = FindIndexX(region[1],X[ai:bi])
                print('MakeMatrix normalize to region {}={},{}={}'.format(region[0],X[ai:bi][ci],region[1],X[ai:bi][di]))
            Y_norm = NormyNormSpecial(X[ai:bi],Y_smthd[ai:bi],region,testing=False)
        else:
            Y_norm = Y_smthd[ai:bi]
        M.append(Y_norm)
    M = np.array(M)
    if np.shape(M)[1] != len(X_cut):
        print('Error! select different xlim to ensure matrix height is length of X_cut.')
    return X_cut,M

# Step (x)
def MakeSERSDF(X,sample_dict,xlim=None):
    '''Sample dict should not be all particles; should be centered particles only. 
    The spectra are cut to a subregion defined by xlim.
    returns df where indices are particles and columns are wavenumbers.
    '''
    df_sers = pd.DataFrame(sample_dict)
    # Check if Nans in df
    if df_sers[df_sers.isna().any(axis=1)].shape[0] > 0:
        print(df_sers[df_sers.isna().any(axis=1)].index, 'contain Nan')
    X_str = ["%.1f" % x for x in X]
    I = list(range(len(X_str)))
    cols = dict(zip(I,X_str))
    df_T = df_sers.rename(index=cols).T
    if xlim is not None:
        x1 = xlim[0]
        x2 = xlim[1]
        ai = FindIndexX(x1,X)
        bi = FindIndexX(x2,X)
        if np.isnan(ai) or np.isnan(bi):
            print('Could not cut matrix to desired xlims, because')
            print('ai:{},bi:{}'.format(ai,bi))
            if ai is None:
                print('spectrum starts at {}'.format(X[0]))
            else:
                print('spectrum ends at {}'.format(X[-1]))
            return X,df_T
        else:
            X_cut = X[ai:bi+1]
            df_cut = df_T.loc[:,("%.1f" % X[ai]):("%.1f" % X[bi])]
            print('shape of cut df:',df_cut.shape)
            return X_cut,df_cut
    else:
        return X,df_T

#                          ~a~a~a~a~a~



# ~s~s~s~s~s~ Using every spectrum of timescan ~s~s~s~s~s~
# Step (1)
# def CenteredS(pth,folder,h5file,sample,readoutnoiselevel,X,power,WLcenter=784.8,threshold=0.05,show=True,save=False):
#     '''
#     h5file must be ..._combined.h5
#     X must be Gen_Wn.
#     Stores every spectrum of timescan in dictionary of centered particles.
#     Scaled offNPoM laser line is subtracted from each spectrum.
#     All negative points in |X| < 20 cm-1 are set to 10**-4.
#     Spectrum is not cut.
#     Plot is on linear scale.
#     '''
#     fullname = pth + folder + h5file
#     Title = sample + ' centered particles' #h5file.split('.')[0] + ' ' + sample
#     not_centered = {}
#     centered = {}
    
#     fig,axes = plt.subplots(nrows=2, ncols=1,figsize=(6,6),sharex=True)
#     fig.suptitle(sample + ' rejecting not-centered particles removes all duds?')
#     axes[0].set_title('centered')
#     axes[1].set_title('not centered')
#     with h5py.File(fullname, 'r') as f: 
#         BG = (f['Reference and Background Spectra']['BG'][:]-readoutnoiselevel) / f['Reference and Background Spectra']['BG'].attrs['Exposure']
#         off = ((f['Reference and Background Spectra']['offNPoM'][:]-(readoutnoiselevel)) / f['Reference and Background Spectra']['offNPoM'].attrs['Exposure']) - BG
#         laser_line = CropLaserLine(off)  
#         group = f['ParticleScannerScan_0']        
#         particles = [key for key in group.keys() if 'Particle' in key]
#         particles_sorted,timestamps = SortKeys(group,particles)
#     #     print(group['Particle_0'].keys()) #'CWL.thumb_image_0', 'SERS_0', 'z_scan_0'
#         for p,particle in enumerate(particles_sorted):
#             particle_dict = {}
#             img_0 = group[particle]['CWL.thumb_image_0']
#             c_c=check_center(img_0)
#             if c_c == True:
#                 ax = axes[0]
#             else:
#                 ax = axes[1]
#             SERS = group[particle]['SERS_0'][:]
#             exposure = group[particle]['SERS_0'].attrs['Exposure']
#             if len(np.shape(SERS)) > 1:
#                 r=0
#                 for row in SERS:
#                     Y = (((row-readoutnoiselevel)/exposure/power) - BG ) #- off
#                     Ynew = SubtractLaserLine(X,Y,laser_line)
#                     Y_clnd = RemoveCosmicRays(X,Ynew,threshold=threshold,plot=False)
#                     Y_irf = IRF(X,Y_clnd,WLcenter)
#                     ax.plot(X,Y_irf)
#                     particle_dict[str(r*exposure)] = Y_irf
#                     r+=1
#             else:
#                 Y = (((SERS-readoutnoiselevel)/exposure/power) - BG )
#                 Ynew = SubtractLaserLine(X,Y,laser_line)
#                 Y_clnd = RemoveCosmicRays(X,Ynew,threshold=threshold,plot=False)
#                 ax.plot(X,Y_clnd)
#                 particle_dict[str(0*exposure)] = Y_clnd
#             if c_c:
#                 centered[particle] = particle_dict
#             else:
#                 not_centered[particle] = particle_dict
#     for ax in axes:
#         ax.set(yscale='linear',xlim=[-1300,1300]) #ylim=[0,2500],
#         ax.set_ylim(bottom = 10**-1)
#     fig.tight_layout()
#     if show == True:
#         plt.show()
#     if save == True:
#         figname = pth+folder+Title+'.png'
#         print(figname)
#         ensure_dir(figname)
#         fig.savefig(figname,format='png',bbox_inches='tight',transparent=False)
#         # plt.close()
#     return centered, not_centered

def CenteredS(pth,folder,h5file,sample,selectK,readoutnoiselevel,X,power,threshold=0.05,show=True,save=False,testing=False):
    '''
    h5file must be ..._combined.h5
    X must be Gen_Wn.
    Stores average of timescan in dictionary of centered particles.
    All negative points in |X| < 20 cm-1 are set to 10**-4.
    Spectrum is not cut.
    Plot is on linear scale.
    Modes: 'nano' or 'ave'
    '''
    fullname = pth + folder + h5file
    Title = sample + ' centered particles' #h5file.split('.')[0] + ' ' + sample
    not_centered = {}
    centered = {}
    fig,axes = plt.subplots(nrows=2, ncols=1,figsize=(6,6),sharex=True)
    fig.suptitle(sample + ' rejecting not-centered particles removes all duds?')
    axes[0].set_title('centered')
    axes[1].set_title('not centered')
    with h5py.File(fullname, 'r') as f: 
        BG = (f['Reference and Background Spectra']['BG'][:]-readoutnoiselevel) / f['Reference and Background Spectra']['BG'].attrs['Exposure']
        group = f['ParticleScannerScan_0']
        for p,particle in enumerate(selectK):
            particle_dict = {}
            SERS = group[particle]['SERS_0'][:]
            exposure = group[particle]['SERS_0'].attrs['Exposure']
            if len(np.shape(SERS)) > 1:
                r=0
                for row in SERS:
                    Y = (((row-readoutnoiselevel)/exposure) - BG ) / power
                    Y_clnd = RemoveCosmicRays(X,Y,threshold=threshold,plot=False,testing=testing)
                    Y_irf = IRF(X,Y_clnd,WLcenter)
                    particle_dict[str(r*exposure)] = Y_irf
                    r+=1
            img_0 = group[particle]['CWL.thumb_image_0']
            c_c=check_center(img_0)
            if c_c == True:
                centered[particle] = particle_dict
                for time in particle_dict.keys():
                    axes[0].plot(X,particle_dict[time])   
            else:
                not_centered[particle] = particle_dict
                for time in X,particle_dict.keys():
                    axes[1].plot(X,X,particle_dict[time])
    for ax in axes:
        ax.set(yscale='log',xlim=[-1300,1300]) 
        ax.set_ylim(bottom = 10**-1)
    fig.tight_layout()
    if show == True:
        plt.show()
    if save == True:
        figname = pth+folder+Title+'.png'
        print(figname)
        ensure_dir(figname)
        fig.savefig(figname,format='png',bbox_inches='tight',transparent=False)
    return centered, not_centered


def SubtractLaserLine(X,Y,laser_line):
    # - - - get scaled laser line - - - 
    # <> add functionality: if max is left or right of 0 cm-1
    # want to actually use side with lower pseudopeak as max?
    ci = FindIndexX(0,X)
    di = FindIndexX(-50,X) # in case there are strong cosmic rays
    mx_ave = max(Y[di:ci])
    laser_line_scaled = NormalizeTo1(laser_line) * mx_ave
    # - - - - - - - - - - - - - - - - -
    # - - - subtract laser line - - - 
    # <> remove areas below 0 ?
    Ynew = Y - laser_line_scaled
    Ynew[(Ynew <= 0.) & (np.abs(X) < 20.)] = 10**-4
    return Ynew


# Step (2)
def HighI_S(X,sample_dict,percent):
    '''Input centered'''
    ai = FindIndexX(100,X)
    bi = FindIndexX(500,X)
    Averages = []
    K = []
    for particle in sample_dict.keys():
        for time in sample_dict[particle].keys():
            Y = sample_dict[particle][time]
            Y_cut = Y[ai:bi]
            Averages.append(np.mean(Y_cut))
            K.append(particle + '_' + time)
    decoratedK = list(zip(Averages,K))
    sorteddecoratedK = sorted(decoratedK)
    sortedK = [tpl[1] for tpl in sorteddecoratedK]
    percentile = int(percent * len(Averages))
    highIkeys = sortedK[percentile:]
    highI = {}
    for particle in sample_dict.keys():
        highI[particle] = {}
        for time in sample_dict[particle].keys():
            new_key = particle + '_' + time
            if new_key in highIkeys:
                highI[particle][time] = sample_dict[particle][time]
    return highI

def LowI_S(X,sample_dict,percent):
    '''Input centered'''
    ai = FindIndexX(100,X)
    bi = FindIndexX(500,X)
    Averages = []
    K = []
    for particle in sample_dict.keys():
        for time in sample_dict[particle].keys():
            Y = sample_dict[particle][time]
            Y_cut = Y[ai:bi]
            Averages.append(np.mean(Y_cut))
            K.append(particle + '_' + time)
    decoratedK = list(zip(Averages,K))
    sorteddecoratedK = sorted(decoratedK)
    # print(sorteddecoratedK)
    sortedK = [tpl[1] for tpl in sorteddecoratedK]
    percentile = int(percent * len(Averages))
    lowIkeys = sortedK[:percentile]
    lowI = {}
    for particle in sample_dict.keys():
        lowI[particle] = {}
        for time in sample_dict[particle].keys():
            new_key = particle + '_' + time
            if new_key in lowIkeys:
                lowI[particle][time] = sample_dict[particle][time]
    return lowI

# Step (3)
def MakeMatrixS(X,sample_dict,xlim=[-1300,1300],normalize=False,smooth=False):
    '''
    * Don't use original sample_dict; use dict of centered particles or highI
    Returns matrix with ordered rows of average SERS of each timescan, 
    cut to specified region,
    as well as cut X
    '''
    x1 = xlim[0]
    x2 = xlim[1]
    ai = FindIndexX(x1,X)
    bi = FindIndexX(x2,X)
    X_cut = X[ai:bi]
    particles_sorted = SortParticles(list(sample_dict.keys()))
    M = []
    for p,particle in enumerate(particles_sorted):
        #particle_dict = sample_dict[particle]
        for key in sample_dict[particle].keys():
            Y = sample_dict[particle][key]
            if smooth:
                Y_smthd = ReduceNoise(X[ai:bi],Y[ai:bi],cutoff = 5000) #Smooth(Y[ai:bi],window_length,polyorder)
            else:
                Y_smthd = Y[ai:bi]
            if normalize == False:
                M.append(Y_smthd)
            else:
                region=[-100,100]
                M.append(NormyNormSpecial(X[ai:bi],Y_smthd,region))
    M = np.array(M)
    if np.shape(M)[1] != len(X_cut):
        print('Error! select different xlim to ensure matrix height is length of X_cut.')
    return X_cut,M


# Step (x)
def MakeSERSDFS(X,sample_dict,xlim=None):
    '''Sample dict should not be all particles; should be centered particles only. 
    The spectra are cut to a subregion defined by xlim.
    returns df where indices are particles and columns are wavenumbers.
    '''
    new_dict = {}
    for particle in sample_dict.keys():
        for time in sample_dict[particle].keys():
            new_key = particle + '_' + time
            new_dict[new_key] = sample_dict[particle][time]
    df_sers = pd.DataFrame(new_dict)
    # Check if Nans in df
    if df_sers[df_sers.isna().any(axis=1)].shape[0] > 0:
        print(df_sers[df_sers.isna().any(axis=1)].index, 'contain Nan')
    X_str = ["%.1f" % x for x in X]
    I = list(range(len(X_str)))
    cols = dict(zip(I,X_str))
    df_T = df_sers.rename(index=cols).T
    if xlim is not None:
        x1 = xlim[0]
        x2 = xlim[1]
        ai = FindIndexX(x1,X)
        bi = FindIndexX(x2,X)
        if np.isnan(ai) or np.isnan(bi):
            print('Could not cut matrix to desired xlims, because')
            print('ai:{},bi:{}'.format(ai,bi))
            if ai is None:
                print('spectrum starts at {}'.format(X[0]))
            else:
                print('spectrum ends at {}'.format(X[-1]))
            return X,df_T
        else:
            X_cut = X[ai:bi+1]
            df_cut = df_T.loc[:,("%.1f" % X[ai]):("%.1f" % X[bi])]
            print('shape of cut df:',df_cut.shape)
            return X_cut,df_cut
    else:
        return X,df_T
#                               ~s~s~s~s~s~







def AveSTD(M):
    STD = []
    STDnorm = []
    AVE = []
    MED = []
    for n in range(np.shape(M)[1]):
        col = M[:,n]
        std = np.std(col)
        STD.append(std)
        ave = np.mean(col)
        AVE.append(ave)
        median = np.median(col)
        MED.append(median)
        stdnorm = std / ave
        STDnorm.append(stdnorm)
    STD = np.array(STD)
    STDnorm = np.array(STDnorm)
    AVE = np.array(AVE)
    MED = np.array(MED)
    return STD,STDnorm,AVE,MED


def PlotAveSTD(X_cut,M,AVE,MED,STD,xlim,ylim,pth,folder,sample,color,title,show=True,save=False):
    fig,ax = plt.subplots(nrows=1, ncols=1,figsize=(6.8*cm,3.4*cm))
    ax.set_title(sample)
    ax.plot(X_cut,AVE,color='k',label='average')
    ax.plot(X_cut,MED,color=color,label='median') # the value separating the higher half from the lower half
    mx = np.max(M,axis=0)  # axis 0 is column
    mn = np.min(M,axis=0)
    ax.fill_between(X_cut,mn,mx,color=color,alpha=0.1,label='mn,mx')
    num = 1
    sigma_up = num*(AVE + STD)
    sigma_down = num*(AVE - STD)
    #         sigma_down_mskd = np.ma.masked_where(sigma_down < 0.,sigma_down)
    ax.fill_between(X_cut,sigma_down,sigma_up,color=color,alpha=0.3,label=str(num)+'*std')
    ax.set(xlim=[xlim[0]-5,xlim[1]+5],ylim=ylim,yscale='log',
           xlabel='Raman Shift (cm$^{-1}$)'#,ylabel='Raman Intensity'
           ) 
    ax.xaxis.set_major_locator(MultipleLocator(200))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(MultipleLocator(100))
    # ax.set_yticks([0.1,ylim[1]])
    # ystr = MakeExponentialStr(ylim[1]) 
    # ax.set_yticklabels(['10$^{-1}$',ystr])#,rotation=90)
    ax.set_yticks([ylim[0],(ylim[1]*0.1)])
    # ax.ticklabel_format(axis='y',style='sci')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='both', which='major', pad=2)
    #     ax.set(ylabel='Manhattan Distance',xlabel='Spectral Region')# (cm$^{-1}$)')
    ax.xaxis.labelpad = 0.5
    ax.yaxis.labelpad = 0.5
    if show == True:
        fig.show()
    if save == True:
        figname = pth+folder+sample+title+'.png'
        print(figname)
        ensure_dir(figname)
        fig.savefig(figname,format='png',bbox_inches='tight',transparent=False)
        # plt.close()
        
        
        
def PlotAveSTDwSelectK(X_cut,M,AVE,MED,STD,Kcut,xlim,ylim,pth,folder,sample,color,title,show=True,save=False):
    fig,ax = plt.subplots(nrows=1, ncols=1,figsize=(6.8*cm,3.4*cm))
    ax.set_title(sample)
    ax.plot(X_cut,AVE,color='k',label='average')
    ax.plot(X_cut,MED,color=color,label='median') # the value separating the higher half from the lower half
    mx = np.max(M,axis=0)  # axis 0 is column
    mn = np.min(M,axis=0)
    ax.fill_between(X_cut,mn,mx,color=color,alpha=0.1,label='mn,mx')
    num = 1
    sigma_up = num*(AVE + STD)
    sigma_down = num*(AVE - STD)
    ax.fill_between(X_cut,sigma_down,sigma_up,color=color,alpha=0.3,label=str(num)+'*std')
    ax.set(xlim=[xlim[0]-5,xlim[1]+5],ylim=ylim,yscale='log',
           xlabel='Raman Shift (cm$^{-1}$)'#,ylabel='Raman Intensity'
           ) 
    colors = CreateColorsList(len(Kcut.keys()))
    for p,particle in enumerate(Kcut.keys()):
        ax.plot(X_cut,Kcut[particle],color=colors[p])
    ax.xaxis.set_major_locator(MultipleLocator(200))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(MultipleLocator(100))
    ax.set_yticks([ylim[0],(ylim[1]*0.1)])
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='both', which='major', pad=2)
    ax.xaxis.labelpad = 0.5
    ax.yaxis.labelpad = 0.5
    if show == True:
        fig.show()
    if save == True:
        figname = pth+folder+sample+title+'.png'
        print(figname)
        ensure_dir(figname)
        fig.savefig(figname,format='png',bbox_inches='tight',transparent=False)
        
# ~ m ~ m ~ m ~ m ~ m ~ Mahalanobis distance ~ m ~ m ~ m ~ m ~ m ~
# Developed in 2023-03-24 jupyter notebook
def RemoveOutliers(pth,folder,sample,X,sample_dict,xlim=[20,1200],mode='all spectra',n_components=3,percentage=0.75,normalize=False,smooth=False,plot=True): #,normalize=True,smooth=True):
    '''
    sample_dict should be highI
    xlim is for selecting sub-region of spectra to use in MakeMatrix and MakeSERSDF
    mode is 'average' or 'all spectra'
    n_components is for PCA and Mahalanobis
    normalize and smooth are for MakeMatrix
    '''
    if mode == 'average':
        # Average
        X_cut,M = MakeMatrix(X,sample_dict,xlim,normalize=normalize,smooth=smooth)
    elif mode == 'all spectra':
        # All spectra
        X_cut,M = MakeMatrixS(X,sample_dict,xlim,normalize=normalize,smooth=smooth) 
    else:
        print('Mode must be average or all spectra')
    Xt2 = PCA(M,n_components)
    m = Outliers(pth,folder,sample, Xt2, n_components, plot=plot) # m = Mahalanobis distance 
    sortedK = SortByMahalanobis(X,sample_dict,m,mode,xlim=[20,30]) # xlim doesnt matter here?
    selectK = SelectSortedK(sortedK,percentage=percentage) # keys: Particle_num_time
    # representative spectra
    if plot:
        fig,ax = plt.subplots(nrows=1, ncols=1,figsize=(3,3))
        fig.suptitle(sample+'_M')
        for row in M:
            ax.plot(X_cut,row)
        ax.set_ylim(bottom=0.)   
        Initialize(sample,M)
        PlotPCs(sample,X_cut,M,n_components)
        PlotRepresentativeSpectra(X,sample_dict,selectK,mode)
    return selectK

def PCA(M,n_components):
    # n_components = 3
    skpca2 = sk_pca(n_components=n_components)#,whiten=True)    
    # Transform on the scaled features  
    Xt2 = skpca2.fit_transform(M) 
    # print('shape feat',np.shape(M))
    # print('shape Xt2',np.shape(Xt2))    
    return Xt2

def Initialize(sample,M):
    # Initialise  
    skpca1 = sk_pca(n_components=10)  
    # Fit the spectral data and extract the explained variance ratio  
    X1 = skpca1.fit(M) #nfeat1)  
    expl_var_1 = X1.explained_variance_ratio_   
    # Plot data  with 
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(3,3))      #(ax1, ax2)
    fig.set_tight_layout(True)    
    fig.suptitle(sample+'_Initialization')
    ax1.plot(expl_var_1,'-o', label="Explained Variance %")      
    ax1.plot(np.cumsum(expl_var_1),'-o', label = 'Cumulative variance %')      
    ax1.set_xlabel("PC number")      
    ax1.set_title(sample + ' SERS data')  
    ax1.set_ylim([0,1.])      
    plt.legend(loc = 'best', fontsize = 7)    

def PlotPCs(sample,X_cut,M,n_components):
    skpca2 = sk_pca(n_components=n_components)#,whiten=True)    
    Xt2 = skpca2.fit_transform(M) # Transform on the scaled features  
    # Plot PCs
    fig,axes = plt.subplots(nrows=2, ncols=1,figsize=(3,5))
    fig.suptitle(sample+'_PCA')
    expl_var_1 = skpca2.explained_variance_ratio_
    axes[0].plot(expl_var_1,'-o', label="Explained Variance %")      
    axes[0].plot(np.cumsum(expl_var_1),'-o', label = 'Cumulative variance %') 
    n=0
    colors = CreateColorsList(n_components,colormap='Set1')
    for comp in skpca2.components_[:n_components]:
        axes[1].plot(X_cut,np.abs(comp),label=str(n+1),color=colors[n])
        if n == 0:
            comp1 = comp
        n+=1
    axes[0].set(title='variance per pc',xlim=[0,n_components])
    axes[0].legend(loc='best',fontsize=7)
    axes[1].set(title=str(n_components) + ' Principal Components (abs)')
    axes[1].set_xlim(left=0.)
    axes[1].legend(loc='best',fontsize=7)   


def Outliers(pth,folder,sample, Xt2, n_components, plot = True):
    # fit a Minimum Covariance Determinant (MCD) robust estimator to data   
    robust_cov = MinCovDet().fit(Xt2[:,:n_components])    
    # Get the Mahalanobis distance  
    m = robust_cov.mahalanobis(Xt2[:,:n_components])
    colors1 = [plt.cm.jet(float(i)/max(m)) for i in m]  
    if plot:
        Title = sample + ' PCA'#' -- 1st derivative'
        fig,ax = plt.subplots(nrows=1, ncols=1,figsize=(3,3))
        # Scatter plot            
        xi = [Xt2[j,0] for j in range(len(Xt2[:,0]))]          
        yi = [Xt2[j,1] for j in range(len(Xt2[:,1]))]          
        mx_x = max(xi)
        mx_y = max(yi)
        if mx_x > mx_y:
            xlim = [-mx_x-2,mx_x+2]
            ylim = xlim
        else:
            ylim = [-mx_y-2,mx_y+2]
            xlim = ylim
        ax.scatter(xi, yi, color=colors1, s=60, edgecolors='k')#,label=str(u))        
        ax.set(title=Title,xlabel='PC1',ylabel='PC2',
              )            
    return m


def SortByMahalanobis(X,sample_dict,m,mode,xlim=[-1300,1300]):
    if mode == 'average':
        X_cut,df_T = MakeSERSDF(X,sample_dict,xlim=xlim)        
    else:
        new_dict = {}
        for particle in sample_dict.keys():
            for time in sample_dict[particle].keys():
                new_key = particle + '_' + time
                new_dict[new_key] = sample_dict[particle][time]
        # df_sers = pd.DataFrame(new_dict)
        X_cut,df_T = MakeSERSDFS(X,sample_dict,xlim=xlim)        
    decoratedK = list(zip(m,list(df_T.index)))
    sorteddecoratedK = sorted(decoratedK)
    sortedK = [tpl[1] for tpl in sorteddecoratedK]
    return sortedK

def SelectSortedK(sortedK,percentage):
    '''percentage=0.75'''
    num = int(len(sortedK) * percentage)
    selectK = sortedK[:num]
    return selectK
    
def PlotRepresentativeSpectra(X,sample_dict,selectK,mode):
    if mode == 'average':
        # Averages
        fig,axes = plt.subplots(nrows=3, ncols=1,figsize=(7,7),sharex=True)
        i=0
        o=0
        total = np.zeros(len(X))
        for particle in sample_dict.keys():
            Y = sample_dict[particle]
            if particle in selectK:
                axes[0].plot(X,Y)
                i+=1
                total += Y                 
            else:
                axes[1].plot(X,Y)
                o+=1
            axes[2].plot(X,Y,color='b',alpha=0.1)
        ave = total / i
        axes[2].plot(X,ave,color='r')
        
        for ax in axes:
            ax.set(xlim=[-1300,1300],ylim=[0.1,10**5],yscale='log')
        axes[0].set(title='representative')
        axes[1].set(title='outliers')
        axes[2].set(title='average representative spectrum')
        print('num representative spectra = {} = {}'.format(i,len(selectK)))
    else:
        # All spectra
        fig,axes = plt.subplots(nrows=3, ncols=1,figsize=(7,7),sharex=True)
        i=0
        o=0
        total = np.zeros(len(X))
        for particle in sample_dict.keys():
            for time in sample_dict[particle].keys():
                Y = sample_dict[particle][time]
                new_key = particle + '_' + time
                if new_key in selectK:
                    axes[0].plot(X,Y)
                    i+=1
                    total += Y             
                else:
                    axes[1].plot(X,Y)
                    o+=1
                axes[2].plot(X,Y,color='b',alpha=0.1)
        ave = total / i
        axes[2].plot(X,ave,color='r')
        for ax in axes:
            ax.set(xlim=[-1300,1300],ylim=[0.1,10**5],yscale='log')
        axes[0].set(title='representative')
        axes[1].set(title='outliers')
        axes[2].set(title='average representative spectrum')
        print('num representative spectra = {} = {}'.format(i,len(selectK)))

        
def Centermost(num,selectK,X,sample_dict,plot_params,mode,plot=True):
    superselectK = selectK[:num]
    if plot:
        fig,ax = plt.subplots(nrows=1, ncols=1,figsize=(5,3),sharex=True)    
        colors = CreateColorsList(len(superselectK))
        if mode == 'average':
            total = np.zeros(len(X))
            for c,particle in enumerate(superselectK):
                Y = sample_dict[particle]
                if c < 3:
                    label = particle
                    zorder=3
                    lw=1
                    alpha=0.8
                else:
                    label = ''
                    zorder=1
                    lw=1
                    alpha=0.5
                ax.plot(X,Y,color=colors[c],alpha=alpha,label=label,zorder=zorder,lw=lw)
                total += Y
            ave2 = total / num
            ax.plot(X,ave2,color='k',lw=2,label='average of centralmost {} spectra'.format(num))
        else:
            # All spectra
            total = np.zeros(len(X))
            for c,new_key in enumerate(superselectK):
                particle = '_'.join(new_key.split('_')[:2])
                time = new_key.split('_')[2]
                Y = sample_dict[particle][time]
                ax.plot(X,Y,color=colors[c],alpha=0.5,label=new_key)
                total += Y
            ave2 = total / num
            ax.plot(X,ave2,color='k',lw=2,label='average of centralmost {} spectra'.format(num))
        ax.set(xlim=plot_params[0],ylim=plot_params[1],yscale=plot_params[2],
               xlabel='Raman Shift (cm$^{-1}$',ylabel='Raman Intensity (counts/mW/s)')
        #ax.legend(loc='best',fontsize=9)            
    return superselectK
#                  ~ m ~ m ~ m ~ m ~ m ~



# DF Analysis
# from basics_Python3 import WNtoWL,Lam
from lmfit.models import GaussianModel#,SkewedGaussianModel
# from lmfit import Parameters
from lmfit import Parameter

WLcenter = 784.8 # nm

def FWHMtoSigma(fwhm):
    sigma = fwhm / 2.3548 
    return sigma


def FitGauss(X,Y):
    model = GaussianModel()
    # pars = Parameters()
#     pars.add('slope',value=0.,vary=False)
#     pars.add('intercept',value=0.,vary=True,min=0.,max=mn)
#     'amplitude', 'center', 'sigma'
    sigma = FWHMtoSigma(300.)
    amp = max(Y) * sigma * np.sqrt(2*np.pi)
    mu0 = X[np.argmax(Y)]
#     pars = model.make_params(amplitude=amp,center=X[np.argmax(Y)], sigma=sigma)#, gamma=0)
#     print('mu0',X[np.argmax(Y)])
#     #{'value': 1, 'min': 0}
#     result = model.fit(Y,pars,x=X)
    
    result = model.fit(Y,x=X,amplitude=Parameter('amplitude',value=amp,min=0),#(amp-1),max=(amp+1)),#,min=(amp*0.5),max=(amp*1.5)),
                             center=Parameter('center',value=mu0,min=(mu0-100),max=(mu0+100)),
                             sigma=Parameter('sigma',value=sigma,min=FWHMtoSigma(25),max=FWHMtoSigma(900))
                      )
    mu = result.params['center'].value
#     print('mufit',mu)
    
    print('amp0',amp,'ampfit',result.params['amplitude'])
    print('mu0',mu0,'mufit',result.params['center'])
    print('sigma0',sigma,'sigmafit',result.params['sigma'])

    Gen_X = np.linspace(X[0],X[-1],100)
    peak_eval = result.eval(x=Gen_X)
    return mu,Gen_X,peak_eval





