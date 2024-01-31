# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 12:55:28 2022

@author: aboeh

"""
import importlib,sys
importlib.reload(sys.modules['basics_forGithub'])
from ERSremoval_forGithub import GetSidesFromWidth #GetHalfWidths
from basics_forGithub import FindIndexX,RemoveNotch,NormalizeTo1,ensure_dir

import h5py
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                                AutoMinorLocator)
import numpy as np
from scipy.signal import find_peaks,savgol_filter
from lmfit.models import LinearModel,GaussianModel,LorentzianModel
from lmfit import Parameters#, fit_report

plt.rcParams["font.family"] = 'Times New Roman'#'Calibri'
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({
    "text.usetex": False,
})
cm = 1/2.54  # centimeters in inches



def GetDFspectra(pth,h5file,Gen_Wn,readout_noise_level,testing=False):
    '''h5file should include folder'''
    DF_dict = {}
    fullname = pth + h5file
    with h5py.File(fullname, 'r') as f: 
        RefData = f['Reference and Background Spectra']
        # BG
        BG_raw = np.array(RefData['BG'][:])  
        BG_exposure = RefData['BG'].attrs['Exposure']
        BG = (BG_raw - readout_noise_level) / BG_exposure
        # White light
        Ref_raw = np.array(RefData['Ref'][:]) #op.Smooth(RefData['Reference'][:],window_length,polyorder)
        Ref_exposure = RefData['Ref'].attrs['Exposure']
        Ref =  ((Ref_raw - readout_noise_level) / Ref_exposure) - BG
        particles = [key for key in f['ParticleScannerScan_0'].keys() if 'Particle' in key]
        p=0
        for particle in particles:
            spe = f['ParticleScannerScan_0'][particle]['z_scan_0'][:]
            s = np.max(spe,axis=0)
            s_corrected = RemoveNotch(Gen_Wn,s)
            Ref_corrected = RemoveNotch(Gen_Wn,Ref)
            s_norm = (s_corrected-readout_noise_level-BG)/Ref_corrected
            DF_dict[particle] = s_norm
            p+=1
            if testing == True:
                if 15 < p < 20:
                    fig,axes = plt.subplots(nrows=2, ncols=1,figsize=(7,7))
                    axes[0].plot(Gen_Wn,s_corrected)
                    axes[0].plot(Gen_Wn,Ref_corrected,color='g')
                    axes[1].plot(Gen_Wn,s_norm)
                    mx_x = Gen_Wn[np.argmax(s_norm)]
                    mx_y = max(s_norm)
                    axes[1].scatter(mx_x,mx_y,marker='X',s=50)
                    for ax in axes:
                        ax.set(title=particle,xlim=[-1550,1600])
                    plt.show()
    return DF_dict
                    

def GetDF_maxpeak(pth,h5file,Gen_Wn,readout_noise_level,testing=False):
    '''h5file should include folder'''
    DF_dict = {}
    fullname = pth  + h5file
    with h5py.File(fullname, 'r') as f: 
        RefData = f['Reference and Background Spectra']
        # BG
        BG_raw = np.array(RefData['BG'][:])  
        BG_exposure = RefData['BG'].attrs['Exposure']
        BG = (BG_raw - readout_noise_level) / BG_exposure
        # White light
        Ref_raw = np.array(RefData['Ref'][:]) #op.Smooth(RefData['Reference'][:],window_length,polyorder)
        Ref_exposure = RefData['Ref'].attrs['Exposure']
        Ref =  ((Ref_raw - readout_noise_level) / Ref_exposure) - BG
        particles = [key for key in f['ParticleScannerScan_0'].keys() if 'Particle' in key]
        p=0
        for particle in particles:
            spe = f['ParticleScannerScan_0'][particle]['z_scan_0'][:]
            s = np.max(spe,axis=0)
            s_corrected = RemoveNotch(Gen_Wn,s)
            Ref_corrected = RemoveNotch(Gen_Wn,Ref)
            s_norm = (s_corrected-readout_noise_level-BG)/Ref_corrected
            mx_x = Gen_Wn[np.argmax(s_norm)]
            mx_y = max(s_norm)
            DF_dict[particle] = (mx_x,mx_y)
            p+=1
            if testing == True:
                if 15 < p < 20:
                    fig,axes = plt.subplots(nrows=2, ncols=1,figsize=(7,7))
                    axes[0].plot(Gen_Wn,s_corrected)
                    axes[0].plot(Gen_Wn,Ref_corrected,color='g')
                    axes[1].plot(Gen_Wn,s_norm)
                    axes[1].scatter(mx_x,mx_y,marker='X',s=50)
                    for ax in axes:
                        ax.set(title=particle,xlim=[-1550,1600])
                    plt.show()
    return DF_dict


def GetDF_maxpeak_1particle(pth,h5file,Gen_Wn,select_particle,readout_noise_level,testing=False):
    '''h5file should include folder'''
    fullname = pth  + h5file
    with h5py.File(fullname, 'r') as f: 
        RefData = f['Reference and Background Spectra']
        # BG
        BG_raw = np.array(RefData['BG'][:])  
        BG_exposure = RefData['BG'].attrs['Exposure']
        BG = (BG_raw - readout_noise_level) / BG_exposure
        # White light
        Ref_raw = np.array(RefData['Ref'][:]) 
        Ref_exposure = RefData['Ref'].attrs['Exposure']
        Ref =  ((Ref_raw - readout_noise_level) / Ref_exposure) - BG
        particles = [key for key in f['ParticleScannerScan_0'].keys() if 'Particle' in key]
        for particle in particles:
            if particle == select_particle:
                spe = f['ParticleScannerScan_0'][particle]['z_scan_0'][:]
                s = np.max(spe,axis=0)
                s_corrected = RemoveNotch(Gen_Wn,s)
                Ref_corrected = RemoveNotch(Gen_Wn,Ref)
                s_norm = (s_corrected-readout_noise_level-BG)/Ref_corrected
                mx_y = max(s_norm)
                peaks, properties = find_peaks(s_norm, height=(mx_y/2), threshold=None, distance=None,#10., 
                                               prominence=None, width=30, wlen=None, rel_height=0.5, plateau_size=None) 
                if len(peaks) > 1:
                    peaks_i,peaks_x,peaks_y = PeakIndices2PeakCoords(peaks,Gen_Wn,s_norm)
                    x_peaks_decorated = [(peaks_y[i],peaks_x[i]) for i in range(len(peaks_x))]
                    x_peaks_decorated_sorted = sorted(x_peaks_decorated,reverse=True)
                    x_peaks_sorted = []
                    for tpl in x_peaks_decorated_sorted:
                        x_peaks_sorted.append(tpl[1])
                else:
                    mx_x = Gen_Wn[np.argmax(s_norm)]
                    peaks_x = [mx_x]
                    peaks_y = [mx_y]
                    x_peaks_sorted = peaks_x
                if testing:
                    fig,axes = plt.subplots(nrows=2, ncols=1,figsize=(7,7))
                    axes[0].plot(Gen_Wn,s_corrected)
                    axes[0].plot(Gen_Wn,Ref_corrected,color='g')
                    axes[1].plot(Gen_Wn,s_norm)
                    # axes[1].scatter(mx_x,mx_y,marker='X',s=50)
                    axes[1].scatter(peaks_x,peaks_y,marker='X',s=50) 
                    for ax in axes:
                        ax.set(title=particle,xlim=[-1550,1600])
                    plt.show()
                return x_peaks_sorted[:2]


def PeakIndices2PeakCoords(peaks_i,X,Y):
    peaks_x = []
    peaks_y = []
    peaks_indices = []
    for j in range(len(peaks_i)):
        x = X[peaks_i[j]]
        y = Y[peaks_i[j]]
        peaks_x.append(x)
        peaks_y.append(y)
        peaks_indices.append(peaks_i[j])
    return peaks_indices,peaks_x,peaks_y


def GetDF_spectrum_1particle(pth,h5file,Gen_Wn,select_particle,readout_noise_level,testing=False):
    '''h5file should include folder'''
    fullname = pth  + h5file
    with h5py.File(fullname, 'r') as f: 
        RefData = f['Reference and Background Spectra']
        # BG
        BG_raw = np.array(RefData['BG'][:])  
        BG_exposure = RefData['BG'].attrs['Exposure']
        BG = (BG_raw - readout_noise_level) / BG_exposure
        # White light
        Ref_raw = np.array(RefData['Ref'][:]) #op.Smooth(RefData['Reference'][:],window_length,polyorder)
        Ref_exposure = RefData['Ref'].attrs['Exposure']
        Ref =  ((Ref_raw - readout_noise_level) / Ref_exposure) - BG
        particles = [key for key in f['ParticleScannerScan_0'].keys() if 'Particle' in key]
        for particle in particles:
            if particle == select_particle:
                spe = f['ParticleScannerScan_0'][particle]['z_scan_0'][:]
                s = np.max(spe,axis=0)
                s_corrected = RemoveNotch(Gen_Wn,s)
                Ref_corrected = RemoveNotch(Gen_Wn,Ref)
                s_norm = (s_corrected-readout_noise_level-BG)/Ref_corrected
                return s_norm
            
        

def RMSE(Y,fit):
    Y_norm = NormalizeTo1(Y)
    fit_norm = NormalizeTo1(fit)
    # mean square error
    summation = 0
    for i in range(len(Y)):
        s = ((Y_norm[i] - fit_norm[i])**2)           
        summation += s
    rmse = np.sqrt((1/len(Y)) * summation) 
    return rmse

def FitGaussian(X,Y,peaks_x,peaks_y):
    mn = min(Y) + 10**-4
    model = LinearModel()#prefix='line_')
    pars = Parameters()
    pars.add('slope',value=0.,vary=False)
    pars.add('intercept',value=0.,vary=True,min=0.,max=mn)
    for i in range(len(peaks_x)):        
        prefix = 'f' + str(i+1) + '_'
        center = peaks_x[i]
        fwhm0 = 500. 
        fwhm_bound1 = 300.
        fwhm_bound2 = 1000. 
        sigma0 = fwhm0 / 2.3548
        sigma_bound1 = fwhm_bound1 / 2.3548
        sigma_bound2 = fwhm_bound2 / 2.3548
        ai = FindIndexX(center,X)
        height = peaks_y[i]
        height_bound1 = height / 10.
        height_bound2 = Y[ai] -  (Y[ai]*0.1)
        height0 = height/2.
        A0 = height0 * sigma0*np.sqrt(2*np.pi) 
        A_bound1 = height_bound1 * sigma_bound1*np.sqrt(2*np.pi) 
        A_bound2 = height_bound2 * sigma_bound2*np.sqrt(2*np.pi) 
        peak = GaussianModel(prefix=prefix) #LorentzianModel(prefix=prefix) #
        # add with tuples:           (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
        pars.add_many((prefix+'center', center, True, (center-5), (center+5), None, None),
                        (prefix+'sigma', sigma0, True, sigma_bound1, sigma_bound2, None, None),
                        (prefix+'amplitude', A0, True, A_bound1, A_bound2, None, None)
                        #(prefix+'height', height+1, True, height, height+2., None, None)
                        )
        model = model + peak
    result = model.fit(Y,pars,x=X)
    fwhms = []
    for i in range(len(peaks_x)):  
        prefix = 'f' + str(i+1) + '_'
        sigma = result.params[prefix+'sigma'].value
        fwhm = sigma * 2.3548
        fwhms.append(fwhm)  
    peaks_eval = result.eval(x=X)
    return peaks_eval,fwhms


def SortByX(X,FWHM):
    decorated = [(X[i],FWHM[i]) for i in range(len(X))]
    decorated_sorted = sorted(decorated,reverse=False)
    X_new = []
    FWHM_new = []
    for tpl in decorated_sorted:
        X_new.append(tpl[0])
        FWHM_new.append(tpl[1])
    return X_new,FWHM_new


def GetDF_peakmaxfwhm_1particle(pth,h5file,Gen_Wn,select_particle,readout_noise_level,thirdround=True,testing=False):
    fullname = pth  + h5file
    with h5py.File(fullname, 'r') as f: 
        RefData = f['Reference and Background Spectra']
        # BG
        BG_raw = np.array(RefData['BG'][:])  
        BG_exposure = RefData['BG'].attrs['Exposure']
        BG = (BG_raw - readout_noise_level) / BG_exposure
        # White light
        Ref_raw = np.array(RefData['Ref'][:]) #op.Smooth(RefData['Reference'][:],window_length,polyorder)
        Ref_exposure = RefData['Ref'].attrs['Exposure']
        Ref =  ((Ref_raw - readout_noise_level) / Ref_exposure) - BG
        particles = [key for key in f['ParticleScannerScan_0'].keys() if 'Particle' in key]
        for particle in particles:
            if particle == select_particle:
                if testing == True:
                    print(particle)
                spe = f['ParticleScannerScan_0'][particle]['z_scan_0'][:]
                s = np.max(spe,axis=0)
                s_corrected = RemoveNotch(Gen_Wn,s)
                Ref_corrected = RemoveNotch(Gen_Wn,Ref)
                Ref_norm = Ref_corrected
                s_norm0 = ((s_corrected-readout_noise_level-BG)/Ref_norm) 
                s_norm = NormalizeTo1(s_norm0)
                mx_y = max(s_norm)
                d = 0.5 
                deriv = savgol_filter(s_norm, 7, polyorder = 5, deriv=1, mode='nearest')
                peaks = np.where((deriv < -d) | (deriv > d))[0]
                if len(peaks) < 1:
                    peaks = [np.argmax(s_norm)]
                cutoff = 1000
                peaks_i,peaks_x,peaks_y = PeakIndices2PeakCoords(peaks,Gen_Wn,s_norm)                    
                # check regions near edges for another peak just off screen
                edge = 300 #500
                mx_y_left_edge = max(s_norm[:edge])
                mx_x_left_edge = Gen_Wn[:edge][np.argmax(s_norm[:edge])]
                mx_y_right_edge = max(s_norm[-edge:])
                mx_x_right_edge = Gen_Wn[-edge:][np.argmax(s_norm[-edge:])]
                check_left = True
                check_right = True
                if check_left == True:
                    if (len(s_norm[np.argmax(s_norm[:edge]):edge]) > 1) and (mx_y_left_edge > (max(peaks_y)/3)): #y_peaks_sorted)/3)):
                        # if derivative of curve is negative to the \right\ of max in edge region, then call that max a peak
                        #deriv_left = np.gradient(s_norm[np.argmax(s_norm[:edge]):edge])
                        deriv_left = savgol_filter(s_norm[np.argmax(s_norm[:edge]):edge], 7, polyorder = 5, deriv=1, mode='nearest')
                        deriv_color = 'red'
                        # don't repeat peaks
                        # distance is between closest x_peak and the position of the max in the left edge
                        distance = min(np.abs(np.array(peaks_x) - mx_x_left_edge))
                        if (np.sum(deriv_left) < 0) and (distance > 10):
                            # x_peaks_sorted.append(mx_x_left_edge)
                            # y_peaks_sorted.append(mx_y_left_edge)
                            peaks_x.insert(0,mx_x_left_edge)
                            peaks_y.insert(0,mx_y_left_edge)
                    else:
                        # max is at right end of left edge, so probs isn't actually a peak
                        deriv_left = np.zeros(len(s_norm[np.argmax(s_norm[:edge]):edge]))
                        deriv_color = 'grey'
                if check_right == True:                        
                    if (len(s_norm[-edge:np.argmax(s_norm[-edge:])]) > 0) and (mx_y_right_edge > (max(peaks_y)/3)):
                        # if derivative of curve is negative to the left/ of max in edge region, then call that max a peak
                        #deriv_right = np.gradient(s_norm[-edge:np.argmax(s_norm[-edge:])])
                        deriv_right = savgol_filter(s_norm[-edge:np.argmax(s_norm[-edge:])], 7, polyorder = 5, deriv=1, mode='nearest')
                        deriv_color = 'red'
                        distance = min(np.abs(np.array(peaks_x) - mx_x_right_edge))
                        if (np.sum(deriv_right) < 0) and (distance > 10):
                            peaks_x.append(mx_x_right_edge)
                            peaks_y.append(mx_y_right_edge)     
                    else:
                        # max is at left end of right edge region, so probs isn't actually a peak
                        deriv_right = np.zeros(len(s_norm[-edge:np.argmax(s_norm[-edge:])]))
                        deriv_color = 'grey'
                # check that there are some peaks
                if len(peaks_x) < 1:
                    print('found 0 peaks in DF spectrum')
                try:
                    peaks_eval,fwhms = FitGaussian(Gen_Wn,s_norm,peaks_x,peaks_y)
                    rmse = RMSE(s_norm,peaks_eval)
                    if testing:
                        print('rmse:',rmse)
                    rmse_cutoff = 0. # 0.09 # 0.18
                    if rmse > rmse_cutoff: 
                        y_new = s_norm - peaks_eval 
                        peaks_new, properties = find_peaks(y_new[edge:-edge], height=0., threshold=None, distance=None,
                                                        prominence=(mx_y*(10**-7)),
                                                        width=50., 
                                                        wlen=None, rel_height=0.5, plateau_size=None) 
                        if len(peaks_new) > 0:
                            peaks_i_new,peaks_x_new,peaks_y_new = PeakIndices2PeakCoords(peaks_new,Gen_Wn[edge:-edge],y_new[edge:-edge]) 
                            # Sort from tallest to shortest
                            x_peaks_new_decorated = [(peaks_y_new[i],peaks_x_new[i]) for i in range(len(peaks_x_new))]# if peaks_x[i] > cutoff]
                            x_peaks_new_decorated_sorted = sorted(x_peaks_new_decorated,reverse=True)
                            x_peaks_new_sorted = []
                            y_peaks_new_sorted = []
                            for tpl in x_peaks_new_decorated_sorted:
                                if tpl[1] not in peaks_x:
                                    x_peaks_new_sorted.append(tpl[1])
                                    y_peaks_new_sorted.append(tpl[0])
                            if len(x_peaks_new_sorted) > 0:
                                peaks_eval_new,fwhms_new = FitGaussian(Gen_Wn,y_new,x_peaks_new_sorted,y_peaks_new_sorted)
                                peaks_eval += peaks_eval_new
                                for i in range(len(x_peaks_new_sorted)):
                                    peaks_y.append(y_peaks_new_sorted[i])
                                    peaks_x.append(x_peaks_new_sorted[i])
                                    fwhms.append(fwhms_new[i])
                except TypeError:
                    peaks_eval = np.zeros(len(Gen_Wn))
                    fwhms = []
                # Remove peaks that're close to each other
                x_peaks_decorated = [(peaks_x[i],peaks_y[i]) for i in range(len(peaks_x))]#
                x_peaks_decorated_sorted = sorted(x_peaks_decorated)
                # print('x_peaks_decorated_sorted',x_peaks_decorated_sorted)
                for i in range(len(x_peaks_decorated_sorted)):
                    if i > 0:
                        if x_peaks_decorated_sorted[i][0] < (x_peaks_decorated_sorted[i-1][0] + 50):
                            peaks_x.remove(x_peaks_decorated_sorted[i][0])
                            peaks_y.remove(x_peaks_decorated_sorted[i][1])
                # Sort from tallest to shortest
                x_peaks_decorated = [(peaks_y[i],peaks_x[i],fwhms[i]) for i in range(len(peaks_x))]# if peaks_x[i] > cutoff]
                x_peaks_decorated_sorted = sorted(x_peaks_decorated,reverse=True)
                x_peaks_sorted = []
                y_peaks_sorted = []
                fwhms_sorted = []
                for tpl in x_peaks_decorated_sorted:
                    x_peaks_sorted.append(tpl[1])
                    y_peaks_sorted.append(tpl[0])
                    fwhms_sorted.append(tpl[2])
                if len(x_peaks_sorted) < 1:
                    if testing == True:
                        print('Found no peaks, so using max of DF spectrum.')
                    mx_x = Gen_Wn[np.argmax(s_norm)]
                    mx_y = max(s_norm)
                    peaks_x = [mx_x]
                    peaks_y = [mx_y]
                    x_peaks_final_cut = peaks_x
                    y_peaks_final_cut = peaks_y
                    fwhms_final_cut = [700] #1000
                # remove points below aS cutoff
                else:
                    # Drop peaks below cutoff
                    x_peaks_final = []
                    y_peaks_final = []
                    fwhms_final = []
                    for i in range(len(x_peaks_sorted)):
                        if (-cutoff < x_peaks_sorted[i]):# and (x_peaks_sorted[i] < 1100.):
                            x_peaks_final.append(x_peaks_sorted[i])
                            y_peaks_final.append(y_peaks_sorted[i])
                            fwhms_final.append(fwhms_sorted[i])
                    if thirdround:
                        if len(x_peaks_final) < 2:
                            # print('meow meow')
                            # repeat subtracting peaks_eval
                            peaks_eval3,fwhms3 = FitGaussian(Gen_Wn,s_norm,x_peaks_final,y_peaks_final)
                            y_new = s_norm - peaks_eval3
                            mn= min(y_new)
                            if mn < 0.:
                                #shift
                                y_new2 = y_new + (np.abs(mn))
                            else:
                                y_new2 = y_new
                            peaks_new, properties = find_peaks(y_new2[edge:-edge], height=0., threshold=None, distance=None, 
                                                            prominence=(max(y_new2)/100),
                                                            width=10., #50, #30.
                                                            wlen=None, rel_height=0.5, plateau_size=None) 
                            if len(peaks_new) > 0:
                                peaks_i_new,peaks_x_new,peaks_y_new = PeakIndices2PeakCoords(peaks_new,Gen_Wn[edge:-edge],y_new[edge:-edge])
                                # Sort from tallest to shortest
                                x_peaks_new_decorated = [(peaks_y_new[i],peaks_x_new[i]) for i in range(len(peaks_x_new))]# if peaks_x[i] > cutoff]
                                x_peaks_new_decorated_sorted = sorted(x_peaks_new_decorated,reverse=True)
                                x_peaks_new_sorted = []
                                y_peaks_new_sorted = []
                                for tpl in x_peaks_new_decorated_sorted:
                                    if tpl[1] not in peaks_x:
                                        x_peaks_new_sorted.append(tpl[1])
                                        y_peaks_new_sorted.append(tpl[0])
                                peaks_eval4,fwhms4 = FitGaussian(Gen_Wn,y_new,x_peaks_new_sorted,y_peaks_new_sorted)
                                peaks_eval += peaks_eval4
                                for i in range(len(y_peaks_new_sorted)):
                                    if np.abs(x_peaks_new_sorted[i]) < cutoff:
                                        y_peaks_final.append(y_peaks_new_sorted[i])
                                        x_peaks_final.append(x_peaks_new_sorted[i])
                                        fwhms_final.append(fwhms4[i])
                                # Remove peaks that're close to each other
                                x_peaks_decorated = [(x_peaks_final[i],y_peaks_final[i]) for i in range(len(x_peaks_final))]#
                                x_peaks_decorated_sorted = sorted(x_peaks_decorated)
                                # print('x_peaks_decorated_sorted',x_peaks_decorated_sorted)
                                for i in range(len(x_peaks_decorated_sorted)):
                                    if i > 0:
                                        if x_peaks_decorated_sorted[i][0] < (x_peaks_decorated_sorted[i-1][0] + 50):
                                            x_peaks_final.remove(x_peaks_decorated_sorted[i][0])
                                            y_peaks_final.remove(x_peaks_decorated_sorted[i][1])
                    if len(x_peaks_final) > 2:         
                        # 2023-08-31
                        x_peaks_final_cut = []
                        y_peaks_final_cut = []
                        fwhms_final_cut = []
                        x_peaks_final_cut.append(x_peaks_final[0])
                        y_peaks_final_cut.append(y_peaks_final[0])
                        fwhms_final_cut.append(fwhms_final[0])
                        if x_peaks_final[1] < 0:
                            if y_peaks_final[2] > (y_peaks_final[1]-(y_peaks_final[1]*0.1)):
                                # if 3rd peak isn't much shorter than 2nd peak
                                f = 2
                            else:
                                f = 1
                        else:
                            f = 1
                        x_peaks_final_cut.append(x_peaks_final[f])
                        y_peaks_final_cut.append(y_peaks_final[f])
                        fwhms_final_cut.append(fwhms_final[f])
                    else:
                        x_peaks_final_cut = x_peaks_final
                        y_peaks_final_cut = y_peaks_final
                        fwhms_final_cut = fwhms_final
                if testing:
                    print('all DF peaks_x in GetDFpeaks',peaks_x)                        
                    print('DF peaks_x sorted in GetDFpeaks',x_peaks_sorted)
                    print('DF x_peaks_final in GetDFpeaks',x_peaks_final,'\n')
                    fig,axes = plt.subplots(nrows=2, ncols=1,figsize=(9*cm,9*cm),sharex=True) 
                    mx_s_corr = max(s_corrected)
                    axes[0].plot(Gen_Wn,s_corrected, label = 'Measured DF',color='b')
                    axes[0].plot(Gen_Wn,NormalizeTo1(Ref_corrected)*mx_s_corr,color='g',label = 'Reference (scaled)')
                    axes[1].plot(Gen_Wn,s_norm,label = 'Referenced DF',color='m')
                    axes[1].scatter(x_peaks_final,y_peaks_final,marker='X',s=50) 
                    axes[1].plot(Gen_Wn,peaks_eval,color='orange',label = 'Fit')
                    if check_left:
                        axes[1].plot(Gen_Wn[np.argmax(s_norm[:edge]):edge],deriv_left,color=deriv_color)
                    if check_right:
                        axes[1].plot(Gen_Wn[-edge:np.argmax(s_norm[-edge:])],deriv_right,color=deriv_color)
                    if rmse > rmse_cutoff:
                        axes[1].plot(Gen_Wn,y_new,color='cyan',label='Residual (shifted)')
                    axes[1].set_yticks([0,0.5,1])
                    axes[1].xaxis.set_major_locator(MultipleLocator(500))
                    axes[1].xaxis.set_major_formatter(FormatStrFormatter('%d'))
                    axes[1].xaxis.set_minor_locator(MultipleLocator(100))
                    axes[1].set(xlabel='Raman Shift (cm$^{-1}$)')
                    for ax in axes:
                        ax.set_ylim(bottom=0.)
                        ax.set(xlim=[-1300,1300])
                        #ax.set(title=particle+' (all peaks)',xlim=[-1550,1600])
                        ax.xaxis.set_ticks_position('bottom')
                        ax.yaxis.set_ticks_position('left')
                        ax.tick_params(axis='both', which='major', pad=2)
                        #axes_list[a].yaxis.labelpad = 0.1#0.5
                        ax.xaxis.labelpad = 0.1#0.5
                        ax.legend(loc='upper right',fontsize=8.,frameon=False,#alignment='right',
                                  handlelength=0.75,handletextpad=0.25
                                  )
                    fig.set_tight_layout(True) 
                    plt.show()
                return x_peaks_final_cut,fwhms_final_cut
            
            
            
# if NOT PT DATA
def GetDF_spectrum_1particle_notPT(pth,h5file,Gen_Wn,DFkey,BGkey,ref_key,readoutnoiselevel,testing=False):
    '''h5file should include folder'''
    fullname = pth  + h5file
    with h5py.File(fullname, 'r') as f: 
        group = f['AndorData']
        BG = (group[BGkey][:]-readoutnoiselevel) / group[BGkey].attrs['Exposure']
        ref = (group[ref_key][:]-readoutnoiselevel) / group[ref_key].attrs['Exposure']
        s = group[DFkey]
        s_corrected = RemoveNotch(Gen_Wn,s[:])
        Ref_corrected = RemoveNotch(Gen_Wn,ref)
        s_norm = (s_corrected-readoutnoiselevel-BG)/Ref_corrected
        return s_norm

def GetDF_peakmaxfwhm_1particle_notPT(pth,h5file,Gen_Wn,DFkey,BGkey,ref_key,readoutnoiselevel,thirdround=True,testing=False):
    fullname = pth  + h5file
    with h5py.File(fullname, 'r') as f: 
        group = f['AndorData']
        BG = (group[BGkey][:]-readoutnoiselevel) / group[BGkey].attrs['Exposure']
        ref = (group[ref_key][:]-readoutnoiselevel) / group[ref_key].attrs['Exposure']
        s = group[DFkey]
        s_corrected = RemoveNotch(Gen_Wn,s[:])
        Ref_corrected = RemoveNotch(Gen_Wn,ref)
        Ref_norm = Ref_corrected
        s_norm0 = ((s_corrected-readoutnoiselevel-BG)/Ref_norm) 
        s_norm = NormalizeTo1(s_norm0)
        mx_y = max(s_norm) 
        d = 0.5 
        deriv = savgol_filter(s_norm, 7, polyorder = 5, deriv=1, mode='nearest')
        peaks = np.where((deriv < -d) | (deriv > d))[0]
        if len(peaks) < 1:
            peaks = [np.argmax(s_norm)]
        # Cutoff: 
        # for both aS and Stokes, because only fitting two peaks and want to prioritize fitting low-v region
        cutoff = 1000
        peaks_i,peaks_x,peaks_y = PeakIndices2PeakCoords(peaks,Gen_Wn,s_norm)                    
        # check regions near edges for another peak just off screen
        edge = 300 #500
        mx_y_left_edge = max(s_norm[:edge])
        mx_x_left_edge = Gen_Wn[:edge][np.argmax(s_norm[:edge])]
        mx_y_right_edge = max(s_norm[-edge:])
        mx_x_right_edge = Gen_Wn[-edge:][np.argmax(s_norm[-edge:])]
        check_left = True
        check_right = True
        if check_left == True:
            if (len(s_norm[np.argmax(s_norm[:edge]):edge]) > 1) and (mx_y_left_edge > (max(peaks_y)/3)): #y_peaks_sorted)/3)):
                # if derivative of curve is negative to the \right\ of max in edge region, then call that max a peak
                #deriv_left = np.gradient(s_norm[np.argmax(s_norm[:edge]):edge])
                deriv_left = savgol_filter(s_norm[np.argmax(s_norm[:edge]):edge], 7, polyorder = 5, deriv=1, mode='nearest')
                deriv_color = 'red'
                # don't repeat peaks
                # distance is between closest x_peak and the position of the max in the left edge
                distance = min(np.abs(np.array(peaks_x) - mx_x_left_edge))
                if (np.sum(deriv_left) < 0) and (distance > 10):
                    peaks_x.insert(0,mx_x_left_edge)
                    peaks_y.insert(0,mx_y_left_edge)
            else:
                # max is at right end of left edge, so probs isn't actually a peak
                deriv_left = np.zeros(len(s_norm[np.argmax(s_norm[:edge]):edge]))
                deriv_color = 'grey'
        if check_right == True:                        
            if (len(s_norm[-edge:np.argmax(s_norm[-edge:])]) > 0) and (mx_y_right_edge > (max(peaks_y)/3)):
                # if derivative of curve is negative to the left/ of max in edge region, then call that max a peak
                #deriv_right = np.gradient(s_norm[-edge:np.argmax(s_norm[-edge:])])
                deriv_right = savgol_filter(s_norm[-edge:np.argmax(s_norm[-edge:])], 7, polyorder = 5, deriv=1, mode='nearest')
                deriv_color = 'red'
                distance = min(np.abs(np.array(peaks_x) - mx_x_right_edge))
                if (np.sum(deriv_right) < 0) and (distance > 10):
                    peaks_x.append(mx_x_right_edge)
                    peaks_y.append(mx_y_right_edge)     
            else:
                # max is at left end of right edge region, so probs isn't actually a peak
                deriv_right = np.zeros(len(s_norm[-edge:np.argmax(s_norm[-edge:])]))
                deriv_color = 'grey'
        if len(peaks_x) < 1:
            print('found 0 peaks in DF spectrum')
        try:
            peaks_eval,fwhms = FitGaussian(Gen_Wn,s_norm,peaks_x,peaks_y)
            rmse = RMSE(s_norm,peaks_eval)
            if testing:
                print('rmse:',rmse)
            rmse_cutoff = 0. 
            if rmse > rmse_cutoff: 
                y_new = s_norm - peaks_eval                       #75
                peaks_new, properties = find_peaks(y_new[edge:-edge], height=0., threshold=None, distance=None,#10., 
                                                prominence=(mx_y*(10**-7)), #10
                                                width=50., #50, #30.
                                                wlen=None, rel_height=0.5, plateau_size=None) 
                if len(peaks_new) > 0:
                    peaks_i_new,peaks_x_new,peaks_y_new = PeakIndices2PeakCoords(peaks_new,Gen_Wn[edge:-edge],y_new[edge:-edge]) 
                    # Sort from tallest to shortest
                    x_peaks_new_decorated = [(peaks_y_new[i],peaks_x_new[i]) for i in range(len(peaks_x_new))]# if peaks_x[i] > cutoff]
                    x_peaks_new_decorated_sorted = sorted(x_peaks_new_decorated,reverse=True)
                    x_peaks_new_sorted = []
                    y_peaks_new_sorted = []
                    for tpl in x_peaks_new_decorated_sorted:
                        if tpl[1] not in peaks_x:
                            x_peaks_new_sorted.append(tpl[1])
                            y_peaks_new_sorted.append(tpl[0])
                    if len(x_peaks_new_sorted) > 0:
                        peaks_eval_new,fwhms_new = FitGaussian(Gen_Wn,y_new,x_peaks_new_sorted,y_peaks_new_sorted)
                        peaks_eval += peaks_eval_new
                        for i in range(len(x_peaks_new_sorted)):
                            peaks_y.append(y_peaks_new_sorted[i])
                            peaks_x.append(x_peaks_new_sorted[i])
                            fwhms.append(fwhms_new[i])
        except TypeError:
            peaks_eval = np.zeros(len(Gen_Wn))
            fwhms = []
        x_peaks_decorated = [(peaks_x[i],peaks_y[i]) for i in range(len(peaks_x))]#
        x_peaks_decorated_sorted = sorted(x_peaks_decorated)
        for i in range(len(x_peaks_decorated_sorted)):
            if i > 0:
                if x_peaks_decorated_sorted[i][0] < (x_peaks_decorated_sorted[i-1][0] + 50):
                    peaks_x.remove(x_peaks_decorated_sorted[i][0])
                    peaks_y.remove(x_peaks_decorated_sorted[i][1])
        # Sort from tallest to shortest
        x_peaks_decorated = [(peaks_y[i],peaks_x[i],fwhms[i]) for i in range(len(peaks_x))]# if peaks_x[i] > cutoff]
        x_peaks_decorated_sorted = sorted(x_peaks_decorated,reverse=True)
        x_peaks_sorted = []
        y_peaks_sorted = []
        fwhms_sorted = []
        for tpl in x_peaks_decorated_sorted:
            x_peaks_sorted.append(tpl[1])
            y_peaks_sorted.append(tpl[0])
            fwhms_sorted.append(tpl[2])
        if len(x_peaks_sorted) < 1:
            if testing == True:
                print('Found no peaks, so using max of DF spectrum.')
            mx_x = Gen_Wn[np.argmax(s_norm)]
            mx_y = max(s_norm)
            peaks_x = [mx_x]
            peaks_y = [mx_y]
            x_peaks_final_cut = peaks_x
            y_peaks_final_cut = peaks_y
            fwhms_final_cut = [700] 
        else:
            # Drop peaks below cutoff
            x_peaks_final = []
            y_peaks_final = []
            fwhms_final = []
            for i in range(len(x_peaks_sorted)):
                if (-cutoff < x_peaks_sorted[i]):# and (x_peaks_sorted[i] < 1100.):
                    x_peaks_final.append(x_peaks_sorted[i])
                    y_peaks_final.append(y_peaks_sorted[i])
                    fwhms_final.append(fwhms_sorted[i])
            if thirdround:
                if len(x_peaks_final) < 2:
                    peaks_eval3,fwhms3 = FitGaussian(Gen_Wn,s_norm,x_peaks_final,y_peaks_final)
                    y_new = s_norm - peaks_eval3
                    mn= min(y_new)
                    if mn < 0.:
                        #shift
                        y_new2 = y_new + (np.abs(mn))
                    else:
                        y_new2 = y_new
                    peaks_new, properties = find_peaks(y_new2[edge:-edge], height=0., threshold=None, distance=None,#10., 
                                                    prominence=(max(y_new2)/100),#(10**-4),#mx_y*(10**-5)), #10
                                                    width=10., #50, #30.
                                                    wlen=None, rel_height=0.5, plateau_size=None) 
                    if len(peaks_new) > 0:
                        peaks_i_new,peaks_x_new,peaks_y_new = PeakIndices2PeakCoords(peaks_new,Gen_Wn[edge:-edge],y_new[edge:-edge]) 
                        
                        # Sort from tallest to shortest
                        x_peaks_new_decorated = [(peaks_y_new[i],peaks_x_new[i]) for i in range(len(peaks_x_new))]# if peaks_x[i] > cutoff]
                        x_peaks_new_decorated_sorted = sorted(x_peaks_new_decorated,reverse=True)
                        x_peaks_new_sorted = []
                        y_peaks_new_sorted = []
                        for tpl in x_peaks_new_decorated_sorted:
                            if tpl[1] not in peaks_x:
                                x_peaks_new_sorted.append(tpl[1])
                                y_peaks_new_sorted.append(tpl[0])
                        
                        peaks_eval4,fwhms4 = FitGaussian(Gen_Wn,y_new,x_peaks_new_sorted,y_peaks_new_sorted)
                        peaks_eval += peaks_eval4
                        for i in range(len(y_peaks_new_sorted)):
                            if np.abs(x_peaks_new_sorted[i]) < cutoff:
                                y_peaks_final.append(y_peaks_new_sorted[i])
                                x_peaks_final.append(x_peaks_new_sorted[i])
                                fwhms_final.append(fwhms4[i])
                        # Remove peaks that're close to each other
                        x_peaks_decorated = [(x_peaks_final[i],y_peaks_final[i]) for i in range(len(x_peaks_final))]#
                        x_peaks_decorated_sorted = sorted(x_peaks_decorated)
                        # print('x_peaks_decorated_sorted',x_peaks_decorated_sorted)
                        for i in range(len(x_peaks_decorated_sorted)):
                            if i > 0:
                                if x_peaks_decorated_sorted[i][0] < (x_peaks_decorated_sorted[i-1][0] + 50):
                                    x_peaks_final.remove(x_peaks_decorated_sorted[i][0])
                                    y_peaks_final.remove(x_peaks_decorated_sorted[i][1])
            if len(x_peaks_final) > 2:         
                # 2023-08-31
                x_peaks_final_cut = []
                y_peaks_final_cut = []
                fwhms_final_cut = []
                x_peaks_final_cut.append(x_peaks_final[0])
                y_peaks_final_cut.append(y_peaks_final[0])
                fwhms_final_cut.append(fwhms_final[0])
                if x_peaks_final[1] < 0:
                    if y_peaks_final[2] > (y_peaks_final[1]-(y_peaks_final[1]*0.1)):
                        # if 3rd peak isn't much shorter than 2nd peak
                        f = 2
                    else:
                        f = 1
                else:
                    f = 1
                x_peaks_final_cut.append(x_peaks_final[f])
                y_peaks_final_cut.append(y_peaks_final[f])
                fwhms_final_cut.append(fwhms_final[f])
            else:
                x_peaks_final_cut = x_peaks_final
                y_peaks_final_cut = y_peaks_final
                fwhms_final_cut = fwhms_final
        if testing:
            print('all DF peaks_x in GetDFpeaks',peaks_x)                        
            print('DF peaks_x sorted in GetDFpeaks',x_peaks_sorted)
            print('DF x_peaks_final in GetDFpeaks',x_peaks_final,'\n')
            fig,axes = plt.subplots(nrows=2, ncols=1,figsize=(9*cm,9*cm),sharex=True)
            mx_s_corr = max(s_corrected)
            axes[0].plot(Gen_Wn,s_corrected, label = 'Measured DF',color='b')
            axes[0].plot(Gen_Wn,NormalizeTo1(Ref_corrected)*mx_s_corr,color='g',label = 'Reference (scaled)')
            axes[1].plot(Gen_Wn,s_norm,label = 'Referenced DF',color='m')
            axes[1].scatter(x_peaks_final,y_peaks_final,marker='X',s=50) 
            axes[1].plot(Gen_Wn,peaks_eval,color='orange',label = 'Fit')
            if check_left:
                axes[1].plot(Gen_Wn[np.argmax(s_norm[:edge]):edge],deriv_left,color=deriv_color)
            if check_right:
                axes[1].plot(Gen_Wn[-edge:np.argmax(s_norm[-edge:])],deriv_right,color=deriv_color)
            if rmse > rmse_cutoff:
                axes[1].plot(Gen_Wn,y_new,color='cyan',label='Residual (shifted)') 
            axes[1].set_yticks([0,0.5,1])
            axes[1].xaxis.set_major_locator(MultipleLocator(500))
            axes[1].xaxis.set_major_formatter(FormatStrFormatter('%d'))
            axes[1].xaxis.set_minor_locator(MultipleLocator(100))
            axes[1].set(xlabel='Raman Shift (cm$^{-1}$)')
            for ax in axes:
                ax.set_ylim(bottom=0.)
                ax.set(xlim=[-1300,1300])
                #ax.set(title=particle+' (all peaks)',xlim=[-1550,1600])
                ax.xaxis.set_ticks_position('bottom')
                ax.yaxis.set_ticks_position('left')
                ax.tick_params(axis='both', which='major', pad=2)
                #axes_list[a].yaxis.labelpad = 0.1#0.5
                ax.xaxis.labelpad = 0.1#0.5
                ax.legend(loc='upper right',fontsize=8.,frameon=False,#alignment='right',
                          handlelength=0.75,handletextpad=0.25
                          )
            fig.set_tight_layout(True) 
            plt.show()
        return x_peaks_final_cut,fwhms_final_cut