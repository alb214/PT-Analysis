# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 16:02:07 2023

@author: aboeh
"""


import importlib,sys
importlib.reload(sys.modules['basics_forGithub'])
from time import process_time
# import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.signal import peak_prominences
from scipy.signal import peak_widths
# from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from lmfit.models import GaussianModel, LinearModel
from lmfit import Parameters
# from lmfit.models import LorentzianModel
# from scipy.integrate import quad
import random
import math
# from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt

import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                                AutoMinorLocator)
# from matplotlib.patches import Polygon

from basics_forGithub import FindIndexX, FindClosesetXbelow_w, Split2, Split3, butterLowpassFiltFilt, ReduceNoise, Smooth, ensure_dir
import ERSremoval_v9 as ers_v9


# laser frequency 
EXACT_WAVELENGTH = 784.8 # nm
EXACT_FREQUENCY = 3.8199*10**14 # Hz

plt.rcParams["font.family"] = 'Times New Roman'#'Calibri'
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({
    "text.usetex": False,
})

cm = 1/2.54  # centimeters in inches

params_list = ['A','T','C','slope','mu1','fwhm1','height1','mu2','fwhm2','height2']


# O o O o O o O o O o O o O o O o O o O o O o O o O o O
def Gaussian(X,mu,fwhm,height):
    sigma = fwhm / 2.3548 # fwhm / (2*np.sqrt(2*np.log(2)))
    amp = height*sigma*np.sqrt(2*np.pi)
    model = GaussianModel()
    params = model.make_params(center=mu,amplitude=amp,sigma=sigma)
    y_eval = model.eval(params, x=X)
    return y_eval


def MultiGauss(X,C,m,mus,fwhms,heights):  
    model = LinearModel()
    pars = Parameters()
    pars.add('slope',value=0.,vary=False)
    pars.add('intercept',value=C) 
    for i in range(len(mus)):        
        prefix = 'f' + str(i+1) + '_'
        center = mus[i] 
        fwhm0 = fwhms[i] 
        sigma0 = fwhm0 / 2.3548
        height0 = heights[i]
        A0 = height0 * sigma0*np.sqrt(2*np.pi)
        peak = GaussianModel(prefix=prefix)
        pars.add_many((prefix+'center', center),#, True, (center-10), (center+10), None, None),
                        (prefix+'sigma', sigma0),#, True, sigma_bound1, sigma_bound2, None, None),
                        (prefix+'amplitude', A0),#, True, A_bound1, A_bound2, None, None)
                        )
        model = model + peak
    y_eval = model.eval(pars,x=X)
    return y_eval

# O o O o O o O o O o O o O o O o O o O o O o O o O o O


def BosePopulation(WN,T):
    '''Returns a vector. Give wn in units cm-1.
    ((6.626e-34)*(2.998e8)*100)/((1.381e-23)*(293))
    = 0.00491
    '''
    # Bose
    h = 6.62607004e-34 # m**2 kg s**-1
    k = 1.38064852e-23 # m**2 kg s**-2 K**-1
    c = 299792458 # m/s
    kT = k * T
    cmperm = 100
    numerator = h*c*cmperm 
    n_bose = [( 1.0/( np.exp((numerator*np.abs(wn)) /kT) - 1) ) for wn in WN] 
    return np.array(n_bose)

def BoseGauss(WN,A,T,C,m,mus,fwhms,heights):    
    n_bose = BosePopulation(WN,T) 
    gaus = MultiGauss(WN,C,m,mus,fwhms,heights)
    ERS_model = np.zeros(len(WN))
    for i in range(len(WN)):
        if WN[i] < 0.:
            ERS_model[i] = (A * n_bose[i]) * gaus[i]#(gaus[i] + 1)
        else:
            ERS_model[i] = (A * (n_bose[i] + 1)) * gaus[i]#(gaus[i] + 1)
    return ERS_model

def OnlyBose(WN,A,T):
    n_bose = BosePopulation(WN,T) 
    ERS_model = np.zeros(len(WN))
    for i in range(len(WN)):
        if WN[i] < 0.:
            ERS_model[i] = (A * n_bose[i]) 
        else:
            ERS_model[i] = (A * (n_bose[i] + 1)) 
    return ERS_model


def StokesHeight(Stokes_WN,Stokes_bg,Stokes_smooth_params,Title,pth,folder,save=False):
    '''Finds approx height of background of Stokes, to set strict  bounds on parameter A.'''
    ci = FindClosesetXbelow_w(500,Stokes_WN)
    # print('500',Stokes_WN[ci])
    ip_bg = Smooth(Stokes_bg[ci:],Stokes_smooth_params[0],Stokes_smooth_params[1]) 
    mn = np.amin(ip_bg)
    if mn < 0:
        di = np.argmin(ip_bg)
        height = np.abs(np.mean(ip_bg[di-10:di+10]))
        label = 'abs(ave(around min of bg)): '
        unsure = True
    else:
        height = mn
        label = 'min: '
        unsure = False
    if save == True:
        fig,ax = plt.subplots(nrows=1, ncols=1,figsize=(5, 3))
        X = Stokes_WN[ci:]#di]#[ai:bi]
        ax.plot(Stokes_WN,Stokes_bg,label='Stokes spectrum')
        ax.plot(X,ip_bg,label='background fit')
        ax.plot(X,np.ones(len(X))*height,label=label+str(height))#average,label='average: '+str(average))
        ax.legend(loc='best',fontsize=8)
        mx = np.amax(Stokes_bg[ci:])#ei])
        ax.set(xlim=[0,Stokes_WN[-1]],ylim=[10**0,mx+5],yscale='log')
        # folder_ext = folder + 'StokesHeight/'
        # figname = pth+folder_ext+Title+'.png'
        plt.show()
    return height, unsure


def Version09(Gen_Wn,spectrum_shifteddown,p0):
    # - - ERSremoval_v9 for comparison
    print('Running ERSremoval version09.')
    increment = 0.9
    mn_allowed_blip = 0. #-10.
    mn = mn_allowed_blip - 1.
    num_iterations = 10
    A = p0[0]
    T = p0[1]    
    ASpopt = (0.,0.)
    AS_WN,AS_spectrum,Stokes_WN,Stokes_spectrum = Split3(Gen_Wn,spectrum_shifteddown,0.0)
    bi = FindIndexX(-21,AS_WN)    
    if np.isnan(bi):
        AS_spectrum_cut = AS_spectrum#[ci:]
        AS_WN_cut = AS_spectrum
    else:
        AS_spectrum_cut = AS_spectrum[:bi]#[ci:bi]
        AS_WN_cut = AS_WN[:bi]#[ci:bi]
    while (num_iterations > 0 and mn < mn_allowed_blip and ASpopt is not None): 
        p0 = [A,T]
        ASpopt,residual,txc =  ers_v9.FitAntiStokes(AS_WN_cut,AS_spectrum_cut,p0)
        if ASpopt is None:
            ERS_2 = None
            ERSremoved_2 = None
        else:
            #  6. Calculate AS and Stokes ERS from fit
            AS_ERS_2 = ers_v9.antiStokes(AS_WN, *ASpopt)
            Stokes_ERS_2 = ers_v9.Stokes(Stokes_WN, *ASpopt) #+ (m*Gen_Wn = b)
            ERS_2 = np.concatenate((AS_ERS_2,Stokes_ERS_2))
            ERSremoved_2 = SubtractERS(Gen_Wn,spectrum_shifteddown,ERS_2,notchedge=None)#spectrum_shifteddown - ERS_2
            blips = []
            for wn in range(int(Gen_Wn[0]),int(Gen_Wn[-1])-25,25):
                ai = FindIndexX(wn,Gen_Wn)
                bi = FindIndexX(wn+10,Gen_Wn)              
                if np.isnan(bi):
                    print(wn,'not in Gen_Wn, so using end of spectrum as end of blip')
                    blip = np.mean(ERSremoved_2[ai:])
                    print('making blip this long:',len(blip),'instead of 10 points long.')
                else:
                    blip = np.mean(ERSremoved_2[ai:bi])
                blips.append(blip)
            mn = min(blips)
        if (mn < mn_allowed_blip) or (ERSremoved_2 is None):
            A = increment*A #A - (A/10.)#5.)#10.)
        else:
            # use the previous A
            A = A / increment
            break
        num_iterations = num_iterations - 1  
    return ASpopt,p0





# /    /    /    /    /    /    /    /    /    /    /    /    /    /    /    
def FitExponential(AS_X,AS_Y,eps=0.,testing=False):
    '''inputs must be arrays'''    
    new_f = np.log(AS_Y + eps)   
    new_f_cut = np.concatenate((new_f[:5],new_f[-5:]))
    AS_X_cut = np.concatenate((AS_X[:5],AS_X[-5:]))
    bounds = ((0.,-np.inf),(np.inf,np.inf))
    fit,pcov = curve_fit(LinearLine,AS_X_cut,new_f_cut,p0=None,bounds=bounds)
    lin_fit = np.poly1d(fit)#LinearLine2(Stokes_WN,b,B)      
    b,lnB = fit #huber.coef_, huber.intercept_ #
    B = np.exp(lnB) #- eps
    if testing == True:
        if np.isnan(b) or np.isnan(B):
            print('b,B =',b,B)
            title = 'b=%e' % b + ', B=%e' % B            
        else:
            print('-b,B =',-1*b,B)
            title = '-b=%e, B=%e' % ((-1.*b), B)
        fig,axes = plt.subplots(nrows=2, ncols=1,figsize=(7,5),sharex=True)
        axes[0].scatter(AS_X,new_f,c='grey')
        axes[0].scatter(AS_X_cut,new_f_cut,c='r')
        axes[0].plot(AS_X,lin_fit(AS_X),color='r',linewidth=3,label='poly1d')
        axes[0].set(title='linear fit '+title)
        exp_fit = Exponential(AS_X,B,b) - eps
        axes[1].plot(AS_X,AS_Y)
        axes[1].plot(AS_X,exp_fit,color='r')
        axes[1].set(title='exp fit')
        axes[1].set_ylim(bottom=-2)
        plt.show()
    return b,B

def Exponential(X,B,b):
    E = np.zeros(len(X))
    for i in range(len(X)):
        E[i] = B * np.exp(-b*np.abs(X[i]))
    return E

def LinearLine(X,m,y0):
    Y = [(m*x)+y0 for x in X]
    return np.array(Y)

# /    /    /    /    /    /    /    /    /    /    /    /    /    /    /    



# ~    ~    ~    ~    ~    ~    ~    ~    ~    ~    ~    ~    ~    ~    ~    ~    

def PeakIndices2PeakCoords2(peaks_i,X,Y):
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


def PeakIndices2PeakCoords(peaks_i,X,Y):
    peaks_x = []
    peaks_y = []
    peaks_indices = []
    for j in range(len(peaks_i)):
        x = X[peaks_i[j]]
        y = Y[peaks_i[j]]
        if (x < -200 or x > 200):
            # skip center peaks
            peaks_x.append(x)
            peaks_y.append(y)
            peaks_indices.append(peaks_i[j])
    return peaks_indices,peaks_x,peaks_y

def GetProminences(peaks_i,X,Y):
    prominences = peak_prominences(Y, peaks_i)[0]
    contour_heights = Y[peaks_i] - prominences
    return contour_heights

def GetFullWidths(peaks_i,X,Y):
    results_full = peak_widths(Y, peaks_i, rel_height=1., prominence_data=None, wlen=None)
    width_heights = results_full[1] #The height of the contour lines at which the widths where evaluated.
    left_ips = np.array([int(yum) for yum in results_full[2]]) # Interpolated positions of left and right intersection points of a horizontal line at the respective evaluation height.
    right_ips = np.array([int(yum) for yum in results_full[3]])
    return width_heights,left_ips,right_ips

def GetHalfWidths(peaks_i,X,Y):
    results_half = peak_widths(Y, peaks_i, rel_height=0.5, prominence_data=None, wlen=None)
    width_heights = results_half[1] #The height of the contour lines at which the widths where evaluated.
    left_ips = np.array([int(yum) for yum in results_half[2]]) # Interpolated positions of left and right intersection points of a horizontal line at the respective evaluation height.
    right_ips = np.array([int(yum) for yum in results_half[3]])
    return width_heights,left_ips,right_ips

def GetSidesFromWidth(width,x,X):
    left = x - (width/2)
    right = x + (width/2)
    left_i = FindIndexX(left,X)
    #if left_i is None:
    if np.isnan(left_i):
        left_i = -1
    right_i = FindIndexX(right,X)
    #if right_i is None:
    if np.isnan(right_i):
        right_i = -1
    return left_i,right_i

def SpectrumWithoutPeaks(X,Y,peak_params,notchedge,testing=False):
    '''Provide spectrum after it's shifted ASto0
    peak_params = 10,10,4 # peak_params = prominence,base width,height
    Returns regions between peaks
    '''
    AS_X,AS_Y,Stokes_X,Stokes_Y = Split2(X,Y,0.)
    Stokes_filtered = butterLowpassFiltFilt(Stokes_Y, cutoff = 5000, fs = 50000, order=1) #cutoff = 6000
    # use guess of params to find peaks
    prom = peak_params[0]
    height = peak_params[2]
    peaks = []
    count = 0
    while len(peaks) < 3:
        if count < 10:
            peaks, properties = find_peaks(Stokes_filtered, height=height, threshold=None, distance=None,#10., 
                                           prominence=prom, width=peak_params[1], wlen=None, rel_height=0.5, plateau_size=None) 
            prom -= max(Stokes_filtered)/100.
            count += 1
        else:
            peaks = [np.argmax(Stokes_filtered)]
            break
    peaks_i,peaks_x,peaks_y = PeakIndices2PeakCoords(peaks,Stokes_X,Stokes_filtered)
    for j,x in enumerate(peaks_x):
        if x < 225:
            peaks_i.remove(peaks_i[j])
            peaks_x.remove(x)
            peaks_y.remove(peaks_y[j])
    if testing == True:
        print('Num peaks found:',len(peaks))
        print('Positions of peaks:',peaks_x)
    contour_heights = GetProminences(peaks_i,Stokes_X,Stokes_filtered)
    width_heights,left_ips,right_ips = GetHalfWidths(peaks_i,Stokes_X,Stokes_filtered)
    if testing == True:
        fig,ax = plt.subplots(nrows=1, ncols=1,figsize=(7,5))
        ax.plot(Stokes_X,Stokes_Y,color='k',ls=':')
        ax.plot(Stokes_X,Stokes_filtered,color='k')
        ax.scatter(peaks_x, peaks_y, c='g',marker='x',s=100) 
        ax.vlines(x=Stokes_X[peaks_i], ymin=contour_heights, ymax=Stokes_filtered[peaks_i])
        ax.hlines(width_heights,Stokes_X[left_ips],Stokes_X[right_ips],colors='b')
    Stokes_masked = np.copy(Stokes_Y)
    widths = []
    for j in range(len(peaks_i)):
        left_i = left_ips[j]
        right_i = right_ips[j]
        fwhm = Stokes_X[right_i] - Stokes_X[left_i]
        width = fwhm * 3.#5. #4. #3.                                                      # THIS IS ALSO A PARAMETER
        left_i,right_i = GetSidesFromWidth(width,peaks_x[j],Stokes_X)
        Stokes_masked[left_i:right_i] = np.ones(len(Stokes_Y[left_i:right_i]), dtype=bool) * np.mean([Stokes_Y[left_i],Stokes_Y[right_i]])
        width = (Stokes_X[right_i] - Stokes_X[left_i]) 
        widths.append(width)
    AS_masked = np.copy(AS_Y)
    for j in range(len(peaks_i)):
        x = -1*np.flip(peaks_x)[j]
        width = np.flip(widths)[j]
        left = x - (width/2)
        right = x + (width/2)
        left_i = FindIndexX(left,AS_X)
        right_i = FindIndexX(right,AS_X)
        AS_masked[left_i:right_i] = np.ones(len(AS_Y[left_i:right_i]), dtype=bool) * np.mean([AS_Y[left_i],AS_Y[right_i]])
    X_bg = X
    Y_bg = np.concatenate((AS_masked,Stokes_masked))
    mask2 = np.abs(X_bg) < notchedge #20. #33. #11. #10.
    X_bg_2 = X_bg[~mask2]
    Y_bg_2 = Y_bg[~mask2]
    if testing == True:
        ax.plot(Stokes_X,Stokes_masked,color='m')
        ax.plot(AS_X,AS_Y,color='k',ls=':')
        ax.plot(AS_X,AS_masked,color='m')
        ax.plot(X_bg_2,Y_bg_2,color='r')
        ax.xaxis.set_major_locator(MultipleLocator(500))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.xaxis.set_minor_locator(MultipleLocator(100))
        xlim = GetXLim(X)
        ax.set(title='SpectrumWoPeaks',xlim=xlim,ylim=[0.001,max(Stokes_Y)/2],yscale='log')
        plt.show()
    return X_bg_2,Y_bg_2



# ~    ~    ~    ~    ~    ~    ~    ~    ~    ~    ~    ~    ~    ~    ~    ~    
def GetMn(X,Y,blip_size=50,testing=False):
    # blip_size = 5 # even number
    halfsz = int(blip_size/2)
    mn_i = np.argmin(Y)
    if mn_i == 0:
        case = 'mn_i = {} = 0'.format(mn_i)
        blip = Y[:(mn_i+blip_size)]
    elif mn_i <= halfsz:
        case = 'mn_i = {} < halfblipsz = {}'.format(mn_i,halfsz)
        blip = Y[:(mn_i+halfsz+np.abs(halfsz-mn_i))]
    elif len(Y) <= (mn_i+halfsz):
        case = 'len(Y) = {} < halfblipsz = {}'.format(len(Y),mn_i+halfsz)
        blip = Y[(mn_i-(halfsz+np.abs(halfsz-mn_i))):]
    else:
        case = 'normal case'
        blip = Y[(mn_i-halfsz):(mn_i+halfsz)]
    mn = np.mean(blip)
    if testing == True:
        print(case)
        print('mn x,y',X[mn_i],Y[mn_i])
        print('blip mn',mn)
    return mn



def PreProcessing(Gen_Wn,Y_orig,region,peak_params,T0,notchedge,fit_gauss,pth,folder,Title,DF_peaks_x,DF_fwhms,removepeaks=True,sv2=False,replace=True,testing=False):
    '''region is for shifting to 0 and for replacing w exponential fit
    region needs to start vvvv close to far left of AS region
    
    '''
    VHGnotchedge = 9.
    mask = np.abs(Gen_Wn) < VHGnotchedge 
    Y_shifted2 = np.ma.masked_array(Y_orig,mask=mask) 
    Y_bg_smthd = ReduceNoise(Gen_Wn,Y_orig,VHGnotchedge,cutoff = 5000)
    if removepeaks:
        # 1. Make Y_bg by removing peaks
        # Fit spectrum without peaks
        X_bg,Y_bg = SpectrumWithoutPeaks(Gen_Wn,Y_bg_smthd,peak_params,notchedge,testing=testing) 
    else:
        mask = np.abs(Gen_Wn) < notchedge #10. 
        X_bg = Gen_Wn[~mask]
        Y_bg = Y_bg_smthd[~mask]
    xlim = [Gen_Wn[0]-10,Gen_Wn[-1]+10]
    Y_to0_2 = Y_bg
    if testing:
        fig,ax = plt.subplots(nrows=1, ncols=1,figsize=(7,5))
        ax.plot(Gen_Wn,Y_orig,color='grey',label='Y_orig')
        ax.plot(Gen_Wn,Y_shifted2,color='k',label='final Y')
        ax.plot(X_bg,Y_to0_2,color='m',label='final Y_bg')
        ax.legend(loc='best',fontsize=9)
        ax.xaxis.set_major_locator(MultipleLocator(500))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.xaxis.set_minor_locator(MultipleLocator(100))
        # axes[1].xaxis.set_minor_formatter(FormatStrFormatter('%d'))
        ax.grid(b=True, which='minor', axis='x', color='gainsboro', linestyle='-')
        ax.set(title='Pre-Processing',xlim=xlim,yscale='log')
        plt.show()
    Stokes_smooth_params  = [101,3]#[301,3] # window_length,polyorder
    zi = FindClosesetXbelow_w(notchedge,X_bg)
    Y_bg_cut = Y_to0_2[zi:]
    A0,unsure = StokesHeight(X_bg[zi:],Y_bg_cut,Stokes_smooth_params,Title,pth,folder,save=sv2)
    mn_i = np.argmin(Y_bg_cut)
    if mn_i == 0:
        blip = Y_bg_cut[:(mn_i+10)]
    elif mn_i == (len(Y_bg_cut)-1):
        blip = Y_bg_cut[(mn_i-10):]
    else:
        blip = Y_bg_cut[(mn_i-5):(mn_i+5)]
    standard_deviation = np.std(blip)
    # factors = [0.05] #[0.4]
    A1_condition = A0 - (standard_deviation * 3.)
    A2_condition = A0 + (standard_deviation * 6)
    if unsure == False:
        if A1_condition < 1.:
            Abound1 = 1.                                                     
        else:

            Abound1 = A1_condition
        Abound2 = A2_condition 
        if testing == True:
            print('std:',standard_deviation)
    else:
        if testing == True:
            print('Unsure about Stokes height.')
        if A1_condition < 1.:
            Abound1 = 1. 
        else:
            Abound1 = A1_condition
        Abound2 = A2_condition  
    if testing:
        print('A0 =',A0)
        print('A bounds:',Abound1,'to',Abound2)
    Tbound1 = 288.15
    Tbound2 = 500 
    if T0 is None:
        T0 = Tbound2 
    if fit_gauss == True:
        C0 = 1.
        Cbound1 = 1.
        Cbound2 = 1.
        m0 = 0.
        mbound1 = 0.
        mbound2 = 0.
        # centers to try
        mus = DF_peaks_x
        fwhms = np.array(DF_fwhms)#* 2. # 
        fwhm_bounds = []
        fwhm0s = [] #fwhms + 499.
        for i in range(len(DF_peaks_x)):                
            fwhm_mn = 600.
            fwhm0 =  700.
            bounds = [fwhm_mn,900] 
            fwhm0s.append(fwhm0)
            fwhm_bounds.append(bounds)
        heights = []
        height_bounds = []
        for i in range(len(DF_peaks_x)):
            mu = mus[i]
            sigma0 = fwhm0 / 2.3548
            mn_height = 0.
            mi = FindIndexX(mu,Gen_Wn)
            weight = 0.5 
            if np.abs(mu) < 100:
                if np.abs(mu) < VHGnotchedge:
                    mi = FindIndexX(-(VHGnotchedge+5),Gen_Wn)
                    mx_height = ((Y_shifted2[mi] / (A0 * BosePopulation(Gen_Wn,290.)[mi])) - 1) * weight
                else:
                    mx_height = ((Y_shifted2[mi] / (A0 * BosePopulation(Gen_Wn,290.)[mi])) - 1) * weight
                if mx_height < 0.:
                    mx_height = 2.
            elif mu > 100.:
                mx_height = ((Y_shifted2[mi] / (A0 * BosePopulation(Gen_Wn,290.)[mi])) - 1) * weight
                if mx_height < 0.:
                    mx_height = 2.                
            else:
                mx_height = ((Y_shifted2[mi] / (A0 * BosePopulation(Gen_Wn,290.)[mi])) - 1) * weight#Tbound2
                if mx_height < 0.:
                    mx_height = 2.#10.
            if testing == True:
                print('\n at mu =',mu,':')
                print('Y_shifted at mu',Y_shifted2[mi])
                print('mx Gauss height =',mx_height)
            height_bounds.append([mn_height,mx_height])
            height0 = mn_height
            heights.append(height0)
        
        p0 = (A0,T0,C0,m0,mus,fwhm0s,heights)
        bounds_lst = [[Abound1,Abound2],[Tbound1,Tbound2],[Cbound1,Cbound2],[mbound1,mbound2]]
        for i in range(len(DF_peaks_x)):
            # mu only allowed to red shift or blue shift? https://pubs.acs.org/doi/10.1021/acsphotonics.5b00707 blue?
            bounds_lst.append([mus[i]-200.,mus[i]+50.])
            bounds_lst.append(fwhm_bounds[i])
            bounds_lst.append(height_bounds[i]) 
        bounds = np.array(bounds_lst)
    else:
        bounds = np.array([[Abound1,Abound2],[Tbound1,Tbound2]])
        p0 = (A0,T0)    
    if fit_gauss == True:
        lsts = p0[4:] # mus, fwhms, heights
        initial_state = list(p0[:4])
        for i in range(len(lsts[0])):
            peak = []
            for lst in lsts:
                peak.append(lst[i])
            initial_state += peak
    else:
        initial_state = list(p0)
    return X_bg,Y_to0_2,Y_shifted2,initial_state,bounds



def CalculateERS(pth,folder,Title,Gen_Wn,Y_orig,region2,DF_peaks_x,DF_fwhms,DF_spectrum,removepeaks=True,peak_params=None,fit_gauss=True,T0=None,replace=False,sv2=False,plot=True,testing=False):
    '''
    Y is irf-removed spectrum.
    Title = sample + key + scan num
    region is for ASto0, and is defined in this function. It's the region of AS spectrum to average as bg height, region = [wn1,wn2]
        * <> and also for blip for calculating standard deviation ?
    Region2 used in simulated_annealing. Actually only the first value is used.
    # The first value is used in get_cost_v2 to evaluate cost, i.e. spectrum below region[0] is ignored.
        
    peak_params = (prominence,base width,height)
    if plot == True, then plot fits
    '''
    region = [-1200,-550]
    if testing == True:
        print('Starting to calculate ERS of ',Title)
        t0 = process_time() # seconds
    # PRE-PROCESSING
    # notchedge = 5. # picocavities
    # # Alkanethiols, CB, proteins
    notchedge = 100.#50. #25.
    X_bg,Y_to0,Y_shifted,initial_state,bounds = PreProcessing(Gen_Wn,Y_orig,region,peak_params,T0,notchedge,
                                                              fit_gauss,pth,folder,Title,DF_peaks_x,DF_fwhms,
                                                              removepeaks=removepeaks,
                                                              sv2=testing,replace=replace,testing=testing)
    if testing == True:
        print('Starting simulated annealing.')
    solution,final_cost = simulated_annealing(initial_state,bounds,X_bg,Y_to0,region2,notchedge,fit_gauss,testing=testing) #Y_bg_smthd,fit_gauss) #Gen_Wn,Y_filled) 
    Areas_dict = None
    initial_cost,Areas_dict = get_cost_v2(initial_state,X_bg,Y_to0,region2,notchedge,Areas_dict,fit_gauss,infcost=True,testing=False,num=1)
    print('initial cost',initial_cost,'final_cost',final_cost)
    cost_too_high = False
    cost_cutoff = 5*10**4# 5*10**3 #10**4
    if math.isinf(initial_cost):
        if final_cost >  cost_cutoff:
            cost_too_high  = True
    elif initial_cost > cost_cutoff:
        if final_cost > (0.5*initial_cost):
            cost_too_high  = True
    else:
        cost_too_high = False
    if cost_too_high:
        # Allow Gaussian peak hights and positions to vary more:
        print('Allowing all Gaussian centers to vary by -400 cm-1 from extracted DF peak position and mx height bc cost was too high.')
        initial_dict = dict(zip(['A0','T0','C0','slope0','mu10','fwhm10','height10','mu20','fwhm20','height20'],initial_state))
        # bounds_dict = dict(zip(['A0','T0','C0','slope0','mu10','fwhm10','height10','mu20','fwhm20','height20'],bounds))
        for k,key in enumerate(initial_dict.keys()):
            if 'mu' in key:
                mu = initial_dict[key]
                mu_bounds_new = [mu-400,mu+10]
                bounds[k] = mu_bounds_new
                if mu > -500:
                    height_bounds_new = [bounds[k+2][0],bounds[k+2][1]*2]
                    bounds[k+2] = height_bounds_new
        solution2,final_cost2 = simulated_annealing(initial_state,bounds,X_bg,Y_to0,region2,notchedge,fit_gauss,testing=testing) #Y_bg_smthd,fit_gauss) #Gen_Wn,Y_filled) 
        if final_cost2 < final_cost:
            solution = solution2
            final_cost = final_cost2
        print('Initial cost',initial_cost,'Final cost',final_cost)
        if math.isinf(initial_cost):
            if final_cost >  cost_cutoff:
                cost_too_high  = True
            else:
                cost_too_high =  False
        elif initial_cost > cost_cutoff:
            if final_cost > (0.75*initial_cost):
                cost_too_high  = True
            else:
                cost_too_high = False
        else: 
            cost_too_high = False
    # If simulated annealing failed, first try different pre-processing, then use ERSremoval_v9
    if (solution == None) or cost_too_high: #(final_cost >  cost_cutoff):
        replace = False
        print('\n \n Changing region from ',region[0],'-',region[1])
        # Make region smaller
        region[0] = region[0] + 100.
        region[1] = region[1] - 50.
        print('to                   ',region[0],'-',region[1])
        X_bg,Y_to0,Y_shifted,initial_state,bounds = PreProcessing(Gen_Wn,Y_orig,region,peak_params,T0,notchedge,
                                                                  fit_gauss,pth,folder,Title,DF_peaks_x,DF_fwhms,
                                                                  removepeaks=removepeaks,
                                                                  sv2=testing,replace=replace,testing=testing)
        print('Shifting Abound1 down from',bounds[0][0])
        A0 = initial_state[0]
        Abound2 = bounds[0][1] #- (A0*0.1)
        Abound1 = bounds[0][0] - (bounds[0][0]*0.1)
        if Abound1 < 1.:
            Abound1 = 1.
        bounds[0] = [Abound1,Abound2]
        print('to',Abound1)
        initial_state[0] = Abound1 + 0.1
        print('New A0 =',initial_state[0])
    
        solution4,final_cost4 = simulated_annealing(initial_state,bounds,X_bg,Y_to0,region2,notchedge,fit_gauss,testing=testing) 
        if final_cost4 < final_cost:
            solution = solution4
            final_cost = final_cost4
        if math.isinf(initial_cost):
            if final_cost >  cost_cutoff:
                cost_too_high  = True
        elif initial_cost > cost_cutoff:
            if final_cost > (0.5*initial_cost):
                cost_too_high  = True
        else:
            cost_too_high = False
        print('Initial cost',initial_cost,'Final cost',final_cost)
    if (solution == None) or cost_too_high:#(final_cost >  cost_cutoff):
        print('\n \n Failed again, trying for 3rd time now')
        print('Shifting Abound1 down from',bounds[0][0])
        A0 = initial_state[0]
        Abound2 = bounds[0][1]
        Abound1 = bounds[0][0] - (A0*0.1)
        if Abound1 < 1.:
            Abound1 = 1.
        bounds[0] = [Abound1,Abound2]
        print('to',Abound1)
        initial_state[0] = Abound1 + 0.1
        print('New A0 =',initial_state[0])  
        
        solution5,final_cost5 = simulated_annealing(initial_state,bounds,X_bg,Y_to0,region2,notchedge,fit_gauss,testing=testing) 
        if final_cost5 < final_cost:
            solution = solution5
            final_cost = final_cost5
        print('Initial cost',initial_cost,'Final cost',final_cost)
    if solution == None:
        version09 = True
        print('Failed to find solution. Reverting to version 09.')
        p0 = initial_state[:2]
        # solution,initial_state = Version09(X_bg,Y_to0,p0) #Gen_Wn,Y,p0) #Y_filled,A0)
        solution,initial_state = Version09(Gen_Wn,Y_shifted,p0) 
        if testing == True:
            print('A0,T0 = ',initial_state)
            print('A,T = ',solution)
        AS_WN,AS_spectrum,Stokes_WN,Stokes_spectrum = Split3(Gen_Wn,Y_shifted,0.0)
        AS_ERS = ers_v9.antiStokes(AS_WN, *solution)
        Stokes_ERS = ers_v9.Stokes(Stokes_WN, *solution) 
        ERS_fit = np.concatenate((AS_ERS,Stokes_ERS)) 
    else:
        version09 = False
        # Rerun simulated annealing and use version with lower cost
        results_dict = {}
        results_dict['result 1'] = (final_cost,solution)
        print('Running simulated annealing again with same settings to get second result.')
        solution2,final_cost2 = simulated_annealing(initial_state,bounds,X_bg,Y_to0,region2,notchedge,fit_gauss,testing=testing) 
        results_dict['result 2'] = (final_cost2,solution2)
        mn_result_key = min(results_dict,key=results_dict.get)
        final_cost,solution = results_dict[mn_result_key]
        print('Final cost',final_cost)
        if testing == True:
            print('which final cost is lower,',results_dict['result 1'][0],'or',results_dict['result 2'][0],'?')
            print(results_dict[mn_result_key][0])
        
        if fit_gauss == True:
            if testing:
                # print('initial state: A0,T0,mu0,fwhm0,height0 = ',initial_state)
                #meow
                print('initial state:',ParamStringFormat(['A0','T0','C0','slope0','mu10','fwhm10','height10','mu20','fwhm20','height20'],initial_state))
                # print('initial bounds: A',Abound1,'-',Abound2,', T',Tbound1,'-',Tbound2)
                print('initial bounds: A',bounds[0][0],'-',bounds[0][1],', T',bounds[1][0],'-',bounds[1][1])
                print('final solution:',ParamStringFormat(['A ','T ','C','slope','mu1 ','fwhm1 ','height1 ','mu2 ','fwhm2 ','height2 '],solution)) #'solution:      A,T,mu,fwhm,height = ',solution)
            A,T,C,m,mus,fwhms,heights = FromStateToParamLsts(solution)
            ERS_fit = BoseGauss(Gen_Wn,A,T,C,m,mus,fwhms,heights)#*solution) 
        else:
            if testing:
                print('initial state: A0,T0 = ',initial_state)
                # print('initial bounds: A',Abound1,'-',Abound2,', T',Tbound1,'-',Tbound2)
                print('initial bounds: A',bounds[0][0],'-',bounds[0][1],', T',bounds[1][0],'-',bounds[1][1])
                print('solution:      A,T = ',solution)
            ERS_fit = OnlyBose(Gen_Wn,*solution) 
            
    VHGnotchedge = 9.             
    Ynew = SubtractERS(Gen_Wn,Y_orig,ERS_fit,VHGnotchedge,testing=testing)

    if plot:
        Y_filtered = ReduceNoise(Gen_Wn,Ynew,notchedge)
        if len(solution) > 2:
            A,T,C,m,mus,fwhms,heights = FromStateToParamLsts(solution)
            gaus_eval = MultiGauss(Gen_Wn,C,m,mus,fwhms,heights)
        if (version09 == False) and (testing == True):
            p0 = initial_state[:2]
            solution_2,intial_state_2 =  Version09(Gen_Wn,Y_shifted,p0) 
            AS_WN,AS_spectrum,Stokes_WN,Stokes_spectrum = Split3(Gen_Wn,Y_shifted,0.0)
            AS_ERS = ers_v9.antiStokes(AS_WN, *solution_2)
            Stokes_ERS = ers_v9.Stokes(Stokes_WN, *solution_2) 
            ERS_2 = np.concatenate((AS_ERS,Stokes_ERS)) 
            if ERS_2 is None:
                print('Did not find a solution by ERSremoval version09.')
                ERS_2 = np.zeros(len(Gen_Wn))
                ERSremoved_2 = np.zeros(len(Gen_Wn))
            else:
                ERSremoved_2 = SubtractERS(Gen_Wn,Y_shifted,ERS_2,notchedge)
                ERSremoved_2_filtered = ReduceNoise(Gen_Wn,ERSremoved_2,notchedge,cutoff = 5000)#butterLowpassFiltFilt(ERSremoved_2, cutoff = 5000, fs = 50000, order=1) 
            print('Plotting the solutions.')
        fig,axes = plt.subplots(nrows=2, ncols=1,figsize=(5,5),sharex=True)
        axes[0].plot(Gen_Wn,np.zeros(len(Gen_Wn)),color='y',ls='--')
        axes[0].plot(Gen_Wn,Y_orig,color='k',ls=':',label='s (not shifted)') 
        if len(solution) > 2:
            label = 'gauss * Bose, sim annealing'
            axes[0].plot(Gen_Wn,gaus_eval,color='gold',label='gaussian')
            txt = ParamStringFormat(['A ','T ','C','slope','mu1 ','fwhm1 ','height1 ','mu2 ','fwhm2 ','height2 '],solution)
            if testing == True:
                axes[0].plot(Gen_Wn,ERS_2,color='m',label='Bose, non-linear least squares') 
                axes[1].plot(Gen_Wn,ERSremoved_2_filtered,color='m',label='non-linear least squares') 
                print('plotted version09 fit')
        else:
            txt = 'A=%.2f, T=%.1f' % tuple(solution)
            if (version09 == False):
                label = 'Bose, sim annealing'
                if testing == True:
                    axes[0].plot(Gen_Wn,ERS_2,color='m',label='Bose, non-linear least squares') 
                    axes[1].plot(Gen_Wn,ERSremoved_2_filtered,color='m',label='non-linear least squares') 
            else:
                label = 'Bose, non-linear least squares'
        axes[0].plot(X_bg,Y_to0,color='b',label='Y_bg')
        axes[0].plot(Gen_Wn,ERS_fit,color='r',label=label) 
        xlim = GetXLim(Gen_Wn)
        axes[0].set(title='Fitting ERS to '+Title,xlim=xlim,ylim=[10**-2,max(Y_shifted)],yscale='log')
        axes[0].set_ylim(bottom= 0.01)
        axes[1].plot(Gen_Wn,np.zeros(len(Gen_Wn)),color='y',ls=':')
        axes[1].plot(Gen_Wn,Y_filtered,color='r',label='simulated annealing') 
        axes[1].text(0.02,0.9,#0.95,0.9, 
        txt,
        fontsize=8,
        horizontalalignment='left',
        verticalalignment='top',
        transform=axes[1].transAxes)
        axes[1].set(ylim=[0.01,max(Y_filtered)],yscale='log')
        if testing:
            suffix = '_1fittingERS_testing_'+'.png'
        else:
            suffix = '_1fittingERS_'+'.png'
        figfolder = pth + folder + 'ERSfit/' + '/'
        ensure_dir(figfolder)
        figname = figfolder + Title + suffix
        print(figname)
        fig.savefig(figname,format='png',bbox_inches='tight',transparent=False)#,dpi=300)
        if testing == False:
            plt.close(fig)
    if testing:
        elapsed = (process_time() - t0)
        print('Finished in '+str(elapsed)+' seconds.')  
    return Ynew,ERS_fit,solution,initial_state 
    




# ~ ~ ~ ~ ~ Simulated Annealing ~ ~ ~ ~ ~
def simulated_annealing(initial_state,bounds,WN,Y,region,notchedge,fit_gauss,testing=False):
    """Peforms simulated annealing to find a solution"""
    # - COOLING SCHEDULE - 
    initial_temp = 10**0
    final_temp = 10**-5 #10**-5 
    alpha1 = 0.99
    alpha2 = 0.97
    current_temp = initial_temp
    # Start by initializing the current state with the initial state
    solution = initial_state #current_state
    if testing == True:
        values_dict = {}
        values_dict['temp'] = []
        values_dict['A'] = []
        values_dict['T'] = []
        if fit_gauss == True:
            values_dict['C'] = []
            values_dict['slope'] = []
            values_dict['mu1'] = []
            values_dict['fwhm1'] = []
            values_dict['height1'] = []
            values_dict['mu2'] = []
            values_dict['fwhm2'] = []
            values_dict['height2'] = []
        values_dict['neighbor_go'] = []
        values_dict['solution_costs'] = []
    i=0
    j=0
    while current_temp > final_temp:
        if i< 5000:
            # print('current solution:',solution)
            neighbor = get_neighbors(solution,bounds,WN) 
            # Check if neighbor is best so far
            if testing == True:
                if (final_temp >= current_temp*alpha2): #alpha1
                    cost_testing = True
                    num = 2
                elif (current_temp == initial_temp):
                    cost_testing = True
                    num = 1
                    Areas_dict = None
                else:
                    cost_testing = False
                    num = 0
            else:
                cost_testing = False
                #num = None
                if (final_temp >= current_temp*alpha2): #alpha1
                    num = 2
                elif (current_temp == initial_temp):
                    num = 1
                    Areas_dict = None
                else:
                    num = 0
                
            infcost = True
            solution_cost,Areas_dict = get_cost_v2(solution,WN,Y,region,notchedge,Areas_dict,fit_gauss,infcost,testing=cost_testing,num=num)
            if (j==0) and (solution_cost != np.inf):
                j+=1
            else:
                pass
            neighbor_cost,Areas_dict = get_cost_v2(neighbor,WN,Y,region,notchedge,Areas_dict,fit_gauss,infcost,testing=False,num=0)
            cost_diff = (solution_cost - neighbor_cost) 
            if cost_diff > 0.0:
                solution = neighbor
                if testing == True:
                    values_dict['neighbor_go'].append(1)         
            else:
                if math.isnan(cost_diff): 
                    if solution_cost == np.inf:
                        solution = neighbor
                        if testing == True:
                            values_dict['neighbor_go'].append(1)
                    else:
                        if random.uniform(0, 1) < 0.5:
                            solution = neighbor
                            if testing == True:
                                values_dict['neighbor_go'].append(1)
                        else:
                            solution = solution
                            if testing == True:
                                values_dict['neighbor_go'].append(0)
                else:
                    if random.uniform(0, 1) < math.exp(cost_diff / current_temp):
                        solution = neighbor
                        if testing == True:
                            values_dict['neighbor_go'].append(1)
                    else:
                        solution = solution  
                        if testing == True:
                            values_dict['neighbor_go'].append(0)
        else:
            print('max iterations reached.')
            if solution_cost is np.inf:
                solution = None                                             
            break           
        if testing:
            values_dict['temp'].append(current_temp)
            values_dict['A'].append(solution[0])
            values_dict['T'].append(solution[1])
            if fit_gauss == True:
                values_dict['C'].append(solution[2])
                values_dict['slope'].append(solution[3])
                values_dict['mu1'].append(solution[4])
                values_dict['fwhm1'].append(solution[5])
                values_dict['height1'].append(solution[6])
                if len(solution) > 8:
                    values_dict['mu2'].append(solution[7])
                    values_dict['fwhm2'].append(solution[8])
                    values_dict['height2'].append(solution[9])
            values_dict['solution_costs'].append(solution_cost)     
        i+=1        
        if current_temp > 10**-3:
            current_temp *= alpha1 # slower
        else:
            current_temp *= alpha2 # faster
    if solution_cost is np.inf:
        solution = None
    if testing == True:
        print('num iterations:',i)
        print('solution cost:',solution_cost)
        if solution is not None:
            if fit_gauss == True:
                c = 3
                r = 4
            else:
                c = 2
                r = 2
            plot_dict = {'solution_costs':{#'vals':values_dict['solution_costs'],
                                          'bounds':[round(np.min(values_dict['solution_costs']),1),round(np.max(values_dict['solution_costs']),1)],
                                          'label':'solution cost'}}
            for v,val in enumerate(solution):
                plot_dict[params_list[v]] = {#'vals':values_dict[params_list[v]],
                                             'bounds':bounds[v],'label':params_list[v]}
            plot_dict['neighbor_go'] = {#'vals':values_dict['neighbor_go'],
                                        'bounds':[0,1],'label':'1=neighbor','ylim':[-0.1,1.1]}
            fig2,axes = plt.subplots(nrows=r, ncols=c,figsize=(13.6*cm,13.6*cm),sharex=True) 
            axes_list = axes.reshape(-1)
            for a,key in enumerate(plot_dict.keys()):
                if a == 0:
                    axes_list[a].plot(values_dict['temp'],values_dict[key],color='g')
                    axes_list[a].set_yticks(np.linspace(plot_dict[key]['bounds'][0],plot_dict[key]['bounds'][1],3))
                    axes_list[a].set(title=plot_dict[key]['label'],xlim=[values_dict['temp'][0],values_dict['temp'][-1]])
                    axes_list[a].set(xscale='log')
                    axes_list[a].set_xticks([10**1,10**-1,10**-3,10**-5])
                elif key == 'neighbor_go': #a == (len(plot_dict.keys())-1):
                    axes_list[a].scatter(values_dict['temp'],values_dict[key],color='g',s=1)
                    axes_list[a].set_yticks(plot_dict[key]['bounds'])
                    axes_list[a].set(title=plot_dict[key]['label'],xlabel='Temp',ylim=plot_dict[key]['ylim'])
                else:
                    axes_list[a].plot(values_dict['temp'],values_dict[key],color='g')
                    axes_list[a].set_yticks(np.linspace(round(plot_dict[key]['bounds'][0],0),round(plot_dict[key]['bounds'][1],0),3))
                    axes_list[a].set(title=plot_dict[key]['label'])#),xlabel='Temp')
                axes_list[a].xaxis.set_ticks_position('bottom')
                axes_list[a].yaxis.set_ticks_position('left')
                axes_list[a].tick_params(axis='both', which='major', pad=2)
                #axes_list[a].yaxis.labelpad = 0.1#0.5
                axes_list[a].xaxis.labelpad = 0.1#0.5
            fig2.set_tight_layout(True)  
            plt.show()
    return solution, solution_cost


def get_cost_v2(state,WN_full,Y_full,region,notchedge,Areas_dict,fit_gauss,infcost,testing=False,num=0):
    """Calculates cost of the argument state for your solution.
    cost = root mean square error
    """
    # CUT off AS tail
    fi = FindIndexX((region[0]+100),WN_full)
    if np.isnan(fi):
        if testing:
            print('for get_cost_v2 in simulated annealing, region[0]+100 is not in given WN_full...')
        for i,wn in enumerate(WN_full):
            if wn > (region[0]+100):
                fi = i
                break
    if np.isnan(fi):
        print('WN_full[0] =',WN_full[0])
    WN = WN_full[fi:]
    Y = Y_full[fi:]   
    if fit_gauss:
        A,T,Cs,slopes,mus,fwhms,heights = FromStateToParamLsts(state)
 
        ERS_fit = BoseGauss(WN,A,T,Cs,slopes,mus,fwhms,heights)
    else:
        A,T = state
        ERS_fit = OnlyBose(WN,A,T)
    Ynew = SubtractERS(WN,Y,ERS_fit,notchedge,testing=False)    
    Bins = [[WN[0],-500],[-500,-notchedge],[notchedge,500],[500,WN[-1]]]
    if num == 1:# and testing:
        # initial guess
        Areas_dict = {}
        for bn in Bins:
            bn_str = str(int(bn[0]))+'-'+str(int(bn[1]))
            # approximate area of each bin as a rectangle 
            halfmx = np.max(Ynew[FindIndexX(bn[0],WN):FindIndexX(bn[1],WN)]) /2
            Areas_dict[bn_str] = halfmx * np.abs(bn[1]-bn[0])
    # mean square error
    summation = 0
    for i in range(len(WN)):
        for bn in Bins:
            if bn[0] <= WN[i] < bn[1]:
                bn_str = str(int(bn[0]))+'-'+str(int(bn[1]))
                area = Areas_dict[bn_str]
                break
            else:
                pass
        s = ((Y[i] - ERS_fit[i])**2) / area        # normalize error (roughly)
        summation += s
    #mse = (1/len(WN)) * summation
    rmse = np.sqrt((1/len(WN)) * summation) 
    if testing == True:
        if fit_gauss == True:
            gaus_eval = MultiGauss(WN,Cs,slopes,mus,fwhms,heights)
        if num == 1:
            title = ' cost testing, initial guess'
            ylim = [10**-2,5*10**4]
        elif num == 2:
            title = ' cost testing, final fit'
            ylim = [10**-2,5*10**4] #[-10,10**3]
        else:
            title = 'not a thing'
        figsize = (7,5)
        fig,axes = plt.subplots(nrows=2, ncols=1,figsize=figsize,sharex=True)
        axes[0].scatter(WN,Y,color='b',marker='.',s=1)
        if fit_gauss == True:
            axes[0].plot(WN,gaus_eval,color='orange')
        axes[0].plot(WN,ERS_fit,color='g')
        # axes[1].plot(WN,Ynew,color='grey')#,ls=':')
        axes[1].scatter(WN,Ynew,color='grey',marker='.',s=1)
    if infcost:
        neg_area = 0.
        num_neg_points = 0 
        nnp = 10**2 
        bn_str = str(int(Bins[0][0]))+'-'+str(int(Bins[0][1]))  # smallest area: far aS from end to -500
        na = Areas_dict[bn_str] * 0.1 # 10% of that area
        for i in range(len(Ynew)):
            wn = WN[i]
            I = Ynew[i]
            if I < 0.:
                neg_area += np.abs(I)
                if 0 < wn < 500.:
                    num_neg_points += (nnp)/3 
                elif -300 < wn < 0:
                    num_neg_points += (nnp)/3
                else:
                    num_neg_points += 1
                if testing == True:
                    axes[1].bar(wn,I,color='r',edgecolor='r')
        if (num_neg_points > nnp) or (neg_area > na): #50.: #10.: #10.: #5.: #2.5: #5.:
            cost = np.inf
        else:
            #cost = mse
            cost = rmse
            # cost = np.tanh(rmse) 
    else:
        cost = rmse
    if testing:
        axes[0].set(title=title,yscale='log',ylim=ylim)
        txt = ParamStringFormat(['A ','T ','C','slope','mu1 ','fwhm1 ','height1 ','mu2 ','fwhm2 ','height2 '],state)
        axes[0].text(0.05,0.9, txt,
        fontsize=8,
        horizontalalignment='left',
        verticalalignment='top',
        transform=axes[0].transAxes)
        txt = 'cost:{:.1f}, neg area:{:.1f}, \n num neg pnts:{}'.format(cost,neg_area,num_neg_points)
        axes[1].text(0.9,0.9,txt,
        horizontalalignment='right',
        verticalalignment='top',
        transform=axes[1].transAxes)
        xlim = GetXLim(WN)
        ylim=[-10,10**3]
        axes[1].xaxis.set_major_locator(MultipleLocator(500))
        axes[1].xaxis.set_major_formatter(FormatStrFormatter('%d'))
        axes[1].xaxis.set_minor_locator(MultipleLocator(100))
        axes[1].set(xlim=xlim,ylim=ylim,yscale='linear')
        axes[0].set_yticks([10**-1,10**1,10**3])
        axes[1].set_yticks([0,10**3])#np.linspace(ylim[0],ylim[1],2))
        for ax in axes:
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            ax.tick_params(axis='both', which='major', pad=2)
            #ax.yaxis.labelpad = 0.1#0.5
            ax.xaxis.labelpad = 0.1#0.5
        fig.set_tight_layout(True)  
        plt.show()
  
    return cost,Areas_dict



def get_neighbors(state,bounds,WN):
    """Returns neighbors of the argument state for your solution."""
    num_points = 10 
    num_segments = 5 
    sizes = bounds[:, 1] - bounds[:, 0]
    segment_sizes = sizes / num_segments
    halfwidths = segment_sizes / 2.
    neighbor = []
    for i in range(len(state)):
        param = state[i]
        # Keep neighbor bounds within set param bounds
        left  = param - np.abs(halfwidths[i])
        if left < bounds[i][0]:
            left = bounds[i][0]
        right = param + np.abs(halfwidths[i])
        if right > bounds[i][1]:
            right = bounds[i][1]
        param_neighbors = np.linspace(left,right,num_points)
        param_new = random.choice(param_neighbors)
        neighbor.append(param_new)
        i+=1
    return tuple(neighbor)


# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~



def FromStateToParamLsts(state):
    A,T,C,m = state[:4]
    mus = []
    fwhms = []
    heights = []
    for i in range(4,len(state)):
        if i in [4,7]:
            mus.append(state[i])
        elif i in [5,8]:
            fwhms.append(state[i])
        else:
            heights.append(state[i])
    return A,T,C,m,mus,fwhms,heights


def GetXLim(WN):
    if np.abs(WN[0]) > WN[-1]:
        lim = np.abs(WN[0])
    else:
        lim = WN[-1]
    xlim = [-1*lim,lim]
    return xlim


def ParamStringFormat(param_strs,state):
    txt = '' 
    for i in range(len(state)):
        param_str = param_strs[i]
        txt += param_str + '=' + '{:<7.1f} '
        if (i == 3) or (i == 6):
            txt += '\n'
    return txt.format(*state)



#     #     #     #     #     #     #     #     #     #     #     #     #     #     #     #     #     #     #     #     #     #     #     #     #     #     #     #     #     #     

def SubtractERS(WN,Y,ERS,notchedge,testing=False):
    # 1. Find peak of Y, only subtract ERS outside of peak
    num = 25.
    ai = FindIndexX(-num,WN)
    bi = FindIndexX(0,WN)
    ci = FindIndexX(num,WN)
    if np.isnan(ai) or np.isnan(bi) or np.isnan(ci):
        notched = True
        i_left = FindIndexX(-(notchedge+1),WN)
        i_right = FindIndexX((notchedge+1),WN)
        if testing:
            print('didnt find {}:'.format(num),'-{} ai={}, 0 bi={}, {} ci={}'.format(num,ai,bi,num,ci))
    else:
        notched = False
        # aS
        Nu_left = np.linspace(WN[ai],WN[bi],100)
        i_mx = np.argmax(np.interp(Nu_left,WN[ai:bi],Y[ai:bi]))
        i_left = FindIndexX(Nu_left[i_mx],WN)
        # Stokes
        Nu_right = np.linspace(WN[bi],WN[ci],100)
        i_mx = np.argmax(np.interp(Nu_right,WN[bi:ci],Y[bi:ci]))
        i_right = FindIndexX(Nu_right[i_mx],WN)
        if testing:
            print('-{}={}, aS intersect={}, 0={}, Stokes intersect={}, {}={}'.format(num,WN[ai],WN[i_left],WN[bi],WN[i_right],num,WN[ci]))
    if testing:
        print('Subtract ERS \n setting region between {} and {} to 10^-4'.format(WN[i_left],WN[i_right]))
    # 2. Segment Y, ERS into three sections: aS, notch, and Stokes
    Y_aS = Y[:i_left+1]
    ERS_aS = ERS[:i_left+1]
    Y_Stokes = Y[i_right+1:]
    ERS_Stokes = ERS[i_right+1:]
    # 3. Subtract (Y-ERS) in aS and Stokes sections
    Ynew_aS = np.subtract(Y_aS,ERS_aS)
    Ynew_Stokes = np.subtract(Y_Stokes,ERS_Stokes)
    # 4. Make notch section 10**-4
    len_notch = len(Y) - (len(Y_aS)+len(Y_Stokes))
    notch = np.ones(len_notch) * (10**-4)
    # 5. Concatenate the three sections together
    Ynew = np.concatenate((Ynew_aS,notch,Ynew_Stokes),axis=0)
    if len(Ynew) != len(Y):
        print('We have a problem in SubtractERS.')
    return Ynew

    

def DivideByERS(Y,ERS):
    '''Dividing accentuates small differences. Result is unitless'''
    Ynew = np.divide(Y,ERS) - 1.
    return Ynew































