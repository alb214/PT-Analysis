# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 11:11:14 2022

@author: alb214


My First Paper
Figure 3
"""

import csv
from basics_Python3 import FindIndexX, LoadCSV, CreateColorsList
from OpenAndPlot_v2 import Smooth, ensure_dir

import numpy as np
from lmfit.models import LinearModel, GaussianModel


def LoadCSV(filename):
    '''Output array: RamanShift, Intensity'''
    with open(filename) as f:
        RamanShifts = [] # cm-1
        Intensities = [] # normalized
        reader = csv.reader(f)
        next(reader, None) # skip headings
        for row in reader:
            wn = float(row[0])
            I = float(row[1])
            RamanShifts.append(wn)
            Intensities.append(I)

    return np.array(RamanShifts), np.array(Intensities)

def LoadDFT(filename):
    '''Output array: RamanShift, Intensity'''
    with open(filename) as f:
        RamanShifts = [] # cm-1
        Intensities = [] # normalized
        reader = csv.reader(f,delimiter=' ')
        next(reader) # skip headings
        for row in reader:
#             print(row)
            wn = float(row[0].strip("'"))
            I = float(row[-1])
            RamanShifts.append(wn)
            Intensities.append(I)

    return np.array(RamanShifts), np.array(Intensities)




# def EvalMultiGaussFromDFT(DFTfullname,eval_dict,sigmas,mol,x_scaling_factor,y_scaling_factor,threshold):
def EvalMultiGaussFromDFT(X_scaled,Y_scaled,eval_dict,sigmas,mol,threshold):
    
    '''threshold is for only include peaks above a threshold intensity'''
    eval_dict[mol] = {}
    
    #DFTfullname = DFT_dict[mol]['DFTfullname']  #DFTp + DFT_p_dict[mol]
    # Xfull,Yfull = LoadDFT(DFTfullname)
    
    # X_scaled = Xfull * x_scaling_factor
    # Y_scaled = Yfull * y_scaling_factor
    
    #left = 5
    right = 200
    #bi = FindIndexX(left,Xfull)
    ci = FindIndexX(right,X_scaled)
    X = X_scaled[:ci] #THzWN
    #print(X[0],X[-1])

    Y = Y_scaled[:ci] #THzY
    # num_peaks = DFT_dict[mol]['num peaks']
    # peaks = DFT_dict[mol]['peaks']



#     for sigma in [1,7,13]:
    for sigma in sigmas: #[2,15,28]:
        key = str(sigma)
        
        model = LinearModel()#prefix='line_')
        pars = model.make_params(slope=0.0, intercept=0.0)
        pars['slope'].set(vary=False)
        pars['intercept'].set(vary=False)
        
        # for n in range(1,num_peaks+1):
        for i in range(len(X)):

            if Y[i] > threshold:

                prefix = 'f' + str(i+1) + '_'

                center = X[i] #         * shift?
                amplitude = Y[i]
                #sigma = 50 #            * arbitrary? iterate andd show three different sigmas?

                peak = GaussianModel(prefix=prefix)
                pars.update(peak.make_params(center=center, sigma=sigma, amplitude=amplitude))

                model = model + peak

        
        Xeval = np.linspace(0,200,201)
        Yeval = model.eval(pars,x=Xeval)
        eval_dict[mol][key] = (Xeval,Yeval) 
    
    return eval_dict




def FitMultiGaus(Xfull,Yfull,DFT_dict,Gauss_dict,mol,scaling_factor):
    scaling_factor_str = str(scaling_factor)
    if mol not in Gauss_dict.keys():
        Gauss_dict[mol] = {}
        Gauss_dict[mol][scaling_factor_str] = {}
    else:
        Gauss_dict[mol][scaling_factor_str] = {}

    left = 13
    right = 200
    bi = FindIndexX(left,Xfull)
    ci = FindIndexX(right,Xfull)
    X = Xfull[bi:ci] #THzWN
    Y = Yfull[bi:ci] #THzY

    if mol in DFT_dict.keys():
        num_peaks = DFT_dict[mol]['num peaks']
        peaks = [peak * scaling_factor for peak in DFT_dict[mol]['peaks']]
        print(peaks)
        # * offset wavenumber positions of DFT peaks ? 
        peakIs = DFT_dict[mol]['normalized peak intensities']
#         print(peakIs)
    else:
        num_peaks = 3 #     * arbitrary?  ?
        peaks = [58,103,148]
        peakIs = [2. for i in range(len(peaks))]   #     * arbitrary?  ?

    model = LinearModel()#prefix='line_')
    pars = model.make_params(slope=0.0, intercept=0.0)
    pars['slope'].set(vary=False)
    pars['intercept'].set(vary=False)

    for n in range(1,num_peaks+1):
#         print(n)
        prefix = 'f' + str(n) + '_'
        peak = GaussianModel(prefix=prefix)
#         if len(peaks) > 0:
        sigma = 10 #    * arbitrary?
        center  =  peaks[n-1] 
#         print(center)
        
        # <check> use DFT Intensity information?
        height_scaling_factor = peakIs[n-1]
        min_height = 0.5 #                               ?
        height = height_scaling_factor * min_height
#         print(height)
        amp = height*sigma*np.sqrt(2*np.pi )
#         print(amp)
#         print('- - '*3)
        
        pars.update(peak.make_params(center=center,amplitude=amp,sigma=sigma)) 
        
        err = 5.
        pars[prefix+'center'].set(min=center-err,max=center+err)
        #pars[prefix+'center'].set(vary=False)
            
        min_fwhm = 6.
        min_sigma  = min_fwhm / 2.3548
        max_fwhm = 75. #30.612
        max_sigma = max_fwhm / 2.3548 # fwhm / (2*np.sqrt(2*np.log(2)))
        pars[prefix+'sigma'].set(min=min_sigma,max=max_sigma)
#         pars[prefix+'sigma'].set(vary=False)

        min_height = 1. #  reset min height from previous  value
        min_amp = min_height*min_sigma*np.sqrt(2*np.pi )
        max_height = 13.
        max_amp = max_height*max_sigma*np.sqrt(2*np.pi )
        pars[prefix+'amplitude'].set(min=min_amp,max=max_amp)
#         pars[prefix+'amplitude'].set(min=0.)
#         pars[prefix+'amplitude'].set(vary=False)

        model = model + peak

    result = model.fit(Y,pars,x=X)
#     plt.plot(X,Y,color='k')
#     plt.plot(X,result.best_fit)
    
    
    comps = result.eval_components(x=X)
    for n in range(1,num_peaks+1):
        prefix = 'f' + str(n) + '_'
        comp_label= 'Gauss ' + str(n)
#         plt.plot(X,comps[prefix],ls='--')
        
    if 'THzWN' not in Gauss_dict.keys():
        Gauss_dict['THzWN'] = X
    Gauss_dict[mol][scaling_factor_str]['fit'] = result.best_fit
    Gauss_dict[mol][scaling_factor_str]['components'] = comps
    
    return Gauss_dict




def wavenumber2frequency(invcm):
    '''converts wavenumber in cm-1 to frequency in THz'''
    c = 299792458 #metres per second
    invm = invcm * 100 
    Hz = invm * c
    THz = Hz * 10**-12
    print(invcm,'cm-1,',THz,'THz')
    return THz

def ConvertAxisInvcm2AxisTHz(axInvCm):
    x1, x2 = axInvCm.get_xlim()
    #print(x1,x2)
    axTHz.set_xlim(wavenumber2frequency(x1), wavenumber2frequency(x2))
    axTHz.figure.canvas.draw() 
    
def IntegratedIntensity(spectrum):
    totalI = 0
    for point in spectrum:
        totalI += point
    return totalI