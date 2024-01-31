
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pylab
import matplotlib.colors as clrs
# import matplotlib.ticker as tck
import matplotlib.patches as patches
# from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               # AutoMinorLocator)
import csv
# from os import listdir
import os
# from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from scipy.signal import butter, filtfilt, savgol_filter
# from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.ndimage import gaussian_filter1d

# import matplotlib.cm as mplcm
# import matplotlib.colors as colors
#from matplotlib.colors import LinearSegmentedColormap 
#from matplotlib.colors import to_rgb



def LoadGen_Wn(filename,header=True):
    '''Given a filename, returns the first column as Gen_Wn array'''
    with open(filename,newline='') as f:
        Gen_Wn = [] # cm-1
        reader = csv.reader(f)#,delimiter='\t')
        if header == True:
            next(reader)
        for row in reader:
            v = float(row[0])
            nutting = float(row[1])
            Gen_Wn.append(v)
    return np.array(Gen_Wn)

def RemoveNotch(X,whitelight,testing=False):
    '''Remove effect of BNFs from WL spectra
    (developed in 2022-07-11 jupyter notebook)
    X is Gen_Wn
    '''
    #testing = False
    if testing == True:
        fig,ax = plt.subplots(nrows=1, ncols=1,figsize=(10,5))
        ax.plot(X,whitelight,label='original')
    # replace filtered part with a line
    # 0. reduce noise
    cutoff = 10000
    fs = 500000
    filtered = butterLowpassFiltFilt(whitelight, cutoff = cutoff, fs = fs)
    
    # 1. find start and end of filtered part
    #deriv = np.diff(op.NormalizeTo1(whitelight),n=1)
    window_length = 21
    polyorder = 5
    deriv = savgol_filter(NormalizeTo1(filtered), window_length, polyorder, deriv=1)#, delta=1.0, axis=-1, mode='interp', cval=0.0)
    for i in range(len(deriv))[500:]:
        if deriv[i] < (min(deriv)+np.std(deriv)): #0.01: #-0.0233230081: #-896.:
            start_i = i - 10
            break
        else:
            start_i = 0
    #for i in reversed(range(len(deriv))):
    for i in reversed(range(len(deriv))[:-500]):
        if deriv[i] > (max(deriv)-np.std(deriv)):#0.01: #0.02: #778.:
            end_i = i + 10
            break
        else:
            end_i = 0
    if testing:
        try:
            print('index i+/-10 ',X[start_i],X[end_i])
        except IndexError:
            print('index i ',X[start_i+10],X[end_i-10])
    # 2. delete this part
    ref_masked = np.ma.masked_where((X >= X[start_i])&(X < X[end_i]),whitelight)
    if testing:
        ax.plot(X,ref_masked,label='masked')
    # 3. Smooth to get better endpoints
    window_length = 21
    polyorder = 5
    # before notch
    ref_smoothed = Smooth(ref_masked[:start_i-1],window_length,polyorder)
    y1 = ref_smoothed[-1]
    # after notch
    ref_smoothed = Smooth(ref_masked[end_i:],window_length,polyorder)
    y2 = ref_smoothed[0]
    if testing:
        ax.plot(X[end_i:],ref_smoothed,label='smoothed')
    # 4. fit a line to deleted part, or interpolate
    line = Line2(X,X[start_i],y1,X[end_i],y2)[start_i+1:end_i-1]
    ref_new = whitelight.copy()
    ref_new[start_i+1:end_i-1] = line
    window_length = 501
    polyorder = 3
    ref_smoothed = Smooth(ref_new,window_length,polyorder) #op.GaussSmooth(ref_new,axis=0) 
    if testing == True:
        ax.plot(X[start_i+1:end_i-1],line,label='line')
        ax.plot(X,ref_smoothed,label='final')
        ax.legend(fontsize=9)
    return ref_smoothed

# <op>
def Smooth(Y,window_length,polyorder):
    '''The length of the filter window must be a positive, odd integer. polyorder must be less than window length. '''
    return savgol_filter(Y, window_length, polyorder, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
    
def GaussSmooth(Y,axis=1):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter1d.html
    smooth_sigma = 1.1
    return gaussian_filter1d(Y,sigma=smooth_sigma,axis=axis)

def ensure_dir(file_path):
    '''https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory'''
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def SortKeys(group,K,meas=None):
    '''K is list of keys.
    https://docs.python.org/3/library/time.html#time.strftime
    '''
    # sort keys by creation timestamp  
    decoratedK = []
    for key in K:
        if meas is None:
            obj = group[key]
        else:
            obj = group[key][meas]
        if isinstance(obj.attrs['creation_timestamp'],str):
            # date = group[key].attrs['creation_timestamp'].split('T')[0]
            # timestamp = group[key].attrs['creation_timestamp']#.split('T')[1]
            datetime_str = obj.attrs['creation_timestamp']
        else:
            # date = group[key].attrs['creation_timestamp'].decode("utf-8").split('T')[0]
            # timestamp = group[key].attrs['creation_timestamp'].decode("utf-8")#.split('T')[1]
            datetime_str = obj.attrs['creation_timestamp'].decode("utf-8")
        # decoratedK.append((timestamp,key))
        datetime_object = datetime.datetime.strptime(datetime_str, '%Y-%m-%dT%H:%M:%S.%f')
        decoratedK.append((datetime_object,key))
    sorteddecoratedK = sorted(decoratedK)
    sortedK = [tpl[1] for tpl in sorteddecoratedK]
    timestamps = [tpl[0] for tpl in sorteddecoratedK]
    return sortedK, timestamps

def SortParticles(K):
    '''K is list of keys.'''
    # sort keys by particle number  
    decoratedK = []
    for key in K:
        number = int(key.split('_')[1])
        decoratedK.append((number,key))
    sorteddecoratedK = sorted(decoratedK)
    sortedK = [tpl[1] for tpl in sorteddecoratedK]
    #numbers = [tpl[0] for tpl in sorteddecoratedK]
    return sortedK

def NormalizeTo1(Y):
    if len(Y) > 1:
        mn = float(np.amin(Y))
        mx = float(np.amax(Y))
        #print "max:", mx
        # normalizedY = [(y/float(mx)) for y in Y]
        normalizedY = [((y-mn)/(mx-mn)) for y in Y]
    else:
        normalizedY = [1]
    return np.array(normalizedY)

def NormalizeTo1Special(X,Y,region):
    '''region = [x1,x2]. returns a numpy array.'''
    x1 = region[0]
    x2 = region[1]
    index1 = FindIndexX(x1,X)
    index2 = FindIndexX(x2,X)
    #print index1,index2
    Y_cut = Y[index1:index2]
    mn = float(np.amin(Y_cut))
    mx = float(np.amax(Y_cut))
    normalizedY = [((y-mn)/(mx-mn)) for y in Y]
    return np.array(normalizedY)

def NormyNormSpecial(X,Y,region,testing=False):
    x1 = region[0]
    x2 = region[1]
    index1 = FindIndexX(x1,X)
    index2 = FindIndexX(x2,X)
    if np.isnan(index1):
        if testing:
            print('In NormyNormSpecial, region left edge is not in given X')
        index1 = 0
    if np.isnan(index2):
        if testing:
            print('In NormyNormSpecial, region right edge is not in given X')
        index2 = -1
    Y_cut = Y[index1:index2]
    mx = float(np.amax(Y_cut))
    normalizedY = [(y/mx) for y in Y]
    return np.array(normalizedY)

def PowerDesc(desc):
    desc_lines = desc.split('\n')
    desc_pow = ''
    for line in desc_lines:
        if ('power' in line.lower() or 'mW' in line or 'uW' in line):
            desc_pow = line.lower()
    return desc_pow

# </op>



def Desc(grp):
    '''given dataset[key], prints description attribute'''
    try:
        desc = grp.attrs['Description'] 
    except KeyError:
        desc = 'no description'
    return desc

def FindIndexX(w,X,testing=False):
    if len(X) > 0:
        X2 = np.array(X)
        indices = np.argwhere(((w-50) < X2) & (X2 < (w+50)))
    #     print('indices',indices)
        distances = []
        if len(indices) > 1:
            for i in range(len(indices)):
    #             print(X2[indices[i][0]])
                dist = np.abs(w - X2[indices[i][0]])
                distances.append((dist,indices[i][0]))
            sorteddecoratedi = sorted(distances)
            sortedi = [tpl[1] for tpl in sorteddecoratedi]
            if np.abs(X[sortedi[0]] - w) < 5:
                return sortedi[0]
            else:
                #print('Given value {} is not found within 5 (units of X )of any value in X'.format(w))
                return np.NAN
        elif len(indices) == 0:
            if testing:
                print('Given value {} is not in X.'.format(w))
            return np.NAN
        else:
            print('len(indices) in FindIndexX',len(indices))
            return indices[0][0]
    else:
        print('Length of X is zero.')
        return np.NAN

def FindClosesetXbelow_w(w,X):
    distances = []
    for x in X:
        if x < w:
            distances.append((w-x))
        else:
            break
    mn_i = np.argmin(distances)
    #return X[mn_i]
    return mn_i
    
def Interpolate(X,Y):
    Xnew = np.arange(X[0],X[-1],0.5)
    f = interp1d(X,Y)
    Ynew = f(Xnew)
#     print 'interpolated: ',len(Xnew),len(Ynew)
    return Xnew,Ynew


def WNtoWL(WN,WLcenter):
    '''center must be center wavelength, in nm'''
    a = 1/WLcenter
    WL = np.zeros(len(WN))
    for i in range(len(WN)):
        b = WN[i]  / (10**7)
        c = a - b
        WL[i] = 1/c
    return WL


def WLtoWN(X,lam):
    '''converts an array of wavelengths to an array of wavenumbers representing Raman shift from laser wavelength lam [nm].
    l must taken from center of laser peak. 
    dw = ( (1/l) - (1/x) ) * (10**7 nm / 1 cm)'''
    a = 1.0/lam
    C = np.zeros(len(X))
    for i in range(len(X)):
        b = 1.0/X[i]
        C[i] = (a - b) * 10**7
    # print('center lambda = ',lam)
    # d = fit.FindIndexX(0,C) is the one in fit.py different?
    d = FindIndexX(0,C)
    # print('0 WN index: ',d)
    # print('WL = ',X[(d-5):(d+5)])
    # print('WN = ',C[(d-5):(d+5)])
    return C 

def Lam(RamanShift,lam0):
    '''Takes a single Raman Shift, returns a single wavelength.'''
    units = 10**-7 #(nm/cm)
    den = (1.0/lam0) - RamanShift * units
    wl = 1.0 / den
    return wl

def CheckSpectIm(d):
    '''Check whether spectrum or image. s = dataset[nam], d=s[:]'''
    h = np.shape(d)
    #print('shape:',h)
    if len(h) == 1:
        a = 'spectrum'
    elif len(h) == 2:
        if h[0] == 512:
            a= 'image'
        else:
            a = 'frames'
    else:
        a= 'what is this group?'
    return a 

def CreateColorsList(length,colormap='gist_rainbow'):
    '''given the length of a list, creates a list of colors w same length. can also choose colormap.'''
    # import pylab
    colors = []
    cm = pylab.get_cmap(colormap)
    for i in range(length):
        color = cm(1.*i/length)  # color will now be an RGBA tuple
        colors.append(color)
    return colors #[::-1]


def butterLowpassFiltFilt(data, cutoff = 1500, fs = 60000, order=1): #cutoff = 1500, fs = 60000, order=1
    '''Smoothes data without shifting it'''
    nyq = 0.5 * fs
    normalCutoff = cutoff / nyq
    b, a = butter(order, normalCutoff, btype='low', analog=False)
    yFiltered = filtfilt(b, a, data)
    return yFiltered

def ReduceNoise(X,Y,notchedge,cutoff = 5000,testing=False):
    '''Use when applying to full spectrum--avoids troubles w nothc'''
    Y_filtered = Y.copy()
    edge = notchedge + 10#25
    if X[0] < 0:
        ai = FindIndexX(-edge,X)#,testing=False)#-33.9
        bi = FindIndexX(edge,X)#,testing=False)#33.9
        if testing:
            print('Reduce Noise',X[ai],X[bi])
        # if ai is None or bi is None:
        if np.isnan(ai) or np.isnan(bi):
            print('Did not find notch edge when trying to ReduceNoise. \n Returning unfiltered spectrum.')
            print(X[ai],X[bi])
        else:
            Y_filtered[:ai] = butterLowpassFiltFilt(Y[:ai], cutoff = cutoff, fs = 50000, order=1)
            Y_filtered[bi:] = butterLowpassFiltFilt(Y[bi:], cutoff = cutoff, fs = 50000, order=1)
    else:
        Y_filtered = butterLowpassFiltFilt(Y, cutoff = cutoff, fs = 50000, order=1)
    return Y_filtered

def PrintCurrentDateTime():
    now = datetime.datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("date and time =", dt_string)










# import matplotlib as mpl
# mpl.style.use('seaborn')
# def ChangeMplStyle(styl):
#     mpl.style.use(styl)
plt.style.use('classic')
def ChangeMplStyle(styl):
    plt.style.use(styl) 
    # https://tonysyu.github.io/raw_content/matplotlib-style-gallery/gallery.html


# plt.style.use('classic')

def MakeExponentialStr(num):
    ystr = str(num)
    if ystr[0] != '1':
        exp_str = ystr[0]+'*10'
    else:
        exp_str = '10'
    z = 0
    for n in ystr:
        if n == '0':
            z+=1
    if z > 1:
        exp_str += '$^{'+str(z)+'}$'
    else:
        exp_str = str(num)
    return exp_str

def joinStrings(separator, strings):
    #return reduce(lambda stringSoFar,s: stringSoFar + separator + s, strings)
    '''given list of strings, join with given separator (a string)'''
    new_string = ''
    for i in range(len(strings)):
        if i==0:
            new_string = strings[i]
        else:
            new_string = new_string + separator + strings[i]
    return new_string



def SmoothSpectrum(X,Y,window_length,polyorder,notchedge):
    '''The length of the filter window must be a positive, odd integer. polyorder must be less than window length. '''
    ai = FindIndexX(-notchedge,X)#,testing=False)
    bi = FindIndexX(notchedge,X)#,testing=False)
    # if ai is None or bi is None:
    if np.isnan(ai) or np.isnan(bi):
        print('Did not find notch edge when trying to ReduceNoise, so did not smooth.')
        return Y
    else:
        Y_smoothed = Y.copy()
        Y_smoothed[:ai] = Smooth(Y[:ai], window_length, polyorder) #savgol_filter(Y[:ai], window_length, polyorder, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
        Y_smoothed[bi:] = Smooth(Y[bi:], window_length, polyorder) #savgol_filter(Y[bi:], window_length, polyorder, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
        return Y_smoothed

# Fitting curves
def Lorentzian(X, amp, cen, wid):
    '''for wide and narrow lorenzian'''
    return (amp*wid**2/((X-cen)**2+wid**2))

def Line(X, m, b):
    '''  '''
    return ((m*np.array(X)) + b) 

def Line2(X,x1,y1,x2,y2):
    #m = (reference[end_i] - reference[start_i]) / (X[end_i] - X[start_i])
    m = (y2 - y1) / (x2 - x1)
    return (m*X) - (m*x1) + y1

def FitLine(X,Y,numpoints):
    '''m,b,Xnew,Y_linfit = FitLine(X,Y,numpoints) 
    ax.plot(Xnew,Y_linfit,label = 'fit: slope=%5.3f, y-intercept=%5.3f' % (m,b) )
    '''
    #p0= [0.3,295] # [m,b]
    popt,pcov = curve_fit(Line,X,Y)#,p0)
    #print popt

    Xnew = np.linspace(X[0],X[-1],numpoints)
    f = interp1d(X,Y)
    #Ynew = f(Xnew)

    Y_linfit = Line(Xnew, *popt)
    m = popt[0]
    b = popt[1]
    return m,b,Xnew,Y_linfit
# - - - - -

def FindOnePeak(X,Y,region):
    '''
    Y should probs already be smoothed.
    region = [x1,x2]. 
    returns mx_x, mx_y'''
    x1 = region[0]
    x2 = region[1]
    index1 = FindIndexX(x1,X)
    index2 = FindIndexX(x2,X)
    X_cut = X[index1:index2]
    Y_cut = Y[index1:index2]
    mx_i = np.argmax(Y_cut)
    mx_x = X_cut[mx_i]
    mx_y = Y_cut[mx_i]
    return mx_x,mx_y 

# * Should depend on resolution of X
def FindIndexX_old(w,X,testing=False):
    '''find index of element w in monotonic array X.
    advised to include print('given element', w,' does not occur within given array') if returned wi is None
    ''' 

    step_size = X[1] - X[0]
    if len(X) != 0:
        if round(X[-1],4) < round(w,4):
            if testing == True:
                print('given element', w,' does not occur within given array')
            else:
                pass
            iw = None 
        else:
            for i in range(len(X)):
                # print 'X[i]:',X[i]
                if round(X[i],4) >= round(w,4):
                    if i != 0:
                        lower = X[i-1]
                        higher = X[i]
                        # pick the closer point, i or i-1
                        if (w - lower) < (higher - w):
                            iw = i-1
                        else:
                            iw = i
                        #print 'i, X[i]: ',i, X[i]     
                        break
                else:
                    iw = i
    else:
        print('length of X is zero.')
        iw = None
    return iw



    
def Split(WN,spectrum,center):
    #print('Given center: '+str(center))
    center_i = FindIndexX(center,WN)
    #print 'Found center: ',WN[center_i]
#     print WN[1010:1030]
    AS_WN = np.array(WN[:center_i]) # [::-1]) * -1.        #        (?) 
    Stokes_WN = np.array(WN[center_i:] )
    AS_spectrum = np.array(spectrum[:center_i])
    Stokes_spectrum = np.array(spectrum[center_i:]) 
    return AS_WN,AS_spectrum,Stokes_WN,Stokes_spectrum

def Split2(WN,Y,center):
    AS_WN = []
    AS_Y = []
    S_WN = []
    S_Y = []
    for i in range(len(WN)):
        wn = WN[i]
        if wn < center:
            AS_WN.append(wn)
            AS_Y.append(Y[i])
        else:
            S_WN.append(wn)
            S_Y.append(Y[i])
    return np.array(AS_WN),np.array(AS_Y),np.array(S_WN),np.array(S_Y)

def Split3(WN,I,center):
    X = np.array(WN)
    Y = np.array(I)
    AS_X = X[X < center]
    S_X = X[X > center]
    AS_Y = Y[X < center]
    S_Y = Y[X > center]
    return AS_X,AS_Y,S_X,S_Y





# cmap = plt.colormaps['bwr']
def TruncateColormap(cmap,minval=0.,maxval=1.,n=100):
    new_cmap = clrs.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name,a=minval,b=maxval),
                                                      cmap(np.linspace(minval,maxval,n)))
    return new_cmap

def MyColormap(colors_list):
    '''Takes a list of colors from matplotlib named colors. https://matplotlib.org/2.1.2/gallery/color/named_colors.html#sphx-glr-gallery-color-named-colors-py 
    Creating a colormap from a list of colors can be done with the from_list() method of LinearSegmentedColormap. You must pass a list of RGB tuples that define the mixture of colors from 0 to 1.'''

    rgb_colors = []
    for color in colors_list:
        c = clrs.to_rgb(color)
        rgb_colors.append(c)

    n_bins = len(colors_list)  # Discretizes the interpolation into bins
    cmap_name = 'my_colormap'
    # Create the colormap
    my_colormap = clrs.LinearSegmentedColormap.from_list(cmap_name, rgb_colors, N=n_bins)
    return my_colormap


    # make N different colors
    #if isinstance(colormap, str):
    #    cm = plt.get_cmap(colormap) # 'gist_rainbow'
    #elif isinstance(colormap,list):
    #    cm=MyColormap(colormap)
    #else:
    #    cm = plt.get_cmap('rainbow')
    #cNorm  = colors.Normalize(vmin=0, vmax=N-1)
    #scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
    ##ax1.set_color_cycle([scalarMap.to_rgba(i) for i in range(N)])
    #ax1.set_prop_cycle('color',[scalarMap.to_rgba(i) for i in range(N)])




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



# def DeleteCosmicRay(i,x_more,side_more,sides_dict,peaks_dict):
#     '''Delete one cosmic ray at at time.'''
#     Y_more = sides_dict[side_more][1] # spectrum (Y)
#     Y_new = np.copy(Y_more)
#     j = peaks_dict[side_more][0][i] # index of ith cosmic ray
#     for k in range(1,11):
#         if (i-k) < 0:
#             # if past first point in cosmic rays list
#             # for ave, use 3 points to left of first point in costmic rays list
#             ai = j - (k+4)
#             bi = j - k
#             break
#         else:
#             if (j-k) in peaks_dict[side_more][0]:
#                 # if index (j-k) in list of indices of cosmic rays
#                 pass
#             else:
#                 # if it's not in list, then also check point to the left 
#                 if (j-(k+1)) in peaks_dict[side_more][0]:
#                     # if that point is in list, then still same cosmic ray
#                     pass
#                 else:
#                     # if both are not in list, then hopefully listtle region to the left is good for averaging
#                     ai = j - (k+4)
#                     bi = j - k
#                     break
#     if ai < 0:
#         ai = None
#     elif bi > len(Y_more):
#         bi = None
#     else:
#         pass
#     if bi < 0:
#         bi = 3
#     # ave = np.mean([Y_more[ai],Y_more[bi]])
#     ave = np.mean(Y_more[ai:bi])
#     # print('ave',ave)
#     ci = j-1
#     di = j+1
#     if ci < 0:
#         ci = None
#     elif di > len(Y_more):
#         di = None
#     else:
#         pass
#     if di < 2:
#         # print('di=',di)
#         di = 2
#     # print('ci=',ci,'di=',di)
#     Y_new[ci:di] = np.ones(2) * ave
#     sides_dict[side_more] = (sides_dict[side_more][0],Y_new)
#     return sides_dict            
        


# def RemoveCosmicRays(X,Y,error = 5.,f=1,plot=False,testing=False):
#     '''
#     '''
#     if plot == True:
#         fig,axes = plt.subplots(nrows=3, ncols=1,figsize=(5,5),sharex=True)
    
#     #region=[-50,50]
#     region = [-200,200]
#     # Ynorm = op.NormyNormSpecial(X,Y,region)
    
#     AS_X,AS_Y,Stokes_X,Stokes_Y = Split2(X,Y,0.) #Ynorm,0.)
#     # sides_dict = {'aS':(AS_X[:-10],AS_Y[:-10]),'Stokes':(Stokes_X[10:],Stokes_Y[10:])}
#     sides_dict = {'aS':(AS_X,AS_Y),'Stokes':(Stokes_X,Stokes_Y)}
#     for side_label in sides_dict.keys():
#         if len(sides_dict[side_label][0]) < 0.:
#             print('Cannot use this method to find cosmic rays, because need both aS and Stokes sides.')
            
    
#     peaks_dict = {}
    
#     # for side in aS and Stokes
#     for side_label in ['aS','Stokes']:#sides_dict.keys():
#         side = sides_dict[side_label]
#         # Normalize to below 50 cm-1
#         sideY_norm = op.NormyNormSpecial(side[0],side[1],region,testing=testing)
#         # Take 1st derivative 
#         deriv = savgol_filter(sideY_norm, 7, polyorder = 5, deriv=1, mode='nearest')
#         # Peaks are where derivative is steeper than 0.015
#         # <> and peak higher than 3 stds? 
#         std = np.std(sideY_norm[:10])
#         # print('std',std)
        
#         threshold = 0.5 #0.1
#         indices = np.where((deriv < -threshold) | (deriv > threshold))[0]   #0.015 #& (sideY_norm > 1*std)
#         # print('indices',indices)
#         if len(indices) > 0:
            
#             # PeakIndices2PeakCoords ignores peaks below 100 cm-1
#             peaks_i,peaks_x,peaks_y = PeakIndices2PeakCoords(indices,side[0],sideY_norm)#side[1])
#             # print('peaks_i',peaks_i)
            
#             if plot == True:
#                 axes[1].plot(side[0],deriv)
                
#                 axes[0].plot(side[0],sideY_norm)
#                 axes[0].scatter(peaks_x,peaks_y,marker='x',color='r',s=50)
        
#             # add these indices flagged as peaks to dictionary
#             if side_label == 'aS':
#                 peaks_dict['aS'] = (peaks_i,peaks_x,peaks_y)
#             else:
#                 aS_peaks_x = peaks_dict['aS'][1]
#                 peaks_dict['Stokes'] = (peaks_i,peaks_x,peaks_y)
                
#                 # find which side has more flagged peaks, aS or Stokes
#                 if len(aS_peaks_x) > len(peaks_x):
#                     # there may be a cosmic ray on aS side
#                     peaks_x_more = aS_peaks_x
#                     peaks_x_less = peaks_x
#                     side_more = 'aS'
#                 else:
#                     peaks_x_more = peaks_x
#                     peaks_x_less = aS_peaks_x
#                     side_more = 'Stokes'
                
#                 # # check if points in peaks_x_more are adjacent
#                 # # if so, get center and width, and delete this
#                 # # else delete each point
#                 # peaks_i_more = peaks_dict[side_more]
#                 # peaks_i_more_new = []
#                 # one_peak = []
#                 # for i in range(1,len(peaks_i_more)):
#                 #     one_peak.append()
#                 #     if np.abs((peaks_i_more[i-1]-1) - peaks_i_more[i]) <= 2:
#                 #         if i == 1:
#                 #             one_peak.append(peaks_i_more[i-1])
#                 #         one_peak.append(peaks_i_more[i])
#                 #     else:
#                 #         peaks_i_more_new = set()
#                 #         one_peak = []
                
#                 for i,x_more in enumerate(peaks_x_more):
#                     if len(peaks_x_less) > 0:
#                         for x_less in peaks_x_less: 
#                             # if any peak x position on the side w less peaks is within +/- 25 cm-1 of a peak on the side with more
#                             # then if it's within the error
#                             # so within +/- 5 cm-1 of the peak on the side w more peaks
#                             # then it's a real peak
#                             # else, Delete that peak from the side with more
#                             if (x_less - 25) < x_more < (x_less + 25):
#                                 # check if aS_x is in ballpark
#                                 if (x_less - error) < x_more < (x_less + error):
#                                     # peak occurs on both sides
#                                     pass
#                                 else:
#                                     sides_dict = DeleteCosmicRay(i,x_more,side_more,sides_dict,peaks_dict)
#                             else:
#                                 # peaks are not in same region of spectrum
#                                 pass
#                     else:
#                         sides_dict = DeleteCosmicRay(i,x_more,side_more,sides_dict,peaks_dict)
#                         # print('Deleting cosmic ray at',x_more)
#                         # Y_more = sides_dict[side_more][1]
#                         # Y_new = np.copy(Y_more)
#                         # j = peaks_dict[side_more][0][i]
#                         # ave = np.mean([Y_more[j-f],Y_more[j+f]])
#                         # Y_new[(j-f):(j+f)] = np.ones((f*2)) * ave
#                         # sides_dict[side_more] = (sides_dict[side_more][0],Y_new)
#         else:
#             sides_dict[side_label] = (sides_dict[side_label][0],sides_dict[side_label][1]) #np.zeros(len(sides_dict[side_label][0])
#             peaks_dict[side_label] = (np.array([]),np.array([]))
#             #pass
#     Ynew = np.concatenate((sides_dict['aS'][1],sides_dict['Stokes'][1]),axis=0)

#     if plot == True:
        
#         # axes[0].plot(X,Y)
#         axes[2].plot(X,Ynew,color='k')
#         axes[0].set(title='',ylim=[0,1.1])
        
#     return Ynew


# #                           ~c~c~c~c~







# ~ ~ ~ plot image ~ ~ ~

def PlotImage(x,im,L,expdate,colormap,y=None,logcmap=None,showticks=None,lims=None,ylabel=None,save=None,molecule=None):
    '''L is title. lims is clims = (min,max). figshape = (width,height) in inches (don't forget space for title).save is True or False.'''
    fig,ax = plt.subplots(nrows=1)
    shape = np.shape(im) # e.g. (600L, 2048L)
    if y is None:
        y = list(range(shape[0]))
    ax.set_ylim([y[0],y[-1]])
    ax.pcolormesh(x, y, im, cmap=colormap, clim=lims, shading='gouraud')# 'nearest' or 'auto')#, vmin=np.min(im), vmax=np.max(im)) # 'RdBu'
    print('min(x),max(x),min(y),max(y), min(im), max(im):',[np.min(x), np.max(x), np.min(y), np.max(y), np.min(im), np.max(im)])
    if logcmap:
    # https://matplotlib.org/users/colormapnorms.html
        if lims is None:
            pcm = ax.pcolor(x, y, im, 
                                norm=clrs.LogNorm(vmin=np.min(im), vmax=np.max(im)),
                                cmap=colormap)
        else:
            pcm = ax.pcolor(x, y, im, 
                                norm=clrs.LogNorm(vmin=lims[0], vmax=lims[1]),
                                cmap=colormap)
        fig.colorbar(pcm, ax=ax, extend='max', spacing='proportional', orientation='horizontal',label=ylabel,use_gridspec=True)
    else:
        pcm = ax.pcolor(x,y,im, cmap=colormap)
        fig.colorbar(pcm, ax=ax, extend='max', orientation='horizontal',aspect=50,label=ylabel,use_gridspec=True)
    plt.minorticks_on()
    if showticks == False:
        ax.tick_params(axis='both', labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    else:
        ax.tick_params(axis='both', labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    plt.show()












def GetExpdate(h5filename):
    date_list = h5filename.split('-')
    expdate = datetime.datetime(int(date_list[0]), int(date_list[1]), int(date_list[2].split('.')[0]))
    return expdate

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]

def FindCSVs(pth,subfolder):
    files = []
    csvfilepth = pth + subfolder #folder + 'ERSfit Bose+Gauss/CSVs/'
    #print(csvfilepth)
    for f in os.listdir(csvfilepth):
        try:
            if f.split('.')[1] == 'csv':
                files.append(f)
        except IndexError:
            print('theres a directory in this subfolder?')
    # print(files)
    return files


def CSVappend(expdate,counts_per_s_per_mW,grating,slit):
    '''append row to csv file containing counts/mW/s'''
    row = [expdate.strftime('%Y-%m-%d'),str(counts_per_s_per_mW),str(grating),str(slit)]
    with open('counts/counts.csv', 'a') as f:
        writer = csv.writer(f,delimiter=',')
        writer.writerow(row)


def LoadCSV(filename,heading=False):
    '''Output array: RamanShift, Intensity'''
    with open(filename) as f:
        RamanShifts = [] # cm-1
        Intensities = [] # normalized
        reader = csv.reader(f)
        if heading == True:
            next(reader)#reader.next() # skip headings
        for row in reader:
            wn = float(row[0])
            I = float(row[1])
            RamanShifts.append(wn)
            Intensities.append(I)
    return np.array(RamanShifts), np.array(Intensities)

def LoadCSV3cols(filename,heading=False):
    '''Output array: col1,col2,col3'''
    with open(filename) as f:
        col1 = [] 
        col2 = [] 
        col3 = []
        reader = csv.reader(f)
        if heading == True:
            next(reader)#reader.next() # skip headings
        for row in reader:
            c1 = float(row[0])
            c2 = float(row[1])
            c3 = float(row[2])
            col1.append(c1)
            col2.append(c2)
            col3.append(c3)
    return np.array(col1), np.array(col2), np.array(col3)

def LoadCSVManyCols(filename,heading=False):
    '''Output array: col1,col2,col3'''
    with open(filename) as f:
        reader = csv.reader(f)
        if heading == True:
            next(reader)#reader.next() # skip headings
        
        reader_lst = list(reader)
        
        cols_dict = {}
        for n in range(len(reader_lst[0])):
            cols_dict['col'+str(n)] = []

        for row in reader_lst:
#             print(row)
            for n,c in enumerate(row):
                cols_dict['col'+str(n)].append(float(c))
        return cols_dict


def exportasCSV2(x,y,fullcsvfilename):
    '''Export as CSV file. '.csv' is added by function.'''

    ensure_dir(fullcsvfilename)
    print(fullcsvfilename)
    with open(fullcsvfilename, 'w', newline='') as csvfile:
        Jwriter = csv.writer(csvfile)
        Jwriter.writerows(list(zip(x,y)))

def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = csv.writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)



def reformat_date(date_str):
    if '-' in date_str:
        date_lst = date_str.split('-')
        date_lst = [date_lst[2],date_lst[1],date_lst[0]]
        return '/'.join(date_lst)
    elif '/' in date_str:
        #date_lst = date_str.split('/')
        return date_str
    else:
        print('is this a date?',date_str)
        
    

def check_for_row(file_name,list_of_elem):
    '''check whether entry already in database. 
    not working yet:                                                                          * <>
    check all columns except b; want to be able to update/overwrite for new fits'''
    # make list_of_elem into str
    for i,value in enumerate(list_of_elem):
        if ~isinstance(value,str):
    #         print(value)
            list_of_elem[i] = str(value)
        if i==0:
            # if date
            list_of_elem[0] = reformat_date(value)
            
    str_of_elem  = ','.join(list_of_elem[:-1])
    
    duplicate = False
    with open(file_name, 'r') as f:
        for i,line in enumerate(f):
            #  skip header
            if i  >= 1:
                line_lst = line.strip().split(',')[:-1]
                line_lst[0] = reformat_date(line_lst[0])
                line_cut = ','.join(line_lst)
                # print('line       ',line_cut)
                # print('str_of_elem',str_of_elem,'\n \n')
                if str_of_elem == line_cut:
                    duplicate = True
    #                 print('line       ',line)
    #                 print('str_of_elem',str_of_elem,'\n \n')
                    break
        return duplicate


def SaveFig(p,expdate,title,molecule=None): 
    expdate_str = expdate.strftime("%Y-%m-%d")
    folder = str(expdate_str) + '/'
    print(folder)
    if 'Si' in title:
        #print('Si is in title.')
        filename = p + 'Si/' + expdate_str + '_' + title + '.png'
        ensure_dir(filename)
        plt.savefig(filename,format='png',bbox_inches='tight')
        filename = p + folder + expdate_str + '_' + title + '.png'
    elif 'BPT' in title:
        #print('BPT is in title.')
        filename = p + 'BPT/' + expdate_str + '_' + title + '.png'
        ensure_dir(filename)
        plt.savefig(filename,format='png',bbox_inches='tight')
        filename = p + folder + expdate_str + '_' + title + '.png'
    elif molecule != None:
        filename = p + molecule + '/' + expdate_str + '_' + title + '.png'
        ensure_dir(filename)
        plt.savefig(filename,format='png',bbox_inches='tight')
        filename = p + folder + expdate_str + '_' + title + '.png'
    else:
        filename = p + folder + expdate_str + '_' + title + '.png'
    print(filename)
    ensure_dir(filename)
    plt.savefig(filename,format='png',bbox_inches='tight',transparent=True)
