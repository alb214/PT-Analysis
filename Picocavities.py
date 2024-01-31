# -*- coding: utf-8 -*-
"""
Separate nanocavity and picocavity of a timescan


Created on Thu Dec  1 15:16:34 2022

@author: alb214
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
from basics import FindIndexX,ReduceNoise,ensure_dir,NormalizeTo1
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)


def MakeImFromTimescanDict(sample,particle,timescan_dict):
    print('Starting to make SERS image.')
    timestamps = list(timescan_dict[sample][particle].keys())
    im_list = []
    for time_str in timestamps:
        X,Y = timescan_dict[sample][particle][time_str]
        Y[Y<=0] = 0.1
        im_list.append(Y) #.tolist())
    im = np.array(im_list)
    return im

def MakeImFromTimescanDict2(particle,timescan_dict,typ='Exp'):
    print('Starting to make SERS image.')
    timestamps = list(timescan_dict[particle].keys())
    im_list = []
    for time_str in timestamps:
        if typ == 'before':
            Y = timescan_dict[particle][time_str]
        elif typ == 'ERS':
            Y,ERS = timescan_dict[particle][time_str]
        else:
            X,Y = timescan_dict[particle][time_str]
        #Y[Y<=0] = 0.1
        im_list.append(Y) #.tolist())
    im = np.array(im_list)
    return im


def MakeMaskedIm(im,num_stds=3,testing=False):
    masked_im = im.copy()
    # print('shape masked_im',np.shape(masked_im))
    for n in range(np.shape(im)[1]):
        # print('n',n)
        column = im[:,n]
        col_ave = np.mean(column)
        col_std = np.std(column)
        i=0
        for m in range(len(column)):
            if (col_ave - column[m]) < (-1*col_std*num_stds): # if col value is above col_ave
                # then mask row i
                masked_im[m,:] = [np.nan for n in range(np.shape(im)[1])]#[0 for n in range(np.shape(im)[1])]
                #print('masked row',m)
                i+=1
        if testing == True:
            if i > (len(column)/2):
                print('masked',str(i),'rows in column',str(n))
    return masked_im


def MakeNanocavityIm(masked_im,multiplier,deg,testing=False):
    #print('Starting to make nanocavity image at '+str(elapsed)+' seconds.')
    nanocavity_im = np.zeros(np.shape(masked_im))
    # - go through image one column (wavelength) at a time
    # m,n = row,column
    # np.shape() = m,n
    for n in range(np.shape(masked_im)[1]):
        column = masked_im[:,n] 
        # masked_column = masked_im[:,n] 
        X = range(len(column))
        # - remove outliers for fit
        # half = int(np.shape(masked_im)[0] / 2) # half num rows
        # halfcolumn = masked_im[:half,n]
        average = np.mean(column) #halfcolumn)
        standard_deviation = np.std(column) #halfcolumn)
        mn = np.min(column)
#         average = np.mean(masked_column)
#         #mn = np.min(column)
#         standard_deviation = np.std(masked_column)
        column_to_fit = column #np.zeros(len(column))
        count=0
        for i in range(len(column)):
            if np.abs(column[i] - average) > (standard_deviation * multiplier):
                column_to_fit[i] = mn
                count+=1
            else:
                column_to_fit[i] = column[i]
        if testing == True:
            if count > (len(column)/2): #30:
                print('-                       -                         -')
                print('replaced',count,'out of',len(column),'points in column',n)#,'of',npom)
                print('-                       -                         -')
        eps = 10**-2
        np.nan_to_num(column_to_fit,nan=eps,copy=False)
        coeff2 = np.polyfit(X, column_to_fit, deg)
        poly2 = np.poly1d(coeff2)
        poly_fit = poly2(X)
        for i in range(len(poly_fit)):
            if poly_fit[i] < 0.:
                poly_fit[i] = eps
        nanocavity_im[:,n] = poly_fit
    return nanocavity_im


def AverageOfRows(im):
    total = np.zeros(len(im[0]))
    # n=0
    for row in im:
        total += row
    average = total / np.shape(im)[0]
    return average


def PlotPicocavities(Gen_Wn,sample,npom,Exp_dict,pico_dict,subfolder,multiplier,degree,num_stds=3,vmin=100,testing=False):
    '''subfolder = pth + all subfolders
    num_stds=3
    '''
    im = MakeImFromTimescanDict(sample,npom,Exp_dict)        
    masked_im = MakeMaskedIm(im,num_stds=num_stds,testing=testing)
    mn = np.amin(im)
    mx = np.amax(im)
    nanocavity_im = MakeNanocavityIm(masked_im,multiplier,degree,testing=testing)
    picocavity_im = np.subtract(im,nanocavity_im) #+ 10**4
    picocavity_im_new = picocavity_im.copy()
    picocavity_im_new[picocavity_im_new <= 0.] = mn
    if sample not in pico_dict.keys():
        pico_dict[sample] = {}
    pico_dict[sample][npom] = {}
    pico_dict[sample][npom]['im'] = im
    pico_dict[sample][npom]['nanocavity im'] = nanocavity_im
    pico_dict[sample][npom]['picocavity im'] = picocavity_im_new
    fig,axes = plt.subplots(nrows=2, ncols=2,figsize=(10,5),sharex=True,sharey=True)
    axes[0,0].pcolormesh(Gen_Wn, range(np.shape(im)[0]), im, cmap='plasma',shading='nearest',
                         norm=clrs.LogNorm(vmin=vmin, vmax=mx))
    axes[0,1].pcolormesh(Gen_Wn, range(np.shape(masked_im)[0]), masked_im, cmap='plasma',shading='nearest',
                         norm=clrs.LogNorm(vmin=vmin, vmax=mx))
    axes[1,0].pcolormesh(Gen_Wn, range(np.shape(nanocavity_im)[0]), nanocavity_im, cmap='plasma',shading='nearest',
                         norm=clrs.LogNorm(vmin=vmin, vmax=mx))
    add = vmin 
    axes[1,1].pcolormesh(Gen_Wn, range(np.shape(picocavity_im_new)[0]), picocavity_im_new, cmap='plasma',shading='nearest',
                         norm=clrs.LogNorm(vmin=add, vmax=mx))
    average_nano = AverageOfRows(nanocavity_im)
    axes[1,0].plot(Gen_Wn,(NormalizeTo1(average_nano)*10)+5,color='c',lw=1)

    titles = ['SERS','SERS masked','nanocavity','picocavity']
    for i,ax in enumerate(axes.reshape(-1)):
        ax.set(title=titles[i],xlim=[-1600,1300],ylim=[-1,31])
    plt.tight_layout()
    return pico_dict


def PlotPicocavities2(Gen_Wn,sample,npom,timescans_dict,typ,pico_dict,subfolder,multiplier,degree,num_stds=3,vmin=100,mask=True,testing=False):
    '''subfolder = pth + all subfolders
    num_stds=3
    '''
    im = MakeImFromTimescanDict2(npom,timescans_dict,typ)  
    print(np.shape(im))
    if mask:      
        masked_im = MakeMaskedIm(im,num_stds=num_stds,testing=testing)
    else:
        masked_im = im
    mn = np.amin(im)
    mx = np.amax(im)
    nanocavity_im = MakeNanocavityIm(masked_im,multiplier,degree,testing=testing)
    picocavity_im = np.subtract(im,nanocavity_im) 
    picocavity_im_new = picocavity_im
    if sample not in pico_dict.keys():
        pico_dict[sample] = {}
    pico_dict[sample][npom] = {}
    pico_dict[sample][npom]['im'] = im
    pico_dict[sample][npom]['nanocavity im'] = nanocavity_im
    pico_dict[sample][npom]['picocavity im'] = picocavity_im_new
    fig,axes = plt.subplots(nrows=2, ncols=2,figsize=(10,5),sharex=True,sharey=True)
    axes[0,0].pcolormesh(Gen_Wn, range(np.shape(im)[0]), im, cmap='plasma',shading='nearest',
                         norm=clrs.LogNorm(vmin=vmin, vmax=mx))
    axes[0,1].pcolormesh(Gen_Wn, range(np.shape(masked_im)[0]), masked_im, cmap='plasma',shading='nearest',
                         norm=clrs.LogNorm(vmin=vmin, vmax=mx))
    axes[1,0].pcolormesh(Gen_Wn, range(np.shape(nanocavity_im)[0]), nanocavity_im, cmap='plasma',shading='nearest',
                         norm=clrs.LogNorm(vmin=vmin, vmax=mx))
    add = vmin 
    axes[1,1].pcolormesh(Gen_Wn, range(np.shape(picocavity_im_new)[0]), picocavity_im_new, cmap='plasma',shading='nearest',
                         norm=clrs.LogNorm(vmin=add, vmax=mx))
    average_nano = AverageOfRows(nanocavity_im)
    axes[1,0].plot(Gen_Wn,(NormalizeTo1(average_nano)*10)+5,color='c',lw=1)
    titles = ['SERS','SERS masked','nanocavity','picocavity']
    for i,ax in enumerate(axes.reshape(-1)):
        ax.set(title=titles[i],xlim=[-1600,1300],ylim=[-1,31])
    plt.tight_layout()
    return pico_dict

def PlotNanocavity(nanocavity_im,title,):
    fig,ax = plt.subplots(nrows=1, ncols=1,figsize=(10,5))
    indices = range(np.shape(nanocavity_im)[1])
    ax.pcolormesh(indices, range(np.shape(nanocavity_im)[0]), nanocavity_im, cmap='plasma',shading='nearest',
                         #norm=clrs.LogNorm(vmin=vmin, vmax=mx)
                         )
    average_nano = AverageOfRows(nanocavity_im)
    ax.plot(indices,(NormalizeTo1(average_nano)*(np.shape(nanocavity_im)[0]/3))+1,color='c',lw=1)
    # ax.set(xlim=[-1600,1300])#,ylim=[-1,31])
    ax.set(title=title)
    
    
    
    
# Figure 4

def Figure4(X,PTsample,pico_dict,particle,exposure_time,pth,title,notchedge=5,save=True):
    start_row = 10
    end_row = None
    im = pico_dict[PTsample][particle]['im'][start_row:end_row]
    nanocavity_im = pico_dict[PTsample][particle]['nanocavity im']#[start_row:]
    picocavity_im = pico_dict[PTsample][particle]['picocavity im'][start_row:end_row]
    # TIMESCAN
    t0 = start_row*exposure_time
    if end_row is not None:
        tf = end_row*exposure_time
    else:
        tf = np.shape(im)[0]*exposure_time
    print('start time:',t0)
    Time = []
    t = t0
    for i in range(np.shape(im)[0]):
        Time.append(t)
        t+=exposure_time
    mask = np.abs(X) < notchedge#23.
    im_dl_lst = []
    for row in im:
        row[mask] = np.ones(len(np.delete(mask, np.where(mask ==False)))) * 0.1
        im_dl_lst.append(row)
    im_dl = np.array(im_dl_lst)
    nano_ave = AverageOfRows(nanocavity_im)
    nano_ave_filtered = ReduceNoise(X,nano_ave,notchedge=1,cutoff = 5000)
    nano_ave_mskd = np.ma.masked_array(nano_ave_filtered, mask=mask)
    fig,axes = plt.subplots(nrows=3, ncols=3,figsize=(6.8*cm,6.8*cm),sharex='col',sharey='row',
                            squeeze=True,
                            #constrained_layout=True,
                            gridspec_kw={'height_ratios': [1., 1, 2],'width_ratios': [1, 2, 1]}
                           )
    fig.subplots_adjust(hspace=0.05,wspace=0.05)  # adjust space between axes
    mid = 360 #199
    xlim_col0 = [-1150,-1*mid]
    xlim_col1 = [-1*mid,mid]
    xlim_col2 = [mid,1150]
    ai = FindIndexX(-1*mid,X)
    bi = FindIndexX(mid,X)
    colorz = 'Oranges'#'bone_r'#'inferno'
    mn = 150 
    mx = np.amax(im)
    pcm = axes[0,0].pcolormesh(X[:ai], Time, im_dl[:,:ai], cmap=colorz, shading='nearest', 
                               norm=clrs.LogNorm(vmin=mn, vmax=mx))#vmin=0, vmax=mx) #
    axes[0,0].tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,
                          labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    pcm = axes[0,1].pcolormesh(X[ai:bi], Time, im_dl[:,ai:bi], cmap=colorz, shading='nearest', 
                               norm=clrs.LogNorm(vmin=mn, vmax=mx))#vmin=0, vmax=mx) #
    # fig.colorbar(pcm, ax=axes[:,:],location='left',shrink=0.15,aspect=5)
    axes[0,1].tick_params(axis='both',which='both', bottom=False,top=False,left=False,right=False,
                          labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    pcm = axes[0,2].pcolormesh(X[bi:], Time, im_dl[:,bi:], cmap=colorz, shading='nearest', 
                               norm=clrs.LogNorm(vmin=mn, vmax=mx))#vmin=0, vmax=mx) #
    axes[0,2].tick_params(axis='both',which='both', bottom=False,top=False,left=False,right=False,
                          labelbottom=False, labeltop=False, labelleft=False, labelright=True)
    axes[0,0].set(ylim=[t0,tf])
    L = list(range(int(t0),int(tf+1),10))
    axes[0,2].set_yticks(L)
    L_labels = [str(l) for l in L]
    axes[0,2].set_yticklabels(L_labels)
    print(L_labels)
    axes[1,0].plot(X[:ai],nano_ave_mskd[:ai],color='k')#,label='Nanocavity average spectrum')
    axes[1,1].plot(X[ai:bi],nano_ave_mskd[ai:bi],color='k')
    axes[1,2].plot(X[bi:],nano_ave_mskd[bi:],color='k')
    for c in [0,1,2]:
        axes[1,c].tick_params(axis='both',which='both',bottom=True,top=False,left=False,right=False,
                              labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    axes[1,0].set(ylim=[-10,3*10**3])

    # PICOCAVITY
    Y0 = picocavity_im[10]
    mx = np.max(Y0)
    offset = 0
    offsetincrement = mx / 2. 
    c=0
    H = []
    time_increment = 10
    for m in range(np.shape(picocavity_im)[0]):
        if m % 2 == 0:
            label = str(m)
        else:
            label = None
        Y_unmasked = picocavity_im[m,:] 
        Y_mskd = np.ma.masked_array(Y_unmasked, mask=mask)
        Y = ReduceNoise(X,Y_mskd,notchedge=1,cutoff = 5000)
        if m % time_increment == 0:
            Yoffset = Y+offset
            H.append(Yoffset[-1])
        axes[2,0].plot(X[:ai],Y[:ai]+offset,label=label,color='k')#colors[c])
        axes[2,1].plot(X[ai:bi],Y[ai:bi]+offset,label=label,color='k')#colors[c])
        axes[2,2].plot(X[bi:],Y[bi:]+offset,label=label,color='k')#colors[c])
        offset += offsetincrement
        c+=1

    for c in [0,1]:
        axes[2,c].tick_params(axis='both',which='both',bottom=True,top=False,left=False,right=False,
                              labelbottom=True, labeltop=False, labelleft=False, labelright=False)
    axes[2,2].tick_params(axis='both', which='both',bottom=True,top=False,left=False,right=True,
                          labelbottom=True, labeltop=False, labelleft=False, labelright=True)
    axes[2,0].set(ylim=[-1,0.75*10**5])
    axes[2,2].set_yticks(H)
    H_labels = [str(int(t)) for t in Time][::time_increment]
    axes[2,2].set_yticklabels(H_labels)
    axes[2,2].yaxis.tick_right()
    for r in [0,1,2]:
        axes[r,0].set_xlim(xlim_col0)
        axes[r,1].set_xlim(xlim_col1)
        axes[r,2].set_xlim(xlim_col2)
    axes[2,0].xaxis.set_major_locator(MultipleLocator(500))
    axes[2,0].xaxis.set_major_formatter(FormatStrFormatter('%d'))
    axes[2,1].xaxis.set_major_locator(MultipleLocator(200))
    axes[2,1].xaxis.set_major_formatter(FormatStrFormatter('%d'))
    axes[2,1].xaxis.set_minor_locator(MultipleLocator(100))
    axes[2,2].xaxis.set_major_locator(MultipleLocator(500))
    axes[2,2].xaxis.set_major_formatter(FormatStrFormatter('%d'))
    # hide the spines 
    axes[1,0].spines['right'].set_visible(False)
    axes[1,1].spines['right'].set_visible(False)
    axes[1,1].spines['left'].set_visible(False)
    axes[1,2].spines['left'].set_visible(False)
    axes[2,0].spines['right'].set_visible(False)
    axes[2,1].spines['right'].set_visible(False)
    axes[2,1].spines['left'].set_visible(False)
    axes[2,2].spines['left'].set_visible(False)
    d = .75  
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=6,#12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    for r in [0,1,2]:
        axes[r,0].plot([1], [0], transform=axes[r,0].transAxes, **kwargs)
        axes[r,1].plot([0, 1], [0,0], transform=axes[r,1].transAxes, **kwargs)
        axes[r,2].plot([0], [0], transform=axes[r,2].transAxes, **kwargs)
    if save:
        figname = pth+particle+' '+title+'.png'
        print(figname)
        ensure_dir(figname)
        fig.savefig(figname,format='png',bbox_inches='tight',transparent=True,dpi=300)
    plt.show()
    
    
cm = 1/2.54  # centimeters in inches
def Figure4_Stokes(X,PTsample,pico_dict,particle,exposure_time,pth,title,notchedge=5,save=True):
    start_row = 0
    end_row = 20
    im = pico_dict[PTsample][particle]['im'][start_row:end_row]
    nanocavity_im = pico_dict[PTsample][particle]['nanocavity im']#[start_row:]
    picocavity_im = pico_dict[PTsample][particle]['picocavity im'][start_row:end_row]
    mask = np.abs(X) < notchedge
    im_dl_lst = []
    for row in im:
        row[mask] = np.ones(len(np.delete(mask, np.where(mask ==False)))) * 0.1
        im_dl_lst.append(row)
    im_dl = np.array(im_dl_lst)
    nano_ave = AverageOfRows(nanocavity_im)
    nano_ave_filtered = ReduceNoise(X,nano_ave,notchedge=1,cutoff = 5000)
    nano_ave_mskd = np.ma.masked_array(nano_ave_filtered, mask=mask)
    fig,axes = plt.subplots(nrows=3, ncols=2,figsize=(6.8*cm,6.8*cm),sharex='col',sharey='row',
                            squeeze=True,
                            #constrained_layout=True,
                            gridspec_kw={'height_ratios': [1., 1, 2],'width_ratios': [1, 1]}
                           )
    fig.subplots_adjust(hspace=0.05,wspace=0.05) 
    mid = 360 
    xlim_col0 = [0,mid]
    xlim_col1 = [mid,1150]
    ai = FindIndexX(0,X)
    bi = FindIndexX(mid,X)
    # TIMESCAN
    Time = []
    t = 0
    for i in range(np.shape(im)[0]):
        Time.append(t)
        t+=exposure_time
    colorz = 'inferno'#'bone'
    mn = 150 #50 #10**2
    mx = np.amax(im)
    pcm = axes[0,0].pcolormesh(X[ai:bi], Time, im_dl[:,ai:bi], cmap=colorz, shading='nearest', 
                               norm=clrs.LogNorm(vmin=mn, vmax=mx))#vmin=0, vmax=mx) #
    axes[0,0].tick_params(axis='both', labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    pcm = axes[0,1].pcolormesh(X[bi:], Time, im_dl[:,bi:], cmap=colorz, shading='nearest', 
                               norm=clrs.LogNorm(vmin=mn, vmax=mx))#vmin=0, vmax=mx) #
    # fig.colorbar(pcm, ax=axes[:,:],location='left',shrink=0.15,aspect=5)
    axes[0,1].tick_params(axis='both', labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    axes[1,0].plot(X[ai:bi],nano_ave_mskd[ai:bi],color='k')#,label='Nanocavity average spectrum')
    axes[1,1].plot(X[bi:],nano_ave_mskd[bi:],color='k')

    # PICOCAVITY
    Y0 = picocavity_im[10]
    mx = np.max(Y0)
    offset = 0
    offsetincrement = mx / 2.
    c=0
    H = []
    time_increment = 10
    for m in range(np.shape(picocavity_im)[0]):
        if m % 2 == 0:
            label = str(m)
        else:
            label = None
        Y_unmasked = picocavity_im[m,:] #_unmasked
    #     Y = np.ma.masked_where(((WN<7)&(WN>-7)), Y_unmasked) 
        Y_mskd = np.ma.masked_array(Y_unmasked, mask=mask)
        Y = ReduceNoise(X,Y_mskd,notchedge=1,cutoff = 5000)
        if m % time_increment == 0:
            Yoffset = Y+offset
            H.append(Yoffset[-1])
        axes[2,0].plot(X[:ai],Y[:ai]+offset,label=label,color='k')
        axes[2,1].plot(X[ai:bi],Y[ai:bi]+offset,label=label,color='k')
        offset += offsetincrement
        c+=1
    axes[0,0].set(ylim=[0,Time[-1]])
    axes[0,1].yaxis.tick_right()
    axes[2,1].set(#ylim=[0.1,40000],#yscale='log',
        #xlim=xlim,
        #title = particle,
    #     xlabel = 'Raman Shift (cm$^{-1}$)'
    )    
    for ax in axes[1:,0]:
        ax.set_ylim(bottom=-1)
    axes[2,0].xaxis.set_major_locator(MultipleLocator(100))
    axes[2,0].xaxis.set_major_formatter(FormatStrFormatter('%d'))
    axes[2,1].xaxis.set_major_locator(MultipleLocator(500))
    axes[2,1].xaxis.set_major_formatter(FormatStrFormatter('%d'))
    axes[2,1].xaxis.set_minor_locator(MultipleLocator(250))
    # hide the spines 
    axes[1,0].spines['right'].set_visible(False)
    axes[1,1].spines['right'].set_visible(False)
    axes[1,1].spines['left'].set_visible(False)
    axes[1,2].spines['left'].set_visible(False)
    axes[2,0].spines['right'].set_visible(False)
    # hide the ticks
    axes[1,0].tick_params(axis='both', labelbottom=True, labeltop=False, labelleft=False, labelright=False, bottom=True, top=False, left=False, right=False)
    axes[1,1].tick_params(axis='both', labelbottom=True, labeltop=False, labelleft=False, labelright=False, bottom=True, top=False, left=False, right=False)
    axes[2,0].tick_params(axis='both', labelbottom=True, labeltop=False, labelleft=False, labelright=False, bottom=True, top=False, left=False, right=False)
    axes[2,1].tick_params(axis='both', labelbottom=True, labeltop=False, labelleft=False, labelright=False, bottom=True, top=False, left=False, right=False)
    # / / 
    d = .75  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    for r in [0,1,2]:
        axes[r,0].plot([1], [0], transform=axes[r,0].transAxes, **kwargs)
        axes[r,1].plot([0, 1], [0,0], transform=axes[r,1].transAxes, **kwargs)
        # axes[r,2].plot([0], [0], transform=axes[r,2].transAxes, **kwargs)
    # /  /
    if save:
        figname = pth+particle+' '+title+'.png'
        print(figname)
        # ensure_dir(figname)
        # fig.savefig(figname,format='png',bbox_inches='tight',transparent=True,dpi=300)
    plt.show()
