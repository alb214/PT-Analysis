# -*- coding: utf-8 -*-

import ERSremoval as ers
from basics import FindIndexX, CreateColorsList, check_for_row, append_list_as_row, ReduceNoise, Smooth, ensure_dir

import importlib,sys
importlib.reload(sys.modules['basics'])

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator,FormatStrFormatter,AutoMinorLocator)
# from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import random
import math

# import statsmodels.api as smapi
# from statsmodels.formula.api import ols

cm = 1/2.54  # centimeters in inches
plt.rcParams["font.family"] = 'Times New Roman'#'Calibri'
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({
    "text.usetex": False,
})


def SubtractExpFit(X,Y,exp_fit,notchedge,testing=False):
    # 1. Find Intersetion of Y and ERS, if notchedge < 20 cm-1
    num = 10.# 20.
    bi = FindIndexX(0,X)
    ci = FindIndexX(num,X)
    if np.isnan(bi) or np.isnan(ci):
        # notchedge is larger than 20 cm-1
        notched = True
        i_right = FindIndexX((notchedge+1),X)
        if testing:
            print('didnt find 0 bi={}, {} ci={}'.format(bi,num,ci))
            print('i_right',i_right)
    else:
        notched = False
        # Stokes
        Nu_right = np.linspace(X[bi],X[ci],100)
        i_mx = np.argmax(np.interp(Nu_right,X[bi:ci],Y[bi:ci]))
        i_right = FindIndexX(Nu_right[i_mx],X)
        #diff_right = np.abs(np.subtract(np.interp(Nu_right,X[bi:ci],Y[bi:ci]),np.interp(Nu_right,X[bi:ci],exp_fit[bi:ci])))
        #i_right = FindIndexX(Nu_right[np.argmin(diff_right)],X)
        # print('left',WN[ai:bi][np.argmin(diff)],'right',WN[bi:ci][np.argmin(diff)])
        #print('left',WN[i_left],'right',WN[i_right])
        if testing:
            print('0={}, Stokes intersect={}, {}={}'.format(X[bi],X[i_right],num,X[ci]))
    if testing:
        print('Subtract exp from Stokes region above {}'.format(X[i_right]))
    # 2. Segment into two regions, below and above itnersection point. Only subtract exp_fit from Stokes side
    Y_aS = Y[:i_right]
    Y_Stokes = Y[i_right:]
    exp_Stokes = exp_fit[i_right:]
    # 3. Subtract (Y-exp_fit) in Stokes section
    Ynew_Stokes = np.subtract(Y_Stokes,exp_Stokes)
    # 5. Concatenate the two sections together
    Ynewnewnew = np.concatenate((Y_aS,Ynew_Stokes),axis=0)
    if len(Ynewnewnew) != len(Y):
        print('We have a problem in SubtractERS.')

    return Ynewnewnew





def FitExponential(folder,name,bg_pnts,full,eps,testing=False):
    '''inputs must be arrays
    full = (Stokes from notch edge to Stokes cutoff)
    bg_pnts = (2 bg pnts)
    
    This is written to only work for Stokes side. Would need to change bounds, and ...
    '''
    # 1. ln(Stokes spectrum bg minima)
    Stokes_X,Stokes_Y = bg_pnts
    new_f = np.log(Stokes_Y)# + eps)

    
    # 2.
    bounds1 = ((-np.inf,0),(0.,2*10**1)) # slope, y-intercept
    fit,pcov = curve_fit(LinearLine,Stokes_X, new_f,p0=None,bounds=bounds1)
    lin_fit = np.poly1d(fit)#LinearLine2(Stokes_WN,b,B)     
    
    # 3.
    # Use curve_fit result as initial guess for simulated annealing
    b0,lnB0 = fit #huber.coef_, huber.intercept_ #
     
    # simulated annealing
    initial_state = [b0,lnB0]
    lnB_bound1 = lnB0 - 1.
    lnB_bound2 = lnB0 + 1.
    bounds2 = np.array([[b0-np.abs(b0*1.),-10**-4],[lnB_bound1,lnB_bound2]])
    if testing:
        print('bounds2 lnB',lnB_bound1,lnB_bound2)
        print('bounds2 B',np.exp(lnB_bound1),np.exp(lnB_bound2))
    solution = simulated_annealing(initial_state,bounds2,full,testing=testing)
    if solution is None:    
        b,lnB = b0,lnB0
        title = 'curve_fit b=%e' % b + ', lnB=%e' % lnB
        print('Did not use sim anneal fit for',name) 
    else:
        b,lnB = solution
        lin_fit_sa = np.poly1d(solution)
        title = 'sim annealing fit b=%e' % b + ', lnB=%e' % lnB  
    B = np.exp(lnB) #- eps

    if testing == True:
        print('curve_fit fit: b={:.3e},lnB={:.3e}-->B={:3e}'.format(b0,lnB0,np.exp(lnB0)))
        print('final fit: b={:.3e},lnB={:.3e}-->B={:3e}'.format(b,lnB,B))
        fig,axes = plt.subplots(nrows=2, ncols=1,figsize=(7,4),sharex=True)
        fig.suptitle('fit exponential')
        axes[0].scatter(Stokes_X,new_f,c='k',label='new_f',s=50)
        axes[0].plot(Stokes_X,lin_fit(Stokes_X),color='m',linewidth=3,label='curve_fit')
        if solution is not None:
            axes[0].plot(Stokes_X,lin_fit_sa(Stokes_X),color='r',linewidth=3,label='simulated_annealing')
        axes[0].set(title=title,xlim=[0,Stokes_X[-1]+10])#,ylim=[-1,1.5])
        axes[0].legend(loc='best')
    
        exp_fit = Exponential(Stokes_X,b,B) #- eps
        axes[1].scatter(Stokes_X,Stokes_Y,label='bg_points',c='k')
        axes[1].plot(Stokes_X,exp_fit,color='r',label='exp_fit')
        #axes[1].set(ylim=None)#[0,2])
        axes[1].set_ylim(bottom=-2)
        axes[1].legend(loc='best')
        
        plt.tight_layout()
        plt.show()
    return b,B


def LineFromTwoPoints(X,x1,y1,x2,y2):
    m = (y2-y1)/(x2-x1)
    Y = []
    for x in X:
        Y.append(m*(x-x1)+y1)
    return np.array(Y)

def LinearLine(X,m,y0):
    Y = [(m*x)+y0 for x in X]
    return np.array(Y)

def Exponential(X,b,B):
    E = np.zeros(len(X))
    for i in range(len(X)):
        E[i] = B * np.exp(b*X[i])
    return E

def CheckForNansOrInfs(X,Y):
    for i in range(len(Y)): 
        if Y[i] is np.nan:
            print(X[i],Y[i],'is nan in')
            allgood = False
        elif Y[i] == np.inf:
            print(X[i],Y[i],'is inf in ')
            allgood = False
        else:
            # print('No nans or infs in given Y')
            allgood = True
    return allgood

def SpectrumWithoutPeaks(X,Y,Stokes_cutoff,peak_params,notchedge,ylim,plot=False):
    '''Provide spectrum before it's shifted down
    
    Stokes_cutoff is highest wavenumber I'll fit w exponential, so its the end i shift to 0.
    
    region is for ASto0. It's the region of AS spectrum to average as bg height, region = [wn1,wn2]
    ^ not using region anymore, not using ASto0 anymore
    '''
    testing = plot
    upperbound = Stokes_cutoff #250.
    if isinstance(Y, np.ndarray):
        pass
    else:
        Y = np.array(Y)
    Y_full_filtered = ReduceNoise(X,Y,1,cutoff = 2000)#,notchedge=10.)
    Mn_x = []
    Mn_y = []
    # 2. anti-Stokes: 
    # 2. a. notch edge
    ai = FindIndexX(-15.,X)
    bi = FindIndexX(-3.,X) 
    notch_i = np.argmax(Y_full_filtered[ai:bi]) #
    notch_l_x = X[ai:bi][notch_i]
    notch_i = FindIndexX(notch_l_x,X)
    if testing:
        print('2.a. aS notch edge (x,y)',X[notch_i],Y_full_filtered[notch_i])
    # 2. b. min btwn -50 cm-1 and notch left edge
    x1 = notch_l_x
    y1 = np.max(Y_full_filtered[ai:bi])
    x2 = -50
    ai = FindIndexX(x2,X)
    y2 = Y_full_filtered[ai]
    X_cut = X[ai:notch_i]
    Y_cut = Y_full_filtered[ai:notch_i] # -50 to notch_l
    # Draw a line from -50 to notch_l_x
    line = LineFromTwoPoints(X_cut,x1,y1,x2,y2)
    # Point furthest from line is minimum
    delta = abs(Y_cut - line)
    mx_i = np.argmax(delta)
    mn_x = X_cut[mx_i]
    mn_y = Y_cut[mx_i]
    if mn_y < 0.:
        ai = FindIndexX(-60,X)
        bi = FindIndexX(-50,X)
        mn_i = np.argmin(Y_full_filtered[ai:bi])
        mn_x = X[ai:bi][mn_i]
        mn_i = FindIndexX(mn_x,X)
        mn_x = X[mn_i]
        mn_y = Y_full_filtered[mn_i]
    if testing:
        print('2.b. aS mn near notch x,y',mn_x,mn_y)
    Mn_x.append(mn_x)
    Mn_y.append(mn_y)
    
    # 2. c. min btwn upperbound and -50
    di = FindIndexX((-1*upperbound), X)
    mn_x = X[di:ai][np.argmin(Y_full_filtered[di:ai])]
    mn_y = np.min(Y_full_filtered[di:ai])
    Mn_x.append(mn_x)
    Mn_y.append(mn_y)
    if testing:
        print('mn btwn -cutoff={} and -50={}: {}'.format(X[di],X[ai],mn_x))
    ai = FindIndexX(3.,X) 
    bi = FindIndexX(15.,X)  
    notch_i = np.argmax(Y_full_filtered[ai:bi])
    notch_r_x = X[ai:bi][notch_i]
    notch_r_y = Y_full_filtered[ai:bi][notch_i]
    notch_i = FindIndexX(notch_r_x,X)   

    x1 = notch_r_x
    y1 = np.max(Y_full_filtered[ai:bi])
    x2 = 50
    ai = FindIndexX(x2,X)
    y2 = Y_full_filtered[ai]
    X_cut = X[notch_i:ai]
    Y_cut = Y_full_filtered[notch_i:ai]
    # Draw a line from notch_r_x to 50
    line = LineFromTwoPoints(X_cut,x1,y1,x2,y2)
    # Point furthest from line is minimum
    delta = abs(Y_cut - line)
    mx_i = np.argmax(delta)
    mn_x = X_cut[mx_i]
    mn_y = Y_cut[mx_i]
    if mn_y < 0.:
        ai = FindIndexX(50,X)
        bi = FindIndexX(60,X)
        mn_i = np.argmin(Y_full_filtered[ai:bi])
        mn_x = X[ai:bi][mn_i]
        mn_i = FindIndexX(mn_x,X)
        mn_x = X[mn_i]
        mn_y = Y_full_filtered[mn_i]
    if testing:
        print('2.b. Stokes mn near notch x,y',mn_x,mn_y)
    Mn_x.append(mn_x)
    Mn_y.append(mn_y)
    # 3. c.  min btwn notch right edge and upperbound cm-1 #500 cm-1
    di = FindIndexX(upperbound, X)
    mn_x = X[ai:di][np.argmin(Y_full_filtered[ai:di])]
    mn_y = np.min(Y_full_filtered[ai:di])
    Mn_x.append(mn_x)
    Mn_y.append(mn_y)    
    if testing:
        print('mn btwn 50={} and cutoff={}: {}'.format(X[ai],X[di],mn_x))
    # 4. 
    redflag = False
    for i,pnt_y in enumerate(Mn_y):
        if pnt_y <= 0:
            print('({},{}) is below zero'.format(Mn_x[i],pnt_y))
            redflag = True
    # ~    ~    ~    ~    ~    ~    ~    ~    ~    ~    ~    ~    ~    ~   
    if plot == True:  
        fig,axes = plt.subplots(nrows=2, ncols=1,figsize=(4,4),sharex=True)
        # fig.suptitle()
        axes[0].plot(X,Y_full_filtered,color='k',ls=':',label='Y orig')    
        axes[0].scatter(Mn_x,Mn_y,c='grey',marker='o')
        # axes[0].scatter(Mn_x,Mn_y_shifted,c='b',marker='o')
        axes[0].set(title='for SpectrumWithoutPeaks in ExpRemoval',xlim=[-550,550],ylim=ylim,yscale='log') #[-10,max(Stokes_spectrum_shifted)]

        axes[1].plot(X,Y_full_filtered,label='Y_full_filtered')
        # axes[1].plot(X,Y_full_filtered_shftd,label='Y_full_filtered_shifted')
        axes[1].legend(loc='best',fontsize=8)
        axes[1].set(ylim=ylim,yscale='linear')
        
        for ax in axes:
            ax.xaxis.set_major_locator(MultipleLocator(100))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax.xaxis.set_minor_locator(MultipleLocator(50))
            # axes[1].xaxis.set_minor_formatter(FormatStrFormatter('%d'))
            ax.grid(b=True, which='minor', axis='x', color='gainsboro', linestyle='-')
        
        plt.show()
    # ~    ~    ~    ~    ~    ~    ~    ~    ~    ~    ~    ~    ~    ~  
    return np.array(Mn_x),np.array(Mn_y),redflag

def RemoveExponential(pth,folder,key,WN,Ynewnew,peak_params,ylim,plot=False,testing=False): #eps = 0.
    '''input the output WN,Ynew from CalculateERS.
    peak_params and ylim are for SpectrumWithoutPeaks3.
        peak_params = prominence,base width,height
    eps is for FitExponential.

    Fits to Stokes bg points btwn 0 and Stokes cutoff. 
    '''
    # 1. Get two points on each side of VHG notch for exponential fit
    print('***',key,'***')
    notchedge = 9.#5.
    Stokes_cutoff = 200. #250. # cm-1
    X_bg,Y_bg,redflag = SpectrumWithoutPeaks(WN,Ynewnew,Stokes_cutoff,peak_params,notchedge,ylim,plot=testing) 
    # 2 points for fit
    if redflag:
        print('What do I do about negative points?')
    # a. two Stokes pnts
    Stokes_Mn_x = X_bg[X_bg>0]
    Stokes_Mn_y = Y_bg[X_bg>0]
    if testing == True:
        print('the Stokes points x:',Stokes_Mn_x)
        print('the Stokes points y:',Stokes_Mn_y)      
    bg_pnts = (Stokes_Mn_x,Stokes_Mn_y)
    # 2. Cut full spectrum to Stokes region from (notch edge) to Stokes cutoff
    Ynewnew_fltrd = ReduceNoise(WN,Ynewnew,5,cutoff = 5000,testing=False)
    ai = FindIndexX((notchedge+4),WN)    
    bi = FindIndexX(Stokes_cutoff,WN)
    Stokes_full = (WN[ai:bi],Ynewnew_fltrd[ai:bi])
    if testing:
        print('Cutting full spectrum to region from {} to {}:'.format(notchedge,Stokes_cutoff),WN[ai],WN[bi])
    # 3. Fit exponential
    eps = 0.
    b,B = FitExponential(folder,key,bg_pnts,Stokes_full,eps,testing=testing)
    exp_fit = Exponential(WN,b,B) 
    fit_params = (b,B)
    # 4. Subtract fit
    Ynewnewnew = SubtractExpFit(WN,Ynewnew,exp_fit,notchedge,testing=testing)
    if plot == True:
        color = '#273db8' 
        Title = 'Fitting Exponential to '+key
        figsize = (4,4)
        fig,axes = plt.subplots(nrows=2, ncols=1,figsize=figsize,sharex=True)
        #fig.suptitle('RemoveExponential')
        axes[0].plot(WN,Ynewnew,color='k',ls=':',alpha=1.,label='ERS-removed',zorder=1)
        axes[0].plot(WN,exp_fit,color='r',label='exponential fit',zorder=1)
        axes[0].scatter(X_bg,Y_bg,c='k',s=10,zorder=2)
        axes[1].plot(WN,Ynewnew,color='k',ls='-',alpha=1.,label='ERS-removed')
        axes[1].plot(WN,np.zeros(len(WN)),color='y',ls='--')
        axes[1].plot(WN,Ynewnewnew,color=color,lw=2,label='Exp and ERS removed')
        xlim = ers.GetXLim(WN)
        axes[0].set(title=Title,xlim=xlim)
        txt = 'b=%.2f, B=%.1f' % tuple(fit_params)
        axes[0].text(0.9,0.9, txt,
        horizontalalignment='right',
        verticalalignment='top',
        transform=axes[0].transAxes)
        lgnd = True
        major = 500
        minor = 100
        axes[1].xaxis.set_major_locator(MultipleLocator(major))
        axes[1].xaxis.set_major_formatter(FormatStrFormatter('%d'))
        axes[1].xaxis.set_minor_locator(MultipleLocator(minor))
        axes[1].grid(b=True, which='minor', axis='x', color='gainsboro', linestyle='-')
        for ax in axes:
            ax.set(ylim=[0.01,50000],yscale='log')
            ax.set_yticks([10**-1,10**1,10**3])
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            ax.tick_params(axis='both', which='major', pad=2)
            ax.xaxis.labelpad = 0.1#0.5
            if lgnd:
                ax.legend(loc='upper left',fontsize=9,frameon=False,framealpha=0.)
        fig.set_tight_layout(True)  

        figfolder = pth + folder + 'Expfit/'
        ensure_dir(figfolder)
        figname = figfolder + key + '_2fittingExponential' + '.png'
        print(figname)
        #fig.savefig(figname,format='png',bbox_inches='tight',transparent=False)
        plt.show()
    return WN,Ynewnewnew, fit_params 



# ~ ~ ~ ~ ~ Simulated Annealing ~ ~ ~ ~ ~
def simulated_annealing(initial_state,bounds,full,testing=False):
    """Peforms simulated annealing to find a solution"""
    X,Y = full
    # - COOLING SCHEDULE - 
    initial_temp = 10**2 
    final_temp = 10**-3
    alpha1 = 0.97 #0.96 
    alpha2 = 0.997    
    current_temp = initial_temp
    # - - 
    # Start by initializing the current state with the initial state
    solution = initial_state #current_state
    if testing == True:
        temp = []
        bs = []
        lnBs = []
        neighbor_go = []
        solution_costs = []
    CD = []
    CDoT = []
    i=0
    j=0
    while current_temp > final_temp:
        if i<5000: 
            neighbor = get_neighbors(solution,bounds,X) 
            if testing == True:
                if (final_temp >= current_temp*alpha1):
                    if testing == True:
                        print('final temp: {}, current temp*alpha1: {}'.format(final_temp,(current_temp*alpha1)))
                    cost_testing = True
                    num = 2
                elif (current_temp == initial_temp):
                    cost_testing = True
                    num = 1
                else:
                    cost_testing = False
                    num = 0
            else:
                cost_testing = False
                num = 0
            solution_cost = get_cost(solution,X,Y,num,testing=cost_testing)
            if (j==0) and (solution_cost != np.inf):
                j+=1
                cost0 = solution_cost
            elif (j==0) and (solution_cost == np.inf):
                cost0 = 1.
            else:
                pass
            neighbor_cost = get_cost(neighbor,X,Y,num,testing=False)
            cost_diff = (solution_cost/cost0) - (neighbor_cost/cost0)
            # if no solution found, then start annealing again
            if (final_temp >= current_temp*alpha1) and (solution_cost is np.inf):
                print('Re-starting annealing, bc cooled without finding a solution (cost=inf)')
                current_temp = initial_temp  
            # if cost_diff is positive (solution cost > neightbor cost),
            # so if the new solution is better, accept it
            if cost_diff > 0.0:
                solution = neighbor
                if testing == True:
                    neighbor_go.append(1)       
            else:
                if math.isnan(cost_diff): 
                    # if both infinite, go to neighbor
                    solution = neighbor
                    if testing == True:
                        neighbor_go.append(1)
                        # move to neightbor anways bc at least then you get a different region to choose from
                else:
                    if testing:
                        CD.append(cost_diff)
                        CDoT.append(math.exp(cost_diff / current_temp))
                    if random.uniform(0, 1) < math.exp(cost_diff / current_temp): 
                        solution = neighbor
                        if testing == True:
                            neighbor_go.append(1)
                    else:
                        solution = solution  
                        if testing == True:
                            neighbor_go.append(0)
        else:
            print('max iterations exceeded.')
            if solution_cost is np.inf:
                solution = None                                                    
            break
        if testing == True:
            temp.append(current_temp)
            bs.append(solution[0])
            lnBs.append(solution[1])
            solution_costs.append(solution_cost)
        i+=1        
        if current_temp > 0.1: 
            current_temp *= alpha1
        elif current_temp < 0.01: 
            current_temp *= alpha1
        else:
            current_temp *= alpha2
    if testing == True:
        print('num iterations:',i)
        print('solution cost:',solution_cost)
        r = 2
        fig2,axes = plt.subplots(nrows=r, ncols=2,figsize=(5,5),sharex=True)
        fig2.suptitle('Simulated annealing')#'Neighbor options')
        axes_list = axes.reshape(-1)
        axes_list[0].plot(temp,solution_costs,label='cost',color='g') #range(i)
        axes_list[0].set(title='solution cost',ylim=None,xlabel='Temp',xlim=[temp[0],temp[-1]])#[0.1,10**7],yscale='log')
        axes_list[1].plot(temp,neighbor_go,label='neighbor go',color='g')
        axes_list[1].set(title='go to neightbor? (Yes=1)',ylim=[-0.1,1.1])        
        axes_list[2].plot(temp,bs,label='b',color='g')
        axes_list[2].set(title='b',ylim=bounds[0])
        axes_list[3].plot(temp,lnBs,label='lnB',color='g')
        axes_list[3].set(title='lnB',ylim=bounds[1])
        for ax in axes_list:
            ax.set(xscale='log')
            ax.set_xticks([10**2,10**0,10**-2])
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            ax.tick_params(axis='both', which='major', pad=2)
            #axes_list[a].yaxis.labelpad = 0.1#0.5
            ax.xaxis.labelpad = 0.1#0.5
        fig2.set_tight_layout(True)  
        plt.show()
        plt.tight_layout()

    return solution

def get_cost(state,X,Y,num,testing=False):
    """Calculates cost of the argument state for your solution.
    full = Stokes from notch edge to Stokes cutoff, filtered
    cost = root mean square error
    """
    b,lnB = state
    B = np.exp(lnB)
    exp_fit = Exponential(X,b,B)
    Ynewnewnew = np.subtract(Y,exp_fit) 
    summation = 0
    for i in range(len(X)):
        x = X[i]
        s = ((Y[i] - exp_fit[i])**2) / Y[i]            
        summation += s
    rmse = np.sqrt((1/len(X)) * summation) 
    if testing == True:
        if num == 1:
            title = 'initial guess'
        elif num == 2:
            title = 'final solution'
        else:
            title = ''
        figsize = (4,4)
        fig,axes = plt.subplots(nrows=2, ncols=1,figsize=figsize,sharex=True) #(2,2) 
        axes[0].scatter(X,Y,color='k',marker='.',label='Stokes from notchedge to Stokes_cutoff, filtered')
        axes[0].plot(X,exp_fit,color='r',label='exponential fit')
        axes[1].plot(X,Ynewnewnew,color='g',lw=2,label='Y - exp_fit',zorder=2)#,ls=':')
    neg_area = 0.
    num_neg_points = 0
    nnp = 10 #10**2
    na = 100 # allow for noise
    for i in range(len(Ynewnewnew)):
        x = X[i]
        I = Ynewnewnew[i]
        # if (15. < np.abs(x) < 100.) and (I < 0.):
        #     neg_area += np.abs(I)*100
        if I < 0.:
            if x < 100:
                num_neg_points += nnp/3
            else:
                num_neg_points += 1
            neg_area += np.abs(I)
            if testing == True:
                axes[1].bar(x,I,color='r',edgecolor='r')
        else:
            pass
    if (num_neg_points > nnp) or (neg_area > na): #50.: #5.: #2.5: #5.:
        cost = np.inf
    else:
        cost = rmse
    if testing == True:
        xlim = [0,X[-1]]
        yscale1 = 'linear'
        axes[0].set(title=title,yscale=yscale1)#,ylim=ylim0)
        # annotation
        txt = 'b={:.2e},lnB={:.2e}'.format(b,lnB)
        axes[0].text(0.9,0.9, txt,
        horizontalalignment='right',
        verticalalignment='top',
        transform=axes[0].transAxes)   
        ylim1=[-np.max(Ynewnewnew),np.max(Ynewnewnew)]
        axes[1].set(xlim=xlim,ylim=ylim1,yscale='linear')
        txt = 'cost:{:.1f}, neg area:{:.1f}, \n num neg pnts:{}'.format(cost,neg_area,num_neg_points)
        axes[1].text(0.9,0.9,txt,
        horizontalalignment='right',
        verticalalignment='top',
        transform=axes[1].transAxes)
        axes[1].xaxis.set_major_locator(MultipleLocator(100))
        axes[1].xaxis.set_major_formatter(FormatStrFormatter('%d'))
        axes[1].xaxis.set_minor_locator(MultipleLocator(50))
        for ax in axes:
            ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0)) # scientific notation
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            ax.tick_params(axis='both', which='major', pad=2)
            #ax.yaxis.labelpad = 0.1#0.5
            ax.xaxis.labelpad = 0.1#0.5
        fig.set_tight_layout(True)  
        plt.show()

    return cost
# def get_cost_ln(state,X,Y,num,testing=False):


def get_neighbors(state,bounds,WN):
    """Returns neighbors of the argument state for your solution."""
    num_points = 10 
    num_segments = 5
    sizes = bounds[:, 1] - bounds[:, 0]
    segment_sizes = sizes / num_segments
    halfwidths = segment_sizes / 2.
    param_names = ['b','lnB']
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



def RMSE(Y,fit):
    
    summation = 0
    for i in range(len(Y)):
        s = ((Y[i] - fit[i])**2)             
        summation += s
    #mse = (1/len(WN)) * summation
    rmse = np.sqrt((1/len(Y)) * summation) 
    return rmse

def RemoveOutliers(X,Y,testing=False):
    print('len X in RemoveOutliers',len(X))
    X_cut = X[1:-2]
    Y_cut = Y[1:-2]
    rmses = []
    for i in range(len(X_cut)):
        X_cut2 = np.delete(X_cut,i)
        Y_cut2 = np.delete(Y_cut,i)
        fit,pcov = curve_fit(LinearLine,X_cut2, Y_cut2) #,p0=None,bounds=bounds1)
        lin_fit = np.poly1d(fit)
        rmse = RMSE(Y_cut2,lin_fit(X_cut2))
        rmses.append((rmse,lin_fit,X_cut2,Y_cut2))
    best_fit = min(rmses)
    lin_fit = best_fit[1]
    X_new = best_fit[2]
    Y_new = best_fit[3]
    X_final = np.insert(X_new,0,X[0])
    Y_final = np.insert(Y_new,0,Y[0])
    X_final = np.append(X_final,[X[-2],X[-1]])
    Y_final = np.append(Y_final,[Y[-2],Y[-1]])
    
    if testing == True:
        fig,ax = plt.subplots(nrows=1, ncols=1,figsize=(5,5))
        fig.suptitle('Remove outliers')
        ax.scatter(X,Y,color='grey')
        ax.scatter(X_final,Y_final,color='b')
        ax.plot(X_cut,lin_fit(X_cut))
        plt.show()
    return X_final,Y_final




    
