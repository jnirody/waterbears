## This code plots ipsilateral vs contralateral phase difference for each leg.

## The plots made from this code appear in: Figure 4d, Figure 5b, Figure S6a

###########################################################################
#!/usr/bin/python
import re, math, sys, os, random
import numpy as np
from matplotlib import collections  as mc
import pandas as pd
from optparse import OptionParser
import matplotlib.pyplot as plt
import glob, csv
from scipy.stats import mode
import seaborn as sns
import scipy
from scipy.optimize import curve_fit
from scipy import stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import kde
from pylab import *
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib import cm
###########################################################################
def grouper(iterable,n):
    args = [iter(iterable)]*n
    return zip(*args)
###########################################################################
def gauss(x,mu,sigma,A,c):
    return A*exp(-(x-mu)**2/2/sigma**2) + c
###########################################################################
stiffness = '10'

upperdir = '/'.join(os.getcwd().split('/')[:-1])
datadir = upperdir + '/Data/ForAnalysis/' + stiffness + 'kPa/Top/Individual/ByStride/'
files = glob.glob(datadir + '*.csv')
avg_file = upperdir + '/Data/ForAnalysis/' + stiffness + 'kPa/Top/bear_averages.csv'
avgdata = pd.read_csv(avg_file)
byframe_dir = upperdir + '/Data/ForAnalysis/' + stiffness + 'kPa/Top/Individual/ByFrame/'

framerate = 50. # frames/sec
resolution = 0.55 # um / pixel (5.5 um/pixel Basler at 10x mag)


rel_swing_lengths = [[] for i in range(6)]
rel_stance_lengths = [[] for i in range(6)]

rel_swing_lengths[5] = [np.mean((avgdata['L1_swing']/avgdata['L1_period']).to_list()),np.std((avgdata['L1_swing']/avgdata['L1_period']).to_list())]
rel_stance_lengths[5] = [np.mean((avgdata['L1_stance']/avgdata['L1_period']).to_list()),np.std((avgdata['L1_stance']/avgdata['L1_period']).to_list())]
rel_swing_lengths[4] = [np.mean((avgdata['R1_swing']/avgdata['R1_period']).to_list()),np.std((avgdata['R1_swing']/avgdata['R1_period']).to_list())]
rel_stance_lengths[4] = [np.mean((avgdata['R1_stance']/avgdata['R1_period']).to_list()),np.std((avgdata['R1_stance']/avgdata['R1_period']).to_list())]
rel_swing_lengths[3] = [np.mean((avgdata['L2_swing']/avgdata['L2_period']).to_list()),np.std((avgdata['L2_swing']/avgdata['L2_period']).to_list())]
rel_stance_lengths[3] = [np.mean((avgdata['L2_stance']/avgdata['L2_period']).to_list()),np.std((avgdata['L2_stance']/avgdata['L2_period']).to_list())]
rel_swing_lengths[2] = [np.mean((avgdata['R2_swing']/avgdata['R2_period']).to_list()),np.std((avgdata['R2_swing']/avgdata['R2_period']).to_list())]
rel_stance_lengths[2] = [np.mean((avgdata['R2_stance']/avgdata['R2_period']).to_list()),np.std((avgdata['R2_stance']/avgdata['R2_period']).to_list())]
rel_swing_lengths[1] = [np.mean((avgdata['L3_swing']/avgdata['L3_period']).to_list()),np.std((avgdata['L3_swing']/avgdata['L3_period']).to_list())]
rel_stance_lengths[1] = [np.mean((avgdata['L3_stance']/avgdata['L3_period']).to_list()),np.std((avgdata['L3_stance']/avgdata['L3_period']).to_list())]
rel_swing_lengths[0] = [np.mean((avgdata['R3_swing']/avgdata['R3_period']).to_list()),np.std((avgdata['R3_swing']/avgdata['R3_period']).to_list())]
rel_stance_lengths[0] = [np.mean((avgdata['R3_stance']/avgdata['R3_period']).to_list()),np.std((avgdata['R3_stance']/avgdata['R3_period']).to_list())]
com = []

phases = []
just_contra  = []

ipsi_phases = [[] for i in range(4)]
contra_phases = [[] for i in range(3)]

for file in files:
    s = '-'
    bear = s.join(file.split('/')[-1].split('_')[0:-1])
    video = file.split('/')[-1].split('-')[-1].split('.')[0]
    dataframe = pd.read_csv(file)
    
    grouped = dataframe.groupby('video',sort=False)
    trial = -1
    for video,data in grouped:
        trial += 1
        print('bear', bear, 'trial', video)
        fbf_file = pd.read_csv(byframe_dir + bear + '_framebyframe.csv')
        if trial > 0:
            subtract = min(np.where(np.array(fbf_file['video']==video))[0])-1
        else:
            subtract = 0
        swing_starts = [[] for i in range(6)]
        stance_starts = [[] for i in range(6)]
        swing_interval_start = []
        swing_interval_end = []
        stance_interval_start = []
        stance_interval_end = []
        # get out all the stance and swing start times to compute out relatives
        swing_starts[5] = list(map(int,data['L1_swing_start'][data['L1_swing_start'] > 0].to_list()))
        stance_starts[5] = list(map(int,data['L1_stance_start'][data['L1_stance_start'] > 0].to_list()))
        swing_starts[4] = list(map(int,data['R1_swing_start'][data['R1_swing_start'] > 0].to_list()))
        stance_starts[4] = list(map(int,data['R1_stance_start'][data['R1_stance_start'] > 0].to_list()))
        swing_starts[3] = list(map(int,data['L2_swing_start'][data['L2_swing_start'] > 0].to_list()))
        stance_starts[3] = list(map(int,data['L2_stance_start'][data['L2_stance_start'] > 0].to_list()))
        swing_starts[2] = list(map(int,data['R2_swing_start'][data['R2_swing_start'] > 0].to_list()))
        stance_starts[2] = list(map(int,data['R2_stance_start'][data['R2_stance_start'] > 0].to_list()))
        swing_starts[1] = list(map(int,data['L3_swing_start'][data['L3_swing_start'] > 0].to_list()))
        stance_starts[1] = list(map(int,data['L3_stance_start'][data['L3_stance_start'] > 0].to_list()))
        swing_starts[0] = list(map(int,data['R3_swing_start'][data['R3_swing_start'] > 0].to_list()))
        stance_starts[0] = list(map(int,data['R3_stance_start'][data['R3_stance_start'] > 0].to_list()))
        
        for leg in range(6):
            if len(swing_starts[leg]) < 2:
                continue
            swing_interval_start.append(swing_starts[leg][0])
            swing_interval_end.append(swing_starts[leg][1])
            for i in range(1,len(swing_starts[leg])-2):
                swing_interval_start.append(swing_starts[leg][i])
                swing_interval_end.append(swing_starts[leg][i+1])
            if len(stance_starts[leg]) < 2:
                continue
            stance_interval_start.append(stance_starts[leg][0])
            stance_interval_end.append(stance_starts[leg][1])
            for i in range(1,len(stance_starts[leg])-2):
                stance_interval_start.append(stance_starts[leg][i])
                stance_interval_end.append(stance_starts[leg][i+1])
                
            for j in range(len(swing_starts[(leg+2)%6])): #stride
                if leg%2 == 0:
                    multiplier = 1
                else:
                    multiplier = -1
                if j > len(swing_starts[leg+1*multiplier])-1:
                    continue
                k = np.where(np.array(swing_interval_start) < swing_starts[(leg+2)%6][j]+1)[0]
                if len(k) == 0:
                    continue
                k = max(k)
                if swing_starts[(leg+2)%6][j] > (swing_interval_end[k]+1):
                    continue
                j2 = np.where(np.array(swing_starts[leg+1*multiplier]) > (swing_interval_start[k]-1))[0]
                if len(j2) == 0:
                    continue
                j2 = min(j2)
                if swing_starts[leg+1*multiplier][j2] > (swing_interval_end[k]+1):
                    continue
                curr_interval = [swing_interval_start[k],swing_interval_end[k]]
                curr_video = fbf_file[fbf_file['video']==video]
                start_idx = np.where(np.array(curr_video['frame'])==swing_interval_start[k])[0]
                contra_phase = (float(swing_starts[leg+1*multiplier][j2]-curr_interval[0]))/ (curr_interval[1]-curr_interval[0])
                ipsi_phase = (float(swing_starts[(leg+2)%6][j]-curr_interval[0]))/ (curr_interval[1]-curr_interval[0])
                end_idx = np.where(np.array(curr_video['frame'])==swing_interval_end[k])[0]
                start_pos = list(map(float,list(curr_video['center_pos'][start_idx+subtract])[0][1:-1].split(',')))
                end_pos = list(map(float,list(curr_video['center_pos'][end_idx+subtract-1])[0][1:-1].split(',')))
                stride_dist = np.sqrt((start_pos[0]-end_pos[0])**2 + (start_pos[1]-end_pos[1])**2)*resolution
                stride_speed = stride_dist/(np.abs(end_idx-start_idx))*framerate
                if leg < 4:
                    ipsi_phases[int(leg)].extend([(stride_speed[0],ipsi_phase)])
                if leg%2 == 0:
                    just_contra.extend([contra_phase])
                    contra_phases[int(leg/2)].extend([(stride_speed[0],contra_phase)])
                phases.extend([(contra_phase,ipsi_phase)])

legs = ['R3','L3','R2','L2','R1','L1']

fig, ax = plt.subplots(2,2)
for j in range(len(ipsi_phases)):
    leg1 = legs[j]
    leg2 = legs[int((j+2)%6)]
    x,y = np.array(ipsi_phases[j]).T
    nbins = 50
    k = kde.gaussian_kde(np.array(ipsi_phases[j]).T)
    xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    phis = ax[int(j/2),int(j%2)].contourf(xi, yi, zi.reshape(xi.shape),vmin=0,vmax=0.025,cmap='Blues')
    ax[int(j/2),int(j%2)].plot(*zip(*ipsi_phases[j]),marker='.',markersize=5,color='k',alpha=0.8,linestyle='')
    N = 20
    s = [i[1] for i in sorted(ipsi_phases[j])]
    q = np.convolve(s, np.ones((N,))/N, mode='valid')
    g = np.linspace(0,300,len(q))
    ax[int(j/2),int(j%2)].plot(g,q,color='red',linewidth=3,alpha=0.8)
    ax[int(j/2),int(j%2)].set_title(leg1 + ' > ' + leg2 )
    ax[int(j/2),int(j%2)].set_xlim([50,300])
    ax[int(j/2),int(j%2)].set_ylim([0,1])
norm = mpl.colors.Normalize(vmin=0, vmax=0.025)
cmap = cm.get_cmap('Blues', nbins)
m = cm.ScalarMappable(cmap=cmap, norm=norm)
#fig.colorbar(m, ax=ax.flat)
plt.tight_layout()
plt.savefig(upperdir + '/Writeup/Figures/' + stiffness + 'kPa_ipsiphases_vs_speed.pdf')

f = 0
for j in range(len(contra_phases)):
    f += len(contra_phases[j])
print(f)
f = 0
for j in range(len(ipsi_phases)):
    f += len(ipsi_phases[j])
print(f)

fig,ax = plt.subplots(1,3, figsize=(9, 3))
for j in range(len(contra_phases)):
    leg1 = legs[j*2]
    leg2 = legs[(j*2)+1]
    x,y = np.array(contra_phases[j]).T
    nbins = 50
    k = kde.gaussian_kde(np.array(contra_phases[j]).T)
    xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    phis = ax[j].contourf(xi, yi, zi.reshape(xi.shape),vmin=0,vmax=0.025,cmap='Blues',alpha=0.7)
    ax[j].plot(*zip(*contra_phases[j]),marker='.',markersize=5,color='k',alpha=0.8,linestyle='')
    N = 50
    s = [i[1] for i in sorted(contra_phases[j])]
    q = np.convolve(s, np.ones((N,))/N, mode='valid')
    g = np.linspace(0,300,len(q))
    ax[j].plot(g,q,color='red',linewidth=3,alpha=0.8)
    ax[j].set_title(leg1 + ' > ' + leg2 )
    ax[j].set_xlim([50,300])
    ax[j].set_ylim([0,1])
norm = mpl.colors.Normalize(vmin=0, vmax=0.025)
cmap = cm.get_cmap('Blues', nbins)
m = cm.ScalarMappable(cmap=cmap, norm=norm)
fig.colorbar(m, ax=ax.flat, orientation='horizontal', aspect=40)
plt.savefig(upperdir + '/Writeup/Figures/' + stiffness + 'kPa_contraphases_vs_speed.pdf')

fig, ax = plt.subplots()
x,y = np.array(phases).T
nbins = 250
k = kde.gaussian_kde(np.array(phases).T)
xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))
phis = plt.contourf(xi, yi, zi.reshape(xi.shape),cmap='viridis')
fig.colorbar(phis, ax=ax)
plt.plot(0.33,0.33,'*',markersize=15, color='white')
plt.plot(0.66,0.33,'*',markersize=15, color='white')
plt.xlabel(r'$\phi_C$ (normalized phase)', fontsize=15)
plt.ylabel(r'$\phi_I$ (normalized phase)', fontsize=15)
plt.savefig(upperdir + '/Writeup/Figures/' + stiffness + 'kPa_phiVphi_nodots.pdf')


fig, ax = plt.subplots()
x,y = np.array(phases).T
nbins = 250
k = kde.gaussian_kde(np.array(phases).T)
xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))
phis = plt.contourf(xi, yi, zi.reshape(xi.shape),cmap='viridis_r')
fig.colorbar(phis, ax=ax)
plt.plot(0.33,0.33,'*',markersize=15,color='white')
plt.plot(0.5,0.5,'*',markersize=15,color='white')
plt.plot(0.5,0.33,'*',markersize=15,color='white')
plt.xlabel(r'$\phi_C$ (normalized phase)', fontsize=15)
plt.ylabel(r'$\phi_I$ (normalized phase)', fontsize=15)
plt.savefig(upperdir + '/Writeup/Figures/' + stiffness + 'kPa_phiVphi.pdf')
