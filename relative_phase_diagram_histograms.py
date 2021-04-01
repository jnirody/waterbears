## This code looks at relative phases between ipsilateral leg pairs and plots the following things: (1) histograms + PDFs testing ipsilateral swing -> swing suppression; (2) CDFs testing ipsilateral swing -> swing suppression; (3) histograms + PDFs testing ipsilateral stance -> swing facilitation; (4) CDFs testing ipsilateral stance -> swing facilitation.

## The plots made from this code appear in: Figures 4a,b,c

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
###########################################################################
def grouper(iterable,n):
    args = [iter(iterable)]*n
    return zip(*args)
###########################################################################
def gauss(x,mu,sigma,A,c):
    return A*exp(-(x-mu)**2/2/sigma**2) + c
###########################################################################
stiffness = '50'

upperdir = '/'.join(os.getcwd().split('/')[:-1])
datadir = upperdir + '/Data/ForAnalysis/' + stiffness + 'kPa/Top/Individual/ByStride/'
files = glob.glob(datadir + '*.csv')
avg_file = upperdir + '/Data/ForAnalysis/' + stiffness + 'kPa/Top/bear_averages.csv'
avgdata = pd.read_csv(avg_file)

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

rel_stance_starts = [[] for i in range(6)]
rel_swing_starts = [[] for i in range(6)]

for file in files:
    s = '-'
    bear = s.join(file.split('/')[-1].split('-')[0:-1])
    video = file.split('/')[-1].split('-')[-1].split('.')[0]
    dataframe = pd.read_csv(file)
    
    grouped = dataframe.groupby('video')
    for video,data in grouped:
        print('bear', bear, 'trial', video)
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
        
        for leg in range(4):
            if len(swing_starts[leg]) < 2:
                continue
            swing_interval_start.append(swing_starts[leg][0])
            swing_interval_end.append(swing_starts[leg][1])
            for i in range(1,len(swing_starts[leg])-1):
                swing_interval_start.append(swing_starts[leg][i])
                swing_interval_end.append(swing_starts[leg][i+1])
            if len(stance_starts[leg]) < 2:
                continue
            stance_interval_start.append(stance_starts[leg][0])
            stance_interval_end.append(stance_starts[leg][1])
            for i in range(1,len(stance_starts[leg])-1):
                stance_interval_start.append(stance_starts[leg][i])
                stance_interval_end.append(stance_starts[leg][i+1])

            for j in range(len(swing_starts[leg+2])): #stride
                k = np.where(np.array(swing_interval_start) < swing_starts[leg+2][j]+1)[0]
                if len(k) == 0:
                    continue
                k = max(k)
                if swing_starts[leg+2][j] > swing_interval_end[k]:
                    continue
                curr_interval = [swing_interval_start[k],swing_interval_end[k]]
                rel_swing_starts[leg+2].append((float(swing_starts[leg+2][j]-curr_interval[0]))/ (curr_interval[1]-curr_interval[0]))
            
composite_swing_starts = []
composite_stance_starts = []
for i in range(2,6):
    composite_swing_starts.extend(rel_swing_starts[i])

print(scipy.stats.ks_2samp(rel_swing_starts[4],rel_swing_starts[5]))
print(scipy.stats.ks_2samp(rel_swing_starts[2],rel_swing_starts[3]))
print(scipy.stats.ks_2samp(rel_swing_starts[2],rel_swing_starts[4]))
print(scipy.stats.ks_2samp(rel_swing_starts[3],rel_swing_starts[5]))
print(scipy.stats.ks_2samp(rel_swing_starts[2],rel_swing_starts[5]))

fig, axes = plt.subplots(2,1)
sns.distplot(composite_swing_starts, ax = axes[0], norm_hist=True, color='deepskyblue')
axes[0].legend([r'$n=$' + str(len(composite_swing_starts))],loc='upper left')
axes[0].set_xlim([0,1])
axes[0].set_xlabel('Phase shift between ipsilateral leg swings',fontname='Georgia', fontsize=13)
axes[0].set_ylabel('Density',fontname='Georgia', fontsize=13)

axins1 = inset_axes(axes[0], width=1.2, height=0.4, bbox_to_anchor=(.68, .5, .06, .5),bbox_transform=axes[0].transAxes)
axins2 = inset_axes(axes[0], width=1.2, height=0.4, bbox_to_anchor=(.68, .5, .31, .5),bbox_transform=axes[0].transAxes)
axinsplot1 = sns.distplot(rel_swing_starts[2], ax=axins1, color = 'r')
axins1.set_ylabel('')
axinsplot2 = sns.distplot(rel_swing_starts[4], ax=axins2, color = 'darkgreen')
axins2.set_ylabel('')
axinsplot1.set_xlim([0,1])
axinsplot2.set_xlim([0,1])
axinsplot1.set_yticks([])
axinsplot2.set_yticks([])
axinsplot1.set_xticks([])
axinsplot2.set_xticks([])

axins3 = inset_axes(axes[0], width=1.2, height=0.4, bbox_to_anchor=(.68, .5, .06, .23),bbox_transform=axes[0].transAxes)
axins4 = inset_axes(axes[0], width=1.2, height=0.4, bbox_to_anchor=(.68, .5, .31, .23),bbox_transform=axes[0].transAxes)
axinsplot3 = sns.distplot(rel_swing_starts[3], ax=axins3, color = 'navy')
axins3.set_ylabel('')
axinsplot4 = sns.distplot(rel_swing_starts[5], ax=axins4, color = 'gold')
axins4.set_ylabel('')
axinsplot3.set_xlim([0,1])
axinsplot3.set_yticks([])
axinsplot4.set_yticks([])
axinsplot4.set_xlim([0,1])
axinsplot3.set_xticks([0,1])
axinsplot4.set_xticks([0,1])
plt.tight_layout()
plt.savefig(upperdir+'/Writeup/Figures/' + stiffness + 'kPa_leg_phase_diff.pdf')

fig, axes = plt.subplots()
axes.set_ylabel(' ', fontname='Georgia', fontsize=13)
axes.set_xlabel('')
sns.ecdfplot(composite_swing_starts, color='deepskyblue', linewidth=4)
axins = inset_axes(axes, width=2, height=1.5, bbox_to_anchor=(.9, .27, .1, .23),bbox_transform=axes.transAxes)
axins.set_ylabel(' ', fontname='Georgia', fontsize=13)
sns.ecdfplot(rel_swing_starts[4], ax = axins, color = 'darkgreen', linewidth=2)
sns.ecdfplot(rel_swing_starts[2], ax = axins, color = 'r',linewidth=2)
sns.ecdfplot(rel_swing_starts[3], ax = axins,  color = 'navy', linewidth=2)
sns.ecdfplot(rel_swing_starts[5], ax = axins, color = 'gold', linewidth=2)
plt.savefig(upperdir+'/Writeup/Figures/' + stiffness + 'kPa_cumdist_ipsilateral.pdf')

for file in files:
    s = '-'
    bear = s.join(file.split('/')[-1].split('-')[0:-1])
    video = file.split('/')[-1].split('-')[-1].split('.')[0]
    dataframe = pd.read_csv(file)

    grouped = dataframe.groupby('video')
    for video,data in grouped:
        print('bear', bear, 'trial', video)
        swing_starts = [[] for i in range(6)]
        stance_starts = [[] for i in range(6)]
        swing_interval_start = []
        swing_interval_end = []
        stance_interval_start = []
        stance_interval_end = []
        # get out all the stance and swing start times to compute out relatives
        swing_starts[0] = list(map(int,data['L1_swing_start'][data['L1_swing_start'] > 0].to_list()))
        stance_starts[0] = list(map(int,data['L1_stance_start'][data['L1_stance_start'] > 0].to_list()))
        swing_starts[1] = list(map(int,data['R1_swing_start'][data['R1_swing_start'] > 0].to_list()))
        stance_starts[1] = list(map(int,data['R1_stance_start'][data['R1_stance_start'] > 0].to_list()))
        swing_starts[2] = list(map(int,data['L2_swing_start'][data['L2_swing_start'] > 0].to_list()))
        stance_starts[2] = list(map(int,data['L2_stance_start'][data['L2_stance_start'] > 0].to_list()))
        swing_starts[3] = list(map(int,data['R2_swing_start'][data['R2_swing_start'] > 0].to_list()))
        stance_starts[3] = list(map(int,data['R2_stance_start'][data['R2_stance_start'] > 0].to_list()))
        swing_starts[4] = list(map(int,data['L3_swing_start'][data['L3_swing_start'] > 0].to_list()))
        stance_starts[4] = list(map(int,data['L3_stance_start'][data['L3_stance_start'] > 0].to_list()))
        swing_starts[5] = list(map(int,data['R3_swing_start'][data['R3_swing_start'] > 0].to_list()))
        stance_starts[5] = list(map(int,data['R3_stance_start'][data['R3_stance_start'] > 0].to_list()))
        
        for leg in range(4):
            if len(swing_starts[leg]) < 2:
                continue
            swing_interval_start.append(swing_starts[leg][0])
            swing_interval_end.append(swing_starts[leg][1])
            for i in range(1,len(swing_starts[leg])-1):
                swing_interval_start.append(swing_starts[leg][i])
                swing_interval_end.append(swing_starts[leg][i+1])
            if len(stance_starts[leg]) < 2:
                continue
            stance_interval_start.append(stance_starts[leg][0])
            stance_interval_end.append(stance_starts[leg][1])
            for i in range(1,len(stance_starts[leg])-1):
                stance_interval_start.append(stance_starts[leg][i])
                stance_interval_end.append(stance_starts[leg][i+1])

            for j in range(len(swing_starts[leg+2])): #stride
                k = np.where(np.array(swing_interval_start) < swing_starts[leg+2][j]+1)[0]
                if len(k) == 0:
                    continue
                k = max(k)
                if swing_starts[leg+2][j] > swing_interval_end[k]:
                    continue
                curr_interval = [swing_interval_start[k],swing_interval_end[k]]
                rel_swing_starts[leg+2].append((float(swing_starts[leg+2][j]-curr_interval[0]))/ (curr_interval[1]-curr_interval[0]))
        for leg in range(4):
            if len(swing_starts[leg]) < 2:
                continue
            swing_interval_start.append(swing_starts[leg][0])
            swing_interval_end.append(swing_starts[leg][1])
            for i in range(1,len(swing_starts[leg])-1):
                swing_interval_start.append(swing_starts[leg][i])
                swing_interval_end.append(swing_starts[leg][i+1])
            if len(stance_starts[leg]) < 2:
                continue
            stance_interval_start.append(stance_starts[leg][0])
            stance_interval_end.append(stance_starts[leg][1])
            for i in range(1,len(stance_starts[leg])-1):
                stance_interval_start.append(stance_starts[leg][i])
                stance_interval_end.append(stance_starts[leg][i+1])
            for j in range(len(stance_starts[leg+2])): #stride
                k = np.where(np.array(swing_interval_start) < stance_starts[leg+2][j]+1)[0]
                if len(k) == 0:
                    continue
                k = max(k)
                if stance_starts[leg+2][j] > swing_interval_end[k]:
                    continue
                k2 = np.where(np.array(stance_interval_start) > swing_interval_start[k]+1)[0]
                if len(k2) == 0:
                    continue
                k2 = min(k2)
                if stance_starts[leg+2][j] > stance_interval_start[k2]:
                    continue
                curr_interval = [swing_interval_start[k],swing_interval_end[k]]
                rel_stance_starts[leg+2].append((float(stance_starts[leg+2][j]-curr_interval[0]))/ (curr_interval[1]-curr_interval[0]))


for i in range(2,6):
    composite_stance_starts.extend(rel_stance_starts[i])
    
fig, axes = plt.subplots(2,1)
sns.distplot(composite_stance_starts, ax = axes[0], norm_hist=True, color='darkorange')
axes[0].set_xlim([0,1])
axins5 = inset_axes(axes[0], width=1.2, height=0.4, bbox_to_anchor=(.68, .5, .06, .5),bbox_transform=axes[0].transAxes)
axins6 = inset_axes(axes[0], width=1.2, height=0.4, bbox_to_anchor=(.68, .5, .31, .5),bbox_transform=axes[0].transAxes)
axinsplot5 = sns.distplot(rel_stance_starts[2], ax=axins5, color = 'r')
axinsplot6 = sns.distplot(rel_stance_starts[3], ax=axins6, color = 'darkgreen')
axins5.tick_params(width = 0, labelbottom=False, labelleft=False)
axins6.tick_params(width = 0, labelleft=False)
axinsplot5.set_xlim([0,1])
axinsplot5.set_yticks([])
axinsplot6.set_xlim([0,1])
axinsplot6.set_yticks([])
axinsplot5.set_xticks([])
axinsplot6.set_xticks([])

axins7 = inset_axes(axes[0], width=1.2, height=0.4, bbox_to_anchor=(.68, .5, .06, .23),bbox_transform=axes[0].transAxes)
axins7.tick_params(labelleft=False)
axins8 = inset_axes(axes[0], width=1.2, height=0.4, bbox_to_anchor=(.68, .5, .31, .23),bbox_transform=axes[0].transAxes)
axins8.tick_params(labelleft=False)
axinsplot7 = sns.distplot(rel_stance_starts[4], ax=axins7, color = 'navy')
axinsplot8 = sns.distplot(rel_stance_starts[5], ax=axins8, color = 'gold')
axinsplot7.set_xlim([0,1])
axinsplot7.set_yticks([])
axinsplot8.set_xlim([0,1])
axinsplot8.set_yticks([])
axinsplot7.set_xticks([0,1])
axinsplot8.set_xticks([0,1])
plt.savefig(upperdir+'/Writeup/Figures/' + stiffness + 'kPa_phaseshift_stanceswing.pdf')

fig, axes = plt.subplots()
axes.set_ylabel(' ', fontname='Georgia', fontsize=13)
axes.set_xlabel('')
sns.ecdfplot(composite_stance_starts, color='deepskyblue', linewidth=4)
axins = inset_axes(axes, width=2, height=1.5, bbox_to_anchor=(.9, .27, .1, .23),bbox_transform=axes.transAxes)
axins.set_ylabel(' ', fontname='Georgia', fontsize=13)
sns.ecdfplot(rel_stance_starts[4], ax = axins, color = 'darkgreen', linewidth=2)
sns.ecdfplot(rel_stance_starts[2], ax = axins, color = 'r',linewidth=2)
sns.ecdfplot(rel_stance_starts[3], ax = axins,  color = 'navy', linewidth=2)
sns.ecdfplot(rel_stance_starts[5], ax = axins, color = 'gold', linewidth=2)
plt.savefig(upperdir+'/Writeup/Figures/' + stiffness + 'kPa_cumdist_stanceswing.pdf')
