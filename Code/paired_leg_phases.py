# This makes the insets in Figure 5c
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
def circular_hist(ax, x, bins=16, density=True, offset=0, gaps=True):
    x = [(i+np.pi) % (2*np.pi) - np.pi for i in x]
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins+1)
    n, bins = np.histogram(x, bins=bins)
    widths = np.diff(bins)
    if density:
        area = n / x.size
        radius = (area/np.pi) ** .5
    else:
        radius = n

    patches = ax.bar(bins[:-1], radius, zorder=1, align='edge', width=widths,
                 edgecolor='C0', fill=False, linewidth=1)

    ax.set_theta_offset(offset)

    if density:
        ax.set_yticks([])

    return n, bins, patches
###########################################################################

stiffness = '10'

upperdir = '/'.join(os.getcwd().split('/')[:-1])
datadir = upperdir + '/Data/ForAnalysis/' + stiffness + 'kPa/Top/Individual/ByStride/'
files = glob.glob(datadir + '*.csv')
avg_file = upperdir + '/Data/ForAnalysis/' + stiffness + 'kPa/Top/bear_averages.csv'
avgdata = pd.read_csv(avg_file)

rel_swing_lengths = [[] for i in range(6)]
rel_stance_lengths = [[] for i in range(6)]

rel_swing_lengths[5] = [np.mean((avgdata['L1_swing']/avgdata['L1_period']).to_list()),np.std((avgdata['L1_swing']/avgdata['L1_period']).to_list())]
rel_stance_lengths[5] = [np.mean((avgdata['L1_stance']/avgdata['L1_period']).to_list()),np.std((avgdata['L1_stance']/avgdata['L1_period']).to_list())]
rel_swing_lengths[2] = [np.mean((avgdata['R1_swing']/avgdata['R1_period']).to_list()),np.std((avgdata['R1_swing']/avgdata['R1_period']).to_list())]
rel_stance_lengths[2] = [np.mean((avgdata['R1_stance']/avgdata['R1_period']).to_list()),np.std((avgdata['R1_stance']/avgdata['R1_period']).to_list())]
rel_swing_lengths[4] = [np.mean((avgdata['L2_swing']/avgdata['L2_period']).to_list()),np.std((avgdata['L2_swing']/avgdata['L2_period']).to_list())]
rel_stance_lengths[4] = [np.mean((avgdata['L2_stance']/avgdata['L2_period']).to_list()),np.std((avgdata['L2_stance']/avgdata['L2_period']).to_list())]
rel_swing_lengths[1] = [np.mean((avgdata['R2_swing']/avgdata['R2_period']).to_list()),np.std((avgdata['R2_swing']/avgdata['R2_period']).to_list())]
rel_stance_lengths[1] = [np.mean((avgdata['R2_stance']/avgdata['R2_period']).to_list()),np.std((avgdata['R2_stance']/avgdata['R2_period']).to_list())]
rel_swing_lengths[3] = [np.mean((avgdata['L3_swing']/avgdata['L3_period']).to_list()),np.std((avgdata['L3_swing']/avgdata['L3_period']).to_list())]
rel_stance_lengths[3] = [np.mean((avgdata['L3_stance']/avgdata['L3_period']).to_list()),np.std((avgdata['L3_stance']/avgdata['L3_period']).to_list())]
rel_swing_lengths[0] = [np.mean((avgdata['R3_swing']/avgdata['R3_period']).to_list()),np.std((avgdata['R3_swing']/avgdata['R3_period']).to_list())]
rel_stance_lengths[0] = [np.mean((avgdata['R3_stance']/avgdata['R3_period']).to_list()),np.std((avgdata['R3_stance']/avgdata['R3_period']).to_list())]

rel_stance_starts = [[] for i in range(6)]
rel_swing_starts = [[] for i in range(6)]

paired_swing_starts = [[] for i in range(6)]

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
        swing_starts[2] = list(map(int,data['L1_swing_start'][data['L1_swing_start'] > 0].to_list()))
        stance_starts[2] = list(map(int,data['L1_stance_start'][data['L1_stance_start'] > 0].to_list()))
        # get out all the stance and swing start times to compute out relatives
        swing_starts[5] = list(map(int,data['R1_swing_start'][data['R1_swing_start'] > 0].to_list()))
        stance_starts[5] = list(map(int,data['R1_stance_start'][data['R1_stance_start'] > 0].to_list()))
        swing_starts[1] = list(map(int,data['L2_swing_start'][data['L2_swing_start'] > 0].to_list()))
        stance_starts[1] = list(map(int,data['L2_stance_start'][data['L2_stance_start'] > 0].to_list()))
        swing_starts[4] = list(map(int,data['R2_swing_start'][data['R2_swing_start'] > 0].to_list()))
        stance_starts[4] = list(map(int,data['R2_stance_start'][data['R2_stance_start'] > 0].to_list()))
        swing_starts[0] = list(map(int,data['L3_swing_start'][data['L3_swing_start'] > 0].to_list()))
        stance_starts[0] = list(map(int,data['L3_stance_start'][data['L3_stance_start'] > 0].to_list()))
        swing_starts[3] = list(map(int,data['R3_swing_start'][data['R3_swing_start'] > 0].to_list()))
        stance_starts[3] = list(map(int,data['R3_stance_start'][data['R3_stance_start'] > 0].to_list()))
        
        for leg in range(2):
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

            for j in range(len(swing_starts[leg+3])): #stride
                k = np.where(np.array(swing_interval_start) < swing_starts[leg+3][j]+1)[0]
                if len(k) == 0:
                    continue
                k = max(k)
                if swing_starts[leg+3][j] > swing_interval_end[k]:
                    continue
                curr_interval = [swing_interval_start[k],swing_interval_end[k]]
                rel_swing_starts[leg+3].append((float(swing_starts[leg+3][j]-curr_interval[0]))/ (curr_interval[1]-curr_interval[0]))
                if leg in [0,1] and j < len(swing_starts[leg+1]):
                    if swing_starts[leg+1][j] < swing_interval_start[k]:
                        continue
                    if swing_starts[leg+1][j] > swing_interval_end[k]:
                        continue
                    paired_swing_starts[leg].append(tuple([(float(swing_starts[leg+1][j]-curr_interval[0]))/ (curr_interval[1]-curr_interval[0]),(float(swing_starts[leg+3][j]-curr_interval[0]))/ (curr_interval[1]-curr_interval[0])]))
                    
                
        for leg in range(3,5):
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
                
            for j in range(len(swing_starts[leg-3])): #stride
                k = np.where(np.array(swing_interval_start) < swing_starts[leg-3][j]+1)[0]
                if len(k) == 0:
                    continue
                k = max(k)
                if swing_starts[leg-3][j] > swing_interval_end[k]:
                    continue
                curr_interval = [swing_interval_start[k],swing_interval_end[k]]
                rel_swing_starts[leg-3].append((float(swing_starts[leg-3][j]-curr_interval[0]))/ (curr_interval[1]-curr_interval[0]))
                if leg in [3,4] and j < len(swing_starts[leg+1]):
                    if swing_starts[leg+1][j] < swing_interval_start[k]:
                        continue
                    if swing_starts[leg+1][j] > swing_interval_end[k]:
                        continue
                    paired_swing_starts[leg].append(tuple([(float(swing_starts[leg+1][j]-curr_interval[0]))/ (curr_interval[1]-curr_interval[0]),(float(swing_starts[leg-3][j]-curr_interval[0]))/ (curr_interval[1]-curr_interval[0])]))


bins = [[] for i in range(2)]
fig, axes = plt.subplots()
for i in [0,1,3,4]:
    for j in range(len(paired_swing_starts[i])):
        if paired_swing_starts[i][j][0] > 0.6 and paired_swing_starts[i][j][0] < 0.95:
            bins[1].extend([paired_swing_starts[i][j][1]*2*np.pi])
        else:
            bins[0].extend([paired_swing_starts[i][j][1]*2*np.pi])
#ax = sns.violinplot(data=bins,inner=None,orient='h')
#for violin,alpha in zip(ax.collections[::1], [0.5,0.5,0.5,0.5]):
#    violin.set_alpha(alpha)
# Plot polar histogram
fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
ax = sns.histplot(data=bins[0],bins=30,color='red')
ax.set_ylabel(' ')
ax.set_yticks([ ])
plt.xticks(np.linspace(0,2*np.pi,9), (' ', ' ', ' ',' ',' ', ' ', ' ', ' ',' '))
plt.savefig(upperdir+'/Writeup/Figures/' + stiffness + 'kPa_conditionalonbound_ipsilateraldistribution.pdf')

fig, ax2 = plt.subplots(subplot_kw=dict(projection='polar'))
ax2 = sns.histplot(data=bins[1],bins=30,color='red')
ax2.set_ylabel(' ')
ax2.set_yticks([ ])
plt.xticks(np.linspace(0,2*np.pi,9), (' ', ' ', ' ',' ',' ', ' ', ' ', ' ',' '))
plt.savefig(upperdir+'/Writeup/Figures/' + stiffness + 'kPa_conditionalontetra_ipsilateraldistribution.pdf')

