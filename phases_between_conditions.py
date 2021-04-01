# This makes the plot in Figure 5c (not insets)

###########################################################################
#!/usr/bin/python
import re, math, sys, os, random
import numpy as np
import pylab as pl
from matplotlib import collections  as mc
import pandas as pd
from optparse import OptionParser
import matplotlib.pyplot as plt
import glob, csv
from scipy.stats import mode
import seaborn as sns
from pylab import *
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

upperdir = '/'.join(os.getcwd().split('/')[:-1])
avg_file50top = upperdir + '/Data/ForAnalysis/50kPa/Top/bear_averages.csv'
avg_file10top = upperdir + '/Data/ForAnalysis/10kPa/Top/bear_averages.csv'

avgdata_conditions = [avg_file50top,avg_file10top]
datadirs = ['/Data/ForAnalysis/50kPa/Top/Individual/ByStride/','/Data/ForAnalysis/10kPa/Top/Individual/ByStride/']

ipsilateral_stance_starts = [[] for k in range(2)]
ipsilateral_swing_starts = [[] for k in range(2)]

for condition in range(len(avgdata_conditions)):
    avgdata = pd.read_csv(avgdata_conditions[condition])
    datadir = upperdir + datadirs[condition]
    files = glob.glob(datadir + '*.csv')
    
    rel_swing_lengths = [[] for k in range(6)]
    rel_stance_lengths = [[] for k in range(6)]
    rel_swing_lengths[0] = [np.mean((avgdata['L1_swing']/avgdata['L1_period']).to_list()),np.std((avgdata['L1_swing']/avgdata['L1_period']).to_list())]
    rel_stance_lengths[0] = [np.mean((avgdata['L1_stance']/avgdata['L1_period']).to_list()),np.std((avgdata['L1_stance']/avgdata['L1_period']).to_list())]
    rel_swing_lengths[3] = [np.mean((avgdata['R1_swing']/avgdata['R1_period']).to_list()),np.std((avgdata['R1_swing']/avgdata['R1_period']).to_list())]
    rel_stance_lengths[3] = [np.mean((avgdata['R1_stance']/avgdata['R1_period']).to_list()),np.std((avgdata['R1_stance']/avgdata['R1_period']).to_list())]
    rel_swing_lengths[1] = [np.mean((avgdata['L2_swing']/avgdata['L2_period']).to_list()),np.std((avgdata['L2_swing']/avgdata['L2_period']).to_list())]
    rel_stance_lengths[1] = [np.mean((avgdata['L2_stance']/avgdata['L2_period']).to_list()),np.std((avgdata['L2_stance']/avgdata['L2_period']).to_list())]
    rel_swing_lengths[4] = [np.mean((avgdata['R2_swing']/avgdata['R2_period']).to_list()),np.std((avgdata['R2_swing']/avgdata['R2_period']).to_list())]
    rel_stance_lengths[4] = [np.mean((avgdata['R2_stance']/avgdata['R2_period']).to_list()),np.std((avgdata['R2_stance']/avgdata['R2_period']).to_list())]
    rel_swing_lengths[2] = [np.mean((avgdata['L3_swing']/avgdata['L3_period']).to_list()),np.std((avgdata['L3_swing']/avgdata['L3_period']).to_list())]
    rel_stance_lengths[2] = [np.mean((avgdata['L3_stance']/avgdata['L3_period']).to_list()),np.std((avgdata['L3_stance']/avgdata['L3_period']).to_list())]
    rel_swing_lengths[5] = [np.mean((avgdata['R3_swing']/avgdata['R3_period']).to_list()),np.std((avgdata['R3_swing']/avgdata['R3_period']).to_list())]
    rel_stance_lengths[5] = [np.mean((avgdata['R3_stance']/avgdata['R3_period']).to_list()),np.std((avgdata['R3_stance']/avgdata['R3_period']).to_list())]

    rel_stance_starts = [[] for k in range(6)]
    rel_swing_starts = [[] for k in range(6)]

    for file in files:
        s = '-'
        bear = s.join(file.split('/')[-1].split('_')[0:-1])
        dataframe = pd.read_csv(file)
        
        grouped = dataframe.groupby('video')
        for video,data in grouped:
            print('bear', bear, 'trial', video)
            swing_starts = [[] for k in range(6)]
            stance_starts = [[] for k in range(6)]
            swing_interval_start = []
            swing_interval_end = []
            stance_interval_start = []
            stance_interval_end = []
            swing_starts[5] = map(int,data['L1_swing_start'][data['L1_swing_start'] > 0].to_list())
            stance_starts[5] = map(int,data['L1_stance_start'][data['L1_stance_start'] > 0].to_list())
            # get out all the stance and swing start times to compute out relatives
            swing_starts[4] = map(int,data['R1_swing_start'][data['R1_swing_start'] > 0].to_list())
            stance_starts[4] = map(int,data['R1_stance_start'][data['R1_stance_start'] > 0].to_list())
            swing_starts[3] = map(int,data['L2_swing_start'][data['L2_swing_start'] > 0].to_list())
            stance_starts[3] = map(int,data['L2_stance_start'][data['L2_stance_start'] > 0].to_list())
            swing_starts[2] = map(int,data['R2_swing_start'][data['R2_swing_start'] > 0].to_list())
            stance_starts[2] = map(int,data['R2_stance_start'][data['R2_stance_start'] > 0].to_list())
            swing_starts[1] = map(int,data['L3_swing_start'][data['L3_swing_start'] > 0].to_list())
            stance_starts[1] = map(int,data['L3_stance_start'][data['L3_stance_start'] > 0].to_list())
            swing_starts[0] = map(int,data['R3_swing_start'][data['R3_swing_start'] > 0].to_list())
            stance_starts[0] = map(int,data['R3_stance_start'][data['R3_stance_start'] > 0].to_list())
            
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
                    
                for j in range(len(stance_starts[leg+2])): #stride
                    k = np.where(np.array(stance_interval_start) < stance_starts[leg+2][j]+1)[0]
                    if len(k) == 0:
                        continue
                    k = max(k)
                    if stance_starts[leg+2][j] > stance_interval_end[k]:
                            continue
                    curr_interval = [stance_interval_start[k],stance_interval_end[k]]
                    rel_stance_starts[leg+2].append((float(stance_starts[leg+2][j]-curr_interval[0]))/(curr_interval[1]-curr_interval[0]))
                
    for j in range(4):
        ipsilateral_swing_starts[condition].extend(rel_swing_starts[j])
        ipsilateral_stance_starts[condition].extend(rel_stance_starts[j])


fig, axes = plt.subplots()
sns.distplot(ipsilateral_swing_starts[1], bins=13, ax = axes, norm_hist=True, color='red',hist_kws={'alpha':0.3})
sns.distplot(ipsilateral_swing_starts[0], bins=13, ax = axes, norm_hist=True, color='dodgerblue',hist_kws={'alpha':0.3})

#ax = sns.violinplot(data=ipsilateral_swing_starts,palette=colors,inner=None,orient='h',scale='width')
#for violin,alpha in zip(ax.collections[::1], [0.5,0.5,0.5,0.5]):
   # violin.set_alpha(alpha)
#ax = sns.swarmplot(data=ipsilateral_swing_starts,palette=colors,orient='h')
axes.set_xlim([0,1])
plt.savefig(upperdir+'/Writeup/Figures/comparative_ipsilateraldistribution.pdf')
