###########################################################################
# Makes averaged gait diagram.
# This makes Fig1c, Fig S3
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
from pylab import *
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib.gridspec as gridspec
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

rel_swing_lengths = [[] for i in range(8)]
rel_stance_lengths = [[] for i in range(8)]

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
rel_swing_lengths[6] = [np.mean((avgdata['L4_swing']/avgdata['L4_period']).to_list()),np.std((avgdata['L4_swing']/avgdata['L4_period']).to_list())]
rel_stance_lengths[6] = [np.mean((avgdata['L4_stance']/avgdata['L4_period']).to_list()),np.std((avgdata['L4_stance']/avgdata['L4_period']).to_list())]
rel_swing_lengths[7] = [np.mean((avgdata['R4_swing']/avgdata['R4_period']).to_list()),np.std((avgdata['R4_swing']/avgdata['R4_period']).to_list())]
rel_stance_lengths[7] = [np.mean((avgdata['R4_stance']/avgdata['R4_period']).to_list()),np.std((avgdata['R4_stance']/avgdata['R4_period']).to_list())]

rel_swing_starts = [[] for i in range(8)]
rel_stance_starts = [[] for i in range(8)]


for file in files:
    s = '-'
    bear = s.join(file.split('/')[-1].split('-')[0:-1])
    video = file.split('/')[-1].split('-')[-1].split('.')[0]
    dataframe = pd.read_csv(file)
    
    grouped = dataframe.groupby('video')
    for video,data in grouped:
        print(file, video)
        swing_starts = [[] for i in range(8)]
        stance_starts = [[] for i in range(8)]
        interval_start = []
        interval_end = []
        swing_starts[0] = map(int,data['L1_swing_start'][data['L1_swing_start'] > 0].to_list())
        stance_starts[0] = map(int,data['L1_stance_start'][data['L1_stance_start'] > 0].to_list())
        # get out all the stance and swing start times to compute out relatives
        swing_starts[3] = map(int,data['R1_swing_start'][data['R1_swing_start'] > 0].to_list())
        stance_starts[3] = map(int,data['R1_stance_start'][data['R1_stance_start'] > 0].to_list())
        swing_starts[1] = map(int,data['L2_swing_start'][data['L2_swing_start'] > 0].to_list())
        stance_starts[1] = map(int,data['L2_stance_start'][data['L2_stance_start'] > 0].to_list())
        swing_starts[4] = map(int,data['R2_swing_start'][data['R2_swing_start'] > 0].to_list())
        stance_starts[4] = map(int,data['R2_stance_start'][data['R2_stance_start'] > 0].to_list())
        swing_starts[2] = map(int,data['L3_swing_start'][data['L3_swing_start'] > 0].to_list())
        stance_starts[2] = map(int,data['L3_stance_start'][data['L3_stance_start'] > 0].to_list())
        swing_starts[5] = map(int,data['R3_swing_start'][data['R3_swing_start'] > 0].to_list())
        stance_starts[5] = map(int,data['R3_stance_start'][data['R3_stance_start'] > 0].to_list())
        swing_starts[6] = map(int,data['L4_swing_start'][data['L4_swing_start'] > 0].to_list())
        stance_starts[6] = map(int,data['L4_stance_start'][data['L4_stance_start'] > 0].to_list())
        swing_starts[7] = map(int,data['R4_swing_start'][data['R4_swing_start'] > 0].to_list())
        stance_starts[7] = map(int,data['R4_stance_start'][data['R4_stance_start'] > 0].to_list())
        
        if len(swing_starts[0]) < 2:
            continue
        interval_start.append(swing_starts[0][0])
        interval_end.append(swing_starts[0][1])
        for i in range(1,len(swing_starts[0])-1):
            interval_start.append(swing_starts[0][i])
            interval_end.append(swing_starts[0][i+1])

        for i in range(len(swing_starts)): #leg
            for j in range(len(swing_starts[i])): #stride
                k = np.where(np.array(interval_start) < swing_starts[i][j]+0.1)[0]
                if len(k) == 0:
                    continue
                k = max(k)
                if swing_starts[i][j] > interval_end[k]:
                    continue
                curr_interval = [interval_start[k],interval_end[k]]
                rel_swing_starts[i].append((float(swing_starts[i][j]-curr_interval[0]))/ (curr_interval[1]-curr_interval[0]))
                
        for i in range(len(stance_starts)): #leg
            for j in range(len(stance_starts[i])): #stride
                k = np.where(np.array(interval_end) > stance_starts[i][j]+0.1)[0]
                if len(k) == 0:
                    continue
                k = max(k)
                if stance_starts[i][j] < interval_start[k]:
                        continue
                curr_interval = [interval_start[k],interval_end[k]]
                rel_stance_starts[i].append((float(stance_starts[i][j]-curr_interval[0]))/ (curr_interval[1]-curr_interval[0]))
                
lines = [[] for i in range(8)]
sdshade = [[] for i in range(8)]
for i in range(len(rel_swing_starts)):
    mode = stats.mode(rel_swing_starts[i])
    m = mode[0][0]
    if m == 0.0:
        c = -1
    elif m == 1.0:
        c = 1
    else:
        c = 0
    expected=(m,0.1,5,c)
    y,x,_ = hist(rel_swing_starts[i],50)
    x = (x[1:]+x[:-1])/2
    params1,cov=curve_fit(gauss,x,y,expected,maxfev=100000)
    sigma=sqrt(diag(cov))
    #plot(x,gauss(x,*params1),color='red')
    
    if i == 0:
        params1[0] = 0
        params1[1] = 0
    sdshade[i] = [[max(-0.5,params1[0]-rel_stance_lengths[i][0]-rel_swing_lengths[i][0]*2-rel_stance_lengths[i][0]*2-rel_swing_lengths[i][1]),max(-0.5,params1[0]-rel_stance_lengths[i][0]-rel_swing_lengths[i][0]*2-rel_stance_lengths[i][0]*2+rel_swing_lengths[i][1])],[max(-0.5,params1[0]-rel_stance_lengths[i][0]-rel_swing_lengths[i][0]*2-rel_stance_lengths[i][0]-params1[1]),max(-0.5,params1[0]-rel_stance_lengths[i][0]-rel_swing_lengths[i][0]*2-rel_stance_lengths[i][0]+params1[1])],[max(-0.5,params1[0]-rel_stance_lengths[i][0]-rel_swing_lengths[i][0]-rel_stance_lengths[i][0]-rel_swing_lengths[i][1]),max(-0.5,params1[0]-rel_stance_lengths[i][0]-rel_swing_lengths[i][0]-rel_stance_lengths[i][0]+rel_swing_lengths[i][1])],[max(-0.5,params1[0]-rel_stance_lengths[i][0]-rel_swing_lengths[i][0]-params1[1]),max(-0.5,params1[0]-rel_stance_lengths[i][0]-rel_swing_lengths[i][0]+params1[1])], [params1[0]-rel_stance_lengths[i][0]-rel_swing_lengths[i][1],params1[0]-rel_stance_lengths[i][0]+rel_swing_lengths[i][1]],[params1[0]-params1[1],params1[0]+params1[1]], [min(1.5,params1[0]+rel_swing_lengths[i][0]-rel_swing_lengths[i][1]),min(1.5,params1[0]+rel_swing_lengths[i][0]+rel_swing_lengths[i][1])],[min(1.5,params1[0]+rel_swing_lengths[i][0]+rel_stance_lengths[i][0]-params1[1]),min(1.5,params1[0]+rel_swing_lengths[i][0]+rel_stance_lengths[i][0]+params1[1])],[min(1.5,params1[0]+rel_swing_lengths[i][0]+rel_stance_lengths[i][0]+rel_swing_lengths[i][0]-rel_swing_lengths[i][1]),min(1.5,params1[0]+rel_swing_lengths[i][0]+rel_stance_lengths[i][0]+rel_swing_lengths[i][0]+rel_swing_lengths[i][1])]]
    lines[i] = [[max(-0.5,params1[0] -rel_stance_lengths[i][0]-rel_swing_lengths[i][0]*2-rel_stance_lengths[i][0]*2),max(-0.5,params1[0]-rel_stance_lengths[i][0]-rel_swing_lengths[i][0]*2-rel_stance_lengths[i][0])],[max(-0.5,params1[0] -rel_stance_lengths[i][0]-rel_swing_lengths[i][0]-rel_stance_lengths[i][0]), max(-0.5,params1[0] -rel_stance_lengths[i][0]-rel_swing_lengths[i][0])],[params1[0]-rel_stance_lengths[i][0],params1[0]], [min(1.5,params1[0]+rel_swing_lengths[i][0]),min(1.5,params1[0]+rel_swing_lengths[i][0]+rel_stance_lengths[i][0])],[min(1.5,params1[0]+rel_swing_lengths[i][0]+rel_stance_lengths[i][0]+rel_swing_lengths[i][0]),min(1.5,params1[0]+rel_swing_lengths[i][0]+rel_stance_lengths[i][0]+rel_swing_lengths[i][0]+rel_stance_lengths[i][0])]]


fig = plt.figure()
ax = fig.add_subplot(111)
plt.axvspan(0,1, facecolor='grey', alpha=0.4)
for leg in range(6):
    for k in range(len(lines[leg])):
        if lines[leg][k][1] - lines[leg][k][0] == 0:
            continue
        plt.plot(lines[leg][k],[leg*-1,leg*-1],'k',linewidth=10, zorder=1)
    for m in range(len(sdshade[leg])):
        if sdshade[leg][m][1] - sdshade[leg][m][0] == 0:
            continue
        plt.plot(sdshade[leg][m],[leg*-1,leg*-1],'r',linewidth=3,alpha=0.8,zorder=1)
ax.set_yticklabels(['','R3','R2','R1','L3','L2','L1'])
ax.set_xticks([-0.5,0,0.5,1,1.5])
ax.set_xlim([-0.5,1.5])
plt.axvspan(-0.54,-0.5, facecolor='white', zorder=2)
plt.axvspan(1.5,1.54, facecolor='white', zorder=2)
plt.yticks(fontname='Georgia',fontsize=18)
plt.xticks(fontname='Georgia',fontsize=18)
ax.set_xlabel('Gait cycle',fontname='Georgia', fontsize=24)
plt.tight_layout()
plt.savefig(upperdir+'/Writeup/Figures/avggaitdiagram_' + stiffness + 'kPa.pdf')

fig = plt.figure()
gs = gridspec.GridSpec(2, 1,height_ratios=[3,1])
ax1 = plt.subplot(gs[0])
ax1.axvspan(0,1, facecolor='grey', alpha=0.4)
for leg in range(6):
    for k in range(len(lines[leg])):
        if lines[leg][k][1] - lines[leg][k][0] == 0:
            continue
        plt.plot(lines[leg][k],[leg*-1,leg*-1],'k',linewidth=10, zorder=1)
    for m in range(len(sdshade[leg])):
        if sdshade[leg][m][1] - sdshade[leg][m][0] == 0:
            continue
        plt.plot(sdshade[leg][m],[leg*-1,leg*-1],'r',linewidth=3,alpha=0.8,zorder=1)
ax2 = plt.subplot(gs[1])
for leg in range(6,8):
    print(lines[leg])
    for k in range(len(lines[leg])):
        if lines[leg][k][1] - lines[leg][k][0] == 0:
            continue
        plt.plot(lines[leg][k],[leg*-1,leg*-1],'k',linewidth=10, zorder=1)
    for m in range(len(sdshade[leg])):
        if sdshade[leg][m][1] - sdshade[leg][m][0] == 0:
            continue
        plt.plot(sdshade[leg][m],[leg*-1,leg*-1],'r',linewidth=3,alpha=0.8,zorder=1)
ax1.set_yticklabels(['','R3','R2','R1','L3','L2','L1'])
ax1.set_xticks([])
ax2.set_xticks([-0.5,0,0.5,1,1.5])
ax2.set_xlim([-0.5,1.5])
ax1.set_xlim([-0.5,1.5])
ax1.axvspan(-0.54,-0.5, facecolor='white', zorder=2)
ax1.axvspan(1.5,1.54, facecolor='white', zorder=2)
ax2.axvspan(0,1, facecolor='grey', alpha=0.4)
ax2.axvspan(-0.54,-0.5, facecolor='white', zorder=2)
ax2.axvspan(1.5,1.54, facecolor='white', zorder=2)
ax2.set_yticklabels(['R4', 'L4'])
ax2.set_ylim([-5.5,-7.5])
ax2.set_yticks([-6,-7])
ax2.set_xlabel('Gait cycle',fontname='Georgia', fontsize=24)
plt.tight_layout()
plt.savefig(upperdir+'/Writeup/Figures/PaperFigures/FigureS3_AverageGaitDiagramAllLegs.pdf')
