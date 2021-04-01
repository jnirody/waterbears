# This makes Figure S5
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
from scipy.stats import kde
import seaborn as sns
###########################################################################
def grouper(iterable,n):
    args = [iter(iterable)]*n
    return zip(*args)
    
def density_estimation(m1, m2):
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    return X, Y, Z
###########################################################################
stiffness = '50'

upperdir = '/'.join(os.getcwd().split('/')[:-1])
datadir = upperdir + '/Data/ForAnalysis/' + stiffness + 'kPa/Top/Individual/ByStride/'
files = glob.glob(datadir + '*.csv')
avg_file = upperdir + '/Data/ForAnalysis/' + stiffness + 'kPa/Top/bear_averages.csv'
avgdata = pd.read_csv(avg_file)
byframe_dir = upperdir + '/Data/ForAnalysis/' + stiffness + 'kPa/Top/Individual/ByFrame/'


framerate = 50. # frames/sec
resolution = 0.55 # um / pixel (5.5 um/pixel Basler at 10x mag)


rel_stance_starts = [[] for i in range(6)]
rel_swing_starts = [[] for i in range(6)]

#[L3, R2, L2, R1, L1] --> R3
tetrapod1 = [0,2./3,1./3,0,2./3,1./3]
tetrapod2 = [0,1./3,1./3,2./3,2./3,0]

tripod = [0,1./2,1./2,0,0,1./2]
wave = [0,1./2,1./6,2./3,1./3,5./6]

tetrapod_count = 0
tripod_count = 0
wave_count = 0
irregular_count = 0

tetrapod_speed = []
tripod_speed = []
wave_speed = []
irregular_speed = []

speeds_with_gaits = []

phases = []

checks = [[] for i in range(4)]
for file in files:
    s = '-'
    bear = s.join(file.split('/')[-1].split('_')[0:-1])
    video = file.split('/')[-1].split('-')[-1].split('.')[0]
    dataframe = pd.read_csv(file)
    
    grouped = dataframe.groupby('video')
    for video,data in grouped:
        print('bear', bear, 'trial', video)
        fbf_file = pd.read_csv(byframe_dir + bear + '_framebyframe.csv')
        swing_starts = [[] for i in range(6)]
        stance_starts = [[] for i in range(6)]
        swing_interval_start = []
        swing_interval_end = []
        stance_interval_start = []
        stance_interval_end = []
        
        # get out all the stance and swing start times to compute out relatives
        swing_starts[5] = list(map(int,data['L1_swing_start'][data['L1_swing_start'] > 0].to_list()))
        swing_starts[4] = list(map(int,data['R1_swing_start'][data['R1_swing_start'] > 0].to_list()))
        swing_starts[3] = list(map(int,data['L2_swing_start'][data['L2_swing_start'] > 0].to_list()))
        swing_starts[2] = list(map(int,data['R2_swing_start'][data['R2_swing_start'] > 0].to_list()))
        swing_starts[1] = list(map(int,data['L3_swing_start'][data['L3_swing_start'] > 0].to_list()))
        swing_starts[0] = list(map(int,data['R3_swing_start'][data['R3_swing_start'] > 0].to_list()))
        
        for leg in range(1):
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
            
        for j in range(len(swing_interval_start)): #stride
            start_idx = np.where(np.array(fbf_file['frame'])==swing_interval_start[j])[0]
            if len(start_idx) > 0:
                start_idx = start_idx[0]
            else:
                continue
            end_idx = np.where(np.array(fbf_file['frame'])==swing_interval_end[j])[0][0]
            start_pos = list(map(float,fbf_file['center_pos'][start_idx][1:-1].split(',')))
            end_pos = list(map(float,fbf_file['center_pos'][end_idx][1:-1].split(',')))
            stride_dist = np.sqrt((start_pos[0]-end_pos[0])**2 + (start_pos[1]-end_pos[1])**2)*resolution
            stride_speed = stride_dist/(np.abs(end_idx-start_idx))*framerate
            stride_legs_down = np.mean(fbf_file['num_feet_down'][start_idx:end_idx])
            
            rel_swing_starts = []
            rel_stance_starts = []
            for leg in range(6):
                if j > len(swing_starts[leg])-1:
                    rel_swing_starts.extend([-1])
                    continue
                k = np.where(np.array(swing_interval_start) < swing_starts[leg][j]+1)[0]
                if len(k) == 0:
                    rel_swing_starts.extend([-1])
                    continue
                k = max(k)
                if swing_starts[leg][j] > swing_interval_end[k]:
                    rel_swing_starts.extend([-1])
                    continue
                curr_interval = [swing_interval_start[k],swing_interval_end[k]]
                rel_swing_starts.extend([(float(swing_starts[leg][j]-curr_interval[0]))/ (curr_interval[1]-curr_interval[0])])
            if len(np.where(np.array(rel_swing_starts)== -1)[0]) > 1:
                continue
            if np.isnan(stride_legs_down) == 1:
                continue
            # ipsi vs contra
            if len(np.where(np.array(rel_swing_starts)== -1)[0]) > 0:
                continue
            else:
                phases.extend([(abs(rel_swing_starts[0]-rel_swing_starts[1]),abs(rel_swing_starts[0]-rel_swing_starts[2]))])
                phases.extend([(abs(rel_swing_starts[0]-rel_swing_starts[1]),abs(rel_swing_starts[1]-rel_swing_starts[3]))])
                phases.extend([(abs(rel_swing_starts[2]-rel_swing_starts[3]),abs(rel_swing_starts[2]-rel_swing_starts[4]))])
                phases.extend([(abs(rel_swing_starts[2]-rel_swing_starts[2]),abs(rel_swing_starts[3]-rel_swing_starts[5]))])
                phases.extend([(abs(rel_swing_starts[4]-rel_swing_starts[5]),abs(rel_swing_starts[0]-rel_swing_starts[4]))])
                phases.extend([(abs(rel_swing_starts[4]-rel_swing_starts[5]),abs(rel_swing_starts[5]-rel_swing_starts[1]))])
            error_tolerance = 0.12
            curr_guess = 'irregular'
            curr_best = 2.
            
            tripod_check = [np.abs(a-b) for a,b in zip(rel_swing_starts,tripod)]
 
            tripod_check = [item for item in tripod_check if item > error_tolerance]
            tripod_check = [item for item in tripod_check if item < (1-error_tolerance)]
            tripod_check.extend([0])
            
            if len(tripod_check) < 3:
                curr_guess = 'tripod'
                curr_best = tripod_check[0]
            
            tetrapod1_check = [np.abs(a-b) for a,b in zip(rel_swing_starts,tetrapod1)]
            
            tetrapod1_check = [item for item in tetrapod1_check if item < (1-error_tolerance)]
            tetrapod1_check = [item for item in tetrapod1_check if item > error_tolerance]

            tetrapod2_check = [np.abs(a-b) for a,b in zip(rel_swing_starts,tetrapod2)]
            
            tetrapod1_check_temp = [item for item in tetrapod1_check if item > error_tolerance]
            tetrapod1_check_temp.extend([item for item in tetrapod1_check if item < (1-error_tolerance)])
            tetrapod1_check = tetrapod1_check_temp

            tetrapod1_check.sort()
            tetrapod1_check.extend([0])
            
            tetrapod2_check = [item for item in tetrapod2_check if item > error_tolerance]
            tetrapod2_check = [item for item in tetrapod2_check if item < (1-error_tolerance)]
            tetrapod2_check.sort()
            tetrapod2_check.extend([0])
            
            if len(tetrapod1_check) < 3:
                if tetrapod1_check[0] < curr_best:
                    curr_guess = 'tetrapod'
                    curr_best = tetrapod1_check[0]
            if len(tetrapod2_check) < 3:
               if tetrapod2_check[0] < curr_best:
                    curr_guess = 'tetrapod'
                    curr_best = tetrapod2_check[0]

            wave_check = [np.abs(a-b) for a,b in zip(rel_swing_starts,wave)]
            wave_check = [item for item in wave_check if item > error_tolerance]
            wave_check = [item for item in wave_check if item < (1-error_tolerance)]
            
            if len(wave_check) < 3:
                if wave_check[0] < curr_best:
                    curr_guess = 'wave'
                    curr_best = wave_check[0]
            
            if curr_guess == 'wave':
                wave_count += 1
                wave_speed.extend([stride_speed])
                speeds_with_gaits.append([stride_speed,'wave'])
            elif curr_guess == 'tripod':
                tripod_count += 1
                tripod_speed.extend([stride_speed])
                speeds_with_gaits.append([stride_speed,'tripod'])
            elif curr_guess == 'tetrapod':
                tetrapod_count += 1
                tetrapod_speed.extend([stride_speed])
                speeds_with_gaits.append([stride_speed,'tetrapod'])
            else:
                irregular_count += 1
                irregular_speed.extend([stride_speed])
                speeds_with_gaits.append([stride_speed,'irregular'])

total = float(tripod_count + tetrapod_count + wave_count + irregular_count)
tripod_ncount = tripod_count/total*100
tetrapod_ncount = tetrapod_count/total*100
wave_ncount = wave_count/total*100
irregular_ncount = irregular_count/total*100

print(tripod_count, tetrapod_count, wave_count, irregular_count)
print(tripod_ncount, tetrapod_ncount, wave_ncount, irregular_ncount)

fig, axes = plt.subplots()

sns.distplot(tripod_speed,hist=False,color='mediumvioletred')
sns.distplot(tetrapod_speed,hist=False,color='dodgerblue')
sns.distplot(wave_speed,hist=False,color='indigo')
sns.distplot(irregular_speed,hist=False,color='dimgray')
axes.set_xlim([-50,350])
#
#sns.distplot(tripod_legs,ax=axes[1],hist=False,color='mediumvioletred')
#sns.distplot(tetrapod_legs,ax=axes[1],hist=False,color='dodgerblue')
#sns.distplot(wave_legs,ax=axes[1],hist=False,color='indigo')
#sns.distplot(irregular_legs,ax=axes[1],hist=False,color='dimgray')
plt.savefig(upperdir+'/Writeup/Figures/gait_propwithspeed_' + stiffness + 'kPa.pdf')



bins = [1, 3, 5]

speeds_with_gaits.sort()
slow = [stride[1] for stride in speeds_with_gaits if stride[0] < 100]
tripod_counts = [float(slow.count('tripod'))/len(slow)*100]
tetrapod_counts = [float(slow.count('tetrapod'))/len(slow)*100]
wave_counts = [float(slow.count('wave'))/len(slow)*100]
unclassified_counts = [float(slow.count('irregular'))/len(slow)*100]

medium = [stride[1] for stride in speeds_with_gaits if (stride[0] > 100 and stride[0] < 200)]
tripod_counts.extend([float(medium.count('tripod'))/len(medium)*100])
tetrapod_counts.extend([float(medium.count('tetrapod'))/len(medium)*100])
wave_counts.extend([float(medium.count('wave'))/len(medium)*100])
unclassified_counts.extend([float(medium.count('irregular'))/len(medium)*100])

fast = [stride[1] for stride in speeds_with_gaits if stride[0] > 200]
tripod_counts.extend([float(fast.count('tripod'))/len(fast)*100])
tetrapod_counts.extend([float(fast.count('tetrapod'))/len(fast)*100])
wave_counts.extend([float(fast.count('wave'))/len(fast)*100])
unclassified_counts.extend([float(fast.count('irregular'))/len(fast)*100])

print(tripod_counts,tetrapod_counts,wave_counts,unclassified_counts)
plt.bar(bins, tripod_counts)
plt.bar(bins, tetrapod_counts, bottom=tripod_counts)
plt.bar(bins, wave_counts, bottom = np.add(tetrapod_counts,tripod_counts).tolist())
plt.bar(bins, unclassified_counts, bottom = np.add(wave_counts,np.add(tetrapod_counts,tripod_counts)).tolist())
plt.xlim([0,6])
plt.xticks([])
plt.savefig(upperdir+'/Writeup/Figures/binnedspeed_gaitprops_' + stiffness + 'kPa.pdf')



# now go in the other direction


#
#labels = ['']
#fig, ax = plt.subplots(1, 1)
#ax.bar(labels,tripod_ncount,width=0.1)
#ax.bar(labels,tetrapod_ncount,bottom=tripod_ncount,width=0.1)
#ax.bar(labels,wave_ncount,bottom=tripod_ncount+tetrapod_ncount,width=0.1)
#ax.bar(labels,irregular_ncount,bottom=tripod_ncount+tetrapod_ncount+wave_ncount,width=0.1)
#ax.set_ylabel('',fontname='Georgia')
#ax.set_ylim(0,110)
#ax.set_xlim(-0.2, 0.2)
#plt.savefig(upperdir+'/Writeup/Figures/50kPa_gait_proportions.pdf')

