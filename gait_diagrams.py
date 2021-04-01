# Just for exposition, not in paper.
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
###########################################################################
def grouper(iterable,n):
    args = [iter(iterable)]*n
    return zip(*args)
###########################################################################
stiffness = '10'

upperdir = '/'.join(os.getcwd().split('/')[:-1])
datadir = upperdir + '/Data/ForAnalysis/' + stiffness + 'kPa/Top/Individual/ByFrame/'
files = glob.glob(datadir + '*.csv')

sides = ['L','R']


colors = ['c','y','r','b','g','k','c']

for file in files:
    print file
    s = '-'
    bear = s.join(file.split('/')[-1].split('_')[0:-1])
    print bear
    dataframe = pd.read_csv(file)
    grouped = dataframe.groupby('video')
    for video,data in grouped:
        swing_starts = [[]]*8
        stance_starts = [[]]*8
        lines = {}
        fig1, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, gridspec_kw={'height_ratios':[4,1,1,1]})
        for leg_pair in range(4):
            for side in range(2):
                lines[2*(leg_pair)+side] = []
                leg = sides[side] + str(leg_pair+1)
                leg_down = map(int,data[leg + '_leg_down'].to_list())
                swing_starts[2*(leg_pair)+side] = (np.where(np.diff(leg_down)==-1)[0]+1).tolist()
                stance_starts[2*(leg_pair)+side] = (np.where(np.diff(leg_down)==1)[0]+2).tolist()
                gait_type = map(int,data['num_feet_down'].to_list())
                back_leg_type = map(int,data['num_rearfeet_down'].to_list())
                if stance_starts[2*(leg_pair)+side][0] > swing_starts[2*(leg_pair)+side][0]:
                    holder = [0] + stance_starts[2*(leg_pair)+side]
                    stance_starts[2*(leg_pair)+side] = holder
                for j in range(min(len(swing_starts[2*(leg_pair)+side]),len(stance_starts[2*(leg_pair)+side]))):
                    lines[2*(leg_pair)+side].append([swing_starts[2*(leg_pair)+side][j],stance_starts[2*(leg_pair)+side][j]])
        for leg in range(6):
            for k in range(len(lines[leg])):
                ax1.plot(lines[leg][k],[leg,leg],'k',linewidth=2)
            ax1.set_ylim(-1,6)
            ax1.set_yticks(np.arange(0,6,1))
            ax1.set_yticks(np.arange(0,6,1))
            ax1.set_yticklabels(['L1','R1','L2','R2','L3','R3'])
        for i in range(len(gait_type)-1):
            ax2.axvline(i,color=colors[gait_type[i]],linewidth=3)
            ax2.set_yticks([])
            ax4.axvline(i,color=colors[back_leg_type[i]],linewidth=2)
        for leg in range(6,8):
            for i in range(len(lines[leg])):
                ax3.plot(lines[leg][i],[leg,leg],'k',linewidth=2)
            ax3.set_ylim(5,8)
            ax3.set_yticks(np.arange(5,8,1))
            ax3.set_yticklabels(['','Lb','Rb'])
        plt.savefig(upperdir + '/Writeup/Figures/GaitDiagrams/' + stiffness + 'kPa/' + bear + '-' + str(video) + '.pdf')
