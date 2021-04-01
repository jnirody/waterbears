# Plot of walking speed distribution between different conditions
# Makes Figure S4a
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

conditions = ['10kpa','50kpa','glass'] #,'roughglass']
datadirs = ['/Data/Tracking/TopTracks/10kPa/','/Data/Tracking/TopTracks/50kPa/','/Data/Tracking/TopTracks/Glass/']

resolution = [0.55,0.55,0.55]
framerate = 150


COM_speeds = [[] for k in range(3)]

for condition in range(len(conditions)):
    datadir = upperdir + datadirs[condition]
    files = glob.glob(datadir + '*.csv')
    for file in files:
        s = '-'
        bear = s.join(file.split('/')[-1].split('_')[0:-1])
        dataframe = pd.read_csv(file)
        if condition < 2:
            grouped = dataframe.groupby('Track')
            xpos = []
            ypos = []
            for leg,data in grouped:
                if leg == 0:
                    start_frame = data['Slice'].iloc[[0][0]]
                    end_frame = data['Slice'].iloc[[1][0]]
                if leg == 11:
                    for i in range(len(data)-1):
                        if data['Slice'].iloc[[i][0]] > (start_frame - 1) and data['Slice'].iloc[[i][0]] < (end_frame + 1):
                            xpos.extend([data['X'].iloc[[i][0]]])
                            ypos.extend([data['Y'].iloc[[i][0]]])
        else:
            xpos = list(map(float,dataframe['X'][dataframe['X']>0].to_list()))
            ypos = list(map(float,dataframe['Y'][dataframe['Y']>0].to_list()))
        window = 55
        number_windows = int((len(xpos)-1)/window)
        for w in range(number_windows):
            x1 = xpos[w*window]
            x2 = xpos[(w+1)*window]
            y1 = ypos[w*window]
            y2 = ypos[(w+1)*window]
            total_dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)*resolution[condition]
            COM_speeds[condition].extend([total_dist/window*50])

fig,axes = plt.subplots()
colors = ['blue','orange','green']
sns.swarmplot(data=COM_speeds, palette=colors)
plt.ylabel('Walking speed ('r'$\mu$m/s)',fontsize=19, fontname='Georgia')
axes.set_xticklabels(['10 kPa gel','50 kPa gel','Glass'], fontsize=14, fontname='Georgia')
plt.savefig(upperdir + '/Writeup/Figures/walkingspeed_across_substrates.pdf')
            
            
