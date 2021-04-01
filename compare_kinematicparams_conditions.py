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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.font_manager import FontProperties
import scipy
from scipy.stats import linregress
from scipy.optimize import curve_fit

# This makes Figure S4b,c

####################################################
def func(x, a, b, c):
    return a*x**(b)+c
####################################################

stiffness = '10'

upperdir = '/'.join(os.getcwd().split('/')[:-1])
alldata_file = upperdir + '/Data/ForAnalysis/' + stiffness + 'kPa/Top/compiled_bears_by_strides.csv'
alldata = pd.read_csv(alldata_file)

# Look at how each leg pair correlates with speed
first_leg_speed = list(map(float,alldata['R1_stride_speed'].tolist()))
first_leg_speed.extend(list(map(float,alldata['L1_stride_speed'].tolist())))
second_leg_speed = list(map(float,alldata['R2_stride_speed'].tolist()))
second_leg_speed.extend(list(map(float,alldata['L2_stride_speed'].tolist())))
third_leg_speed = list(map(float,alldata['R3_stride_speed'].tolist()))
third_leg_speed.extend(list(map(float,alldata['L3_stride_speed'].tolist())))
back_leg_speed = list(map(float,alldata['R4_stride_speed'].tolist()))
back_leg_speed.extend(list(map(float,alldata['L4_stride_speed'].tolist())))

loc_legs_speed = first_leg_speed + second_leg_speed + third_leg_speed

first_step = list(map(float,alldata['R1_step_length'].tolist()))
first_step.extend(list(map(float,alldata['L1_step_length'].tolist())))
idx1 = (np.array(first_step) > 0) & (np.array(first_leg_speed) > 20)

second_step = list(map(float,alldata['R2_step_length'].tolist()))
second_step.extend(list(map(float,alldata['L2_step_length'].tolist())))
idx2 = (np.array(second_step) > 0) & (np.array(second_leg_speed) > 20)

third_step = list(map(float,alldata['R3_step_length'].tolist()))
third_step.extend(list(map(float,alldata['L3_step_length'].tolist())))
idx3 = (np.array(third_step) > 0) & (np.array(third_leg_speed) > 20)

back_step = list(map(float,alldata['R4_step_length'].tolist()))
back_step.extend(list(map(float,alldata['L4_step_length'].tolist())))
idx4 = (np.array(back_step) > 0) & (np.array(back_leg_speed) > 20)

loc_legs_step = first_step + second_step + third_step
idx = (np.array(loc_legs_step) > 0) & (np.array(loc_legs_speed) > 20)

fig, axes = plt.subplots()
xdata = np.array(loc_legs_speed)[idx]
ydata = np.array(loc_legs_step)[idx]
corr = scipy.stats.spearmanr(xdata,ydata)
corrtowrite = str(round(corr[0],2))
if corr[1] < 0.05:
    sig = 'p < 0.05'
    if corr[1] < 0.01:
        sig = 'p < 0.01'
    if corr[1] < 0.001:
        sig = 'p < 0.001'
else:
    sig = 'p = ' + str(round(corr[1],2))
plt.plot(xdata,ydata, color = 'darkorange', linestyle = '', marker = '.', alpha=0.4, label= '10 kPa: ' + r'$\rho$ = ' + corrtowrite + ', ' + sig)

stiffness = '50'

upperdir = '/'.join(os.getcwd().split('/')[:-1])
alldata_file = upperdir + '/Data/ForAnalysis/' + stiffness + 'kPa/Top/compiled_bears_by_strides.csv'
alldata = pd.read_csv(alldata_file)

# Look at how each leg pair correlates with speed
first_leg_speed = list(map(float,alldata['R1_stride_speed'].tolist()))
first_leg_speed.extend(list(map(float,alldata['L1_stride_speed'].tolist())))
second_leg_speed = list(map(float,alldata['R2_stride_speed'].tolist()))
second_leg_speed.extend(list(map(float,alldata['L2_stride_speed'].tolist())))
third_leg_speed = list(map(float,alldata['R3_stride_speed'].tolist()))
third_leg_speed.extend(list(map(float,alldata['L3_stride_speed'].tolist())))
back_leg_speed = list(map(float,alldata['R4_stride_speed'].tolist()))
back_leg_speed.extend(list(map(float,alldata['L4_stride_speed'].tolist())))

loc_legs_speed = first_leg_speed + second_leg_speed + third_leg_speed

first_step = list(map(float,alldata['R1_step_length'].tolist()))
first_step.extend(list(map(float,alldata['L1_step_length'].tolist())))
idx1 = (np.array(first_step) > 0) & (np.array(first_leg_speed) > 20)

second_step = list(map(float,alldata['R2_step_length'].tolist()))
second_step.extend(list(map(float,alldata['L2_step_length'].tolist())))
idx2 = (np.array(second_step) > 0) & (np.array(second_leg_speed) > 20)

third_step = list(map(float,alldata['R3_step_length'].tolist()))
third_step.extend(list(map(float,alldata['L3_step_length'].tolist())))
idx3 = (np.array(third_step) > 0) & (np.array(third_leg_speed) > 20)

back_step = list(map(float,alldata['R4_step_length'].tolist()))
back_step.extend(list(map(float,alldata['L4_step_length'].tolist())))
idx4 = (np.array(back_step) > 0) & (np.array(back_leg_speed) > 20)

loc_legs_step = first_step + second_step + third_step
idx = (np.array(loc_legs_step) > 0) & (np.array(loc_legs_speed) > 20)

xdata = np.array(loc_legs_speed)[idx]
ydata = np.array(loc_legs_step)[idx]
corr = scipy.stats.spearmanr(xdata,ydata)
corrtowrite = str(round(corr[0],2))
if corr[1] < 0.05:
    sig = 'p < 0.05'
    if corr[1] < 0.01:
        sig = 'p < 0.01'
    if corr[1] < 0.001:
        sig = 'p < 0.001'
else:
    sig = 'p = ' + str(round(corr[1],2))
plt.plot(xdata,ydata, color = 'blue', linestyle = '', marker = '.', alpha=0.4, label= '50 kPa: '+ r'$\rho$ = ' + corrtowrite + ', ' + sig)

plt.legend(fontsize=16)
plt.xlim([0,400])
plt.ylim([0,200])
plt.yticks(fontname='Georgia',fontsize=13)
plt.xticks(fontname='Georgia',fontsize=13)
axes.set_ylabel('Step Amplitude ('r'$\mu$m)', fontname='Georgia', fontsize=18)
axes.set_xlabel(' ', fontname='Georgia', fontsize=13)
axes.legend(prop=dict(size=10))
plt.savefig(upperdir + '/Writeup/Figures/steplength_across_stiffness.pdf')



stiffness = '10'

upperdir = '/'.join(os.getcwd().split('/')[:-1])
alldata_file = upperdir + '/Data/ForAnalysis/' + stiffness + 'kPa/Top/compiled_bears_by_strides.csv'
alldata = pd.read_csv(alldata_file)

# Look at how each leg pair correlates with speed
first_leg_speed = list(map(float,alldata['R1_stride_speed'].tolist()))
first_leg_speed.extend(list(map(float,alldata['L1_stride_speed'].tolist())))
second_leg_speed = list(map(float,alldata['R2_stride_speed'].tolist()))
second_leg_speed.extend(list(map(float,alldata['L2_stride_speed'].tolist())))
third_leg_speed = list(map(float,alldata['R3_stride_speed'].tolist()))
third_leg_speed.extend(list(map(float,alldata['L3_stride_speed'].tolist())))
back_leg_speed = list(map(float,alldata['R4_stride_speed'].tolist()))
back_leg_speed.extend(list(map(float,alldata['L4_stride_speed'].tolist())))

loc_legs_speed = first_leg_speed + second_leg_speed + third_leg_speed

first_period = list(map(float,alldata['R1_period'].tolist()))
first_period.extend(list(map(float,alldata['L1_period'].tolist())))
idx1 = (np.array(first_period) > 0) & (np.array(first_leg_speed) > 20)

second_period = list(map(float,alldata['R2_period'].tolist()))
second_period.extend(list(map(float,alldata['L2_period'].tolist())))
idx2 = (np.array(second_period) > 0) & (np.array(second_leg_speed) > 20)

third_period = list(map(float,alldata['R3_period'].tolist()))
third_period.extend(list(map(float,alldata['L3_period'].tolist())))
idx3 = (np.array(third_period) > 0) & (np.array(third_leg_speed) > 20)

loc_legs_period = first_period + second_period + third_period
idx = (np.array(loc_legs_period) > 0) & (np.array(loc_legs_speed) > 20)

fig, axes = plt.subplots()
xdata = np.array(loc_legs_speed)[idx]
ydata = np.array(loc_legs_period)[idx]
corr = scipy.stats.spearmanr(xdata,ydata)
corrtowrite = str(round(corr[0],2))
if corr[1] < 0.05:
    sig = 'p < 0.05'
    if corr[1] < 0.01:
        sig = 'p < 0.01'
    if corr[1] < 0.001:
        sig = 'p < 0.001'
else:
    sig = 'p = ' + str(round(corr[1],2))
plt.plot(xdata,ydata, color = 'darkorange', linestyle = '', marker = '.', alpha=0.4, label= '10 kPa: ' + r'$\rho$ = ' + corrtowrite + ', ' + sig)

stiffness = '50'

upperdir = '/'.join(os.getcwd().split('/')[:-1])
alldata_file = upperdir + '/Data/ForAnalysis/' + stiffness + 'kPa/Top/compiled_bears_by_strides.csv'
alldata = pd.read_csv(alldata_file)

# Look at how each leg pair correlates with speed
first_leg_speed = list(map(float,alldata['R1_stride_speed'].tolist()))
first_leg_speed.extend(list(map(float,alldata['L1_stride_speed'].tolist())))
second_leg_speed = list(map(float,alldata['R2_stride_speed'].tolist()))
second_leg_speed.extend(list(map(float,alldata['L2_stride_speed'].tolist())))
third_leg_speed = list(map(float,alldata['R3_stride_speed'].tolist()))
third_leg_speed.extend(list(map(float,alldata['L3_stride_speed'].tolist())))
back_leg_speed = list(map(float,alldata['R4_stride_speed'].tolist()))
back_leg_speed.extend(list(map(float,alldata['L4_stride_speed'].tolist())))

loc_legs_speed = first_leg_speed + second_leg_speed + third_leg_speed

first_period = list(map(float,alldata['R1_period'].tolist()))
first_period.extend(list(map(float,alldata['L1_period'].tolist())))
idx1 = (np.array(first_period) > 0) & (np.array(first_leg_speed) > 20)

second_period = list(map(float,alldata['R2_period'].tolist()))
second_period.extend(list(map(float,alldata['L2_period'].tolist())))
idx2 = (np.array(second_period) > 0) & (np.array(second_leg_speed) > 20)

third_period = list(map(float,alldata['R3_period'].tolist()))
third_period.extend(list(map(float,alldata['L3_period'].tolist())))
idx3 = (np.array(third_period) > 0) & (np.array(third_leg_speed) > 20)

loc_legs_period = first_period + second_period + third_period
idx = (np.array(loc_legs_period) > 0) & (np.array(loc_legs_speed) > 20)

xdata = np.array(loc_legs_speed)[idx]
ydata = np.array(loc_legs_period)[idx]
corr = scipy.stats.spearmanr(xdata,ydata)
corrtowrite = str(round(corr[0],2))
if corr[1] < 0.05:
    sig = 'p < 0.05'
    if corr[1] < 0.01:
        sig = 'p < 0.01'
    if corr[1] < 0.001:
        sig = 'p < 0.001'
else:
    sig = 'p = ' + str(round(corr[1],2))
plt.plot(xdata,ydata, color = 'blue', linestyle = '', marker = '.', alpha=0.4, label= '50 kPa: '+ r'$\rho$ = ' + corrtowrite + ', ' + sig)

plt.legend(fontsize=16)
plt.yticks(fontname='Georgia',fontsize=13)
plt.xticks(fontname='Georgia',fontsize=13)
axes.set_ylabel('Period (s)', fontname='Georgia', fontsize=18)
axes.set_xlabel(' ', fontname='Georgia', fontsize=13)
axes.legend(prop=dict(size=10))
plt.savefig(upperdir + '/Writeup/Figures/period_across_stiffness.pdf')
