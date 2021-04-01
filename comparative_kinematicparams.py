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

# This makes Figure 5a

####################################################
def func(x, a, b, c):
    return a*x**(b)+c
####################################################

stiffness1 = '50'
stiffness2 = '10'

upperdir = '/'.join(os.getcwd().split('/')[:-1])
alldata_file1 = upperdir + '/Data/ForAnalysis/' + stiffness1 + 'kPa/Top/compiled_bears_by_strides.csv'
alldata_file2 = upperdir + '/Data/ForAnalysis/' + stiffness2 + 'kPa/Top/compiled_bears_by_strides.csv'

alldata1 = pd.read_csv(alldata_file1)
alldata2 = pd.read_csv(alldata_file2)


# Look at how each leg pair correlates with speed
first_leg_speed1 = list(map(float,alldata1['R1_stride_speed'].tolist()))
first_leg_speed1.extend(list(map(float,alldata1['L1_stride_speed'].tolist())))
second_leg_speed1 = list(map(float,alldata1['R2_stride_speed'].tolist()))
second_leg_speed1.extend(list(map(float,alldata1['L2_stride_speed'].tolist())))
third_leg_speed1 = list(map(float,alldata1['R3_stride_speed'].tolist()))
third_leg_speed1.extend(list(map(float,alldata1['L3_stride_speed'].tolist())))
back_leg_speed1 = list(map(float,alldata1['R4_stride_speed'].tolist()))
back_leg_speed1.extend(list(map(float,alldata1['L4_stride_speed'].tolist())))
loc_legs_speed1 = first_leg_speed1 + second_leg_speed1 + third_leg_speed1
idx = np.array(loc_legs_speed1) > 20
loc_legs_speed1 = np.array(loc_legs_speed1)[idx]
#print(np.mean(loc_legs_speed1),np.std(loc_legs_speed1))

first_leg_speed2 = list(map(float,alldata2['R1_stride_speed'].tolist()))
first_leg_speed2.extend(list(map(float,alldata2['L1_stride_speed'].tolist())))
second_leg_speed2 = list(map(float,alldata2['R2_stride_speed'].tolist()))
second_leg_speed2.extend(list(map(float,alldata2['L2_stride_speed'].tolist())))
third_leg_speed2 = list(map(float,alldata2['R3_stride_speed'].tolist()))
third_leg_speed2.extend(list(map(float,alldata2['L3_stride_speed'].tolist())))
back_leg_speed2 = list(map(float,alldata2['R4_stride_speed'].tolist()))
back_leg_speed2.extend(list(map(float,alldata2['L4_stride_speed'].tolist())))
loc_legs_speed2 = first_leg_speed2 + second_leg_speed2 + third_leg_speed2
idx = np.array(loc_legs_speed2) > 20
loc_legs_speed2 = np.array(loc_legs_speed2)[idx]
#print(np.mean(loc_legs_speed2),np.std(loc_legs_speed2))

first_leg_pullback1 = list(map(float,alldata1['R1_pull_back'].tolist()))
first_leg_pullback1.extend(list(map(float,alldata1['L1_pull_back'].tolist())))
second_leg_pullback1 = list(map(float,alldata1['R2_pull_back'].tolist()))
second_leg_pullback1.extend(list(map(float,alldata1['L2_pull_back'].tolist())))
third_leg_pullback1 = list(map(float,alldata1['R3_pull_back'].tolist()))
third_leg_pullback1.extend(list(map(float,alldata1['L3_pull_back'].tolist())))
back_leg_pullback1 = list(map(float,alldata1['R4_pull_back'].tolist()))
back_leg_pullback1.extend(list(map(float,alldata1['L4_pull_back'].tolist())))
loc_legs_pullback1 = first_leg_pullback1 + second_leg_pullback1 + third_leg_pullback1
idx = np.array(loc_legs_pullback1) > 20
loc_legs_pullback1 = np.array(loc_legs_pullback1)[idx]

first_leg_pullback2 = list(map(float,alldata2['R1_pull_back'].tolist()))
first_leg_pullback2.extend(list(map(float,alldata2['L1_pull_back'].tolist())))
second_leg_pullback2 = list(map(float,alldata2['R2_pull_back'].tolist()))
second_leg_pullback2.extend(list(map(float,alldata2['L2_pull_back'].tolist())))
third_leg_pullback2 = list(map(float,alldata2['R3_pull_back'].tolist()))
third_leg_pullback2.extend(list(map(float,alldata2['L3_pull_back'].tolist())))
back_leg_pullback2 = list(map(float,alldata2['R4_pull_back'].tolist()))
back_leg_pullback2.extend(list(map(float,alldata2['L4_stride_pullback'].tolist())))
loc_legs_pullback2 = first_leg_pullback2 + second_leg_pullback2 + third_leg_pullback2
idx = np.array(loc_legs_pullback2) > 20
loc_legs_pullback2 = np.array(loc_legs_pullback2)[idx]

first_step1 = list(map(float,alldata1['R1_step_length'].tolist()))
first_step1.extend(list(map(float,alldata1['L1_step_length'].tolist())))
second_step1 = list(map(float,alldata1['R2_step_length'].tolist()))
second_step1.extend(list(map(float,alldata1['L2_step_length'].tolist())))
third_step1 = list(map(float,alldata1['R3_step_length'].tolist()))
third_step1.extend(list(map(float,alldata1['L3_step_length'].tolist())))
loc_legs_step1 = first_step1 + second_step1 + third_step1
idx = (np.array(loc_legs_step1) > 0)
loc_legs_step1 = np.array(loc_legs_step1)[idx]

first_step2 = list(map(float,alldata2['R1_step_length'].tolist()))
first_step2.extend(list(map(float,alldata2['L1_step_length'].tolist())))
second_step2 = list(map(float,alldata2['R2_step_length'].tolist()))
second_step2.extend(list(map(float,alldata2['L2_step_length'].tolist())))
third_step2 = list(map(float,alldata2['R3_step_length'].tolist()))
third_step2.extend(list(map(float,alldata2['L3_step_length'].tolist())))
loc_legs_step2 = first_step2 + second_step2 + third_step2
idx = (np.array(loc_legs_step2) > 0)
loc_legs_step2 = np.array(loc_legs_step2)[idx]

first_period1 = list(map(float,alldata1['R1_period'].tolist()))
first_period1.extend(list(map(float,alldata1['L1_period'].tolist())))
second_period1 = list(map(float,alldata1['R2_period'].tolist()))
second_period1.extend(list(map(float,alldata1['L2_period'].tolist())))
third_period1 = list(map(float,alldata1['R3_period'].tolist()))
third_period1.extend(list(map(float,alldata1['L3_period'].tolist())))
loc_legs_period1 = first_period1 + second_period1 + third_period1
idx = (np.array(loc_legs_period1) > 0)
loc_legs_period1 = np.array(loc_legs_period1)[idx]

first_period2 = list(map(float,alldata2['R1_period'].tolist()))
first_period2.extend(list(map(float,alldata2['L1_period'].tolist())))
second_period2 = list(map(float,alldata2['R2_period'].tolist()))
second_period2.extend(list(map(float,alldata2['L2_period'].tolist())))
third_period2 = list(map(float,alldata2['R3_period'].tolist()))
third_period2.extend(list(map(float,alldata2['L3_period'].tolist())))
loc_legs_period2 = first_period2 + second_period2 + third_period2
idx = (np.array(loc_legs_period2) > 0)
loc_legs_period2 = np.array(loc_legs_period2)[idx]

first_swing1 = list(map(float,alldata1['R1_swing'].tolist()))
first_swing1.extend(list(map(float,alldata1['L1_swing'].tolist())))
second_swing1 = list(map(float,alldata1['R2_swing'].tolist()))
second_swing1.extend(list(map(float,alldata1['L2_swing'].tolist())))
third_swing1 = list(map(float,alldata1['R3_swing'].tolist()))
third_swing1.extend(list(map(float,alldata1['L3_swing'].tolist())))
loc_legs_swing1 = first_swing1 + second_swing1 + third_swing1

first_swing2 = list(map(float,alldata2['R1_swing'].tolist()))
first_swing2.extend(list(map(float,alldata2['L1_swing'].tolist())))
second_swing2 = list(map(float,alldata2['R2_swing'].tolist()))
second_swing2.extend(list(map(float,alldata2['L2_swing'].tolist())))
third_swing2 = list(map(float,alldata2['R3_swing'].tolist()))
third_swing2.extend(list(map(float,alldata2['L3_swing'].tolist())))
loc_legs_swing2 = first_swing2 + second_swing2 + third_swing2

first_stance1 = list(map(float,alldata1['R1_stance'].tolist()))
first_stance1.extend(list(map(float,alldata1['L1_stance'].tolist())))
second_stance1 = list(map(float,alldata1['R2_stance'].tolist()))
second_stance1.extend(list(map(float,alldata1['L2_stance'].tolist())))
third_stance1 = list(map(float,alldata1['R3_stance'].tolist()))
third_stance1.extend(list(map(float,alldata1['L3_stance'].tolist())))
loc_legs_stance1 = first_stance1 + second_stance1 + third_stance1
idx = ((np.array(loc_legs_swing1) > 0) & (np.array(loc_legs_stance1) > 0))
loc_legs_stance1 = np.array(loc_legs_stance1)[idx]
loc_legs_swing1 = np.array(loc_legs_swing1)[idx]

first_stance2 = list(map(float,alldata2['R1_stance'].tolist()))
first_stance2.extend(list(map(float,alldata2['L1_stance'].tolist())))
second_stance2 = list(map(float,alldata2['R2_stance'].tolist()))
second_stance2.extend(list(map(float,alldata2['L2_stance'].tolist())))
third_stance2 = list(map(float,alldata2['R3_stance'].tolist()))
third_stance2.extend(list(map(float,alldata2['L3_stance'].tolist())))
loc_legs_stance2 = first_stance2 + second_stance2 + third_stance2
idx = ((np.array(loc_legs_stance2) > 0) & (np.array(loc_legs_swing2) > 0) )
loc_legs_stance2 = np.array(loc_legs_stance2)[idx]
loc_legs_swing2 = np.array(loc_legs_swing2)[idx]

fig, axes = plt.subplots(2,2)
colors=['blue', 'red']
df = [loc_legs_step1,loc_legs_step2]
ax = sns.violinplot(data=df,orient='h', inner='quartiles',ax=axes[0][0], palette=colors)
for l in ax.lines:
    l.set_linestyle('--')
    l.set_linewidth(1)
    l.set_color('yellow')
axes[0][0].set_yticks([])
axes[0][0].set_xlabel('Step Amplitude ('r'$\mu$m)',fontname='Georgia', fontsize=13)

df = [loc_legs_period1,loc_legs_period2]
ax = sns.violinplot(data=df,orient='h',inner='quartiles', ax=axes[0][1],palette=colors)
for l in ax.lines:
    l.set_linestyle('--')
    l.set_linewidth(1)
    l.set_color('yellow')
axes[0][1].set_yticks([])
axes[0][1].set_xlabel('Period (s)',fontname='Georgia', fontsize=13)

df = [loc_legs_swing1,loc_legs_swing2]
ax = sns.violinplot(data=df,orient='h',inner='quartiles',ax=axes[1][0],palette=colors)
for l in ax.lines:
    l.set_linestyle('--')
    l.set_linewidth(1)
    l.set_color('yellow')
axes[1][0].set_yticks([])
axes[1][0].set_xlabel('Swing (s)',fontname='Georgia', fontsize=13)

df = [loc_legs_stance1,loc_legs_stance2]
ax = sns.violinplot(data=df,orient='h',inner='quartiles',ax=axes[1][1],palette=colors)
for l in ax.lines:
    l.set_linestyle('--')
    l.set_linewidth(1)
    l.set_color('yellow')
axes[1][1].set_yticks([])
axes[1][1].set_xlabel('Stance (s)',fontname='Georgia', fontsize=13)

plt.tight_layout()
plt.savefig(upperdir+'/Writeup/Figures/' + 'compare_kinematics_by_stiffness.pdf')

#

#
#fig, axes = plt.subplots()
#xdata = np.array(first_leg_speed)[idx1]
#ydata = np.array(first_step)[idx1]
#pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
#residuals = ydata - func(xdata, *pars)
#ss_res = np.sum(residuals**2)
#ss_tot = np.sum((ydata-np.mean(ydata))**2)
#r_squared = 1 - (ss_res / ss_tot)
#corrtowrite1 = str(round(r_squared,2))
#plt.plot(xdata,ydata, color = 'blue', linestyle = '', marker = '.', alpha=0.4,label='First leg pair, ' + r'$R^2$ = ' + corrtowrite1)
#plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color='blue')
#
#xdata = np.array(second_leg_speed)[idx2]
#ydata = np.array(second_step)[idx2]
#pars, cov = curve_fit(func,xdata,ydata, p0=[1, -0.1, 1], maxfev=100000)
#residuals = ydata - func(xdata, *pars)
#ss_res = np.sum(residuals**2)
#ss_tot = np.sum((ydata-np.mean(ydata))**2)
#r_squared = 1 - (ss_res / ss_tot)
#corrtowrite2 = str(round(r_squared,2))
#plt.plot(xdata,ydata, color = 'green', linestyle = '', marker = '.', alpha=0.4, label='Second leg pair, ' + r'$R^2$ = ' + corrtowrite2)
#plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color = 'green')
#
#xdata = np.array(third_leg_speed)[idx3]
#ydata = np.array(third_step)[idx3]
#pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
#residuals = ydata - func(xdata, *pars)
#ss_res = np.sum(residuals**2)
#ss_tot = np.sum((ydata-np.mean(ydata))**2)
#r_squared = 1 - (ss_res / ss_tot)
#corrtowrite3 = str(round(r_squared,2))
#plt.plot(xdata,ydata, color = 'gold', linestyle = '', marker = '.', alpha=0.4,label='Third leg pair, ' + r'$R^2$ = ' + corrtowrite3)
#plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color = 'gold')
#
#xdata = np.array(back_leg_speed)[idx4]
#ydata = np.array(back_step)[idx4]
#pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
#residuals = ydata - func(xdata, *pars)
#ss_res = np.sum(residuals**2)
#ss_tot = np.sum((ydata-np.mean(ydata))**2)
#r_squared = 1 - (ss_res / ss_tot)
#corrtowrite4 = str(round(r_squared,2))
#plt.plot(xdata,ydata, color = 'red', linestyle = '', marker = '.', alpha=0.4, label='Back leg pair, ' + r'$R^2$ = ' + corrtowrite4)
#plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color='red')
#
#plt.legend()
#plt.yticks(fontname='Georgia',fontsize=11)
#plt.xticks(fontname='Georgia',fontsize=11)
#plt.xlim([0,400])
#plt.ylim([0,200])
#axes.set_ylabel('Step Amplitude ('r'$\mu$m)', fontname='Georgia', fontsize=13)
#axes.set_xlabel(' ', fontname='Georgia', fontsize=13)
#plt.savefig(upperdir+'/Writeup/Figures/' + stiffness + 'kPa_byleg_steplength_vs_speed.pdf')
#
#fig, axes = plt.subplots()
#xdata = np.array(loc_legs_speed)[idx]
#ydata = np.array(loc_legs_step)[idx]
##pars, cov = curve_fit(func,xdata,ydata, p0=[0,6,0], maxfev=100000)
##residuals = ydata - func(xdata, *pars)
##ss_res = np.sum(residuals**2)
##ss_tot = np.sum((ydata-np.mean(ydata))**2)
##r_squared = 1 - (ss_res / ss_tot)
##corrtowrite = str(round(r_squared,2))
#corr = scipy.stats.spearmanr(xdata,ydata)
#corrtowrite = str(round(corr[0],2))
#if corr[1] < 0.05:
#    sig = 'p < 0.05'
#    if corr[1] < 0.01:
#        sig = 'p < 0.01'
#    if corr[1] < 0.001:
#        sig = 'p < 0.001'
#else:
#    sig = 'p = ' + str(round(corr[1],2))
#plt.plot(xdata,ydata, color = 'green', linestyle = '', marker = '.', alpha=0.7, label= r'$\rho$ = ' + corrtowrite + ', ' + sig)
##plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color = 'green')
#
#plt.legend(fontsize=16)
#plt.xlim([0,400])
#plt.ylim([0,200])
#plt.yticks(fontname='Georgia',fontsize=13)
#plt.xticks(fontname='Georgia',fontsize=13)
#axes.set_ylabel('Step Amplitude ('r'$\mu$m)', fontname='Georgia', fontsize=18)
#axes.set_xlabel(' ', fontname='Georgia', fontsize=13)
#plt.savefig(upperdir+'/Writeup/Figures/' + stiffness + 'kPa_steplength_vs_speed.pdf')
#
## speed vs period
#first_period = list(map(float,alldata['R1_period'].tolist()))
#first_period.extend(list(map(float,alldata['L1_period'].tolist())))
#idx1 = (np.array(first_period) > 0) & (np.array(first_leg_speed) > 20)
#
#second_period = list(map(float,alldata['R2_period'].tolist()))
#second_period.extend(list(map(float,alldata['L2_period'].tolist())))
#idx2 = (np.array(second_period) > 0) & (np.array(second_leg_speed) > 20)
#
#third_period = list(map(float,alldata['R3_period'].tolist()))
#third_period.extend(list(map(float,alldata['L3_period'].tolist())))
#idx3 = (np.array(third_period) > 0) & (np.array(third_leg_speed) > 20)
#
#back_period = list(map(float,alldata['R4_period'].tolist()))
#back_period.extend(list(map(float,alldata['L4_period'].tolist())))
#idx4 = (np.array(back_period) > 0) & (np.array(back_leg_speed) > 20)
#
#loc_legs_period = first_period + second_period + third_period
#idx = (np.array(loc_legs_period) > 0) & (np.array(loc_legs_speed) > 20)
#
#fig, axes = plt.subplots()
#xdata = np.array(first_leg_speed)[idx1]
#ydata = np.array(first_period)[idx1]
#pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
#residuals = ydata - func(xdata, *pars)
#ss_res = np.sum(residuals**2)
#ss_tot = np.sum((ydata-np.mean(ydata))**2)
#r_squared = 1 - (ss_res / ss_tot)
#corrtowrite1 = str(round(r_squared,2))
#plt.plot(np.array(first_leg_speed)[idx1],np.array(first_period)[idx1], color = 'blue', linestyle = '', marker = '.', alpha=0.4,label='First leg pair, ' + r'$R^2$ = ' + corrtowrite1)
#plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color='blue')
#
#xdata = np.array(second_leg_speed)[idx2]
#ydata = np.array(second_period)[idx2]
#pars, cov = curve_fit(func,xdata,ydata, p0=[1, -0.1, 1], maxfev=100000)
#residuals = ydata - func(xdata, *pars)
#ss_res = np.sum(residuals**2)
#ss_tot = np.sum((ydata-np.mean(ydata))**2)
#r_squared = 1 - (ss_res / ss_tot)
#corrtowrite2 = str(round(r_squared,2))
#plt.plot(second_leg_speed,second_period, color = 'green', linestyle = '', marker = '.', alpha=0.4, label='Second leg pair, ' + r'$R^2$ = ' + corrtowrite2)
#plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color = 'green')
#
#xdata = np.array(third_leg_speed)[idx3]
#ydata = np.array(third_period)[idx3]
#pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
#residuals = ydata - func(xdata, *pars)
#ss_res = np.sum(residuals**2)
#ss_tot = np.sum((ydata-np.mean(ydata))**2)
#r_squared = 1 - (ss_res / ss_tot)
#corrtowrite3 = str(round(r_squared,2))
#plt.plot(third_leg_speed,third_period, color = 'gold', linestyle = '', marker = '.', alpha=0.4,label='Third leg pair, ' + r'$R^2$ = ' + corrtowrite3)
#plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color = 'gold')
#
#xdata = np.array(back_leg_speed)[idx4]
#ydata = np.array(back_period)[idx4]
#pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
#residuals = ydata - func(xdata, *pars)
#ss_res = np.sum(residuals**2)
#ss_tot = np.sum((ydata-np.mean(ydata))**2)
#r_squared = 1 - (ss_res / ss_tot)
#corrtowrite4 = str(round(r_squared,2))
#plt.plot(back_leg_speed,back_period, color = 'red', linestyle = '', marker = '.', alpha=0.4, label='Back leg pair, ' + r'$R^2$ = ' + corrtowrite4)
#plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color='red')
#
#plt.legend()
#plt.yticks(fontname='Georgia',fontsize=11)
#plt.xticks(fontname='Georgia',fontsize=11)
#plt.xlim([0,400])
#plt.ylim([0,2.5])
#axes.set_ylabel('Period ('r's)', fontname='Georgia', fontsize=13)
#axes.set_xlabel(' ', fontname='Georgia', fontsize=13)
#plt.savefig(upperdir+'/Writeup/Figures/' + stiffness + 'kPa_byleg_period_vs_speed.pdf')
#
#fig, axes = plt.subplots()
#xdata = np.array(loc_legs_speed)[idx]
#ydata = np.array(loc_legs_period)[idx]
##pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
##residuals = ydata - func(xdata, *pars)
##ss_res = np.sum(residuals**2)
##ss_tot = np.sum((ydata-np.mean(ydata))**2)
##r_squared = 1 - (ss_res / ss_tot)
##corrtowrite = str(round(r_squared,2))
#corr = scipy.stats.spearmanr(xdata,ydata)
#corrtowrite = str(round(corr[0],2))
#if corr[1] < 0.05:
#    sig = 'p < 0.05'
#    if corr[1] < 0.01:
#        sig = 'p < 0.01'
#    if corr[1] < 0.001:
#        sig = 'p < 0.001'
#else:
#    sig = 'p = ' + str(round(corr[1],2))
#plt.plot(np.array(loc_legs_speed)[idx],np.array(loc_legs_period)[idx], color = 'purple', linestyle = '', marker = '.', alpha=0.7, label= r'$\rho$ = ' + corrtowrite + ', ' + sig)
##plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color = 'purple')
#
#plt.legend(fontsize=16)
#plt.xlim([0,400])
#plt.ylim([0,2.5])
#plt.yticks(fontname='Georgia',fontsize=13)
#plt.xticks(fontname='Georgia',fontsize=13)
#axes.set_ylabel('Period (s)', fontname='Georgia', fontsize=18)
#axes.set_xlabel(' ', fontname='Georgia', fontsize=13)
#plt.savefig(upperdir+'/Writeup/Figures/' + stiffness + 'kPa_period_vs_speed.pdf')
#
#
## speed vs stance
#first_stance = list(map(float,alldata['R1_stance'].tolist()))
#first_stance.extend(list(map(float,alldata['L1_stance'].tolist())))
#idx1 = (np.array(first_stance) > 0) & (np.array(first_leg_speed) > 20)
#
#second_stance = list(map(float,alldata['R2_stance'].tolist()))
#second_stance.extend(list(map(float,alldata['L2_stance'].tolist())))
#idx2 = (np.array(second_stance) > 0) & (np.array(second_leg_speed) > 20)
#
#third_stance = list(map(float,alldata['R3_stance'].tolist()))
#third_stance.extend(list(map(float,alldata['L3_stance'].tolist())))
#idx3 = (np.array(third_stance) > 0) & (np.array(third_leg_speed) > 20)
#
#back_stance = list(map(float,alldata['R4_stance'].tolist()))
#back_stance.extend(list(map(float,alldata['L4_stance'].tolist())))
#idx4 = (np.array(back_stance) > 0) & (np.array(back_leg_speed) > 20)
#
#loc_legs_stance = first_stance + second_stance + third_stance
#idx = (np.array(loc_legs_stance) > 0) & (np.array(loc_legs_speed) > 20)
#
#
#fig, axes = plt.subplots()
#xdata = np.array(first_leg_speed)[idx1]
#ydata = np.array(first_stance)[idx1]
#pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
#residuals = ydata - func(xdata, *pars)
#ss_res = np.sum(residuals**2)
#ss_tot = np.sum((ydata-np.mean(ydata))**2)
#r_squared = 1 - (ss_res / ss_tot)
#corrtowrite1 = str(round(r_squared,2))
#plt.plot(first_leg_speed,first_stance, color = 'blue', linestyle = '', marker = '.', alpha=0.4,label='First leg pair, ' + r'$R^2$ = ' + corrtowrite1)
#plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color='blue')
#
#xdata = np.array(second_leg_speed)[idx2]
#ydata = np.array(second_stance)[idx2]
#pars, cov = curve_fit(func,xdata,ydata, p0=[0, 0, 0], maxfev=100000)
#residuals = ydata - func(xdata, *pars)
#ss_res = np.sum(residuals**2)
#ss_tot = np.sum((ydata-np.mean(ydata))**2)
#r_squared = 1 - (ss_res / ss_tot)
#corrtowrite2 = str(round(r_squared,2))
#plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color = 'green')
#plt.plot(second_leg_speed,second_stance, color = 'green', linestyle = '', marker = '.', alpha=0.4, label='Second leg pair, ' + r'$R^2$ = ' + corrtowrite2)
#
#xdata = np.array(third_leg_speed)[idx3]
#ydata = np.array(third_stance)[idx3]
#pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
#residuals = ydata - func(xdata, *pars)
#ss_res = np.sum(residuals**2)
#ss_tot = np.sum((ydata-np.mean(ydata))**2)
#r_squared = 1 - (ss_res / ss_tot)
#corrtowrite3 = str(round(r_squared,2))
#plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color = 'gold')
#plt.plot(third_leg_speed,third_stance, color = 'gold', linestyle = '', marker = '.', alpha=0.4,label='Third leg pair, ' + r'$R^2$ = ' + corrtowrite3)
#
#xdata = np.array(back_leg_speed)[idx4]
#ydata = np.array(back_stance)[idx4]
#pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
#residuals = ydata - func(xdata, *pars)
#ss_res = np.sum(residuals**2)
#ss_tot = np.sum((ydata-np.mean(ydata))**2)
#r_squared = 1 - (ss_res / ss_tot)
#corrtowrite4 = str(round(r_squared,2))
#plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color = 'red')
#plt.plot(back_leg_speed,back_stance, color = 'red', linestyle = '', marker = '.', alpha=0.4, label='Back leg pair, ' + r'$R^2$ = ' + corrtowrite4)
#plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color='red')
#
#plt.xlim([0,400])
#plt.ylim([0,2.5])
#
#plt.legend()
#plt.yticks(fontname='Georgia',fontsize=11)
#plt.xticks(fontname='Georgia',fontsize=11)
#axes.set_ylabel('Stance duration ('r's)', fontname='Georgia', fontsize=13)
#axes.set_xlabel(' ', fontname='Georgia', fontsize=13)
#plt.savefig(upperdir+'/Writeup/Figures/' + stiffness + 'kPa_byleg_stance_vs_speed.pdf')
#
#fig, axes = plt.subplots()
#xdata = np.array(loc_legs_speed)[idx]
#ydata = np.array(loc_legs_stance)[idx]
#pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
#residuals = ydata - func(xdata, *pars)
#ss_res = np.sum(residuals**2)
#ss_tot = np.sum((ydata-np.mean(ydata))**2)
#r_squared = 1 - (ss_res / ss_tot)
#corrtowrite_stance = str(round(r_squared,2))
##plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color = 'blue')
#plt.plot(np.array(loc_legs_speed)[idx],np.array(loc_legs_stance)[idx], color = 'blue', linestyle = '', marker = '.', alpha=0.7, label= r'$R^2$ = ' + corrtowrite_stance)
#
#plt.legend(fontsize=16)
#plt.xlim([0,400])
#plt.ylim([0,2.5])
#plt.yticks(fontname='Georgia',fontsize=13)
#plt.xticks(fontname='Georgia',fontsize=13)
#axes.set_ylabel('Stance duration (s)', fontname='Georgia', fontsize=18)
#axes.set_xlabel(' ', fontname='Georgia', fontsize=13)
#plt.savefig(upperdir+'/Writeup/Figures/' + stiffness + 'kPa_stance_vs_speed.pdf')
#
##speed vs swing
#first_swing = list(map(float,alldata['R1_swing'].tolist()))
#first_swing.extend(list(map(float,alldata['L1_swing'].tolist())))
#idx1 = (np.array(first_swing) > 0) & (np.array(first_leg_speed) > 20)
#
#second_swing = list(map(float,alldata['R2_swing'].tolist()))
#second_swing.extend(list(map(float,alldata['L2_swing'].tolist())))
#idx2 = (np.array(second_swing) > 0) & (np.array(second_leg_speed) > 20)
#
#third_swing = list(map(float,alldata['R3_swing'].tolist()))
#third_swing.extend(list(map(float,alldata['L3_swing'].tolist())))
#idx3 = (np.array(third_swing) > 0) & (np.array(third_leg_speed) > 20)
#
#back_swing = list(map(float,alldata['R4_swing'].tolist()))
#back_swing.extend(list(map(float,alldata['L4_swing'].tolist())))
#idx4 = (np.array(back_swing) > 0) & (np.array(back_leg_speed) > 20)
#
#loc_legs_swing = first_swing + second_swing + third_swing
#idx = (np.array(loc_legs_swing) > 0) & (np.array(loc_legs_speed) > 20)
#
#fig, axes = plt.subplots()
#xdata = np.array(first_leg_speed)[idx1]
#ydata = np.array(first_swing)[idx1]
#pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
#residuals = ydata - func(xdata, *pars)
#ss_res = np.sum(residuals**2)
#ss_tot = np.sum((ydata-np.mean(ydata))**2)
#r_squared = 1 - (ss_res / ss_tot)
#corrtowrite1 = str(round(r_squared,2))
#plt.plot(first_leg_speed,first_swing, color = 'blue', linestyle = '', marker = '.', alpha=0.4,label='First leg pair, ' + r'$R^2$ = ' + corrtowrite1)
#plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color='blue')
#
#xdata = np.array(second_leg_speed)[idx2]
#ydata = np.array(second_swing)[idx2]
#pars, cov = curve_fit(func,xdata,ydata, p0=[0, 0, 0], maxfev=100000)
#residuals = ydata - func(xdata, *pars)
#ss_res = np.sum(residuals**2)
#ss_tot = np.sum((ydata-np.mean(ydata))**2)
#r_squared = 1 - (ss_res / ss_tot)
#corrtowrite2 = str(round(r_squared,2))
#plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color = 'green')
#plt.plot(second_leg_speed,second_swing, color = 'green', linestyle = '', marker = '.', alpha=0.4, label='Second leg pair, ' + r'$R^2$ = ' + corrtowrite2)
#
#xdata = np.array(third_leg_speed)[idx3]
#ydata = np.array(third_swing)[idx3]
#pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
#residuals = ydata - func(xdata, *pars)
#ss_res = np.sum(residuals**2)
#ss_tot = np.sum((ydata-np.mean(ydata))**2)
#r_squared = 1 - (ss_res / ss_tot)
#corrtowrite3 = str(round(r_squared,2))
#plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color = 'gold')
#plt.plot(third_leg_speed,third_swing, color = 'gold', linestyle = '', marker = '.', alpha=0.4,label='Third leg pair, ' + r'$R^2$ = ' + corrtowrite3)
#
#
#xdata = np.array(back_leg_speed)[idx4]
#ydata = np.array(back_swing)[idx4]
#pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
#residuals = ydata - func(xdata, *pars)
#ss_res = np.sum(residuals**2)
#ss_tot = np.sum((ydata-np.mean(ydata))**2)
#r_squared = 1 - (ss_res / ss_tot)
#corrtowrite3 = str(round(r_squared,2))
#plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color = 'red')
#plt.plot(back_leg_speed,back_swing, color = 'red', linestyle = '', marker = '.', alpha=0.4, label='Back leg pair, ' + r'$R^2$ = ' + corrtowrite4)
#plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color='red')
#
#plt.legend()
#plt.xlim([0,400])
#plt.ylim([0,2.5])
#plt.yticks(fontname='Georgia',fontsize=11)
#plt.xticks(fontname='Georgia',fontsize=11)
#axes.set_ylabel('Swing duration (s)', fontname='Georgia', fontsize=13)
#axes.set_xlabel(' ', fontname='Georgia', fontsize=13)
#plt.savefig(upperdir+'/Writeup/Figures/' + stiffness + 'kPa_byleg_swing_vs_speed.pdf')
#
#
#fig, axes = plt.subplots()
#xdata = np.array(loc_legs_speed)[idx]
#ydata = np.array(loc_legs_swing)[idx]
#pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
#residuals = ydata - func(xdata, *pars)
#ss_res = np.sum(residuals**2)
#ss_tot = np.sum((ydata-np.mean(ydata))**2)
#r_squared = 1 - (ss_res / ss_tot)
#corrtowrite = str(round(r_squared,2))
##plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color = 'red')
#plt.plot(np.array(loc_legs_speed)[idx],np.array(loc_legs_swing)[idx], color = 'red', linestyle = '', marker = '.', alpha=0.4, label= r'$R^2$ = ' + corrtowrite)
#
#plt.legend()
#plt.xlim([0,400])
#plt.ylim([0,2.5])
#plt.yticks(fontname='Georgia',fontsize=11)
#plt.xticks(fontname='Georgia',fontsize=11)
#axes.set_ylabel('Swing duration (s)', fontname='Georgia', fontsize=13)
#axes.set_xlabel(' ', fontname='Georgia', fontsize=13)
#plt.savefig(upperdir+'/Writeup/Figures/' + stiffness + 'kPa_swing_vs_speed.pdf')
#
#fig, axes = plt.subplots()
#xdata = np.array(loc_legs_speed)[idx]
#ydata = np.array(loc_legs_swing)[idx]
##pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
##residuals = ydata - func(xdata, *pars)
##ss_res = np.sum(residuals**2)
##ss_tot = np.sum((ydata-np.mean(ydata))**2)
##r_squared = 1 - (ss_res / ss_tot)
##corrtowrite = str(round(r_squared,2))
##plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color = 'red')
#corr = scipy.stats.spearmanr(xdata,ydata)
#corrtowrite = str(round(corr[0],2))
#if corr[1] < 0.05:
#    sig = 'p < 0.05'
#    if corr[1] < 0.01:
#        sig = 'p < 0.01'
#    if corr[1] < 0.001:
#        sig = 'p < 0.001'
#else:
#    sig = 'p = ' + str(round(corr[1],2))
#plt.plot(np.array(loc_legs_speed)[idx],np.array(loc_legs_swing)[idx], color = 'red', linestyle = '', marker = '.', alpha=0.8, label= 'Swing: ' + r'$\rho$ = ' + corrtowrite + ', ' + sig)
#xdata = np.array(loc_legs_speed)[idx]
#ydata = np.array(loc_legs_stance)[idx]
##pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
##residuals = ydata - func(xdata, *pars)
##ss_res = np.sum(residuals**2)
##ss_tot = np.sum((ydata-np.mean(ydata))**2)
##r_squared = 1 - (ss_res / ss_tot)
##corrtowrite_stance = str(round(r_squared,2))
##plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color = 'blue')
#corr = scipy.stats.spearmanr(xdata,ydata)
#corrtowrite = str(round(corr[0],2))
#if corr[1] < 0.05:
#    sig = 'p < 0.05'
#    if corr[1] < 0.01:
#        sig = 'p < 0.01'
#    if corr[1] < 0.001:
#        sig = 'p < 0.001'
#else:
#    sig = 'p = ' + str(round(corr[1],2))
#plt.plot(np.array(loc_legs_speed)[idx],np.array(loc_legs_stance)[idx], color = 'blue', linestyle = '', marker = '.', alpha=0.7, label= 'Stance: ' + r'$\rho$ = ' + corrtowrite + ', ' + sig)
#
#plt.legend(fontsize=16)
#plt.xlim([0,400])
#plt.ylim([0,2.5])
#plt.yticks(fontname='Georgia',fontsize=13)
#plt.xticks(fontname='Georgia',fontsize=13)
#axes.set_ylabel('Phase duration (s)', fontname='Georgia', fontsize=18)
#axes.set_xlabel(' ', fontname='Georgia', fontsize=13)
#plt.savefig(upperdir+'/Writeup/Figures/' + stiffness + 'kPa_swingstance_vs_speed.pdf')
