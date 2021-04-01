# This makes the inset to Figure 3
###########################################################################
#!/usr/bin/python
import re, math, sys, os, random,scipy
import numpy as np
import pylab as pl
from matplotlib import collections  as mc
import pandas as pd
from optparse import OptionParser
import matplotlib.pyplot as plt
import glob, csv
from scipy.stats import mode
import seaborn as sns
import matplotlib.pylab as pylab
import matplotlib
import scipy
from matplotlib.font_manager import FontProperties
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

###########################################################################
def grouper(iterable,n):
    args = [iter(iterable)]*n
    return zip(*args)
###########################################################################
def func(x, a, b, c):
    return a*x**(b)+c
###########################################################################

# Say, "the default sans-serif font is COMIC SANS"
matplotlib.rcParams['font.sans-serif'] = 'Georgia'
# Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams['font.family'] = "sans-serif"
params = {'legend.fontsize': 'large',
 'figure.figsize': (10, 5),
'axes.labelsize': 'xx-large',
'axes.titlesize':'x-large',
'xtick.labelsize':'x-large',
'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

stiffness = '50'

upperdir = '/'.join(os.getcwd().split('/')[:-1])
alldata_file = upperdir + '/Data/ForAnalysis/' + stiffness + 'kPa/Top/compiled_bears_by_strides.csv'
alldata = pd.read_csv(alldata_file)

first_leg_speed = list(map(float,alldata['R1_stride_speed'].tolist()))
first_leg_speed.extend(list(map(float,alldata['L1_stride_speed'].tolist())))
second_leg_speed = list(map(float,alldata['R2_stride_speed'].tolist()))
second_leg_speed.extend(list(map(float,alldata['L2_stride_speed'].tolist())))
third_leg_speed = list(map(float,alldata['R3_stride_speed'].tolist()))
third_leg_speed.extend(list(map(float,alldata['L3_stride_speed'].tolist())))
back_leg_speed = list(map(float,alldata['R4_stride_speed'].tolist()))
back_leg_speed.extend(list(map(float,alldata['L4_stride_speed'].tolist())))

loc_legs_speed = first_leg_speed + second_leg_speed + third_leg_speed

# speed vs df
first_df = list(map(float,alldata['R1_duty_factor'].tolist()))
first_df.extend(list(map(float,alldata['L1_duty_factor'].tolist())))
idx1 = (np.array(first_df) > 0) & (np.array(first_leg_speed) > 20)

second_df = list(map(float,alldata['R2_duty_factor'].tolist()))
second_df.extend(list(map(float,alldata['L2_duty_factor'].tolist())))
idx2 = (np.array(second_df) > 0) & (np.array(second_leg_speed) > 20)

third_df = list(map(float,alldata['R3_duty_factor'].tolist()))
third_df.extend(list(map(float,alldata['L3_duty_factor'].tolist())))
idx3 = (np.array(third_df) > 0) & (np.array(third_leg_speed) > 20)

back_df = list(map(float,alldata['R4_duty_factor'].tolist()))
back_df.extend(list(map(float,alldata['L4_duty_factor'].tolist())))
idx4 = (np.array(back_df) > 0) & (np.array(back_leg_speed) > 20)

loc_legs_df = first_df + second_df + third_df
idx = (np.array(loc_legs_df) > 0) & (np.array(loc_legs_speed) > 20)


fig, axes = plt.subplots()
xdata = np.array(first_leg_speed)[idx1]
ydata = np.array(first_df)[idx1]
pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
residuals = ydata - func(xdata, *pars)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((ydata-np.mean(ydata))**2)
r_squared = 1 - (ss_res / ss_tot)
corrtowrite1 = str(round(r_squared,2))
plt.plot(first_leg_speed,first_df, color = 'blue', linestyle = '', marker = '.', alpha=0.4,label='First leg pair, ' + r'$R^2$ = ' + corrtowrite1)
plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color='blue')

xdata = np.array(second_leg_speed)[idx2]
ydata = np.array(second_df)[idx2]
pars, cov = curve_fit(func,xdata,ydata, p0=[0, 0, 0], maxfev=100000)
residuals = ydata - func(xdata, *pars)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((ydata-np.mean(ydata))**2)
r_squared = 1 - (ss_res / ss_tot)
corrtowrite2 = str(round(r_squared,2))
plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color = 'green')
plt.plot(second_leg_speed,second_df, color = 'green', linestyle = '', marker = '.', alpha=0.4, label='Second leg pair, ' + r'$R^2$ = ' + corrtowrite2)

xdata = np.array(third_leg_speed)[idx3]
ydata = np.array(third_df)[idx3]
pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
residuals = ydata - func(xdata, *pars)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((ydata-np.mean(ydata))**2)
r_squared = 1 - (ss_res / ss_tot)
corrtowrite3 = str(round(r_squared,2))
plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color = 'gold')
plt.plot(third_leg_speed,third_df, color = 'gold', linestyle = '', marker = '.', alpha=0.4,label='Third leg pair, ' + r'$R^2$ = ' + corrtowrite3)

xdata = np.array(back_leg_speed)[idx4]
ydata = np.array(back_df)[idx4]
pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
residuals = ydata - func(xdata, *pars)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((ydata-np.mean(ydata))**2)
r_squared = 1 - (ss_res / ss_tot)
corrtowrite4 = str(round(r_squared,2))
plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color = 'red')
plt.plot(back_leg_speed,back_df, color = 'red', linestyle = '', marker = '.', alpha=0.4, label='Back leg pair, ' + r'$R^2$ = ' + corrtowrite4)
plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color='red')

plt.legend()
plt.xlim([0,400])
plt.ylim([0,2.5])
plt.yticks(fontname='Georgia',fontsize=11)
plt.xticks(fontname='Georgia',fontsize=11)
axes.set_ylabel('df duration (s)', fontname='Georgia', fontsize=13)
axes.set_xlabel(' ', fontname='Georgia', fontsize=13)
plt.savefig(upperdir+'/Writeup/Figures/' + stiffness + 'kPa_byleg_dutyfactor_vs_speed.pdf')
######
    
fig, axes = plt.subplots()
xdata = np.array(loc_legs_speed)[idx]
ydata = np.array(loc_legs_df)[idx]
print(np.mean(ydata),np.std(ydata))
slope, intercept, r, p, stderr = scipy.stats.linregress(xdata,ydata)
#pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
#residuals = ydata - func(xdata, *pars)
#ss_res = np.sum(residuals**2)
#ss_tot = np.sum((ydata-np.mean(ydata))**2)
#r_squared = 1 - (ss_res / ss_tot)
#corrtowrite = str(round(r_squared,2))
#plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color = 'black')
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
plt.plot(xdata,ydata, color = 'black', linestyle = '', marker = '.', alpha=0.4)
sns.regplot(xdata, ydata, line_kws={'alpha':0.4}, scatter_kws={'color':'black', 'marker':'o','alpha':0.0}, marker='s', color='black')
print(len(xdata))


plt.xlim([0,400])
plt.ylim([0,1.0])
plt.yticks(fontname='Georgia',fontsize=11)
plt.xticks(fontname='Georgia',fontsize=11)
axes.set_ylabel('Duty Factor', fontname='Georgia', fontsize=13)
axes.set_xlabel('Walking speed ' r'($\mu$m/s)', fontname='Georgia', fontsize=13)
axes.text(310,0.93, r'$\rho$ = ' + corrtowrite + ', ' + sig, style='italic', fontname = 'Georgia', fontsize = 13, bbox={'facecolor': 'white', 'alpha': 1, 'pad': 3})

######

upperdir = '/'.join(os.getcwd().split('/')[:-1])
datadir = upperdir + '/Data/'
file = datadir + 'Comparative_DutyFactor.xlsx'

xls = pd.ExcelFile(file)
drosophila = pd.read_excel(xls, 'Drosophila')
stickinsect = pd.read_excel(xls,'Carausius_morosus')
woodant = pd.read_excel(xls,'Cataglyphis_fortis')
earthworm = pd.read_excel(xls,'Lumbricus_terrestris')
spider = pd.read_excel(xls,'Cupiennius_salei')
spider2 = pd.read_excel(xls,'Grammostola_mollicoma')
caterpillar = pd.read_excel(xls,'Manduca_sexta')

axins = inset_axes(axes, width=4.5, height=1.7, bbox_to_anchor=(.9, .32, .1, .23),bbox_transform=axes.transAxes)

corr = scipy.stats.spearmanr(drosophila['Speed'],drosophila['Duty Factor'])
corrtowrite = str(round(corr[0],2))
axins.plot(drosophila['Speed'],drosophila['Duty Factor'],linestyle = '', marker='s',color='orange',label='drosophila',alpha=0.4,fillstyle='none')
if corr[1] < 0.05:
    sns.regplot(drosophila['Speed'], drosophila['Duty Factor'], line_kws={'alpha':0.4}, scatter_kws={'color':'orange', 'marker':'o','alpha':0.4}, marker='s', color='orange')
    sig = 'p < 0.05'
    if corr[1] < 0.01:
        sig = 'p < 0.01'
    if corr[1] < 0.001:
        sig = 'p < 0.001'
    axins.text(2,1.0, r'$\rho$ = ' + corrtowrite + ', ' + sig, style='italic', fontname = 'Georgia', fontsize = 9, bbox={'facecolor': 'white', 'alpha': 1, 'pad': 3})
else:
    sns.regplot(drosophila['Speed'], drosophila['Duty Factor'], ax=axins, marker='s', line_kws={'alpha':0.7, 'linestyle':'--'}, ci=None, scatter_kws={'edgecolor':'none','facecolor':'white'}, color='orange')
    axins.text(2,1.0, r'$\rho$ = ' + corrtowrite + ', p = ' + str(round(corr[1],2) ), style='italic', fontname = 'Georgia', color='grey', fontsize = 9, bbox={'facecolor': 'white', 'alpha': 1, 'pad': 3,'edgecolor': 'grey','linestyle':'--'})
    
corr = scipy.stats.spearmanr(caterpillar['Speed'],caterpillar['Duty Factor'])
corrtowrite = str(round(corr[0],2))
axins.plot(caterpillar['Speed'],caterpillar['Duty Factor'],linestyle = '',color='blue',label='caterpillar',alpha=0.4,fillstyle='none')
if corr[1] < 0.05:
    sns.regplot(caterpillar['Speed'], caterpillar['Duty Factor'], ax=axins, line_kws={'alpha':0.4}, scatter_kws={'color':'blue', 'marker':'o','alpha':0.4}, color='blue',truncate=True)
    sig = 'p < 0.05'
    if corr[1] < 0.01:
        sig = 'p < 0.01'
    if corr[1] < 0.001:
        sig = 'p < 0.001'
    axins.text(11.5,0.27, r'$\rho$ = ' + corrtowrite + ', ' + sig, style='italic', fontname = 'Georgia', fontsize = 9, bbox={'facecolor': 'white', 'alpha': 1, 'pad': 3})
else:
    sns.regplot(caterpillar['Speed'], caterpillar['Duty Factor'], marker='s', line_kws={'alpha':0.7, 'linestyle':'--'}, ci=None, ax=axins, scatter_kws={'edgecolor':'none','facecolor':'white'}, color='blue')
    axins.text(11.5,0.27, r'$\rho$ = ' + corrtowrite + ', p = ' + str(round(corr[1],2) ), style='italic', fontname = 'Georgia', color='grey', fontsize = 9, bbox={'facecolor': 'white', 'alpha': 1, 'pad': 3,'edgecolor': 'grey','linestyle':'--'})

corr = scipy.stats.spearmanr(woodant['Speed'],woodant['Duty Factor'])
corrtowrite = str(round(corr[0],2))
axins.plot(woodant['Speed'],woodant['Duty Factor'], marker='s', linestyle = '', color='green',label='wood ant',alpha=0.4)
if corr[1] < 0.05:
    sns.regplot(woodant['Speed'], woodant['Duty Factor'], marker='s', ax=axins, line_kws={'alpha':0.4}, scatter_kws={'color':'green','alpha':0.8}, color='green',truncate=True)
    sig = 'p < 0.05'
    if corr[1] < 0.01:
        sig = 'p < 0.01'
    if corr[1] < 0.001:
        sig = 'p < 0.001'
    axins.text(20,0.47, r'$\rho$ = ' + corrtowrite + ', ' + sig, style='italic', fontname = 'Georgia', fontsize = 9, bbox={'facecolor': 'white', 'alpha': 1, 'pad': 3})
else:
    sns.regplot(woodant['Speed'], woodant['Duty Factor'], line_kws={'alpha':0.8, 'linestyle':'--'}, ci=None, marker = 's', scatter_kws={'edgecolor':'none','facecolor':'white'}, color='green')
    plt.text(20,0.47, r'$\rho$ = ' + corrtowrite + ', p = ' + str(round(corr[1],2) ), style='italic', fontname = 'Georgia', color='grey', fontsize = 9, bbox={'facecolor': 'white', 'alpha': 1, 'pad': 3,'edgecolor': 'grey','linestyle':'--'})

#corr = scipy.stats.spearmanr(earthworm['Speed'],earthworm['Duty Factor'])
#corrtowrite = str(round(corr[0],2))
#plt.plot(earthworm['Speed'],earthworm['Duty Factor'],marker='o',linestyle = '', color='blue',label='earthworm',alpha=0.4,fillstyle='none')
#if corr[1] < 0.05:
#    sns.regplot(earthworm['Speed'], earthworm['Duty Factor'], marker='o', line_kws={'alpha':0.4}, scatter_kws={'color':'blue','alpha':0.8}, color='blue', truncate=True)
#    sig = 'p < 0.05'
#    if corr[1] < 0.01:
#        sig = 'p < 0.01'
#    if corr[1] < 0.001:
#        sig = 'p < 0.001'
#    plt.text(-4,0.1, r'$\rho$ = ' + corrtowrite + ', ' + sig, style='italic', fontname = 'Georgia', fontsize = 18, bbox={'facecolor': 'white', 'alpha': 1, 'pad': 3})
#else:
#    sns.regplot(earthworm['Speed'], earthworm['Duty Factor'], marker='o', line_kws={'alpha':0.7, 'linestyle':'--'}, ci=None, scatter_kws={'edgecolor':'none','facecolor':'white'}, color='blue',truncate=True)
#    plt.text(-4,0.1, r'$\rho$ = ' + corrtowrite + ', p = ' + str(round(corr[1],2) ), style='italic', fontname = 'Georgia', color='grey', fontsize = 18, bbox={'facecolor': 'white', 'alpha': 1, 'pad': 3,'edgecolor': 'grey','linestyle':'--'})
    
corr = scipy.stats.spearmanr(spider['Speed'],spider['Duty Factor'])
corrtowrite = str(round(corr[0],2))
axins.plot(spider['Speed'],spider['Duty Factor'],marker='s',linestyle = '', color='darkred',label='spider',alpha=0.2)
if corr[1] < 0.05:
    sns.regplot(spider['Speed'], spider['Duty Factor'], marker='s', ax=axins, line_kws={'alpha':0.2}, scatter_kws={'color':'darkred', 'alpha':0.6}, color='darkred',truncate=True)
    sig = 'p < 0.05'
    if corr[1] < 0.01:
        sig = 'p < 0.01'
    if corr[1] < 0.001:
        sig = 'p < 0.001'
    axins.text(45,0.31, r'$\rho$ = ' + corrtowrite + ', ' + sig, style='italic', fontname = 'Georgia', fontsize = 9, bbox={'facecolor': 'white', 'alpha': 1, 'pad': 3})
else:
    sns.regplot(spider['Speed'], spider['Duty Factor'], marker='s', ax=axins, line_kws={'alpha':0.7, 'linestyle':'--'}, ci=None, scatter_kws={'edgecolor':'none', 'facecolor':'white'}, color='darkred')
    axins.text(45,0.31, r'$\rho$ = ' + corrtowrite + ', p = ' + str(round(corr[1],2) ), style='italic', fontname = 'Georgia', color='grey', fontsize = 9, bbox={'facecolor': 'white', 'alpha': 1, 'pad': 3,'edgecolor': 'grey','linestyle':'--'})
    
corr = scipy.stats.spearmanr(stickinsect['Speed'],stickinsect['Duty Factor'])
corrtowrite = str(round(corr[0],2))
axins.plot(stickinsect['Speed'],stickinsect['Duty Factor'],marker='s',linestyle = '', color='purple',label='stick insect',alpha=0.4,fillstyle='none')
if corr[1] < 0.05:
    sns.regplot(stickinsect['Speed'], stickinsect['Duty Factor'], ax=axins, line_kws={'alpha':0.4}, scatter_kws={'color':'purple', 'marker':'s','alpha':1.0}, color='black',truncate=True)
    sig = 'p < 0.05'
    if corr[1] < 0.01:
        sig = 'p < 0.01'
    if corr[1] < 0.001:
        sig = 'p < 0.001'
    axins.text(22,1.0, r'$\rho$ = ' + corrtowrite + ', ' + sig, style='italic', fontname = 'Georgia', fontsize = 9, bbox={'facecolor': 'white', 'alpha': 1, 'pad': 3,})
else:
    sns.regplot(stickinsect['Speed'], stickinsect['Duty Factor'], ax=axins, line_kws={'alpha':0.4}, scatter_kws={'color':'purple', 'marker':'s','alpha':1.0}, color='black',truncate=True)
    axins.text(22,1.0, r'$\rho$ = ' + corrtowrite + ', p = ' + str(round(corr[1],2) ), style='italic', fontname = 'Georgia', color='grey', fontsize = 9, bbox={'facecolor': 'white', 'alpha': 1, 'pad': 3,'edgecolor': 'grey','linestyle':'--'})

plt.xlabel(' ',fontsize=10)
plt.ylabel(' ',fontsize=10)
plt.ylim([0,1.15])
plt.xlim([-5,65])
plt.yticks(fontname='Georgia',fontsize=7)
plt.xticks(fontname='Georgia',fontsize=7)
plt.savefig(upperdir+'/Writeup/Figures/' + stiffness + 'kPa_dutyfactor_vs_speed.pdf')
