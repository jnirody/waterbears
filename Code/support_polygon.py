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
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.geometry import MultiPoint
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
datadir = upperdir + '/Data/ForAnalysis/' + stiffness + 'kPa/Top/Individual/ByFrame/'
files = glob.glob(datadir + '*.csv')
avg_file = upperdir + '/Data/ForAnalysis/' + stiffness + 'kPa/Top/bear_averages.csv'
avgdata = pd.read_csv(avg_file)

falses = 0
total = 0
inpts = []
outpts = []
resolution = 0.55
for file in files:
    s = '-'
    bear = s.join(file.split('/')[-1].split('-')[0:-1])
    video = file.split('/')[-1].split('-')[-1].split('.')[0]
    dataframe = pd.read_csv(file)
    
    grouped = dataframe.groupby('video')
    trial = -1
    for video,data in grouped:
        trial += 1
        fbf_file = pd.read_csv(file)
        if trial > 0:
            subtract = min(np.where(np.array(fbf_file['video']==video))[0])
        else:
            subtract = 0
        print(file, video)
        sides = ['L','R']
        pairs = ['1','2','3','4']
        for frame in range(len(data['center_pos'])-1):
            support_points = []
            for side in sides:
                for pair in pairs:
                    leg = side + pair
                    if data[leg+'_leg_down'][frame+subtract] == 1:
                        support_points.extend([eval(data[leg+'_leg_loc'][frame+subtract])])
            if len(support_points) > 2:
                support_polygon = Polygon(support_points)
                ch_area = MultiPoint(support_points).convex_hull
                if type(data['center_pos'][frame+subtract]) is str:
                    COM = eval(data['center_pos'][frame+subtract])
                else:
                    COM = 0
                if type(COM) is tuple:
                    total += 1
                    COM = Point(COM)
                    if ch_area.contains(COM) == False:
                        outpts.extend([-1*resolution*support_polygon.exterior.distance(COM)])
                        falses += 1
                    else:
                        inpts.extend([resolution*support_polygon.exterior.distance(COM)])
#                        x,y = ch_area.exterior.xy
#                        plt.plot(x,y)
#                        plt.plot(eval(data['center_pos'][frame+subtract])[0],eval(data['center_pos'][frame+subtract])[1],'r*')
#                        plt.show()
        print(falses,total)
plt.hist(inpts,100,color='green')
plt.hist(outpts,100,color='red')
plt.xlim([-100,100])
plt.savefig(upperdir+'/Writeup/Figures/COM_stability_polygon.pdf')
