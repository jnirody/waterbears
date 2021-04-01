# writes out csvs for analysis
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
###########################################################################
def grouper(iterable,n):
    args = [iter(iterable)]*n
    return zip(*args)
###########################################################################
stiffness = '10'

currdir = os.getcwd()
upperdir = '/'.join(currdir.split('/')[:-1])
directory = upperdir + '/Data/Tracking/TopTracks/' + stiffness + 'kPa/'
files = glob.glob(directory + '*.csv')

framerate = 50. # frames/sec
resolution = 0.55 # um / pixel (5.5 um/pixel Basler at 10x mag)

beardata = {} # create a dictionary to then write out from
for file in files:
    print file
    s = '-'
    bear = s.join(file.split('/')[-1].split('-')[0:-1])
    video = file.split('/')[-1].split('-')[-1].split('.')[0]
    dataframe = pd.read_csv(file)
    # constrct the dictionary with all the variables to write out
    if bear in beardata.keys():
        beardata[bear][video] = {}
    else:
        beardata[bear] = {}
        beardata[bear][video] = {}
    grouped = dataframe.groupby('Track')
    hx = []
    hy = []
    for leg,data in grouped:
        if leg == 0:
            beardata[bear][video][leg] = {}
            beardata[bear][video][leg]['start_frame'] = data['Slice'].iloc[[0][0]]
            start_frame = beardata[bear][video][leg]['start_frame']
            end_frame = data['Slice'].iloc[[1][0]]
        elif leg == 1:
            beardata[bear][video][leg] = {}
            bx1 = data['X'].iloc[[0][0]]
            bx2 = data['X'].iloc[[1][0]]
            by1 = data['Y'].iloc[[0][0]]
            by2 = data['Y'].iloc[[1][0]]
            lx1 = data['X'].iloc[[2][0]]
            lx2 = data['X'].iloc[[3][0]]
            ly1 = data['Y'].iloc[[2][0]]
            ly2 = data['Y'].iloc[[3][0]]
            beardata[bear][video][leg]['body_length'] = np.sqrt((bx1-bx2)**2 + (by1-by2)**2)*resolution
            beardata[bear][video][leg]['limb_length'] = np.sqrt((lx1-lx2)**2 + (ly1-ly2)**2)*resolution
            beardata[bear][video][leg]['gait_type1'] = []
            beardata[bear][video][leg]['gait_type2'] = []
        elif leg == 10:
            beardata[bear][video][leg] = {}
            beardata[bear][video][leg]['head_pos'] = []
            for i in range(len(data)-1):
                if data['Slice'].iloc[[i][0]] > (start_frame-1) and data['Slice'].iloc[[i][0]] < (end_frame + 1):
                    x1 = data['X'].iloc[[i][0]]
                    x2 = data['X'].iloc[[i+1][0]]
                    y1 = data['Y'].iloc[[i][0]]
                    y2 = data['Y'].iloc[[i+1][0]]
                    hx.extend([x1])
                    hy.extend([y1])
                    dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)*resolution
                    beardata[bear][video][leg]['head_pos'].append((x1,y1))
        elif leg == 11:
            beardata[bear][video][leg] = {}
            beardata[bear][video][leg]['center_pos'] = []
            beardata[bear][video][leg]['distance'] = []
            beardata[bear][video][leg]['velocity'] = []
            for i in range(len(data)-1):
                if data['Slice'].iloc[[i][0]] > (start_frame-1) and data['Slice'].iloc[[i][0]] < (end_frame + 1):
                    cx1 = data['X'].iloc[[i][0]]
                    cx2 = data['X'].iloc[[i+1][0]]
                    cy1 = data['Y'].iloc[[i][0]]
                    cy2 = data['Y'].iloc[[i+1][0]]
                    beardata[bear][video][leg]['center_pos'].append((cx1,cy1))
                    dist = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)*resolution
                    beardata[bear][video][leg]['distance'].append(dist)
                    beardata[bear][video][leg]['velocity'].append(dist*framerate)
        elif leg == 12:
            beardata[bear][video][leg] = {}
            beardata[bear][video][leg]['tail_pos'] = []
            for i in range(len(data)-1):
                if data['Slice'].iloc[[i][0]] > (start_frame-1) and data['Slice'].iloc[[i][0]] < (end_frame + 1):
                    tx = data['X'].iloc[[i][0]]
                    ty = data['Y'].iloc[[i][0]]
                    beardata[bear][video][leg]['tail_pos'].append((tx,ty))
        else:
            stroke_num = 0
            beardata[bear][video][leg] = {}
            beardata[bear][video][leg]['to_shift'] = []
            beardata[bear][video][leg]['leg_down'] = [0]*(end_frame - start_frame + 2)
            beardata[bear][video][leg]['leg_down_loc'] = [0]*(end_frame - start_frame + 2)
            beardata[bear][video][leg]['liftoff_time'] = []
            beardata[bear][video][leg]['duty_factor'] = []
            beardata[bear][video][leg]['period'] = []
            beardata[bear][video][leg]['step_length'] = []
            beardata[bear][video][leg]['stride_length'] = []
            beardata[bear][video][leg]['stride_center_speed'] = []
            beardata[bear][video][leg]['pull_back'] = []
            beardata[bear][video][leg]['stance'] = []
            beardata[bear][video][leg]['swing'] = []
            beardata[bear][video][leg]['liftoff_time'] = []
            beardata[bear][video][leg]['liftoff_loc'] = []
            beardata[bear][video][leg]['touchdown_time'] = []
            beardata[bear][video][leg]['touchdown_loc'] = []
            last_down_frame = start_frame - 1
            for i in range(len(data)):
                if data['Slice'].iloc[[i][0]] > (start_frame -1) and data['Slice'].iloc[[i][0]] < (end_frame + 1):
                    if sum(beardata[bear][video][leg]['leg_down']) == 0 and data['Slice'].iloc[[i][0]] > start_frame:
                        beardata[bear][video][leg]['touchdown_time'].append(data['Slice'].iloc[[i][0]])
                        beardata[bear][video][leg]['touchdown_loc'].append((data['X'].iloc[[i][0]],data['Y'].iloc[[i][0]]))
                    beardata[bear][video][leg]['leg_down'][data['Slice'].iloc[[i][0]] - start_frame] = 1
                    beardata[bear][video][leg]['leg_down_loc'][data['Slice'].iloc[[i][0]] - start_frame] = (data['X'].iloc[[i][0]],data['Y'].iloc[[i][0]])
                    if data['Slice'].iloc[[i][0]] > data['Slice'].iloc[[i-1][0]] + 1 and data['Slice'].iloc[[i-1][0]] > start_frame:
                        beardata[bear][video][leg]['liftoff_time'].append(data['Slice'].iloc[[i-1][0]])
                        beardata[bear][video][leg]['liftoff_loc'].append((data['X'].iloc[[i-1][0]],data['Y'].iloc[[i-1][0]]))
                        beardata[bear][video][leg]['touchdown_time'].append(data['Slice'].iloc[[i][0]])
                        beardata[bear][video][leg]['touchdown_loc'].append((data['X'].iloc[[i][0]],data['Y'].iloc[[i][0]]))
                    if i == len(data)-1 and data['Slice'].iloc[[i][0]] < (end_frame) and data['Slice'].iloc[[i][0]] > start_frame:
                        beardata[bear][video][leg]['liftoff_time'].append(data['Slice'].iloc[[i][0]])
                        beardata[bear][video][leg]['liftoff_loc'].append((data['X'].iloc[[i-1][0]],data['Y'].iloc[[i-1][0]]))
    # calculate out for each frame how many legs are on the ground (both for first 6, and then for two back legs)
    temp_style1 = [0]*len(beardata[bear][video][2]['leg_down'])
    temp_style2 = [0]*len(beardata[bear][video][2]['leg_down'])
    for k in range(2,8):
        temp_style1 = [y + z for y,z in zip(beardata[bear][video][k]['leg_down'],temp_style1)]
    for m in range(8,10):
       temp_style2 = [x + y for x,y in zip(beardata[bear][video][m]['leg_down'],temp_style2)]
    beardata[bear][video][1]['gait_type1'] = temp_style1
    beardata[bear][video][1]['gait_type2'] = temp_style2
    
    # write individual frame-by-frame csvs to analyse
    with open(upperdir + '/Data/ForAnalysis/' + stiffness + 'kPa/Top/Individual/ByFrame/' + bear + '_framebyframe.csv','w') as outfile:
        bear_writer = csv.writer(outfile, delimiter=',')
        bear_writer.writerow(['bear','video','frame','body_length','limb_length','num_feet_down', 'num_rearfeet_down','L1_leg_down', 'L1_leg_loc','R1_leg_down','R1_leg_loc','L2_leg_down','L2_leg_loc', 'R2_leg_down','R2_leg_loc','L3_leg_down', 'L3_leg_loc','R3_leg_down','R3_leg_loc', 'L4_leg_down','L4_leg_loc','R4_leg_down','R4_leg_loc','head_pos', 'center_pos','COM_dist','COM_speed','tail_pos'])
        for video in beardata[bear]:
            row = []
            for leg in beardata[bear][video]:
                if leg == 0:
                    continue
                elif leg == 1:
                    for i in range(len(beardata[bear][video][leg]['gait_type2'])):
                        row.append([bear,video,beardata[bear][video][0]['start_frame']+i, beardata[bear][video][leg]['body_length'], beardata[bear][video][leg]['limb_length'],beardata[bear][video][leg]['gait_type1'][i],beardata[bear][video][leg]['gait_type2'][i]])
                elif leg == 10:
                    for i in range(len(beardata[bear][video][leg]['head_pos'])):
                        row[i].extend([beardata[bear][video][leg]['head_pos'][i]])
                elif leg == 11:
                    for i in range(len(beardata[bear][video][leg]['center_pos'])):
                        row[i].extend([beardata[bear][video][leg]['center_pos'][i],beardata[bear][video][leg]['distance'][i],beardata[bear][video][leg]['velocity'][i]])
                elif leg == 12:
                    for i in range(len(beardata[bear][video][leg]['tail_pos'])):
                        row[i].extend([beardata[bear][video][leg]['tail_pos'][i]])
                else:
                    for i in range(len(beardata[bear][video][leg]['leg_down'])):
                        row[i].extend([beardata[bear][video][leg]['leg_down'][i],beardata[bear][video][leg]['leg_down_loc'][i]])
            for j in range(len(row)):
                bear_writer.writerow(row[j])
            # calculate the values for 'by stride' calculations
            for leg in beardata[bear][video]:
                if leg > 1 and leg < 10:
                    strides = []
                    if beardata[bear][video][leg]['touchdown_time'][0] < beardata[bear][video][leg]['liftoff_time'][0]:
                        strides.append(['',beardata[bear][video][leg]['touchdown_time'][0], '','','','','','','','',beardata[bear][video][leg]['touchdown_loc'][0], ''])
                        beardata[bear][video][leg]['to_shift'] = 1
                    else:
                        beardata[bear][video][leg]['to_shift'] = 0
                    for i in range(len(beardata[bear][video][leg]['liftoff_time'])-1):
                        if beardata[bear][video][leg]['liftoff_time'][i+1]-start_frame > len(beardata[bear][video][11]['center_pos'])-1:
                            continue
                        beardata[bear][video][leg]['period'].extend([float(beardata[bear][video][leg]['liftoff_time'][i+1]-beardata[bear][video][leg]['liftoff_time'][i])/framerate])
                        PEP1x = beardata[bear][video][leg]['liftoff_loc'][i][0]
                        PEP1y = beardata[bear][video][leg]['liftoff_loc'][i][1]
                        PEP2x = beardata[bear][video][leg]['liftoff_loc'][i+1][0]
                        PEP2y = beardata[bear][video][leg]['liftoff_loc'][i+1][1]
                        AEP1x = beardata[bear][video][leg]['touchdown_loc'][i+beardata[bear][video][leg]['to_shift']][0]
                        AEP1y = beardata[bear][video][leg]['touchdown_loc'][i+beardata[bear][video][leg]['to_shift']][1]
                        cpos1  = beardata[bear][video][11]['center_pos'][beardata[bear][video][leg]['liftoff_time'][i]-beardata[bear][video][0]['start_frame']]
                        cpos2  = beardata[bear][video][11]['center_pos'][beardata[bear][video][leg]['liftoff_time'][i+1]-beardata[bear][video][0]['start_frame']]
                        dist = np.sqrt((cpos1[0]-cpos2[0])**2 + (cpos1[1]-cpos2[1])**2)*resolution
                        beardata[bear][video][leg]['stride_center_speed'].append(dist/(float(beardata[bear][video][leg]['liftoff_time'][i+1]-beardata[bear][video][leg]['liftoff_time'][i])/framerate))
                        if i + beardata[bear][video][leg]['to_shift']+1 < len(beardata[bear][video][leg]['touchdown_loc']):
                            AEP2x = beardata[bear][video][leg]['touchdown_loc'][i+beardata[bear][video][leg]['to_shift']+1][0]
                            AEP2y = beardata[bear][video][leg]['touchdown_loc'][i+beardata[bear][video][leg]['to_shift']+1][1]
                            beardata[bear][video][leg]['stride_length'].append(np.sqrt((AEP2x-AEP1x)**2+ (AEP2y-AEP1y)**2)*resolution)
                        else:
                            AEP2x = AEP1x
                            AEP2y = AEP1y
                            beardata[bear][video][leg]['stride_length'].append(np.mean( beardata[bear][video][leg]['stride_length']))
                        beardata[bear][video][leg]['pull_back'].append(np.sqrt((PEP2x-AEP1x)**2+(PEP2y-AEP1y)**2)*resolution)
                        beardata[bear][video][leg]['step_length'].append(np.sqrt((PEP1x-AEP1x)**2+(PEP1y-AEP1y)**2)*resolution)
                        beardata[bear][video][leg]['swing'].append( float(beardata[bear][video][leg]['touchdown_time'][i+beardata[bear][video][leg]['to_shift']] - beardata[bear][video][leg]['liftoff_time'][i])/framerate)
                        beardata[bear][video][leg]['stance'].append(float(beardata[bear][video][leg]['liftoff_time'][i+1] - beardata[bear][video][leg]['touchdown_time'][i+beardata[bear][video][leg]['to_shift']])/framerate)
                        beardata[bear][video][leg]['duty_factor'].append(beardata[bear][video][leg]['stance'][i]/float(beardata[bear][video][leg]['period'][i]))
                        strides.append([beardata[bear][video][leg]['liftoff_time'][i], beardata[bear][video][leg]['touchdown_time'][i+beardata[bear][video][leg]['to_shift']],beardata[bear][video][leg]['swing'][i],beardata[bear][video][leg]['stance'][i],beardata[bear][video][leg]['period'][i],beardata[bear][video][leg]['duty_factor'][i],beardata[bear][video][leg]['step_length'][i],beardata[bear][video][leg]['pull_back'][i],beardata[bear][video][leg]['stride_length'][i],beardata[bear][video][leg]['stride_center_speed'][i],beardata[bear][video][leg]['touchdown_loc'][i+beardata[bear][video][leg]['to_shift']],beardata[bear][video][leg]['liftoff_loc'][i]])
                    if beardata[bear][video][leg]['touchdown_time'][-1] < end_frame:
                        strides.append([beardata[bear][video][leg]['liftoff_time'][-1],'', '','','','','','','','','', beardata[bear][video][leg]['liftoff_loc'][-1]])
                    beardata[bear][video][leg]['strides'] = strides
                
    # write individual stride-by-stride csvs to analyse
    with open(upperdir + '/Data/ForAnalysis/' + stiffness + 'kPa/Top/Individual/ByStride/' + bear + '_stridebystride.csv','w') as outfile:
        bear_writer = csv.writer(outfile, delimiter=',')
        bear_writer.writerow(['bear','video','body_length','limb_length','L1_swing_start','L1_stance_start','L1_swing','L1_stance', 'L1_period','L1_duty_factor','L1_step_length','L1_pull_back','L1_stride_length','L1_stride_speed','L1_AEP','L1_PEP', 'R1_swing_start','R1_stance_start','R1_swing','R1_stance','R1_period', 'R1_duty_factor','R1_step_length','R1_pull_back', 'R1_stride_length','R1_stride_speed', 'R1_AEP','R1_PEP', 'L2_swing_start','L2_stance_start','L2_swing','L2_stance','L2_period','L2_duty_factor', 'L2_step_length','L2_pull_back','L2_stride_length','L2_stride_speed','L2_AEP','L2_PEP', 'R2_swing_start','R2_stance_start','R2_swing','R2_stance', 'R2_period','R2_duty_factor','R2_step_length','R2_pull_back','R2_stride_length','R2_stride_speed','R2_AEP','R2_PEP', 'L3_swing_start','L3_stance_start','L3_swing','L3_stance', 'L3_period','L3_duty_factor','L3_step_length','L3_pull_back','L3_stride_length','L3_stride_speed','L3_AEP','L3_PEP', 'R3_swing_start','R3_stance_start','R3_swing','R3_stance','R3_period', 'R3_duty_factor','R3_step_length','R3_pull_back','R3_stride_length','R3_stride_speed','R3_AEP','R3_PEP', 'L4_swing_start','L4_stance_start','L4_swing','L4_stance','L4_period','L4_duty_factor','L4_step_length', 'L4_pull_back','L4_stride_length','L4_stride_speed','L4_AEP','L4_PEP', 'R4_swing_start','R4_stance_start','R4_swing','R4_stance','R4_period','R4_duty_factor','R4_step_length','R4_pull_back','R4_stride_length','R4_stride_speed','R4_AEP','R4_PEP'])
        for video in beardata[bear]:
            row = []
            for leg in beardata[bear][video]:
                if leg == 1:
                    longest_run = max(len(beardata[bear][video][2]['strides']),len(beardata[bear][video][3]['strides']),len(beardata[bear][video][4]['strides']),len(beardata[bear][video][5]['strides']),len(beardata[bear][video][6]['strides']),len(beardata[bear][video][7]['strides']),len(beardata[bear][video][8]['strides']),len(beardata[bear][video][9]['strides']))
                    for i in range(longest_run):
                        row.append([bear,video,beardata[bear][video][leg]['body_length'],beardata[bear][video][leg]['limb_length']])
                elif leg > 1 and leg < 10:
                    for i in range(longest_run):
                        try:
                            row[i].extend(beardata[bear][video][leg]['strides'][i])
                        except IndexError:
                            row[i].extend(['','','','','','','', '', '','', '',''])
            for j in range(len(row)):
                bear_writer.writerow(row[j])
    bear_average_rows = []
    for bear in beardata:
        bear_average_rows.append([bear])
        body_length = []
        c_body_length = []
        limb_length = []
        cum_swing = []
        cum_stance = []
        cum_period = []
        cum_duty_factor = []
        cum_step_length = []
        cum_pull_back = []
        cum_stride_length = []
        swing = []
        stance = []
        period = []
        duty_factor = []
        step_length = []
        pull_back = []
        stride_length = []
        speed = []
        for video in beardata[bear]:
            for leg in beardata[bear][video]:
                if leg == 0:
                    continue
                elif leg == 1:
                    body_length.extend([beardata[bear][video][leg]['body_length']])
                    limb_length.extend([beardata[bear][video][leg]['limb_length']])
                elif leg == 10:
                    continue
                elif leg == 11:
                    # should this be replaced with COM speed?
                    speed.extend([(np.cumsum(beardata[bear][video][leg]['distance'][1:-1])[-1])/(len(beardata[bear][video][leg]['distance'][1:-1])/framerate)])
                elif leg > 11:
                    continue
                else:
                    swing.append([beardata[bear][video][leg]['swing']])
                    stance.append([beardata[bear][video][leg]['stance']])
                    period.append([beardata[bear][video][leg]['period']])
                    duty_factor.append([beardata[bear][video][leg]['duty_factor']])
                    step_length.append([beardata[bear][video][leg]['step_length']])
                    pull_back.append([beardata[bear][video][leg]['pull_back']])
                    stride_length.append([beardata[bear][video][leg]['stride_length']])
                    if leg < 7:
                        cum_swing.extend(beardata[bear][video][leg]['swing'])
                        cum_stance.extend(beardata[bear][video][leg]['stance'])
                        cum_period.extend(beardata[bear][video][leg]['period'])
                        cum_duty_factor.extend(beardata[bear][video][leg]['duty_factor'])
                        cum_step_length.extend(beardata[bear][video][leg]['step_length'])
                        cum_pull_back.extend(beardata[bear][video][leg]['pull_back'])
                        cum_stride_length.extend(beardata[bear][video][leg]['stride_length'])
        bear_average_rows[-1].extend([np.mean(body_length),np.std(body_length),np.mean(limb_length),np.std(limb_length)])
        for i in range(8):
            bear_average_rows[-1].extend([np.mean(swing[i]),np.std(swing[i]),np.mean(stance[i]),np.std(stance[i]),np.mean(period[i]), np.std(stance[i]),np.mean(duty_factor[i]),np.std(duty_factor[i]),np.mean(step_length[i]),np.std(step_length[i]),np.mean(pull_back[i]),np.std(pull_back[i]),np.mean(stride_length[i]),np.std(stride_length[i])])
        bear_average_rows[-1].extend([np.mean(cum_swing),np.std(cum_swing), np.mean(cum_stance),np.std(cum_stance), np.mean(cum_period),np.std(cum_period),np.mean(cum_duty_factor), np.std(cum_duty_factor),np.mean(cum_step_length), np.std(cum_step_length),np.mean(cum_pull_back),np.std(cum_pull_back),np.mean(cum_stride_length), np.std(cum_stride_length)])
        bear_average_rows[-1].extend([np.mean(speed),np.std(speed),((np.mean(speed)**2)/(np.mean(limb_length)*980000))])
    
with open(upperdir + '/Data/ForAnalysis/' + stiffness + 'kPa/Top/' + 'bear_averages.csv', 'w') as outfile:
    bear_writer = csv.writer(outfile, delimiter=',')
    bear_writer.writerow(['bear','body_length','body_length_sd','limb_length','limb_length_sd','L1_swing','L1_swing_sd','L1_stance','L1_stance_sd','L1_period','L1_period_sd','L1_duty_factor','L1_duty_factor_sd', 'L1_step_length','L1_step_length_sd','L1_pull_back','L1_pull_back_sd','L1_stride_length','L1_stride_length_sd','R1_swing','R1_swing_sd','R1_stance','R1_stance_sd','R1_period','R1_period_sd','R1_duty_factor','R1_duty_factor_sd', 'R1_step_length','R1_step_length_sd','R1_pull_back','R1_pull_back_sd','R1_stride_length','R1_stride_length_sd','L2_swing', 'L2_swing_sd','L2_stance','L2_stance_sd','L2_period','L2_period_sd','L2_duty_factor','L2_duty_factor_sd', 'L2_step_length','L2_step_length_sd','L2_pull_back','L2_pull_back_sd','L2_stride_length','L2_stride_length_sd','R2_swing', 'R2_swing_sd','R2_stance','R2_stance_sd','R2_period','R2_period_sd','R2_duty_factor','R2_duty_factor_sd', 'R2_step_length','R2_step_length_sd','R2_pull_back','R2_pull_back_sd','R2_stride_length','R2_stride_length_sd','L3_swing', 'L3_swing_sd','L3_stance','L3_stance_sd','L3_period','L3_period_sd','L3_duty_factor','L3_duty_factor_sd', 'L3_step_length','L3_step_length_sd','L3_pull_back','L3_pull_back_sd','L3_stride_length','L3_stride_length_sd','R3_swing', 'R3_swing_sd','R3_stance','R3_stance_sd','R3_period','R3_period_sd','R3_duty_factor','R3_duty_factor_sd', 'R3_step_length','R3_step_length_sd','R3_pull_back','R3_pull_back_sd','R3_stride_length','R3_stride_length_sd','L4_swing', 'L4_swing_sd','L4_stance','L4_stance_sd','L4_period','L4_period_sd','L4_duty_factor','L4_duty_factor_sd', 'L4_step_length','L4_step_length_sd','L4_pull_back','L4_pull_back_sd','L4_stride_length','L4_stride_length_sd','R4_swing', 'R4_swing_sd','R4_stance','R4_stance_sd','R4_period','R4_period_sd','R4_duty_factor','R4_duty_factor_sd', 'R4_step_length','R4_step_length_sd','R4_pull_back','R4_pull_back_sd','R4_stride_length','R4_stride_length_sd','cum_swing','cum_swing_sd','cum_stance','cum_stance_sd', 'cum_period','cum_period_sd', 'cum_duty_factor','cum_duty_factor_sd', 'cum_step_length','cum_step_length_sd', 'cum_pull_back','cum_pull_back_sd','cum_stride_length','cum_stride_length_sd', 'speed','speed_sd','Froude_number'])
    for j in range(len(bear_average_rows)):
        bear_writer.writerow(bear_average_rows[j])

concatdir = upperdir + '/Data/ForAnalysis/' + stiffness + 'kPa/Top/Individual/ByStride/'
filenames = glob.glob(concatdir + "/*.csv")
big_df = []
for file in filenames:
    df = pd.read_csv(file, index_col=None, header=0)
    big_df.append(df)

frame = pd.concat(big_df, axis=0, ignore_index=True)
frame.to_csv(upperdir + '/Data/ForAnalysis/' + stiffness + 'kPa/Top/' + 'compiled_bears_by_strides.csv')
    
