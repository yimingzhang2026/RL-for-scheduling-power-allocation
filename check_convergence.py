# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 04:01:09 2022

@author: yzhang4
"""
import numpy as np
import matplotlib.pyplot as plt
import json
import matplotlib
import argparse
import seaborn as sns
import pandas as pd
import matplotlib.animation as animation



parser = argparse.ArgumentParser(description='give test scenarios.')
parser.add_argument("--seeds", default=[1], nargs='+', type=int)              
parser.add_argument("--episode-timeslots", default=75000, type=int) 
parser.add_argument("--timeslots", default=75000, type=int)   
parser.add_argument("--mode", default="traffic")
parser.add_argument("--reset-gains", default=True, type=bool) 
parser.add_argument("--max-rates", default=[15], nargs='+', type=float)
parser.add_argument("--K", default= 5, type=int)   
parser.add_argument("--N", default= 10, type=int) 
parser.add_argument("--M", default=1,  type=int)   
parser.add_argument("--Mobility", default= True,  type=bool) 
parser.add_argument('--json-file-policy', type=str, default='dqn200_100_50',
                   help='json file for the hyperparameters')
args = parser.parse_args()
args.seed = args.seeds[0]
json_file_policy = args.json_file_policy

for data_rate in args. max_rates:

    folder_name = './simulations/N%dK%dM%dseed%dtimeslots%depisode_timeslots%ddr%dm%s'%(args.N,args.K,args.M,args.seed,args.timeslots,args.episode_timeslots,data_rate,args.Mobility)
    
    np_save_path = '%s/%scheckweights.npz'%(folder_name,args.json_file_policy)
    
    data = np.load(np_save_path)
    
    
    var_1_all = data['arr_0']
    var_2_all = data['arr_1']
    var_3_all = data['arr_2']
    b_3_all = data['arr_3']
    throughput = data['arr_4']
    reward_all = data['arr_5']
    RX_loc = data['arr_6']
    utilization = data['arr_7']
    time_calculating_strategy_takes = data['arr_8']
    time_optimization_at_each_slot_takes = data['arr_9']
    utilization = utilization / args.timeslots
    reward_curve = np.sum(reward_all, axis = (1,2))
    all_x = list(var_3_all[:,2,2])
    x = all_x[0:len(all_x):50]
    all_y = var_3_all[:,1,0]
    y = all_y[0:len(all_y):50]
    
    fig = plt.figure(tight_layout=True)
    def plot_curve(fig,x):
        ax = fig.gca()
        ax.set_xticks(np.arange(0, args.timeslots, 5000))
        plt.plot(x)
        plt.grid(ls="--")
        
    for i in range(3):
        for j in range(3):
            data = var_3_all[:,i * np.random.randint(0, 10), j * np.random.randint(0,2)]
            plot_curve(fig,data)
    fig = plt.figure(tight_layout=True)
    for i in range(3):
            data = b_3_all[:,np.random.randint(0, 10)]
            plot_curve(fig,data)
            
    fig = plt.figure(tight_layout=True) 
    plot_curve(fig,reward_curve)
    
    fig = plt.figure(tight_layout=True)
    plot_curve(fig,time_calculating_strategy_takes)
    fig = plt.figure(tight_layout=True)
    plot_curve(fig,time_optimization_at_each_slot_takes)
    
    
    
    
    # fig = plt.figure(tight_layout=True)
    # plt.plot(x,y)
    # point_ani, = plt.plot(x[0], y[0], "ro")
    # plt.grid(ls="--")
    
    
    # def update_points(num):
    #     point_ani.set_data(x[num], y[num])
    #     return point_ani,
    
    
    # ani = animation.FuncAnimation(fig, update_points, frames = np.arange(0, 6000), interval=10, blit=True)
    
    # plt.show()
    
    # x_loc = RX_loc[0,:]
    # y_loc = RX_loc[1,:]
    # fig = plt.figure(tight_layout=True)
    # plt.plot(x,y)
    # point_ani, = plt.plot(x_loc[0], x_loc[0], "ro")
    # plt.grid(ls="--")
    
    
    # def update_points_loc(num):
    #     point_ani.set_data(x_loc[num], y_loc[num])
    #     return point_ani,
    
    
    # ani = animation.FuncAnimation(fig, update_points_loc, frames = np.arange(0, 5000), interval=100, blit=True)
    
    # plt.show()