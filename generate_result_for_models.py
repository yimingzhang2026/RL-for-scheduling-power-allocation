# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 13:32:16 2022

@author: yzhang4
"""
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import json
import matplotlib
# matplotlib.use('Qt5Agg')
import argparse
import seaborn as sns
import pandas as pd

parser = argparse.ArgumentParser(description='give data set under policy trained.')
parser.add_argument("--seeds", default=[1], nargs='+', type=int)              
parser.add_argument("--episode-timeslots", default=2500, type=int)
parser.add_argument("--timeslots", default=50000, type=int)   
parser.add_argument("--mode", default="traffic")
parser.add_argument("--reset-gains", default=True, type=bool) 
parser.add_argument("--K", default= 5, type=int)   
parser.add_argument("--N", default= 10, type=int) 
parser.add_argument("--M", default=1,  type=int)   
parser.add_argument("--Mobility", default= True,  type=bool) 
parser.add_argument("--modelsims", default= [0,5,20], nargs='+', type=float) # test models
parser.add_argument("--episodes", default=[1, 5, 10, 15, 19], nargs='+', type=float)
parser.add_argument("--logs", default=True, type=bool)

parser.add_argument('--json-file-policy', type=str, default='dqn200_100_50',
                   help='json file for the hyperparameters')


kwargs = {'cumulative': True}
args = parser.parse_args()
max_rate = 5
modelsims = args.modelsims
json_file_policy = args.json_file_policy
min_pkts = args.episode_timeslots * 10
for seed in args.seeds:
    args.seed = seed
    folder_name = './simulations/N%dK%dM%dseed%dtimeslots%depisode_timeslots%ddr%dm%s'%(args.N,args.K,args.M,args.seed,args.timeslots,args.episode_timeslots,max_rate,args.Mobility)
    for modelsim in args.modelsims:

        delay = {}
        p = {}
    
        r = {}
        ue = {}
        ave_delay_all = {}
        total_p_all = {}
        total_reward_all = {}
        through_put_all = {}
        

        for idxep,ep in enumerate(args.episodes):
           
                
            # np_save_path = '%s/%s%s.npz'%(folder_name_traffic,policy_folder_name.split('./simulations/')[1],json_file_policy)
            np_save_path = '%s/%s%dmodelsim%dep%d.npz'%(folder_name,json_file_policy,args.seed,modelsim,ep)
            
            data = np.load(np_save_path)
            
            p_strategy_all = data['arr_1']
            user_strategy_all = data['arr_2']
            all_delay_time = data['arr_3']
            all_reward = data['arr_4']
            total_packets = data['arr_5']
            total_Mbits = data['arr_6']
            actual_p = data['arr_7']
            state_all = data['arr_8']


            
            min_pkts = min(len(all_delay_time),min_pkts)

            p['eps {}'.format(ep)] = np.mean(p_strategy_all,axis = 1)
            r['eps {}'.format(ep)] = np.mean(all_reward, axis = (1,2))
            ue['eps {}'.format(ep)] = user_strategy_all[:,0]
            ave_delay = np.array(np.mean(all_delay_time,axis = 0))
            #total_p = np.sum(actual_p,axis = (0,1))
            total_p = np.sum(actual_p,axis = (0,1,2))
            
            total_reward = np.sum(all_reward,axis = (0,1,2))

            
            ave_delay_all['eps {}'.format(ep)] = ave_delay
            total_p_all['eps {}'.format(ep)] = total_p
            
            total_reward_all['eps {}'.format(ep)] = total_reward

            
    
            
    
    
            #load bench mark
            np_save_path_benchmark = '%s/%sep%d.npz'%(folder_name,'benchmark',ep)
            
            data = np.load(np_save_path_benchmark)
            
            all_delay_time_benchmark = data['arr_0']
            all_reward_benchmark = data['arr_1']
            p_strategy_all_benchmark = data['arr_2']


                    
            average_reward_benchmark = np.mean(all_reward_benchmark, axis = (1,2))

        
        
        


            p['bm eps {}'.format(ep)] = np.mean(p_strategy_all_benchmark,axis = 1)
            r['bm eps {}'.format(ep)] = np.mean(all_reward_benchmark, axis = (1,2))
            total_p_benchmark = np.sum(p_strategy_all_benchmark,axis = (0,1))
            total_p_all['bm eps {}'.format(ep)] = total_p_benchmark
        
        
            ave_delay = np.array(np.mean(all_delay_time_benchmark,axis = 0))
            ave_delay_all['bm eps {}'.format(ep)] = ave_delay
        

        
            total_reward_benchmark = np.sum(all_reward_benchmark, axis = (0,1,2))

        

        
            total_reward_all['bm eps {}'.format(ep)] = total_reward_benchmark


        for idxep,ep in enumerate(args.episodes):
            np_save_path = '%s/%s%dmodelsim%dep%d.npz'%(folder_name,json_file_policy,args.seed,modelsim,ep)
            data = np.load(np_save_path)
            all_delay_time = data['arr_3']
            delay['eps {}'.format(ep)] = all_delay_time[:min_pkts]
            
            np_save_path_benchmark = '%s/%sep%d.npz'%(folder_name,'benchmark',ep)
            data = np.load(np_save_path_benchmark)
            all_delay_time_benchmark = data['arr_0']
            delay['bm eps {}'.format(ep)] = all_delay_time_benchmark[:min_pkts]
        #generate df and plot figure

        df_ave_delay = pd.DataFrame(ave_delay_all, index=[0])        
        df_total_p = pd.DataFrame(total_p_all,index=[0])  
        df_delay = pd.DataFrame(delay)
        df_p = pd.DataFrame(p)      
        df_r = pd.DataFrame(r)    
        df_ue = pd.DataFrame(ue)  
        # df_1 = pd.DataFrame(total_reward_bit_all,index=[0])
        # df_2 = pd.DataFrame(total_reward_p_all,index=[0])
        # df_integrated_reward = pd.concat([df_1,df_2])
        
        # df = pd.DataFrame({'reward from bit': df_integrated_reward.iloc[0],
        #                    'reward from p': df_integrated_reward.iloc[1]},
        #                   index = df_integrated_reward.columns)
        
        # df.plot(kind='bar', stacked=True, color=['green', 'skyblue'])
        
        # plt.ylim((-200.0,0.0))
        # # labels for x & y axis
        # plt.xlabel('eps')
        # plt.ylabel('total reward')
         
        # # title of plot
        # plt.title('total reward model{}'.format(modelsim))
        # plt.show()
        
        
        plt.figure()
        sns.displot(df_delay, 
                    kind="ecdf")
        plt.xlabel('avarage delay time (seconds)')
        plt.ylabel('CDF')
        plt.title("avarage delay time")
        plt.xlim((0.0, 0.5))
        plt.ylim((0.0, 1.1))
        plt.show()

        
        
        plt.figure()
        sns.barplot(data = df_ave_delay)
        plt.xlabel('eps')
        plt.ylabel('avarage delay time (seconds)')
        plt.title("avarage delay time")
        # plt.xlim((0.0, 5))
        plt.ylim((0.0, 0.5))
        plt.show()
        
        
        
        
        
        
        plt.figure()
        sns.barplot(data = df_total_p)
        plt.xlabel('eps')
        plt.ylabel('total power')
        plt.title("total power")
        # # plt.xlim((0.0, 5))
        # plt.ylim((0.0, 0.1))
        plt.show()







