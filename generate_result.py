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
parser.add_argument("--episode-timeslots", default=75000, type=int)  
parser.add_argument("--timeslots", default=75000, type=int)   
parser.add_argument("--mode", default="traffic")
parser.add_argument("--reset-gains", default=True, type=bool) 
parser.add_argument("--K", default= 5, type=int)   
parser.add_argument("--N", default= 10, type=int) 
parser.add_argument("--M", default=1,  type=int)   
parser.add_argument("--Mobility", default= True,  type=bool) 
parser.add_argument("--modelsims", default= [0,5,10,15,20,25,30], nargs='+', type=float) # test models

parser.add_argument("--logs", default=True, type=bool)

parser.add_argument('--json-file-policy', type=str, default='dqn200_100_50',
                   help='json file for the hyperparameters')


kwargs = {'cumulative': True}
args = parser.parse_args()
data_rate = 15
modelsims = args.modelsims
json_file_policy = args.json_file_policy

for seed in args.seeds:
    args.seed = seed

    delay = {}
    p = {}

    r = {}
    ue = {}
    ave_delay_all = {}
    total_p_all = {}
    total_reward_all = {}
    throughput_all ={}
    
    for idxep, modelsim in enumerate(modelsims):
        folder_name = './simulations/N%dK%dM%dseed%dtimeslots%depisode_timeslots%ddr%dm%s'%(args.N,args.K,args.M,args.seed,args.timeslots,args.episode_timeslots,data_rate,args.Mobility)
            
        # np_save_path = '%s/%s%s.npz'%(folder_name_traffic,policy_folder_name.split('./simulations/')[1],json_file_policy)
        np_save_path = '%s/%s%dmodelsim%d.npz'%(folder_name,json_file_policy,args.seed,modelsim)
        
        data = np.load(np_save_path)
  
        throughputs = data['arr_1']
        p_strategy_all = data['arr_2']
        user_strategy_all = data['arr_3']
        all_delay_time = data['arr_4']
        all_reward = data['arr_5']
        total_packets = data['arr_6']
        total_Mbits = data['arr_7']

        spec_effs = data['arr_8']
        interfs  = data['arr_9']
        actual_p = data['arr_10']
        state_all = data['arr_11']
        strategy_all =  data['arr_12']

        

        delay['NN {}'.format(modelsim)] = all_delay_time
        p['NN {}'.format(modelsim)] = np.mean(p_strategy_all,axis = 1)
        r['NN {}'.format(modelsim)] = np.mean(all_reward, axis = (1,2))
        ue['NN {}'.format(modelsim)] = user_strategy_all[:,0]
        ave_delay = np.array(np.mean(all_delay_time,axis = 0))
        throughput_all['NN {}'.format(modelsim)] = np.sum(throughputs,axis = (0,1))
        #total_p = np.sum(actual_p,axis = (0,1))
        total_p = np.sum(actual_p,axis = (0,1,2))
        
        total_reward = np.sum(all_reward,axis = (0,1,2))

        
        ave_delay_all['NN {}'.format(modelsim)] = ave_delay
        total_p_all['NN {}'.format(modelsim)] = total_p
        
        total_reward_all['NN{}'.format(modelsim)] = total_reward

        

        


#load bench mark
np_save_path_benchmark = '%s/%s.npz'%(folder_name,'benchmark')

data = np.load(np_save_path_benchmark)


throughputs_benchmark = data['arr_0']
p_strategy_all_benchmark = data['arr_1']
user_strategy_all_benchmark = data['arr_2']
all_delay_time_benchmark = data['arr_3']
all_reward_benchmark = data['arr_4']
spec_effs_benchmark = data['arr_5']
interfs_benchmark  = data['arr_6']
total_packets_benchmark = data['arr_7']
total_Mbits_benchmark = data['arr_8']

        
average_reward_benchmark = np.mean(all_reward_benchmark, axis = (1,2))
average_p_benchmark = np.mean(p_strategy_all_benchmark,axis = 1)




delay['bm'] = all_delay_time_benchmark
p['bench_mark'] = np.mean(p_strategy_all_benchmark,axis = 1)
r['bench_mark'] = np.mean(all_reward_benchmark, axis = (1,2))
ue['bench_mark'] = user_strategy_all_benchmark[:,0]
throughput_all['bm'] = np.sum(throughputs_benchmark,axis = (0,1))       


ave_delay = np.array(np.mean(all_delay_time_benchmark,axis = 0))
ave_delay_all['bm'] = ave_delay

total_p_benchmark = np.sum(p_strategy_all_benchmark,axis = (0,1))

total_reward_benchmark = np.sum(all_reward_benchmark, axis = (0,1,2))

total_p_all['bm'] = total_p_benchmark

total_reward_all['bm'] = total_reward_benchmark


# total_bit_reward['bm'] = total_reward_benchmark + total_p_benchmark * args.price
#titanic = sns.load_dataset('titanic')

# df_wt = pd.DataFrame(wt)       
df_ave_delay = pd.DataFrame(ave_delay_all, index=[0])        
df_total_p = pd.DataFrame(total_p_all,index=[0])  
df_delay = pd.DataFrame(delay)
df_p = pd.DataFrame(p)      
df_r = pd.DataFrame(r)    
df_ue = pd.DataFrame(ue)  
dr_thrput = pd.DataFrame(throughput_all, index=[0])   

# df = pd.DataFrame({'reward from bit': df_integrated_reward.iloc[0],
#                    'reward from p': df_integrated_reward.iloc[1]},
#                   index = df_integrated_reward.columns)

# df.plot(kind='bar', stacked=True, color=['green', 'skyblue'])

# plt.ylim((-500.0,0.0))
# # labels for x & y axis
# plt.xlabel('eps')
# plt.ylabel('total reward')
 
# # title of plot
# plt.title('total reward')
# plt.show()


# plt.figure()
# sns.displot(df_delay, 
#             kind="ecdf")
# plt.xlabel('avarage delay time (seconds)')
# plt.ylabel('CDF')
# plt.title("avarage delay time")
# # plt.xlim((0.0, 0.3))
# plt.ylim((0.0, 1.1))
# plt.show()


plt.figure()
sns.barplot(data = df_ave_delay)
plt.xlabel('eps')
plt.ylabel('avarage delay time (seconds)')
plt.title("avarage delay time")
# plt.xlim((0.0, 5))
plt.ylim((0.0, 0.5))
plt.show()

# plt.figure()
# sns.barplot(data = dr_thrput)
# plt.xlabel('eps')
# plt.ylabel('throughput')
# plt.title("throughput")
# # plt.xlim((0.0, 5))
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
sns.barplot(data = df_total_p)
plt.xlabel('eps')
plt.ylabel('total power')
plt.title("total power")
# # plt.xlim((0.0, 5))
# plt.ylim((0.0, 0.1))
plt.show()







