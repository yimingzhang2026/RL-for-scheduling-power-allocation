# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 14:46:12 2022

@author: yzhang4
"""
import os
import numpy as np
import project_backend as pb
import time
import collections
import json
import DQN as DQN
import tensorflow as tf
import argparse
import tensorflow as tf
from itertools import cycle
#define test deployment
parser = argparse.ArgumentParser(description='give test scenarios.')
parser.add_argument("--seeds", default=[1], nargs='+', type=int)              
parser.add_argument("--episode-timeslots", default=2500, type=int)
parser.add_argument("--slow-cycle", default= 0, type=int)   
parser.add_argument("--timeslots", default=75000, type=int)   
parser.add_argument("--mode", default="traffic")
parser.add_argument("--reset-gains", default=True, type=bool) 
parser.add_argument("--K", default= 5, type=int)   
parser.add_argument("--N", default= 10, type=int) 
parser.add_argument("--M", default=1,  type=int)  
parser.add_argument("--Mobility", default= True,  type=bool)  
parser.add_argument("--modelsims", default= [30], nargs='+', type=float) # test models
#parser.add_argument("--episodes", default=[0,1,5,10,15,20,25,30,35,40], nargs='+', type=float) # test episodes
parser.add_argument("--compare", default=True, type=bool) 
# parser.add_argument("--num-potential-APs", default=3,  type=int)

parser.add_argument('--json-file-policy', type=str, default='dqn200_100_50',
                   help='json file for the hyperparameters')

#define test model

parser.add_argument("--policy-seeds", default=[1], nargs='+', type=int)  
parser.add_argument("--policy-episode-timeslots", default=75000, type=int)    
parser.add_argument("--policy-timeslots", default=75000, type=int)   
parser.add_argument("--policy-mode", default="traffic")
parser.add_argument("--policy-reset-gains", default=True, type=bool) 
parser.add_argument("--policy-K", default=5, type=int)  
parser.add_argument("--policy-N", default=10, type=int)  
parser.add_argument("--policy-M", default=1, type=int) 
parser.add_argument("--policy-Mobility", default= True,  type=bool) 

parser.add_argument("--logs", default=True, type=bool)


args = parser.parse_args()
json_file_policy = args.json_file_policy
with open ('./config/policy/'+json_file_policy+'.json','r') as f:
    options_policy = json.load(f)   

if not options_policy['cuda']:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    
#define data rates    
max_rate = 15
data_rates = max_rate * np.ones(args.N)
operate_APs = [ [i,i] for i in range(args.K)]
operate_APs_pool = cycle(sum(operate_APs,[]))
mobility_update_interval = 50 #1s
re_association_interval = mobility_update_interval * 50 #about 50m
model_save_interval = 2500

for seed in args.seeds:
    args.seed = seed
    args.policy_seed = seed


    policy_folder_name = './simulations/N%dK%dM%dseed%dtimeslots%depisode_timeslots%ddr%dm%s'%(args.policy_N,args.policy_K,args.policy_M,args.seed,args.policy_timeslots,args.policy_episode_timeslots,max_rate,args.policy_Mobility)

    folder_name = './simulations/N%dK%dM%dseed%dtimeslots%depisode_timeslots%ddr%dm%s'%(args.N,args.K,args.M,args.seed,args.timeslots,args.episode_timeslots,max_rate,args.Mobility)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        
    for modelsim in args.modelsims:
        tf.reset_default_graph()
        # tf.set_random_seed(100+args.seed)
        import env
        envs = []
        # env[0] [1] [2] for NN， FP， WMMSE respectively
        envs.append(env.wireless_env(N = args.N,
                            M = args.M,
                            K = args.K,
                            data_rates = data_rates,
                            reset_gains = args.reset_gains,
                            seed = args.seed,
                            duration = args.timeslots,
                            Num_neighbors = options_policy["AP_neighbors"],
                            ))
        envs.append(env.wireless_env(N = args.N,
                            M = args.M,
                            K = args.K,
                            data_rates = data_rates,
                            reset_gains = args.reset_gains,
                            seed = args.seed,
                            duration = args.timeslots,
                            Num_neighbors = options_policy["AP_neighbors"],
                            ))
        envs.append(env.wireless_env(N = args.N,
                            M = args.M,
                            K = args.K,
                            data_rates = data_rates,
                            reset_gains = args.reset_gains,
                            seed = args.seed,
                            duration = args.timeslots,
                            Num_neighbors = options_policy["AP_neighbors"],
                            ))
        
        policy = DQN.DQN(args,options_policy,envs[0]) 


        time_fps_takes = []
        time_wmmse_takes = []
        time_NN_takes = []
        throughput = []
        p_strategy_all = []
        user_strategy_all = []
        all_delay_time = []
        all_reward = []
        
        # actual_p = []
        
        # state_all = []
        # strategy_all = []
        
        if args.compare == True:
            # weights_pfs = np.ones(env[1].K)
            # weights_wmmse = np.ones(env[2].K)
            p_strategy_all_pfs = []
            all_delay_time_pfs = []
            all_reward_pfs = []
            weights_pfs = []
            p_strategy_all_wmmse = []
            all_delay_time_wmmse = []
            all_reward_wmmse = []
            weights_wmmse = []

        
        if args.logs:
            total_packets = []
            total_Mbits = []

            average_wait_time = []
            total_served = []
            average_delay = []
            spec_effs = []
            # interfs = []
            
            if args.compare == True:

                average_delay_pfs  = []
                spec_effs_pfs  = []
                # interfs_benchmark  = []
                total_packets_pfs = []
                total_Mbits_pfs = []
                weights_pfs_all = []
                weights_wmmse_all = []
                average_delay_wmmse  = []
                spec_effs_wmmse  = []
                # interfs_wmmse  = []
                total_packets_wmmse = []
                total_Mbits_wmmse = []
        
        with tf.Session() as sess:
            sess.run(policy.init)
            # Start iterating over timeslots
            for sim in range (args.timeslots):
                # save an instance per training episode for testing purposes.
                if(sim == 0):
                    model_destination = ('%s/%s_sim%d.ckpt'%(
                            policy_folder_name,json_file_policy,modelsim * model_save_interval)).replace('[','').replace(']','')
                    policy.load(sess,model_destination)

                    for env in envs:
                        env.reset()
                if(sim %args.episode_timeslots < 1):
                    #initialize p and user_selection
                    p_strategy = envs[0].Pmax * np.ones(envs[0].K)
                    user_strategy = np.random.randint(envs[0].users_per_cell,size = envs[0].K)
                    # clean_queues = 0
                    if args.compare == True:
                        p_strategy_pfs = envs[1].Pmax * np.ones(envs[1].N)
                        p_strategy_wmmse = envs[2].Pmax * np.ones(envs[2].N)
                        # user_strategy_benchmark = np.zeros(env.K).reshape(env.K).astype(int)
                        # RL_win = 0
                        # benchmark_win = 0
                    
                if (sim %args.episode_timeslots >= 1):
                    if args.Mobility == True:
                        if(sim % mobility_update_interval == 0):
                            for env in envs:
                                env.mobility_update(random_move = env.move_random_state)

                    # for NN method
                    state = envs[0].get_state()
                    # time_NN = 0
                    for agent in range (envs[0].K):                       
                        current_local_state = state[agent,0,:]
#                        a_time = time.time() 
                        strategy= policy.act_noepsilon(sess,current_local_state,sim,agent)
                        # time_NN += time.time()-a_time
                        # strategy_all.append(strategy)
                        # Pick the action
                        if policy.strategy_translation[strategy] != 'off' :
                            user_strategy[agent] = policy.strategy_translation[strategy][0]
                            p_strategy[agent] = policy.strategy_translation[strategy][1]

                        else:
                            p_strategy[agent] = 0
                            user_strategy[agent] = 999 # no user selected since AP is off
                    # time_NN_takes.append(time_NN)

                    for k in range(envs[0].K):
                        remain_packets = []
                        for ue in envs[0].user_mapping[k]:
                            remain_packets.append(sum(envs[0].packets[ue]))
                        if sum(remain_packets) == 0:
                            p_strategy[k] = 0
                            user_strategy[k] = 999 
                            
                    # for pfs and wmmse method
                    if args.compare == True:
                        # a_time = time.time() 
                        p_strategy_pfs = pb.FP_algorithm_weighted(envs[1].N, envs[1].H[-1], envs[1].Pmax, envs[1].noise_var,envs[1].weights)
                        # time_fps_takes.append(time.time()-a_time)
                        weights_wmmse = np.ones(envs[2].N)
                        for n in range(envs[2].N):
                            if len(envs[2].packets[n]) == 0:
                                weights_wmmse[n] = 0
                        # a_time = time.time() 
                        # print('weights are {}'.format(weights_wmmse))
                        p_strategy_wmmse = pb.WMMSE_algorithm_weighted(envs[2].N, envs[2].H[-1], envs[2].Pmax, envs[2].noise_var,weights_wmmse)
                        # time_wmmse_takes.append(time.time()-a_time)
                        # print(p_strategy_wmmse)

                if (sim %args.episode_timeslots >= 1):
                    #store the output
                    # throughput.append(np.array(envs[0].throughput))
                    p_strategy_all.append(np.array(p_strategy))
                    user_strategy_all.append(np.array(user_strategy))
    
                
                    if args.compare == True:
                        #throughput_benchmark.append(np.array(env.throughput_benchmark))
                        p_strategy_all_pfs.append(np.array(p_strategy_pfs))
                        weights_pfs_all.append(envs[2].weights)
                        p_strategy_all_wmmse.append(np.array(p_strategy_wmmse))
                        weights_wmmse_all.append(weights_wmmse)
                        #user_strategy_all_benchmark.append(np.array(user_strategy_benchmark))
    
    
                
    
                    
                    
                    if args.logs:
                        total_packets.append([len(ar) for ar in envs[0].packets])
                        total_Mbits.append([sum(ar)/1e6 for ar in envs[0].packets])
#                        average_wait_time.append([env.t - sum(ar)/len(ar) if len(ar) > 0 else 0 for ar in env.packets_t])
#                        total_served.append([len(ar) for ar in env.processed_packets_t])
#                        average_delay.append([sum(ar)/len(ar) if len(ar) > 0 else 0 for ar in env.processed_packets_t])
    
    
                    
                        if args.compare == True:
#                            average_wait_time_benchmark.append([env.t - sum(ar)/len(ar) if len(ar) > 0 else 0 for ar in env.packets_t_benchmark])
#                            total_served_benchmark.append([len(ar) for ar in env.processed_packets_t_benchmark])
#                            average_delay_benchmark.append([sum(ar)/len(ar) if len(ar) > 0 else 0 for ar in env.processed_packets_t_benchmark])
                            spec_effs_pfs.append(np.sum(envs[1].spec_eff,axis=1))
                            total_packets_pfs.append([len(ar) for ar in envs[1].packets])
                            total_Mbits_pfs.append([sum(ar)/1e6 for ar in envs[1].packets])
                            
                            spec_effs_wmmse.append(np.sum(envs[2].spec_eff,axis=1))
                            total_packets_wmmse.append([len(ar) for ar in envs[2].packets])
                            total_Mbits_wmmse.append([sum(ar)/1e6 for ar in envs[2].packets])
                            
                            

                envs[0].step(p_strategy.reshape(envs[0].K,envs[0].M), user_strategy.reshape(envs[0].K,envs[0].M))
                envs[1].step_pfs(p_strategy_pfs.reshape(envs[1].N,envs[1].M))
                envs[2].step_wmmse(p_strategy_wmmse.reshape(envs[2].N,envs[2].M))
                # print(sim)
                
                        
                        
                if args.logs and sim %args.episode_timeslots >= 1:
                    spec_effs.append(np.sum(envs[0].spec_eff,axis=(0,1)))
                    reward = np.array(envs[0].get_reward())
                    all_reward.append(reward)
                    if args.compare == True :
                        spec_effs_pfs.append(np.sum(envs[1].spec_eff,axis=(0,1)))
                        spec_effs_wmmse.append(np.sum(envs[2].spec_eff,axis=(0,1)))

                    
                if(sim % args.episode_timeslots == 0):
                    print('Timeslot %d'%(sim))
                    if(sim > 0):
                        # save 
                        np_save_path = '%s/%s%dmodelsim%dep%d.npz'%(folder_name,json_file_policy,args.policy_seed,modelsim,int(sim / args.episode_timeslots))
                        print(np_save_path)
                        for n in range(len(envs[0].processed_packets_t)):
                            all_delay_time = np.concatenate((all_delay_time,envs[0].processed_packets_t[n]), axis = None)
                            unprocessed_packets_t = envs[0].packets_t[n]
                            unprocessed_delay = [envs[0].t - x for x in unprocessed_packets_t]
                            all_delay_time = np.concatenate((all_delay_time,unprocessed_delay), axis = None)
                        all_delay_time = list(all_delay_time * 0.02)
                        sumrate = [ i * envs[0].bw *envs[0].T for i in spec_effs]
                        np.savez(np_save_path,
                                 options_policy,
                                 p_strategy_all,
                                 user_strategy_all,
                                 all_delay_time,
                                 all_reward,
                                 total_packets,
                                 total_Mbits,
                                 sumrate)
                        #clean record data and start next record
                        envs[0].packets = [[] for i in range(envs[0].N)]
                        envs[0].packets_t = [[] for i in range(envs[0].N)] 
                        envs[0].processed_packets_t = [[] for i in range(envs[0].N)]
                        p_strategy_all = []
                        user_strategy_all = []
                        all_delay_time = []
                        all_reward = []
                        total_packets = []
                        total_Mbits = []
                        spec_effs = []
                        


                        
                        if args.compare == True:
                            #save pfs
                            np_save_path_pfs = '%s/%sep%d.npz'%(folder_name,'benchmark_pfs',int(sim / args.episode_timeslots))
                            for n in range(len(envs[1].processed_packets_t)):
                                all_delay_time_pfs = np.concatenate((all_delay_time_pfs,envs[1].processed_packets_t[n]), axis = None)
                                unprocessed_packets_t_pfs = envs[1].packets_t[n]
                                unprocessed_delay_pfs = [envs[1].t - x for x in unprocessed_packets_t_pfs]
                                all_delay_time_pfs = np.concatenate((all_delay_time_pfs,unprocessed_delay_pfs),axis = None)
                            all_delay_time_pfs = list(all_delay_time_pfs * 0.02)
                            sumrate_pfs = [ i * envs[1].bw *envs[1].T for i in spec_effs_pfs]
                            np.savez(np_save_path_pfs,  
                                      all_delay_time_pfs,
                                      weights_pfs,                     
                                      p_strategy_all_pfs,
                                      total_packets_pfs,
                                      total_Mbits_pfs,
                                      sumrate_pfs)
                            envs[1].packets = [[] for i in range(envs[1].N)]
                            envs[1].packets_t = [[] for i in range(envs[1].N)] 
                            envs[1].processed_packets_t = [[] for i in range(envs[1].N)]
                            all_delay_time_pfs = []
                            weights_pfs = []
                            p_strategy_all_pfs  = []
                            total_packets_pfs = []
                            total_Mbits_pfs = []
                            spec_effs_pfs = []
                            
                            #save wmmse
                            np_save_path_wmmse = '%s/%sep%d.npz'%(folder_name,'benchmark_wmmse',int(sim / args.episode_timeslots))
                            for n in range(len(envs[2].processed_packets_t)):
                                all_delay_time_wmmse = np.concatenate((all_delay_time_wmmse,envs[2].processed_packets_t[n]), axis = None)
                                unprocessed_packets_t_wmmse = envs[2].packets_t[n]
                                unprocessed_delay_wmmse = [envs[2].t - x for x in unprocessed_packets_t_wmmse]
                                all_delay_time_wmmse = np.concatenate((all_delay_time_wmmse,unprocessed_delay_wmmse),axis = None)
                            all_delay_time_wmmse = list(all_delay_time_wmmse * 0.02)
                            sumrate_wmmse = [ i * envs[2].bw *envs[2].T for i in spec_effs_wmmse]
                            np.savez(np_save_path_wmmse,  
                                      all_delay_time_wmmse,
                                      weights_wmmse,                     
                                      p_strategy_all_wmmse,
                                      total_packets_wmmse,
                                      total_Mbits_wmmse,
                                      sumrate_wmmse)
                            envs[2].packets = [[] for i in range(envs[2].N)]
                            envs[2].packets_t = [[] for i in range(envs[2].N)] 
                            envs[2].processed_packets_t = [[] for i in range(envs[2].N)]
                            all_delay_time_wmmse = []
                            weights_wmmse = []
                            p_strategy_all_wmmse  = []
                            total_packets_wmmse = []
                            total_Mbits_wmmse = []
                            spec_effs_wmmse = []
            print('Test is over')

np_save_path_time = '%s/%s%dtime.npz'%(folder_name,json_file_policy,args.timeslots)
np.savez(np_save_path_time,
         time_fps_takes,
         time_wmmse_takes,
         time_NN_takes)
         

        
    

    
    



