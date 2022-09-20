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
#define test deployment
parser = argparse.ArgumentParser(description='give test scenarios.')
parser.add_argument("--seeds", default=[1], nargs='+', type=int)              
parser.add_argument("--episode-timeslots", default=5000, type=int)
parser.add_argument("--timeslots", default=5000, type=int)   
parser.add_argument("--mode", default="traffic")
parser.add_argument("--reset-gains", default=True, type=bool) 
parser.add_argument("--K", default= 5, type=int)   
parser.add_argument("--N", default= 10, type=int) 
parser.add_argument("--M", default=1,  type=int)  
parser.add_argument("--Mobility", default= True,  type=bool)  
parser.add_argument("--modelsims", default= [0,1,5,10,15,20,25,30], nargs='+', type=float) # test models
#parser.add_argument("--episodes", default=[0,1,5,10,15,20,25,30,35,40], nargs='+', type=float) # test episodes
parser.add_argument("--compare", default=True, type=bool) 
# parser.add_argument("--num-potential-APs", default=3,  type=int)

parser.add_argument('--json-file-policy', type=str, default='dqn200_100_50',
                   help='json file for the hyperparameters')

#define test model

parser.add_argument("--policy-seeds", default=[1], nargs='+', type=int)  
parser.add_argument("--policy-episode-timeslots", default=75000, type=int)   
parser.add_argument("--policy-slow-cycle", default= 0, type=int)  
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
max_rate = 15
data_rates = max_rate * np.ones(args.N)
mobility_update_interval = 50 #1s
re_association_interval = mobility_update_interval * 50 #about 50m
model_save_interval = 2500

for seed in args.seeds:
    args.seed = seed
    args.policy_seed = seed

    policy_folder_name = './simulations/N%dK%dM%dseed%dtimeslots%depisode_timeslots%ddr%dm%s'%(args.policy_N,args.policy_K,args.policy_M,args.seed,args.policy_timeslots,args.policy_episode_timeslots,max_rate,args.policy_Mobility)

    folder_name = policy_folder_name
        
    for modelsim in args.modelsims:
        tf.reset_default_graph()
        # tf.set_random_seed(100+args.seed)
        import env
        env = env.wireless_env(N = args.N,
                            M = args.M,
                            K = args.K,
                            data_rates = data_rates,
                            reset_gains = args.reset_gains,
                            seed = args.seed,
                            duration = args.timeslots,
                            Num_neighbors = options_policy["AP_neighbors"],
                            )
        
        policy = DQN.DQN(args,options_policy,env)  


        
        throughput = []
        p_strategy_all = []
        user_strategy_all = []
        all_delay_time = []
        all_reward = []
        all_reward_bit = []
        all_reward_p = []
        actual_p = []
        
        state_all = []
        strategy_all = []
        
        if args.compare == True:
            throughput_benchmark = []
            p_strategy_all_benchmark = []
            user_strategy_all_benchmark = []
            all_delay_time_benchmark = []
            all_reward_benchmark = []
            all_reward_bit_benchmark = []
            all_reward_p_benchmark = []
        
        if args.logs:
            total_packets = []
            total_Mbits = []
            
            
            average_wait_time = []
            total_served = []
            average_delay = []
            spec_effs = []
            interfs = []
            
            if args.compare == True:
                average_wait_time_benchmark = []
                total_served_benchmark  = []
                average_delay_benchmark  = []
                spec_effs_benchmark  = []
                interfs_benchmark  = []
                total_packets_benchmark = []
                total_Mbits_benchmark = []
        
        with tf.Session() as sess:
            sess.run(policy.init)
            # Start iterating over timeslots
            for sim in range (args.timeslots):
                # save an instance per training episode for testing purposes.
                if(sim == 0):
                    model_destination = ('%s/%s_sim%d.ckpt'%(
                            policy_folder_name,json_file_policy,modelsim * model_save_interval)).replace('[','').replace(']','')
                    policy.load(sess,model_destination)

                    if args.compare == True:
                        env.reset_compare()
                    else:
                        env.reset()
                if(sim %args.episode_timeslots < 1):
                    #initialize p and user_selection
                    p_strategy = env.Pmax * np.ones(env.K)
                    user_strategy = np.random.randint(env.users_per_cell,size = env.K)

                    if args.compare == True:
                        p_strategy_benchmark = env.Pmax * np.ones(env.K)
                        user_strategy_benchmark = np.zeros(env.K).reshape(env.K).astype(int)

                    
                if (sim %args.episode_timeslots >= 1):
                    if args.Mobility == True:
                        if(sim % mobility_update_interval == 0):
                            env.mobility_update(random_move = env.move_random_state)


                    state = env.get_state()
                    state_all.append(np.squeeze(state, axis=1))
                    # reward_f = env.get_reward_inclue_p()
                    # if (sim % 200 == 0):
                    #     # print('timeslot {}, the reward is {}'.format(sim,reward_f))
                    #     #print('timeslot {}, the queue length is \n{}'.format(sim,int(list(env.packets))))
                    #     print('timeslot {}, the queue length is \n{}'.format(sim,env.packets))
                    for agent in range (env.K):
                        current_local_state = state[agent,0,:]
                        strategy= policy.act_noepsilon(sess,current_local_state,sim,agent)
                        strategy_all.append(strategy)
                        # Pick the action
                        if policy.strategy_translation[strategy] != 'off' :
                            user_strategy[agent] = policy.strategy_translation[strategy][0]
                            p_strategy[agent] = policy.strategy_translation[strategy][1]

                        else:
                            p_strategy[agent] = 0
                            user_strategy[agent] = 999 # no user selected since AP is off
                    

                            
                            
                            
                        if args.compare == True:
                        # allocate power for benchmark algorithm, i.e. max power for non-empty queue
                            user = []
                            remain_packets = []
                            for ue in env.user_mapping[agent]:
                                user.append(ue)
                                remain_packets.append(sum(env.packets_benchmark[ue]))
                                
                            if sum( remain_packets ) != 0:
                                p_strategy_benchmark[agent] = env.Pmax
                                selected_user = user[remain_packets.index(max(remain_packets))]
                                user_strategy_benchmark[agent] = selected_user
                            else:
                                p_strategy_benchmark[agent] = 0
                                user_strategy_benchmark[agent] = 999 # no user selected since AP is off
                    

                for k in range(env.K):
                    remain_packets = []
                    for ue in env.user_mapping[k]:
                        remain_packets.append(sum(env.packets[ue]))
                    if sum(remain_packets) == 0:
                        p_strategy[k] = 0
                        user_strategy[k] = 999 
                            
                            
                
                    

                    

                        

                if (sim %args.episode_timeslots >= 1):
                    #store the output
                    throughput.append(np.array(env.throughput))
                    p_strategy_all.append(np.array(p_strategy))
                    user_strategy_all.append(np.array(user_strategy))
    
                
                    if args.compare == True:
                        throughput_benchmark.append(np.array(env.throughput_benchmark))
                        p_strategy_all_benchmark.append(np.array(p_strategy_benchmark))
                        user_strategy_all_benchmark.append(np.array(user_strategy_benchmark))
    
    
                
    
                    
                    
                    if args.logs:
                        total_packets.append([len(ar) for ar in env.packets])
                        total_Mbits.append([sum(ar)/1e6 for ar in env.packets])
#                        average_wait_time.append([env.t - sum(ar)/len(ar) if len(ar) > 0 else 0 for ar in env.packets_t])
#                        total_served.append([len(ar) for ar in env.processed_packets_t])
#                        average_delay.append([sum(ar)/len(ar) if len(ar) > 0 else 0 for ar in env.processed_packets_t])
    
    
                    
                        if args.compare == True:
#                            average_wait_time_benchmark.append([env.t - sum(ar)/len(ar) if len(ar) > 0 else 0 for ar in env.packets_t_benchmark])
#                            total_served_benchmark.append([len(ar) for ar in env.processed_packets_t_benchmark])
#                            average_delay_benchmark.append([sum(ar)/len(ar) if len(ar) > 0 else 0 for ar in env.processed_packets_t_benchmark])
                            spec_effs_benchmark.append(np.sum(env.spec_eff_benchmark,axis=1))
                            interfs_benchmark.append(np.array(env.total_interf_benchmark))
                            total_packets_benchmark.append([len(ar) for ar in env.packets_benchmark])
                            total_Mbits_benchmark.append([sum(ar)/1e6 for ar in env.packets_benchmark])
                            
                            
                if args.compare == True:
                    env.step_compare(p_strategy.reshape(env.K,env.M), user_strategy.reshape(env.K,env.M),p_strategy_benchmark.reshape(env.K,env.M), user_strategy_benchmark.reshape(env.K,env.M))
                else:
                    env.step(p_strategy.reshape(env.K,env.M), user_strategy.reshape(env.K,env.M))
                
                        
                if args.logs and sim %args.episode_timeslots >= 1:
                    spec_effs.append(np.sum(env.spec_eff,axis=1))
                    interfs.append(np.array(env.total_interf))
                    actual_p.append(np.array(env.p))

                    #reward = np.array(env.get_reward_include_p())
                    
                    reward = np.array(env.get_reward())

                    all_reward.append(reward)
                    if args.compare == True :
                        reward_benchmark = np.array(env.get_reward_compare())
                        all_reward_benchmark.append(reward_benchmark)
                    
                if(sim % args.episode_timeslots == 0):
                    print('Timeslot %d'%(sim))
                            
            print('Test is over')
        # End Train Phase
        # np_save_path = '%s/%s%s.npz'%(folder_name,policy_folder_name.split('./simulations/')[1],json_file_policy)
        np_save_path = '%s/%s%dmodelsim%d.npz'%(policy_folder_name,json_file_policy,args.policy_seed,modelsim)

        print(np_save_path)
        for n in range(len(env.processed_packets_t)):
            all_delay_time = np.concatenate((all_delay_time,env.processed_packets_t[n]), axis = None)
            unprocessed_packets_t = env.packets_t[n]
            #ignore last 0.1 * args.timeslots arrivals, e.g. 50s, ignore last 5s
            #unprocessed_packets_t = [x for x in unprocessed_packets_t if x < 0.9* args.timeslots]
            unprocessed_delay = [env.t - x for x in unprocessed_packets_t]
#            unprocessed_delay = np.clip(unprocessed_delay, 0, args.episode_timeslots*0.1)
            all_delay_time = np.concatenate((all_delay_time,unprocessed_delay), axis = None)
        all_delay_time = list(all_delay_time * 0.02)
        
        
        if args.compare == True:
            for n in range(len(env.processed_packets_t_benchmark)):
                all_delay_time_benchmark = np.concatenate((all_delay_time_benchmark,env.processed_packets_t_benchmark[n]), axis = None)
                unprocessed_packets_t_benchmark = env.packets_t_benchmark[n]
                #unprocessed_packets_t_benchmark = [x for x in unprocessed_packets_t_benchmark if x < 0.9* args.timeslots]
                unprocessed_delay_benchmark = [env.t - x for x in unprocessed_packets_t_benchmark]
                all_delay_time_benchmark = np.concatenate((all_delay_time_benchmark,unprocessed_delay_benchmark),axis = None)
            all_delay_time_benchmark = list(all_delay_time_benchmark * 0.02)

        if not args.logs:
            np.savez(np_save_path,options_policy,throughput,p_strategy_all,user_strategy_all,
                         all_delay_time)
            if args.compare == True:
                np_save_path_benchmark = '%s/%s.npz'%(policy_folder_name,'benchmark')
                np.savez(np_save_path_benchmark,throughput_benchmark,p_strategy_all_benchmark,user_strategy_all_benchmark,
                         all_delay_time_benchmark)
        else:
            
            np.savez(np_save_path,
                     options_policy,
                     throughput,
                     p_strategy_all,
                     user_strategy_all,
                     all_delay_time,
                     all_reward,
                     total_packets,
                     total_Mbits,
                     spec_effs,
                     interfs,
                     actual_p,
                     state_all,
                     strategy_all)
            

            
            if args.compare == True:
                np_save_path_benchmark = '%s/%s.npz'%(policy_folder_name,'benchmark')
                np.savez(np_save_path_benchmark,  
                         throughput_benchmark,
                         p_strategy_all_benchmark,
                         user_strategy_all_benchmark,
                         all_delay_time_benchmark,
                         all_reward_benchmark,
                         # average_wait_time_benchmark,
                         # total_served_benchmark,
                         # average_delay_benchmark,
                         spec_effs_benchmark,
                         interfs_benchmark,
                         total_packets_benchmark,
                         total_Mbits_benchmark)
        
    

    
    


