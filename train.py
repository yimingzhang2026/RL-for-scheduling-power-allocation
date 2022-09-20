
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 16:32:52 2022

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
import copy
from itertools import cycle

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
json_file_policy = args.json_file_policy
with open ('./config/policy/'+json_file_policy+'.json','r') as f:
    options_policy = json.load(f)   

if not options_policy['cuda']:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    
# set data rates
#make simulation folder           
if not os.path.exists("./simulations"):
    os.makedirs("./simulations")
    
writer=tf.summary.FileWriter('./logs', tf.get_default_graph())

mobility_update_interval = 50 #1s

   
model_save_interval = 2500

pre_SINR = True
# reward_thresh2 = 2e6 * args.K / 8e6 / 25
#20 M data clear in one slot, indicate human clearance
reward_gap = 0.5
#stack more than 20M, clean queue and restart training
clean_thres = 100e6
for max_rate in args.max_rates:
    data_rates = max_rate * np.ones(args.N)
    for seed in args.seeds:
        args.seed = seed
        # train start
        folder_name = './simulations/N%dK%dM%dseed%dtimeslots%depisode_timeslots%ddr%dm%s'%(args.N,args.K,args.M,args.seed,args.timeslots,args.episode_timeslots,max_rate,args.Mobility)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            
        
        tf.reset_default_graph()  
        tf.set_random_seed(100 + args.seed)
        import env
        env = env.wireless_env(N = args.N,
                            M = args.M,
                            K = args.K,
                            data_rates = data_rates,
                            reset_gains = args.reset_gains,
                            seed = args.seed,
                            duration = args.episode_timeslots,
                            Num_neighbors = options_policy["AP_neighbors"],
                            )
        policy = DQN.DQN(args,options_policy,env)  
            
        
        p_strategy_all = []
        user_strategy_all = []
        utilization = np.zeros(env.K)
        #record data
        throughput = []
        reward_all = []
        state_all = []
        w_1_all = []
        w_2_all = []
        w_3_all = []
        b_3_all = []
        RX_loc = []
        old_reward = np.zeros(env.K)
        time_calculating_strategy_takes = []
        time_optimization_at_each_slot_takes = []
        
    
        with tf.Session() as sess:
            sess.run(policy.init)
    
            policy.initialize_updates(sess) 
            # Start iterating over timeslots
            for sim in range (args.timeslots):
                policy.check_memory_restart(sess,sim)       
                policy.update_handler(sess,sim)
                # save an instance per training episode for testing purposes.
                # if(sim %args.episode_timeslots == 0):
                #     model_destination = ('%s/%s_episode%d.ckpt'%(
                #             folder_name,json_file_policy,int(float(sim)/args.episode_timeslots))).replace('[','').replace(']','')
                #     policy.save(sess,model_destination)
                #     env.reset()
                if(sim == 0):
                    env.reset()
                    #olicy.update_epsilon(reset = True)
                
                if(sim % model_save_interval == 0):
                    model_destination = ('%s/%s_sim%d.ckpt'%(
                            folder_name,json_file_policy,int(float(sim)))).replace('[','').replace(']','')
                    policy.save(sess,model_destination)
                #every 1 sec mobility
    
                if(sim %args.episode_timeslots < 1):
                    #initialize p and user_selection
                    p_strategy = env.Pmax * np.random.rand(env.K)
                    user_strategy = np.random.randint(env.users_per_cell,size = env.K)

                    
                    
                if (sim % args.episode_timeslots >= 1):
                    #add mobility
                    if args.Mobility == True:
                    #add slow cycle
                        if(sim % mobility_update_interval == 0):
                            env.mobility_update(random_move = None)
                            record_rx_loc = copy.deepcopy(env.RX_loc)
                            RX_loc.append(record_rx_loc)
                            #print('sim {} q0 {} q1 {} q2 {} q3 {}'.format(sim,len(env.packets[0]),len(env.packets[1]),len(env.packets[2]),len(env.packets[3])))




                    #Each agent picks its strategy.
                    state = env.get_state()
                    state_all.append(state)
    #                reward = env.get_reward_include_p()
                    reward = env.get_reward()
                    # print(sum(reward_f_bit))

                    reward_all.append(reward)
    
                    
                    for agent in range (env.K):
                        current_local_state = state[agent,0,:]
    
                        
                        if (sim %args.episode_timeslots > 2 ): 
                            # value = max((abs(sum(old_reward) - sum(reward_f_bit))),value)
                            # current_reward = reward_f[agent] 
    
                            #discard outlier reward
                            if (abs(sum(old_reward) - sum(reward)) < reward_gap):
    
                                current_reward = reward[agent] 
                                policy.remember(agent,current_local_state,current_reward)

                        a_time = time.time() 
                        strategy= policy.act(sess,current_local_state,sim,agent)
                        time_calculating_strategy_takes.append(time.time()-a_time)
                                
                        # np.random.seed(sim)
                        # rd = np.random.randint(0,env.K-1,1)
                        # if agent == (rd[0]): 
    
                        #     # train for a minibatch
                        #     policy.train(sess,sim)                           
                                  
                        # Pick the action
                        if policy.strategy_translation[strategy] != 'off' :
                            user_strategy[agent] = policy.strategy_translation[strategy][0]
                            p_strategy[agent] = policy.strategy_translation[strategy][1]
    
                        else:
                            p_strategy[agent] = 0
                            user_strategy[agent] = 999 # no user selected since AP is off
    
                        # Add current state to the short term memory to observe it during the next state
                        policy.previous_state[agent,:] = current_local_state
                        policy.previous_action[agent] = strategy
                    # Only train it once per timeslot
                    a_time = time.time() 
                    policy.train(sess,sim)
                    time_optimization_at_each_slot_takes.append(time.time()-a_time)
                    
                    # vars = tf.trainable_variables()
                    # # print(vars) #some infos about variables...
                    w_1 = [v for v in tf.trainable_variables() if v.name == "Fweights/Variable_1:0"][0]
                    w_2 = [v for v in tf.trainable_variables() if v.name == "Fweights/Variable_2:0"][0]
                    w_3 = [v for v in tf.trainable_variables() if v.name == "Fweights/Variable_3:0"][0]
                    b_3 = [v for v in tf.trainable_variables() if v.name == "Fbiases/Variable_3:0"][0]
                    # var_2_all.append(var_2.eval())
                    # var_3_all.append(var_3.eval())
                    # print(w_1_all.eval())
                    # print(w_2_all.eval())
                    # print(w_3_all.eval())
                    #print(b_3_all.eval())
                    w_1_all.append(w_1.eval())
                    w_2_all.append(w_2.eval())
                    w_3_all.append(w_3.eval())
                    b_3_all.append(b_3.eval())

                    
                    


                for k in range(env.K):
                    remain_packets = []
                    for ue in env.user_mapping[k]:
                        remain_packets.append(sum(env.packets[ue]))
                    if sum(remain_packets) == 0:
                        p_strategy[k] = 0
                        user_strategy[k] = 999 
                        policy.previous_action[k] = 0 # action should be off
                    if p_strategy[k] != 0:
                         utilization[k] += 1
                        
                #store the output
                throughput.append(np.array(env.throughput))
                p_strategy_all.append(np.array(p_strategy))
                user_strategy_all.append(np.array(user_strategy))
                # if sum(p_strategy) > env.K * env.Pmax * 0.95:
                #     print('all 1 sim %d'%(sim))
                # if sum(p_strategy) < env.K * env.Pmax * 0.01 and sum([len(arr) for arr in env.packets])!=0:
                #     print('all 0 sim %d'%(sim))
                if sum([len(arr) for arr in env.packets])==0:
                    print('clean queues sim %d'%(sim))
                old_reward = env.get_reward()
                all_packets = sum([sum(queue) for queue in env.packets])
                if (sim > 0.1 * args.episode_timeslots and all_packets > clean_thres):
                    env.packets = [[] for i in range(env.N)]
                    print('manually cleaned')
    
                # print('old reward is %f'%sum(old_reward))
                env.step(p_strategy.reshape(env.K,env.M), user_strategy.reshape(env.K,env.M))
                
    
    
                # feedbacks = env.feedbacks #number of countinuous failed transmissions
                # if(sim % args.episode_timeslots == 0):
                #     print('Timeslot %d'%(sim))
                        
            policy.equalize(sess)
            print('Train is over')
                
            model_destination = ('%s/%s_sim%d.ckpt'%(
                    folder_name,json_file_policy,int(float(sim+1)))).replace('[','').replace(']','')
            policy.save(sess,model_destination)
            
    
        
# In[*]
        # End Train Phase
        np_save_path = '%s/%scheckweights.npz'%(folder_name,json_file_policy)
        print(np_save_path)
        # np.savez(np_save_path,options_policy,throughput,p_strategy_all,user_strategy_all)
        np.savez(np_save_path,w_1_all,w_2_all,w_3_all,b_3_all,throughput,reward_all,RX_loc,utilization,time_calculating_strategy_takes,time_optimization_at_each_slot_takes)

    
    
    
    
    
    
    
    
    
    
    
    
    
    