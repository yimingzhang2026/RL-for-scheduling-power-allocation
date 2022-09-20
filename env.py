# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 23:56:49 2022

@author: yzhang4
"""
import numpy as np
from numpy import random
from scipy import special
import project_backend as pb
import collections
import math
import copy


class wireless_env():
    def __init__(self,
             N = 20,
             K = 5,
             M = 1,
             T = 0.02,
             Pmax_dBm = 23.0,
             n0_dBm = -114.0,
             rayleigh_var = 1.0, 
             shadowing_dev = 8.0,
             R = 500, # meter
             min_dist = 35,
             equal_number_for_BS = True,
             reset_gains = False,
             data_rates = None,
             Num_neighbors = 3,
             seed = 2022,
             f_d = 10.0,
             traffic_levels = 5,
             packet_size = 5e5,
             bw = 10e6, #Hz
             traffic_seed = None,
             duration = 100000,
             step = 1):
        
        self.min_dist = min_dist 
        self.data_rates = data_rates
        self.step_size = step
        self.seed = seed
        self.N = N
        self.K = K
        self.M = M
        self.rayleigh_var = rayleigh_var
        self.shadowing_dev = shadowing_dev
        self.R = R
        self.equal_number_for_BS = equal_number_for_BS
        if equal_number_for_BS == True:
            self.users_per_cell = int(np.ceil(self.N/self.K))
        # self.min_dist = min_dist
        # self.equal_number_for_BS = equal_number_for_BS

        self.pre_SINRmax_db = 30
        self.link_activation = np.zeros(N)
        #self.feedbacks = np.zeros((self.N,self.M))
        
        self.correlation = special.j0(2.0*np.pi*f_d*T)
        self.Pmax_dB = Pmax_dBm - 30
        self.Pmax = np.power(10.0,(Pmax_dBm - 30)/10)
        self.noise_var = np.power(10.0,(n0_dBm - 30)/10)
        
        self.SINR = np.zeros((self.N, self.M))
        
        # Just store the current instance and one past instance of the channel
        self.H_cell2user = collections.deque([],2) # downlink channel gain
        self.H = collections.deque([],2) # aggregate interference
        
        self.priorities = np.ones(self.N)
        self.p = self.Pmax * np.ones((self.N,self.M))
        self.spec_eff = np.zeros((self.N, self.M))
        self.total_interf = np.zeros((self.N, self.M))
        self.p_action = self.Pmax * np.ones((self.K,self.M))
        self.p_action_benchmark = self.Pmax * np.ones((self.K,self.M))
        
        
        np.random.seed(self.seed)
        self.channel_random_state = np.random.RandomState(self.seed + 2021)
        self.traffic_random_state = np.random.RandomState(self.seed + 4042)
        self.move_random_state = np.random.RandomState(self.seed + 8084)
        
        self.data_rates = data_rates
        self.duration = duration
        self.bw = bw
        self.packet_size = packet_size
        self.traffic_levels = traffic_levels
        self.T = T
        
        self.Num_neighbors = Num_neighbors

        
        self.reset_gains = reset_gains
        self.t = -1
        

        scale_g_dB = - (128.1 + 37.6* np.log10(0.001 * self.R))
        self.scale_gain = np.power(10.0,scale_g_dB/10.0)
        
        
        self.spec_eff_benchmark = np.zeros((self.N, self.M))
        self.total_interf_benchmark = np.zeros((self.N, self.M))
        
        # self.activation = np.ones(self.K)
        
        self.weights = np.ones(self.N)
        self.beta = 0.01
        self.average_spec_eff = np.zeros(self.N)
        
    def create_traffic(self):
        self.packets = [[] for i in range(self.N)]
        self.packets_t = [[] for i in range(self.N)] # store when the packet arrived
        
        
        self.arrival_rates = self.data_rates / self.traffic_levels * self.traffic_random_state.choice(range(1, 1+self.traffic_levels),size=self.N) 
        #self.arrival_rates = self.data_rates
#        self.arrival_rates = self.max_rate / self.traffic_levels * self.traffic_random_state.choice(range(1, 1+self.traffic_levels),size=self.N)*self.T
        # self.poisson_process = [ np.zeros(self.duration) for i in range(self.N)]    
        self.poisson_process = [[] for i in range(self.N)]
        
        # time = np.arange(0, self.duration + 1, 1)
        # amplitude = 1 + 0.5 * np.sin(2 * np.pi / 2500 * time)
        for i in range(self.N):
            arrival_rate = self.arrival_rates[i]
            self.poisson_process[i] = np.random.poisson(arrival_rate* self.T, self.duration + 2)

        
        self.throughput = np.zeros(self.N)
        self.processed_packets_t = [[] for i in range(self.N)] # stores delays
        
        
    def create_traffic_compare(self):
        np.random.seed(self.seed)
        self.create_traffic()
#         self.packets = [[] for i in range(self.N)]
#         self.packets_t = [[] for i in range(self.N)] # store when the packet arrived
        
        self.packets_benchmark = [[] for i in range(self.N)]
        self.packets_t_benchmark = [[] for i in range(self.N)] # store when the packet arrived
        #print('benchmark declared')
        
#         self.arrival_rates = self.data_rates / self.traffic_levels * self.traffic_random_state.choice(range(1, 1+self.traffic_levels),size=self.N) 
# #        self.arrival_rates = self.max_rate / self.traffic_levels * self.traffic_random_state.choice(range(1, 1+self.traffic_levels),size=self.N)*self.T
#         # self.poisson_process = [ np.zeros(self.duration) for i in range(self.N)]    
#         self.poisson_process = [[] for i in range(self.N)]

#         for i in range(self.N):
#             arrival_rate = self.arrival_rates[i]
#             self.poisson_process[i] = np.random.poisson(arrival_rate* self.T, self.duration + 1)
        
#         self.throughput = np.zeros(self.N)
#         self.processed_packets_t = [[] for i in range(self.N)] # stores delays
        
        self.throughput_benchmark = np.zeros(self.N)
        self.processed_packets_t_benchmark = [[] for i in range(self.N)] # stores delays
        #print('declared')
            
    def load_traffic(self):
        # load traffic
        for n in range(self.N):
            num_incoming = int(self.poisson_process[n][self.t])
            for i in range(num_incoming):
                self.packets[n].append(self.packet_size)
                self.packets_t[n].append(self.t)
                
    def load_traffic_compare(self):
        # load traffic
        for n in range(self.N):
            num_incoming = int(self.poisson_process[n][self.t])
            for i in range(num_incoming):
                arrive_packet = self.packet_size
                self.packets[n].append(self.packet_size)
                self.packets_t[n].append(self.t)
                # print(self.t, n)
                # print('packets_t{}'.format(self.packets_t))
                self.packets_benchmark[n].append(arrive_packet)
                self.packets_t_benchmark[n].append(self.t)
                # print('packets_t_benchmark{}'.format(self.packets_t_benchmark))
        
    def process_traffic(self):
        for n in range(self.N):
            tmp = int(np.sum(self.spec_eff[n], axis = -1)* self.bw * self.T) 
            tmp_init = tmp
            while tmp > 0 and len(self.packets[n]) > 0:
                if tmp >= self.packets[n][0]:
                    tmp -= self.packets[n][0]
                    self.processed_packets_t[n].append(self.t - self.packets_t[n][0])
                    del(self.packets[n][0])
                    del(self.packets_t[n][0])
                else:
                    self.packets[n][0] -= tmp
                    tmp = 0
                    
            self.throughput[n] = tmp_init - tmp
            
    def process_traffic_compare(self):
        for n in range(self.N):
            tmp = int(np.sum(self.spec_eff[n], axis = -1)* self.bw * self.T) 
            tmp_init = tmp
            
            tmp_benchmark = int(np.sum(self.spec_eff_benchmark[n], axis = -1)* self.bw * self.T) 
            tmp_init_benchmark = tmp_benchmark
            
            
            while tmp > 0 and len(self.packets[n]) > 0:
                if tmp >= self.packets[n][0]:
                    tmp -= self.packets[n][0]
                    self.processed_packets_t[n].append(self.t - self.packets_t[n][0])
                    # print('passing')
                    del(self.packets[n][0])
                    del(self.packets_t[n][0])
                else:
                    # print('no passing')
                    self.packets[n][0] -= tmp
                    tmp = 0
                    
            self.throughput[n] = tmp_init - tmp
            
            while tmp_benchmark > 0 and len(self.packets_benchmark[n]) > 0:
                if tmp_benchmark >= self.packets_benchmark[n][0]:
                    tmp_benchmark -= self.packets_benchmark[n][0]
                    # check the correctness of this calculation.
                    self.processed_packets_t_benchmark[n].append(self.t - self.packets_t_benchmark[n][0])
                    del(self.packets_benchmark[n][0])
                    del(self.packets_t_benchmark[n][0])
                else:
                    self.packets_benchmark[n][0] -= tmp_benchmark
                    tmp_benchmark = 0
                    
            self.throughput_benchmark[n] = tmp_init_benchmark - tmp_benchmark
        
    def channel_step(self):
        self.gl_map = pb.find_global_local_mapping(self.user_mapping,self.gains_cell2user,self.users_per_cell)
        self.state_cell2user = pb.get_markov_rayleigh_variable(
                                state = self.state_cell2user,
                                correlation = self.correlation,
                                N = self.N,
                                random_state = None,
                                M = self.M, 
                                K = self.K)
        self.H_cell2user.append(np.multiply(np.sqrt(self.gains_cell2user), abs(self.state_cell2user)))
        tmp_H = np.zeros((self.N,self.N,self.M))
        for n in range(self.N):
            tmp_H[n,:,:] = np.multiply(np.sqrt(self.gains[n,:,:]), abs(self.state_cell2user[n,self.cell_mapping,:]))
        self.H.append(tmp_H)
        
    def channel_step_compare(self):
        self.state_cell2user = pb.get_markov_rayleigh_variable(
                                state = self.state_cell2user,
                                correlation = self.correlation,
                                N = self.N,
                                random_state = self.channel_random_state,
                                M = self.M, 
                                K = self.K)
        self.H_cell2user.append(np.multiply(np.sqrt(self.gains_cell2user), abs(self.state_cell2user)))
        tmp_H = np.zeros((self.N,self.N,self.M))
        for n in range(self.N):
            tmp_H[n,:,:] = np.multiply(np.sqrt(self.gains[n,:,:]), abs(self.state_cell2user[n, self.cell_mapping, :]))
        self.H.append(tmp_H)
    

    def reset(self):
        # set the environment
        if self.t == -1 or self.reset_gains:
            channel_parameters = pb.generate_Cellular_CSI(N = self.N, 
                                            K = self.K, 
                                            random_state_seed = None,
                                            M = self.M, 
                                            rayleigh_var = 1.0, 
                                            shadowing_dev = 8.0,
                                            R = self.R, 
                                            min_dist = self.min_dist,
                                            equal_number_for_BS = True)

            self.gains, self.gains_cell2user, self.cell_mapping, self.user_mapping, self.TX_loc, self.RX_loc = channel_parameters
            pb.plot_deployment(self.TX_loc,self.RX_loc)
            
        self.distance_matrix_UE_AP = pb.get_UE_AP_distance(self.TX_loc, self.RX_loc)  
        self.large_f_matrix = - (128.1 + 37.6 * np.log10(0.001 * self.distance_matrix_UE_AP))
                
        self.distance_matrix_AP = pb.get_AP_distance(self.TX_loc)  
        self.AP_neighbors = pb.get_AP_neighbor(self.distance_matrix_AP, self.Num_neighbors)         
        # associate each UE with single 1 AP
            # Compute rayleigh fading for time slot = 0
        self.state_cell2user = pb.get_random_rayleigh_variable(N = self.N, 
                                                                    random_state = self.channel_random_state,
                                                                    M = self.M, 
                                                                    K = self.K, 
                                                                    rayleigh_var = 1.0)
        self.H_cell2user.append(np.multiply(np.sqrt(self.gains_cell2user), abs(self.state_cell2user)))
        tmp_H = np.zeros((self.N,self.N,self.M))
        for n in range(self.N):
            tmp_H[n,:,:] = np.multiply(np.sqrt(self.gains[n,:,:]), abs(self.state_cell2user[n,self.cell_mapping,:]))
        self.H.append(tmp_H)
        self.t = 0
        self.SINR, self.spec_eff, self.total_interf = pb.sumrate_multi_list_clipped(self.H[-1], self.p, self.noise_var)
        self.create_traffic()
        
        self.channel_step() # need last 2 time slot channel state
        self.load_traffic()

        
        return
    


    def reset_compare(self):
        # set the environment
        if self.t == -1 or self.reset_gains:
            channel_parameters = pb.generate_Cellular_CSI(N = self.N, 
                                            K = self.K, 
                                            random_state_seed = None,
                                            M = self.M, 
                                            rayleigh_var = 1.0, 
                                            shadowing_dev = 8.0,
                                            R = self.R, 
                                            min_dist = self.min_dist,
                                            equal_number_for_BS = True)

            self.gains, self.gains_cell2user, self.cell_mapping, self.user_mapping, self.TX_loc, self.RX_loc = channel_parameters
            pb.plot_deployment(self.TX_loc,self.RX_loc)
            
        self.distance_matrix_UE_AP = pb.get_UE_AP_distance(self.TX_loc, self.RX_loc)  
        self.large_f_matrix = - (128.1 + 37.6 * np.log10(0.001 * self.distance_matrix_UE_AP))
                
        self.distance_matrix_AP = pb.get_AP_distance(self.TX_loc)  
        self.AP_neighbors = pb.get_AP_neighbor(self.distance_matrix_AP, self.Num_neighbors)         
        # associate each UE with single 1 AP
            # Compute rayleigh fading for time slot = 0
        self.state_cell2user = pb.get_random_rayleigh_variable(N = self.N, 
                                                                    random_state = self.channel_random_state,
                                                                    M = self.M, 
                                                                    K = self.K, 
                                                                    rayleigh_var = 1.0)
        self.H_cell2user.append(np.multiply(np.sqrt(self.gains_cell2user), abs(self.state_cell2user)))
        tmp_H = np.zeros((self.N,self.N,self.M))
        for n in range(self.N):
            tmp_H[n,:,:] = np.multiply(np.sqrt(self.gains[n,:,:]), abs(self.state_cell2user[n,self.cell_mapping,:]))
        self.H.append(tmp_H)
        self.t = 0
        self.SINR, self.spec_eff, self.total_interf = pb.sumrate_multi_list_clipped(self.H[-1], self.p, self.noise_var)
        
        self.create_traffic_compare()
        
        self.channel_step() # need last 2 time slot channel state
        self.load_traffic_compare()

        
        return
        
        
    def step(self, p_action, user_action):
        assert p_action.shape == (self.K, self.M), "action shape should be (K,M)"
        assert user_action.shape == (self.K, self.M) # local index
        self.t += 1
        self.p = np.zeros((self.N,self.M))
        self.p_action = p_action
        for k in range(self.K):
            for m in range(self.M):
                dic = self.gl_map[k][0]
                if user_action[k][m] != 999:
                    selected_user_index = dic[user_action[k][m]]#find corresponding global index
                    if selected_user_index != 'Idle':
                        self.p[selected_user_index][m] = self.p_action[k][m]
        self.SINR, self.spec_eff, self.total_interf = pb.sumrate_multi_list_clipped(self.H[-1], self.p, self.noise_var)
        # print(self.p)
        self.process_traffic()            
        self.channel_step()
        
        self.load_traffic()
        #print(self.scale_gain)
        
        return 
    
    def step_pfs(self, p_action):
        assert p_action.shape == (self.N, self.M), "action shape should be (K,M)"
        self.t += 1
        self.p = p_action
        self.SINR, self.spec_eff, self.total_interf = pb.sumrate_multi_list_clipped(self.H[-1], self.p, self.noise_var)
        # print(self.p)
        self.process_traffic()
        self.average_spec_eff = (1.0-self.beta)*self.average_spec_eff+self.beta*np.sum(self.spec_eff, axis = 1)
        self.weights = 1.0 / self.average_spec_eff            
        self.channel_step()
        
        self.load_traffic()
        #print(self.scale_gain)
        
        return 
    
    def step_wmmse(self, p_action):
        assert p_action.shape == (self.N, self.M), "action shape should be (K,M)"
        self.t += 1
        self.p = p_action
        self.SINR, self.spec_eff, self.total_interf = pb.sumrate_multi_list_clipped(self.H[-1], self.p, self.noise_var)
        # print(self.p)
        for n in range(self.N):
            if len(self.packets[n]) == 0:
                self.weights[n] = 0
        self.load_traffic()
        self.process_traffic()            
        self.channel_step()

        #print(self.scale_gain)
        
        return 
    
    def step_compare(self, p_action, user_action, p_action_benchmark, user_action_benchmark):
        assert p_action.shape == (self.K, self.M), "action shape should be (K,M)"
        assert user_action.shape == (self.K, self.M) 
        assert p_action_benchmark.shape == (self.K, self.M), "action shape should be (K,M)"
        assert user_action_benchmark.shape == (self.K, self.M) # local index
        self.t += 1
        self.p = np.zeros((self.N,self.M))
        self.p_action = p_action
        
        for k in range(self.K):
            for m in range(self.M):
                dic = self.gl_map[k][0]
                if user_action[k][m] != 999:
                    selected_user_index = dic[user_action[k][m]]#find corresponding global index
                    if selected_user_index != 'Idle':
                        self.p[selected_user_index][m] = self.p_action[k][m]
                        
                   
        self.SINR, self.spec_eff, self.total_interf = pb.sumrate_multi_list_clipped(self.H[-1], self.p, self.noise_var)
        # print(self.p)    
        # print(self.spec_eff) 
        
        self.p_benchmark = np.zeros((self.N,self.M))
        self.p_action_benchmark = p_action_benchmark
        for k in range(self.K):
            for m in range(self.M):
                if user_action_benchmark[k][m] != 999:
                    selected_user_index = user_action_benchmark[k][m]#find corresponding global index
                    self.p_benchmark[selected_user_index][m] = self.p_action_benchmark[k][m]
        self.SINR_benchmark, self.spec_eff_benchmark, self.total_interf_benchmark = pb.sumrate_multi_list_clipped(self.H[-1], self.p_benchmark, self.noise_var)

        self.process_traffic_compare()            
        self.channel_step_compare()
        
        self.load_traffic_compare()
        
        return 
    


        
    def get_state(self):
        self.state_dim_user = 4 + self.Num_neighbors  
        self.state_dim_AP = self.state_dim_user * self.users_per_cell
        state_AP = np.zeros((self.K,self.M,self.state_dim_AP))
        
        for k in range(self.K):
            for m in range(self.M):
                
                users = self.gl_map[k][0]
                state_user = np.zeros((self.users_per_cell,self.M,self.state_dim_user))
                vector = []
            
                for local_index,global_index in users.items():
                    # key is the local_index, value is the global index
                    if global_index != 'Idle':
                        cursor = 0
                        #queue length
                        state_user[local_index,m,cursor] = np.log2( 1 + sum(self.packets[global_index])/self.packet_size + np.sign(sum(self.packets[global_index])))
                        cursor += 1
                        #spectral efficiency
                        state_user[local_index,m,cursor] = self.spec_eff[global_index] 
                        cursor += 1
                        #total interference
                        state_user[local_index,m,cursor] = self.total_interf[global_index] / self.scale_gain
                        cursor += 1
                        # downlink_channel gain from associate AP

                        d_c_gain = self.H_cell2user[-1][global_index,k,m] ** 2
                        #print(d_c_gain)
 #                       d_c_gain = np.clip(d_c_gain, a_min = 1e-13, a_max = 1e-10)
#                        state_user[local_index,m,cursor]  =  np.log10(d_c_gain/self.scale_gain)
                        d_c_gain  =  np.log10(d_c_gain/self.scale_gain)
                        state_user[local_index,m,cursor] = np.clip(d_c_gain, a_min = -1.3, a_max = 1.69)
                        
                        # find potential neighbor AP
                        neighbors = self.AP_neighbors[k]
                        if len(neighbors) != 0:
                            for i in range(len(neighbors)):
                                cursor += 1
                                nei = neighbors[i]
                                d_c_gain = self.H_cell2user[-1][global_index,nei,m] ** 2
                                d_c_gain  =  np.log10(d_c_gain/self.scale_gain)
                                state_user[local_index,m,cursor] = np.clip(d_c_gain, a_min = -1.3, a_max = 1.69)




                        # #agrregated interference
                        # state_user[local_index,m,cursor]  =  np.log10((self.H[-1][global_index,global_index,m] ** 2)/self.scale_gain)
                        # cursor += 1
                    vector = np.concatenate((vector,state_user[local_index,m,:]), axis = None)
                state_AP[k,m,:] = vector
        return state_AP
                    
                    
    def get_reward(self):
        reward = np.zeros((self.K,self.M))
        for k in range(self.K):
            for m in range(self.M):
                this_reward = 0
                nei = []
                for n in self.user_mapping[k]:
                    this_reward -= sum(self.packets[n]) / 8e6
                    for neigh in self.AP_neighbors[k]:
                        for ue in self.user_mapping[neigh]:
                            nei.append(ue)
                for ue in nei:
                    this_reward -= sum(self.packets[ue]) / 8e6
                # energy_cost_p =  self.price * self.p_action[k]
                reward[k,m] = this_reward
                # reward[k,m] = this_reward - energy_cost_p
        return reward / 25
    


    
    def get_reward_compare(self):
        reward = np.zeros((self.K,self.M))
        for k in range(self.K):
            for m in range(self.M):
                this_reward = 0
                nei = []
                for n in self.user_mapping[k]:
                    this_reward -= sum(self.packets_benchmark[n]) / 8e6
                    for neigh in self.AP_neighbors[k]:
                        for ue in self.user_mapping[neigh]:
                            nei.append(ue)
                for ue in nei:
                    this_reward -= sum(self.packets_benchmark[ue]) / 8e6
                reward[k,m] = this_reward
        return reward / 25
    
    
    def mobility_update(self,random_move = None):
        #add mobility for user(Tx)
        if random_move is None:
            for i in range(len(self.RX_loc[0])):
                angle = np.random.randint(360)
                step = np.random.uniform(0.5 * self.step_size, 1.5 * self.step_size)
                rad= np.deg2rad(angle)
        
                x = step * math.cos(rad)
                y = step * math.sin(rad)
    
                self.RX_loc[0][i] = np.clip(self.RX_loc[0][i] + x, self.TX_loc[0][self.cell_mapping[i]] - self.R ,self.TX_loc[0][self.cell_mapping[i]] + self.R)
                self.RX_loc[1][i] = np.clip(self.RX_loc[1][i] + y, self.TX_loc[1][self.cell_mapping[i]] - self.R ,self.TX_loc[1][self.cell_mapping[i]] + self.R)
        else:
            for i in range(len(self.TX_loc[0])):
                angle = random_move.randint(360)
                step = random_move.uniform(0.5 * self.step_size, 1.5 * self.step_size)
                rad= np.deg2rad(angle)
        
                x = step * math.cos(rad)
                y = step * math.sin(rad)
    
                self.RX_loc[0][i] = np.clip(self.RX_loc[0][i] + x, self.TX_loc[0][self.cell_mapping[i]] - self.R ,self.TX_loc[0][self.cell_mapping[i]] + self.R)
                self.RX_loc[1][i] = np.clip(self.RX_loc[1][i] + y, self.TX_loc[1][self.cell_mapping[i]] - self.R ,self.TX_loc[1][self.cell_mapping[i]] + self.R)
            
            
            
        self.distance_matrix_UE_AP = pb.get_UE_AP_distance(self.TX_loc, self.RX_loc)  
        self.large_f_matrix = - (128.1 + 37.6 * np.log10(0.001 * self.distance_matrix_UE_AP))
        
        # update downlink channel gain
        # repeat large scale fading to M subbands (g stands for path loss, fading = g * h^2, where h is the small scale fading)
        g_dB_cell2user = np.repeat(np.expand_dims(self.large_f_matrix, axis = 2), self.M , axis = -1)

        g_dB_user2user = np.zeros((self.N,self.N,self.M))
        for n in range(self.N):
            g_dB_user2user[n, :, :] = g_dB_cell2user[n, self.cell_mapping, :]
            
        self.g_user2user = np.power(10, g_dB_user2user / 10)

        self.g_cell2user = np.power(10, g_dB_cell2user / 10)
        
        self.H_cell2user.append(np.multiply(np.sqrt(self.g_cell2user), abs(self.state_cell2user)))
        tmp_H = np.zeros((self.N,self.N,self.M))
        for n in range(self.N):
            tmp_H[n,:,:] = np.multiply(np.sqrt(self.g_user2user[n,:,:]), abs(self.state_cell2user[n, self.cell_mapping, :]))
        self.H.append(tmp_H)
        
        return
        
        
    
    
    
    # def get_reward_include_p(self):
    #     reward = np.zeros((self.K,self.M))
    #     for k in range(self.K):
    #         for m in range(self.M):
    #             this_reward = 0
    #             nei = []
    #             for n in self.user_mapping[k]:
    #                 this_reward -= sum(self.packets[n]) / 8e6
    #                 # print(self.p)
    #                 # print(sum(self.packets[n]))
    #                 #include energy consuming
    #                 this_reward -= self.p[n] * self.price
    #                 for neigh in self.AP_neighbors[k]:
    #                     for ue in self.user_mapping[neigh]:
    #                         nei.append(ue)
    #             for ue in nei:
    #                 this_reward -= sum(self.packets[ue]) / 8e6
    #             reward[k,m] = this_reward
    #     return reward / 25
        
        
        
        
        
        
        
        
                     