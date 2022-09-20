# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 21:46:17 2022

@author: yzhang4
"""
import numpy as np
import project_backend as pb
import tensorflow as tf
import collections
import copy
import itertools


class DQN:
    def __init__(self, args,options_policy,env):
        tf.reset_default_graph()        
        self.timeslots = args.timeslots
        self.train_episodes = args.episode_timeslots
        self.seed = args.seed
        self.N = env.N
        self.K = env.K
        # self.M = env.M
        self.Pmax = env.Pmax
        self.noise_var = env.noise_var
        self.args = args
        self.env = env
        
        # power actions
        sing_ac1 = options_policy['power_actions'] + 1
        powers = np.zeros(sing_ac1)
        powers[0] = 0.0 # Tx power 0
        Pmin_dB = 0.0-30
        # Calculate steps in dBm
        if sing_ac1 > 2:
            strategy_translation_dB_step = (env.Pmax_dB-Pmin_dB)/(sing_ac1-2)
        for i in range(1,sing_ac1-1):
            powers[i] = np.power(10.0,((Pmin_dB+(i-1)*strategy_translation_dB_step))/10)
        powers[-1] = env.Pmax
        powers = powers[1:]
        # user selection action
        sing_ac2 = env.users_per_cell
        users = [i for i in range(sing_ac2)]
        
        def myfunc(list1, list2):
            return [np.append(i,j) for i in list1 for j in list2]
        
        strategies = myfunc(users,powers)
        # add 1 idle term, i.e. no user selection and zero power
        strategies.insert(0, 'off')
    
        self.strategy_translation = strategies
        
        self.num_output = self.num_actions = len(self.strategy_translation) #     number of actions
        self.discount_factor = options_policy['discount_factor']
        
        self.AP_neighbors = env.Num_neighbors
        # each in the server pool reports (4,) including downlink channel gain, aggregated interference,
        #power weight, and queue length, if not enough use zero-padding
        #4 + self.Num_neighbors  * 2
        self.num_input = (4 + env.Num_neighbors ) * sing_ac2
        
        learning_rate_0 = options_policy['learning_rate_0']
        learning_rate_decay = options_policy['learning_rate_decay']
        learning_rate_min = options_policy['learning_rate_min']
        self.learning_rate_all = [learning_rate_0]
        for i in range(1,self.timeslots):
            # if i % self.train_episodes == 0:
            #     self.learning_rate_all.append(learning_rate_0)
            # else:
            self.learning_rate_all.append(max(learning_rate_min,learning_rate_decay*self.learning_rate_all[-1]))
    
        # learning_rate_0 = options_policy['learning_rate_0_s']
        # learning_rate_decay = options_policy['learning_rate_decay_s']
        # learning_rate_min = options_policy['learning_rate_min_s']
        # self.learning_rate_all_s = [learning_rate_0]
        # for i in range(1,self.timeslots):
        #     # if i % self.train_episodes == 0:
        #     #     self.learning_rate_all.append(learning_rate_0)
        #     # else:
        #     self.learning_rate_all_s.append(max(learning_rate_min,learning_rate_decay*self.learning_rate_all[-1]))    


        # epsilon greedy algorithm
        max_epsilon = options_policy['max_epsilon']
        epsilon_decay = options_policy['epsilon_decay']
        min_epsilon = options_policy['min_epsilon']
        # epsilon greedy algorithm       
        self.epsilon_all=[max_epsilon]
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        
        
        # quasi-static target network update
        self.target_update_count = options_policy['target_update_count']
        self.time_slot_to_pass_weights = options_policy['time_slot_to_pass_weights'] # 50 slots needed to pass the weights
        
        
        n_hidden_1 = options_policy['n_hiddens'][0]
        n_hidden_2 = options_policy['n_hiddens'][1]
        n_hidden_3 = options_policy['n_hiddens'][2]
        
        
        self.batch_size = options_policy['batch_size']
        memory_per_agent = options_policy['memory_per_agent']
        self.memory_len = memory_per_agent * env.K
        
        # Experience replay memory
        self.memory = {}
        self.memory['s'] = collections.deque([],self.memory_len + self.K)
        self.memory['s_prime'] = collections.deque([],self.memory_len)
        self.memory['rewards'] = collections.deque([],self.memory_len)
        self.memory['actions'] = collections.deque([],self.memory_len)
        
        # self.memory_s = {}
        # self.memory_s['s'] = collections.deque([],self.memory_len+self.K)
        # self.memory_s['s_prime'] = collections.deque([],self.memory_len)
        # self.memory_s['rewards'] = collections.deque([],self.memory_len)
        # self.memory_s['actions'] = collections.deque([],self.memory_len)
        
        self.previous_state = np.zeros((self.K,self.num_input))
        self.previous_action = np.ones(self.K) * self.num_actions
        
        # required for session to know whether dictionary is train or test
        self.is_train = tf.placeholder("bool")  
        
        self.x_policy = tf.placeholder("float", [None, self.num_input])
        self.y_policy = tf.placeholder("float", [None, 1])
        with tf.name_scope("Fweights"):
            self.weights_policy = pb.initial_weights (self.num_input, n_hidden_1,
                                               n_hidden_2, n_hidden_3, self.num_output,self.seed)
        with tf.name_scope("Ftarget_weights"): 
            self.weights_target_policy = pb.initial_weights (self.num_input, n_hidden_1,
                                               n_hidden_2, n_hidden_3, self.num_output,self.seed)
        with tf.name_scope("Ftmp_weights"): 
            self.weights_tmp_policy = pb.initial_weights (self.num_input, n_hidden_1,
                                               n_hidden_2, n_hidden_3, self.num_output,self.seed)
        with tf.name_scope("Fbiases"):
            self.biases_policy = pb.initial_biases (n_hidden_1, n_hidden_2, n_hidden_3,
                                          self.num_output,self.seed)
        with tf.name_scope("Ftarget_biases"): 
            self.biases_target_policy = pb.initial_biases (n_hidden_1, n_hidden_2, n_hidden_3,
                                          self.num_output,self.seed)
        with tf.name_scope("Ftmp_biases"): 
            self.biases_tmp_policy = pb.initial_biases (n_hidden_1, n_hidden_2, n_hidden_3,
                                          self.num_output,self.seed)
        # initialize the neural network for each agent
        self.QNN= pb.neural_net(self.x_policy, self.weights_policy, self.biases_policy)
        self.QNN_target = pb.neural_net(self.x_policy, self.weights_target_policy,
                                            self.biases_target_policy)
        self.actionslatten = tf.placeholder(tf.int32, self.batch_size)
        self.actions_one_hot = tf.one_hot(self.actionslatten, self.num_actions, 1.0, 0.0)
        self.single_q = tf.reshape(tf.reduce_sum(tf.multiply(self.QNN, self.actions_one_hot), reduction_indices=1),(self.batch_size,1))
        # loss function is simply least squares cost
        self.loss = tf.reduce_sum(tf.square(self.y_policy - self.single_q))
        self.learning_rate = (tf.placeholder('float'))
        # RMSprop algorithm used
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.9,
                                              epsilon=1e-10).minimize(self.loss)
        
        
        
        # self.x_policy_s = tf.placeholder("float", [None, self.num_input_s])
        # self.y_policy_s = tf.placeholder("float", [None, 1])
        # with tf.name_scope("weights"):
        #     self.weights_policy_s = pb.initial_weights (self.num_input_s, n_hidden_1,
        #                                        n_hidden_2, n_hidden_3, self.num_output_s)
        # with tf.name_scope("target_weights"): 
        #     self.weights_target_policy_s = pb.initial_weights (self.num_input_s, n_hidden_1,
        #                                        n_hidden_2, n_hidden_3, self.num_output_s)
        # with tf.name_scope("tmp_weights"): 
        #     self.weights_tmp_policy_s = pb.initial_weights (self.num_input_s, n_hidden_1,
        #                                        n_hidden_2, n_hidden_3, self.num_output_s)
        # with tf.name_scope("biases"):
        #     self.biases_policy_s = pb.initial_biases (n_hidden_1, n_hidden_2, n_hidden_3,
        #                                   self.num_output_s)
        # with tf.name_scope("target_biases"): 
        #     self.biases_target_policy_s = pb.initial_biases (n_hidden_1, n_hidden_2, n_hidden_3,
        #                                   self.num_output_s)
        # with tf.name_scope("tmp_biases"): 
        #     self.biases_tmp_policy_s = pb.initial_biases (n_hidden_1, n_hidden_2, n_hidden_3,
        #                                   self.num_output_s)
        # # initialize the neural network for each agent
        # self.QNN_s= pb.neural_net(self.x_policy_s, self.weights_policy_s, self.biases_policy_s)
        # self.QNN_target_s = pb.neural_net(self.x_policy_s, self.weights_target_policy_s,
        #                                     self.biases_target_policy_s)
        # self.actionslatten_s = tf.placeholder(tf.int32, self.batch_size)
        # self.actions_one_hot_s = tf.one_hot(self.actionslatten_s, self.num_actions, 1.0, 0.0)
        # self.single_q_s = tf.reshape(tf.reduce_sum(tf.multiply(self.QNN_s, self.actions_one_hot_s), reduction_indices=1),(self.batch_size,1))
        # # loss function is simply least squares cost
        # self.loss_s = tf.reduce_sum(tf.square(self.y_policy_s - self.single_q_s))
        # self.learning_rate_s = (tf.placeholder('float'))
        # # RMSprop algorithm used
        # self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate_s, decay=0.9,
        #                                       epsilon=1e-10).minimize(self.loss_s)
        
        
        self.init = tf.global_variables_initializer()
        # quasi-static target update simulation counter = 0
        self.saver = tf.train.Saver()
        
        
        
    def check_memory_restart(self,sess,sim):   
        if(sim %self.train_episodes == 0 and sim != 0): # Restart experience replay.
            self.memory = {}
            self.memory['s'] = collections.deque([],self.memory_len + self.K)
            self.memory['s_prime'] = collections.deque([],self.memory_len + self.K)
            self.memory['rewards'] = collections.deque([],self.memory_len + self.K)
            self.memory['actions'] = collections.deque([],self.memory_len + self.K)
            
            self.previous_state = np.zeros((self.K,self.num_input))
            self.previous_action = np.ones(self.K) * self.num_actions   
            
            ## fix lack of experience of slow
            
    def update_handler(self,sess,sim):
        # Quasi-static target Algorithm
        # First check whether target network has to be changed.
        self.simulation_target_update_counter -= 1
        if (self.simulation_target_update_counter == 0):
            for update_instance in self.update_class1:
                sess.run(update_instance)
            self.simulation_target_update_counter = self.target_update_count
            self.process_weight_update = True

        if self.process_weight_update:
            self.simulation_target_pass_counter -= 1
        
        if (self.simulation_target_pass_counter <= 0):
            for update_instance in self.update_class2:
                sess.run(update_instance)
            self.process_weight_update = False
            self.simulation_target_pass_counter = self.time_slot_to_pass_weights       
            
    def update_epsilon(self,reset):
        if reset:
            self.epsilon_all.append(self.max_epsilon)
        else:
            self.epsilon_all.append(max(self.min_epsilon,self.epsilon_decay*self.epsilon_all[-1]))
        
    def initialize_updates(self,sess): # Keed to rund this before calling quasi static.
        self.saver = tf.train.Saver(tf.global_variables())
        self.update_class1 = []
        for (w,tmp_w) in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Fweights'),
             tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Ftmp_weights')):
            self.update_class1.append(tf.assign(tmp_w,w))
            sess.run(self.update_class1[-1])
        for (b,tmp_b) in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Fbiases'),
             tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Ftmp_biases')):
            self.update_class1.append(tf.assign(tmp_b,b))
            sess.run(self.update_class1[-1])
        self.update_class2 = []
        for (tmp_w,t_w) in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Ftmp_weights'),
             tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Ftarget_weights')):
            self.update_class2.append(tf.assign(t_w,tmp_w))
            sess.run(self.update_class2[-1])
        for (tmp_b,t_b) in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Ftmp_biases'),
             tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Ftarget_biases')):
            self.update_class2.append(tf.assign(t_b,tmp_b))
            sess.run(self.update_class2[-1])
        self.simulation_target_update_counter = self.target_update_count
        self.process_weight_update = False
        self.simulation_target_pass_counter = self.time_slot_to_pass_weights
        print('first update')
        
        
    def act(self,sess,current_local_state,sim,agent):
        # Current QNN outputs for all available actions
        current_QNN_outputs = sess.run(self.QNN_target, feed_dict={self.x_policy: current_local_state.reshape(1,self.num_input), self.is_train: False})
        # print('sim{} epsilon{}'.format(self.env.t, self.epsilon_all[-1]))
        # epsilon greedy algorithm
        if np.random.rand() < self.epsilon_all[-1]:
            strategy = np.random.randint(self.num_actions)
        else:
            strategy = np.argmax(current_QNN_outputs)
        return strategy
    
    def act_noepsilon(self,sess,current_local_state,sim,agent):
        # Current QNN outputs for all available actions
        current_QNN_outputs = sess.run(self.QNN_target, feed_dict={self.x_policy: current_local_state.reshape(1,self.num_input), self.is_train: False})
        return np.argmax(current_QNN_outputs)
    
    def remember(self,agent,current_local_state,current_reward):
        self.memory['s'].append(copy.copy(self.previous_state[agent,:]).reshape(self.num_input))
        self.memory['actions'].append(copy.copy(self.previous_action[agent]))
        self.memory['rewards'].append(copy.copy(current_reward))
        self.memory['s_prime'].append(copy.copy(current_local_state))

    def train(self,sess,sim):
        # mod 
        if len(self.memory['s']) >= self.batch_size+self.K:
            # Minus K ensures that experience samples from previous timeslots been used, not this time slot.
            #print(len(self.memory['rewards']))
            idx = np.random.randint(len(self.memory['rewards'])-self.K,size=self.batch_size)
            c_QNN_outputs = sess.run(self.QNN_target, feed_dict={self.x_policy: np.array(self.memory['s_prime'])[idx, :].reshape(self.batch_size,self.num_input),
                                                                 self.is_train: False})
            opt_y = np.array(self.memory['rewards'])[idx].reshape(self.batch_size) + self.discount_factor * np.max(c_QNN_outputs,axis=1)
            actions = np.array(self.memory['actions'])[idx]
            (tmp,tmp_mse) = sess.run([self.optimizer, self.loss], feed_dict={self.learning_rate:self.learning_rate_all[sim],self.actionslatten:actions,
                                self.x_policy: np.array(self.memory['s'])[idx, :],
                                self.y_policy: opt_y.reshape(self.batch_size,1), self.is_train: True})
            
            self.update_epsilon(reset = False)

    def equalize(self,sess):
        for update_instance in self.update_class1:
            sess.run(update_instance)
        for update_instance in self.update_class2:
            sess.run(update_instance)
        
    def save(self,sess,model_destination):
        self.saver = tf.train.Saver(tf.global_variables())
        save_path = self.saver.save(sess, model_destination)
        print("Model saved in path: %s" % save_path)
        
    def load(self,sess,model_destination):
        self.saver = tf.train.Saver(tf.global_variables())
        self.saver.restore(sess, model_destination)
        print('Model loaded from: %s' %(model_destination))
        
        
        
        
        
        
        

    