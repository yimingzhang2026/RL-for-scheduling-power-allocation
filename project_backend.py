# -*- coding: utf-8 -*-
"""
@author: anonymous
"""
import numpy as np
import time
import tensorflow as tf
import matplotlib.pyplot as plt
global treshold_sinr
from numpy import random
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.patches import Polygon
import math
import heapq
import copy
from itertools import cycle


max_SINR = 10*np.log10(1000)
max_spectual_eff = np.log2(1.0+1000)
AVOID_DIV_BY_ZERO = 1e-15



def neural_net(x, weights, biases): # Create model
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])   # x = w*x+b
    layer_1 = tf.nn.relu(layer_1)                                 # x = max(0, x)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)

    # Kumber of neurons at the layer3 is basically number of possible actions.
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
#    out_layer = tf.nn.relu(out_layer)
#    out_layer = tf.nn.tanh(out_layer)
    return out_layer


def initial_weights (num_input, n_hidden_1, n_hidden_2, n_hidden_3, num_output, seed = None):

    weights = {
        'h1': tf.Variable(tf.random_uniform([num_input, n_hidden_1], -1./np.sqrt(n_hidden_1),1./np.sqrt(n_hidden_1),seed=seed+1000)),
        'h2': tf.Variable(tf.random_uniform([n_hidden_1, n_hidden_2], -1./ np.sqrt(n_hidden_2), 1./ np.sqrt(n_hidden_2),seed=seed+2000)),
        'h3': tf.Variable(tf.random_uniform([n_hidden_2, n_hidden_3], -1/ np.sqrt(n_hidden_3), 1./ np.sqrt(n_hidden_3),seed=seed+3000)),
        'out': tf.Variable(tf.random_uniform([n_hidden_3, num_output], -0.003, 0.003,seed=seed+4000)),
    }
    return weights


def update_weights (source_weights, destination_weights):
    destination_weights['h1']=tf.identity(source_weights['h1'])
    destination_weights['h2']=tf.identity(source_weights['h2'])
    destination_weights['h3']=tf.identity(source_weights['h3'])
    destination_weights['out']=tf.identity(source_weights['out'])
    #return 0

def initial_biases (n_hidden_1, n_hidden_2, n_hidden_3, num_output, seed = None):
    biases = {
        'b1': tf.Variable(tf.random_uniform([n_hidden_1], -1./np.sqrt(n_hidden_1),1./np.sqrt(n_hidden_1),seed=seed+5000)),
        'b2': tf.Variable(tf.random_uniform([n_hidden_2], -1./ np.sqrt(n_hidden_2), 1./ np.sqrt(n_hidden_2),seed=seed+6000)),
        'b3': tf.Variable(tf.random_uniform([n_hidden_3], -1/ np.sqrt(n_hidden_3), 1./ np.sqrt(n_hidden_3),seed=seed+7000)),
        'out': tf.Variable(tf.random_uniform([num_output], -0.003, 0.003,seed=seed+8000)),
    }
    return biases


def update_biases (source_biases, destination_biases):
    destination_biases['b1']=tf.identity(source_biases['b1'])
    destination_biases['b2']=tf.identity(source_biases['b2'])
    destination_biases['b3']=tf.identity(source_biases['b3'])
    destination_biases['out']=tf.identity(source_biases['out'])
    #return None



# In[*] 


def voronoi_finite_polygons_2d(vor, radius=None):

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def random_color(as_str=True, alpha=0.5):
    rgb = [random.randint(0,255),
           random.randint(0,255),
           random.randint(0,255)]
    if as_str:
        return "rgba"+str(tuple(rgb+[alpha]))
    else:
        # Normalize & listify
        return list(np.array(rgb)/255) + [alpha]
    


def get_AP_distance(TX_loc):
    K = len(TX_loc[0])
    distance_vector = np.zeros((K,K))
    for i in range(K):
        for j in range(K):
            distance_vector[i,j] = np.sqrt(np.square(TX_loc[0,i]-TX_loc [0, j])+np.square(TX_loc[1,i]-TX_loc [1, j]))
    return distance_vector 
    
def get_AP_neighbor(distance_matrix_AP,NumNeighbor): 
    num = len(distance_matrix_AP[0])
    AP_neighbors = [] 
    for i in range(num):
        distance_vec = list(distance_matrix_AP[i])
        closest = heapq.nsmallest(NumNeighbor + 1, distance_vec)
        subindex = []
        for d in closest:
            subindex.append(distance_vec.index(d))
            distance_vec[distance_vec.index(d)] = 0
        subindex.remove(i)
        AP_neighbors.append(subindex)
        
    return AP_neighbors

#calculate distance vector
def get_UE_AP_distance(TX_loc, RX_loc):
    N = len(RX_loc[0])
    K = len(TX_loc[0])
    distance_vector = np.zeros((N,K))
    for ap in range(K):
        for ue in range (N):
            distance_vector[ue,ap] = np.sqrt(np.square(RX_loc[0,ue]-TX_loc [0, ap])+np.square(RX_loc[1,ue]-TX_loc [1, ap]))
    return distance_vector


    


def find_global_local_mapping(valid_user_association, large_f_matrix, max_serve_pool):
    K = len(large_f_matrix[0])
    gl_map = [[] for i in range(K)]
    for i in range(K):
        local_index = [i for i in range(max_serve_pool)]
        large_f_vec = [large_f_matrix[j][i] for j in valid_user_association[i]]
        dic_large_f = dict(zip(valid_user_association[i],large_f_vec))
        sorted_dic_large_f = sorted(dic_large_f.items(), key = lambda item: item[1] , reverse= True)
        sorted_UE = [key for key,value in sorted_dic_large_f]
        while(len(sorted_UE) < max_serve_pool):
            sorted_UE.append('Idle')
        global_index = sorted_UE
        AP_map = dict(zip(local_index, global_index))
        gl_map[i].append(AP_map)
        
    return gl_map


# In[*] 
# given a random_state, generate the same channel condition
def get_random_rayleigh_variable(N,
                                 K,
                                 random_state = None,
                                 M=1, 
                                 rayleigh_var=1.0):
    if random_state is None:
        return np.sqrt(2.0/np.pi) * (rayleigh_var * np.random.randn(N, K, M) +
                                                1j * rayleigh_var * random_state.randn(N, N, M))
    else:
        return np.sqrt(2.0/np.pi) * (rayleigh_var * random_state.randn(N, K, M) +
                                                1j * rayleigh_var * random_state.randn(N, K, M))
    
def get_markov_rayleigh_variable(state,
                                 correlation,
                                 N,
                                 K,
                                 random_state = None,
                                 M=1, 
                                 rayleigh_var=1.0):
    if random_state is None:
        return correlation*state +np.sqrt(1-np.square(correlation)) * np.sqrt(2.0/np.pi) * (rayleigh_var * np.random.randn(N, K, M) +
                                                1j * rayleigh_var * np.random.randn(N, K, M))
    else:
        return correlation*state +np.sqrt(1-np.square(correlation)) * np.sqrt(2.0/np.pi) * (rayleigh_var * random_state.randn(N, K, M) +
                                                1j * rayleigh_var * random_state.randn(N, K, M))
    
# Calculate sum_rate with given channel and power allocation
def sumrate_multi_list_clipped(H,p,noise_var):
    # Take pow 2 of abs_H, no need to take it again and again
    H_2 = H ** 2
    N = H.shape[1] # number of links
    M = H.shape[2] # number of channels
    
    sum_rate1 = np.zeros((N, M)) #calculate SINR
    sum_rate2 = np.zeros((N, M)) #calculate spec_eff
    total_interf = np.zeros((N, M))
    for out_loop in range(M):
        for loop in range (N):
            tmp_1 = H_2[loop, loop, out_loop] * p[loop, out_loop]
            tmp_2 = np.matmul(H_2[loop, :, out_loop], p[:, out_loop]) + noise_var - tmp_1
            total_interf[loop,out_loop] = tmp_2
            if tmp_1 == 0:
                sum_rate1[loop,out_loop] = 0.0
                sum_rate2[loop,out_loop] = 0.0
            else:
                sum_rate1[loop,out_loop] = 10*np.log10(tmp_1/tmp_2)
                sum_rate2[loop,out_loop] = np.log2(1.0+tmp_1/tmp_2)
    sum_rate1 = np.clip(sum_rate1, a_min = None, a_max = max_SINR)
    sum_rate2 = np.clip(sum_rate2, a_min = None, a_max = max_spectual_eff)
    return sum_rate1, sum_rate2, total_interf

def generate_Cellular_CSI(N, 
                          K,
                          random_state_seed = None,
                          M = 1, 
                          rayleigh_var = 1.0, 
                          shadowing_dev = 8.0,
                          R = 200, 
                          min_dist = 35,
                          equal_number_for_BS = True):

    assert not equal_number_for_BS or N % K == 0, 'N needs to be divisible by UE_perBS!'

            
    # IMAC Case: we have the mirror BS at the same location.
    max_dist = R
    x_hexagon=R*np.array([0, -np.sqrt(3)/2, -np.sqrt(3)/2, 0, np.sqrt(3)/2, np.sqrt(3)/2, 0])
    y_hexagon=R*np.array([-1, -0.5, 0.5, 1, 0.5, -0.5, -1])

    TX_loc = np.zeros((2,K))
    TX_xhex = np.zeros((7,K))
    TX_yhex = np.zeros((7,K))
    
    RX_loc = np.zeros((2,N))
    cell_mapping = np.zeros(N).astype(int)
    user_mapping = [[] for idx in range(K)]
    
    ############### DROP ALL txers    
    generated_hexagons = 0
    i = 0

    TX_loc [0, generated_hexagons] = 0.0
    TX_loc [1, generated_hexagons] = 0.0
    TX_xhex [:,generated_hexagons] = x_hexagon
    TX_yhex [:,generated_hexagons] = y_hexagon
    generated_hexagons += 1

    while(generated_hexagons < K):
        for j in range(6):
            tmp_xloc = TX_loc [0, i]+np.sqrt(3)*R*np.cos(j*np.pi/(3))
            tmp_yloc = TX_loc [1, i]+np.sqrt(3)*R*np.sin(j*np.pi/(3))
            tmp_xhex = tmp_xloc+x_hexagon
            tmp_yhex = tmp_yloc+y_hexagon
            was_before = False
            for inner_loop in range(generated_hexagons):
                if (abs(tmp_xloc-TX_loc [0, inner_loop*1])<R*1e-2 and abs(tmp_yloc-TX_loc [1, inner_loop*1])<R*1e-2):
                    was_before = True
                    break
            if (not was_before):
                TX_loc [0, generated_hexagons] = tmp_xloc
                TX_loc [1, generated_hexagons] = tmp_yloc
                TX_xhex [:,generated_hexagons] = tmp_xhex
                TX_yhex [:,generated_hexagons] = tmp_yhex      
                generated_hexagons += 1
            if(generated_hexagons>= K):
                break
        i += 1
    if random_state_seed != None:
        np.random.seed(random_state_seed)
    ############### DROP USERS
    for i in range(N):
        # Randomly assign initial cell placement
        if equal_number_for_BS:
            UE_perBS = N//K
            cell_mapping[i] = int(i/UE_perBS)
        else:
            cell_mapping[i] = np.random.randint(K)
        this_cell = cell_mapping[i]
        user_mapping[this_cell].append(i)
        
        # Place UE within that cell.
        constraint_minx_UE=min(TX_xhex[:,this_cell])
        constraint_maxx_UE=max(TX_xhex[:,this_cell])
        constraint_miny_UE=min(TX_yhex[:,this_cell])
        constraint_maxy_UE=max(TX_yhex[:,this_cell])
        inside_checker = True

        while (inside_checker):
            RX_loc[0,i]= np.random.uniform(constraint_minx_UE,constraint_maxx_UE)
            RX_loc[1,i]= np.random.uniform(constraint_miny_UE,constraint_maxy_UE)
            tmp_distance2center = np.sqrt(np.square(RX_loc[0,i]-TX_loc [0, this_cell])+np.square(RX_loc[1,i]-TX_loc [1, this_cell]))
            if(_inside_hexagon(RX_loc[0,i],RX_loc[1,i],TX_xhex[:,this_cell],TX_yhex[:,this_cell])
                and tmp_distance2center>min_dist and tmp_distance2center<max_dist):
                inside_checker = False

    distance_vector = _get_distance(K,N,TX_loc, RX_loc)
    
    # Get 2D distance pathloss, original pathloss tried in the previous versions
    # Get channel gains
    g_dB2_cell2user = - (128.1 + 37.6* np.log10(0.001*distance_vector))
    shadowing_vector = np.random.randn(N,K)
    
    # rayleigh_vector = np.zeros((N,K,M))
    # rayleigh_rand_var = abs(get_random_rayleigh_variable(N, M))
    
    g_dB2_cell2user = g_dB2_cell2user + shadowing_dev * shadowing_vector
    # Repeat the small scale fading for M subbands
    g_dB2_cell2user = np.repeat(np.expand_dims(g_dB2_cell2user,axis=2),M,axis=-1)
    
    g_dB2 = np.zeros((N,N,M))
    for n in range(N):
        g_dB2[n,:,:] = g_dB2_cell2user[n,cell_mapping,:]
        # rayleigh_vector[n,:,:] = rayleigh_rand_var[cell_mapping[sample_idx],n,:]
        
    gains = np.power(10.0, g_dB2 / 10.0)
    gains_cell2user = np.power(10.0, g_dB2_cell2user / 10.0)
    
    # H_all[sample_idx] = np.multiply(np.sqrt(np.repeat(np.expand_dims(gains,axis=2),M,axis=-1)),rayleigh_vector)


    return gains, gains_cell2user, cell_mapping, user_mapping, TX_loc, RX_loc

def _inside_hexagon(x,y,TX_xhex,TX_yhex):
    n = len(TX_xhex)-1
    inside = False
    p1x,p1y = TX_xhex[0],TX_yhex[0]
    for i in range(n+1):
        p2x,p2y = TX_xhex[i % n],TX_yhex[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside

def _get_distance(K,N,TX_loc, RX_loc):
    distance_vector = np.zeros((N,K))
    # tmp_TX_loc = np.zeros((2,N))

        
    # tmp_TX_loc = TX_loc[:,cell_mapping]
    
    for k in range (K):
        distance_vector[:,k]=np.sqrt(np.square(TX_loc[0,k]-RX_loc[0,:])+
                       np.square(TX_loc[1,k]-RX_loc[1,:]))   
            
    return distance_vector

def plot_deployment(TX_loc,RX_loc):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    K = len(TX_loc[0])
    N = len(RX_loc[0])
    for i in range(1):
        plt.plot(TX_loc[0,i],TX_loc[1,i],'g^', label = 'Base Station')
        plt.text(TX_loc[0,i],TX_loc[1,i], 'AP{}'.format(i), fontsize=10)
    #circ = plt.Circle((TX_loc[0,i],TX_loc[1,i]),min_dist,color='k',fill=False)
    #ax.add_patch(circ)
    for i in range(1,K):
        plt.plot(TX_loc[0,i],TX_loc[1,i],'g^')
        plt.text(TX_loc[0,i],TX_loc[1,i], 'AP{}'.format(i), fontsize=10)
        #circ = plt.Circle((TX_loc[0,i],TX_loc[1,i]),min_dist,color='k',fill=False)
        #ax.add_patch(circ)
    for i in range(1):
        plt.plot(RX_loc[0,i],RX_loc[1,i],'ro',label = 'User')
        plt.text(RX_loc[0,i],RX_loc[1,i], 'UE{}'.format(i), fontsize=10)
    for i in range(1,N): 
        plt.plot(RX_loc[0,i],RX_loc[1,i],'ro')     
        plt.text(RX_loc[0,i],RX_loc[1,i], 'UE{}'.format(i), fontsize=10)
        
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))   
    plt.tight_layout()
    plt.xlabel('x axis position (meters)')
    plt.ylabel('y axis position (meters)')
    # plt.legend(loc=4)
    #plt.savefig('./fig/deployment_K{}N{}seed{}.png'.format(K,N,seed), format='png', dpi=5000)
    plt.show()
    
    
def FP_algorithm_weighted(N, H, Pmax, noise_var,weights):
    f_new = 0
    gamma = np.zeros(N)
    y = np.zeros(N)
    p_init = Pmax * np.ones(N)
    # Initial power is just all transmitters transmit with full power
    p = np.array(p_init)
    # Take pow 2 of abs_H, no need to take it again and again
    H_2 = H ** 2
    H_2 = np.squeeze(H_2)


    for i in range(N):
        tmp_1 = H_2[i, i] * p[i]
        tmp_2 = np.matmul(H_2[i, :], p) + noise_var
        # Initialize gamma
        gamma[i] = tmp_1 / (tmp_2 - tmp_1)
    for iter in range(100):
        f_old = f_new
        for i in range(N):
            tmp_1 = H_2[i, i] * p[i]
            tmp_2 = np.matmul(H_2[i, :], p) + noise_var
            # Update y
            y[i] = np.sqrt(weights[i] * (1 + gamma[i]) * tmp_1) / (tmp_2)
            # Update gamma
            gamma[i] = tmp_1 / (tmp_2 - tmp_1)


        f_new = 0
        for i in range(N):
            # Update p
            p[i] = min (Pmax, (y[i] ** 2) * weights[i] * (1 + gamma[i]) * H_2[i,i] / np.square(np.matmul(np.square(y), H_2[:,i])))
        for i in range(N):
            # Get new result
            f_new = f_new + 2 * y[i] * np.sqrt(weights[i] * (1+gamma[i]) * H_2[i,i] * p[i]) - (y[i] ** 2) * (np.matmul(H_2[i, :], p)
                                                                                                            + noise_var)
        #Look for convergence
        if f_new - f_old <= 0.001:
            break
    # Return optimum result after convergence
    return p


def WMMSE_algorithm_weighted(N, H, Pmax, var_noise, weights):
    
    vnew = 0
    # random initialization gives much lower performance.
    b = np.sqrt(Pmax) * np.ones(N)#* np.random.rand(N) 
    f = np.zeros(N)
    w = np.zeros(N)
    H = np.squeeze(H)
    for i in range(N):
        f[i] = H[i, i] * b[i] / (np.matmul(np.square(H[i, :]), np.square(b)) + var_noise)
        w[i] = 1.0 / (1 - f[i] * b[i] * H[i, i])
        vnew = vnew + np.log2(w[i])

#    VV = np.zeros(100)
    for iter in range(100):
        vold = vnew
        for i in range(N):
            btmp = weights[i]* w[i] * f[i] * H[i, i] /(AVOID_DIV_BY_ZERO + sum(weights * w * np.square(f) * np.square(H[:, i])))
            b[i] = min(btmp, np.sqrt(Pmax)) + max(btmp, 0) - btmp

        vnew = 0
        for i in range(N):
            f[i] = H[i, i] * b[i] / (np.matmul(np.square(H[i, :]) ,np.square(b)) + var_noise)
            w[i] = 1.0 / (AVOID_DIV_BY_ZERO + 1 - f[i] * b[i] * H[i, i])
            vnew = vnew + np.log2(w[i])

        if vnew - vold <= 0.01:
            break
        
    p_opt = np.square(b)

    return p_opt