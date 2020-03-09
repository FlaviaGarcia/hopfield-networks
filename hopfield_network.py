# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 12:25:28 2020

@author: flaviagv
"""

import math
import numpy as np 
import matplotlib.pyplot as plt




 

def weight_calc(patterns, do_scaling=True, disp_W=False, zeros_diagonal=True):
    # Check for 1-D pattern shape = (N,)
    if patterns.size == patterns.shape[0]:
        n_units = patterns.size
    else:
        n_units = patterns.shape[1]
    # This is the same that summing all the outer products of each pattern with itself
    W = np.dot(patterns.T,patterns)

    if do_scaling:
        W = W / n_units # Changed because /= was rasing an error for some reason...
    
    if zeros_diagonal:
        np.fill_diagonal(W,0)

    if disp_W:
        # Display weight matrix as greyscale image
        plt.title('Color representation of weight matrix')
        plt.imshow(W,  cmap='jet') 
        plt.colorbar()
        plt.show()
    return W



def check_fixed_point_found(patterns_new, patterns_prev): 
    fixed_points_in_patterns = np.all(patterns_new == patterns_prev, axis = 1)
    fixed_point_found = np.any(fixed_points_in_patterns)
    return fixed_point_found


def degraded_recall_epochs_multiple_patterns(patterns_prev, W, epochs=1000, show_energy_per_epoch=False):
    n_patterns = patterns_prev.shape[0]
    n_nodes = patterns_prev.shape[1]
    patterns_new = np.zeros((n_patterns, n_nodes))
    for idx_pattern in range(n_patterns):   
        print("Index pattern: " + str(idx_pattern))
        this_pattern_prev = patterns_prev[idx_pattern]
        this_pattern_new = degraded_recall_epochs(this_pattern_prev, W, epochs, show_energy_per_epoch)
        patterns_new[idx_pattern] = this_pattern_new
    return patterns_new
    


    
def degraded_recall_epochs(pattern_prev, W, epochs=1000, show_energy_per_epoch=False):
    """
    pattern_prev: it's an array with one n_nodes values
    """
    n_nodes = len(pattern_prev)
    energy_per_epoch = []
    pattern_new = np.zeros(n_nodes)

    for epoch in range(epochs):
        print("Epoch: " + str(epoch))
        energy = calculate_energy(pattern_prev,W)
        print("Energy: " + str(energy))
        energy_per_epoch.append(energy)
        
        for idx_node_i in range(n_nodes):
            result_sum = 0
            for idx_node_j in range(n_nodes):
                result_sum += W[idx_node_i,idx_node_j] * pattern_prev[idx_node_j]
            pattern_new[idx_node_i] = our_sign(result_sum)    
        print("Pattern new:")
        print(pattern_new)
        
        
        if stability_reached(pattern_prev, pattern_new):
            print("Stability reached in " + str(epoch+1) + " epochs.")
            stability_epochs = epoch
            break
        
        pattern_prev = pattern_new.copy()  
    
    if show_energy_per_epoch:
        plt.plot(range(1, stability_epochs+2), energy_per_epoch, "c", linestyle='--', marker='o')
        plt.xlabel("Number of recall iterations")
        plt.ylabel("Energy")
        plt.show()
        print(energy_per_epoch)
    
    return pattern_new


def our_sign(x):
    if x>=0:
        return 1 
    else:
        return -1


def stability_reached(pattern_prev, pattern_new):
    return np.all(pattern_prev == pattern_new)
    


def degraded_recall(image_vec, W, epochs, print_step):
    # Plot input image
    plt.title("degraded input")
    img = image_vec.reshape(int(math.sqrt(image_vec.size)),-1)
    plt.imshow(img, cmap='gray')
    plt.show()
    
    for i in range(epochs):
        image_vec = np.sign(np.dot(W,image_vec.T).T)
        if (i+1)%print_step == 0:
            plt.title("update # %i" %(i+1))
            img = image_vec.reshape(int(math.sqrt(image_vec.size)),-1)
            plt.imshow(img, cmap='gray')
            plt.show()  


def calculate_energy(pattern,W):
    return - pattern @ W @ pattern.T

