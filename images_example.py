# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 10:31:13 2020

@author: flaviagv
"""


import math
import numpy as np 
import matplotlib.pyplot as plt
import hopfield_network as hf



def plot_original_and_recall_imgs(patterns, patterns_recall):
    for idx_pattern in range(len(patterns)):
        txt_title_orig = "Image " + str(idx_pattern+1) + " input"
        plt.title(txt_title_orig)
        img = patterns[idx_pattern].reshape(int(math.sqrt(patterns.shape[1])),-1)
        plt.imshow(img, cmap='gray')
        plt.show()
        
        txt_title_recall = "Image " + str(idx_pattern+1) + " recall"
        plt.title(txt_title_recall)
        img = patterns_recall[idx_pattern].reshape(int(math.sqrt(patterns.shape[1])),-1)
        plt.imshow(img, cmap='gray')
        plt.show()
        
        
if __name__ == "__main__":
    
    # Load data from pict
    pict = np.genfromtxt('pict.dat', delimiter=',').reshape(-1,1024)
    
    # Show images being used for training
    show_images = False
    if show_images:
        for i in range(pict.shape[0]):
            img = pict[i].reshape(int(math.sqrt(pict.shape[1])),-1)
            plt.imshow(img,  cmap='gray')
            plt.show()
    
    # Calculate Weight Matrix
    n_patterns = 3
    disp_W = True
    
    pict_for_learning=pict[:n_patterns]
    
    W = hf.weight_calc(pict_for_learning, disp_W, zeros_diagonal=True)
    pict_recall = hf.degraded_recall_epochs_multiple_patterns(pict_for_learning, W)
    
    stability_check = np.all(pict_for_learning==pict_recall)
    
    print("Are the patterns stable? " + str(stability_check))
    
    
    
    plot_original_and_recall_imgs(pict_for_learning, pict_recall)
    
    ### Recall degraded patterns ###
    print(" \n\n ############### p10 recall ############### ")
    p10 = pict[9]
    p10_recall = hf.degraded_recall_epochs(p10, W, show_energy_per_epoch=True)
    plt.title("Image p10 input")
    plt.imshow(p10.reshape(int(math.sqrt(pict.shape[1])),-1),  cmap='gray')
    plt.show()
    plt.title("Image p10 recall")
    plt.imshow(p10_recall.reshape(int(math.sqrt(pict.shape[1])),-1),  cmap='gray')
    plt.show()
    
    
    print(" \n\n ############### p11 recall ############### ")
    p11 = pict[10]
    p11_recall = hf.degraded_recall_epochs(p11, W, show_energy_per_epoch=True)
    
    plt.title("Image p11 input")
    plt.imshow(p11.reshape(int(math.sqrt(pict.shape[1])),-1),  cmap='gray')
    plt.show()
    plt.title("Image p11 recall")
    plt.imshow(p11_recall.reshape(int(math.sqrt(pict.shape[1])),-1),  cmap='gray')
    plt.show()
        
    print(" \n\n ############### random image recall ############### ")
    rand_vec = np.random.randint(2, size=1024)
    rand_vec[rand_vec == 0] = -1
    rand_recall = hf.degraded_recall_epochs(rand_vec, W, 100)
    plt.imshow(rand_recall.reshape(int(math.sqrt(pict.shape[1])),-1),  cmap='gray')
    plt.show()
    
    
    
    
    print("Energy at p0: " + str(hf.calculate_energy(pict_for_learning[0],W)))
    print("Energy at p1: ", hf.calculate_energy(pict_for_learning[1],W))
    print("Energy at p2: ", hf.calculate_energy(pict_for_learning[2],W))
    
    
    print("Energy at p10: " + str(hf.calculate_energy(p10, W)))
    print("Energy at p11: " + str(hf.calculate_energy(p11,W)))
    
    
   