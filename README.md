# Hopfield Networks

In `hopfield_network.py` we implement a hopfield network from scratch. 

Hopfield Networks are autoassociative networks with symmentric weights. This means that the input is `x` and the ouput is also `x`. The main applications of this type of network is to reproduce the orginial version from a noisy version or a partial version. 

Hopfield networks use hebb's rule to learn the weights. Their nodes have a bipolar coding {-1,1} and has a `sign` activation function. 

`images_example.py` shows the performance of Hopfield Networks. The example network is trained with three images and it shows how the network is able to reconstract noisy and partial versions of the original images. The update method used is the synchronous.    


