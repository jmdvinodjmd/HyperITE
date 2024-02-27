'''
This file contains utilities used in the development of hypernetworks.

@author:
    Vinod Kumar Chauhan
    Institute of Biomedical Engineering
    University of Oxford, UK
'''
import torch.nn as nn
import torch.nn.functional as F


def MLPFunctional(inputs, weights, in_size=8, layers=[100, 100, 1], activations=[F.elu, F.elu, None], dropout_rate=0.0):
    '''
        Generic function to produce output through an MLP functional of desired architecture 
        using given parameters and inputs.
        Inputs:
            inputs          :input data to pass through the MLP
            weights         :weights for the MLP
            in_size         :size of the input data
            layers          :list of number of neurons in different layers of the desired MLP
            activations     :activations corresponding to each layer as used in 'layers'
            dropout_rate    :droput rate to be used in the MLP
        
        Output:
            x               :output after passing inputs through the MLP

    '''
    # split the given weights into layers
    w = shapeWeights(weights, in_size, layers)
    x = inputs

    # processing through hidden layers
    for i in range(len(layers) - 1):
        x = nn.functional.linear(x, weight=w[i][0], bias=w[i][1])
        if activations[i] is not None:
            x = activations[i](x)
        if i>0:
            x = nn.functional.dropout(x, dropout_rate)

    # processing through the output layer
    x = nn.functional.linear(x, weight=w[-1][0], bias=w[-1][1])
    if activations[-1] is not None:
        x = activations[-1](x)

    return x

def shapeWeights(weights, in_size, layers):
    '''
    This function is used to split the given weights into layers as per the architecture of MLP
    Inputs: 
        weights     : total weights of the MLP
        in_size     : input size for the MLP
        layers      : list of number of neurons of the desired MLP
    
    Outputs:
        w           : list of weights where each element is a set of weights to be used in the corresponding layer of MLP
    '''
    w = []
    t=0
    for i in range(len(layers)):
        if i==0:
            size = layers[i] + layers[i] * in_size
            wt = weights[t:t+size]
            w.append([wt[:-layers[i]].view(layers[i], in_size), wt[-layers[i]]])
        else:
            size = layers[i] + layers[i] * layers[i-1]
            wt = weights[t:t+size]
            w.append([wt[:-layers[i]].view(layers[i], layers[i-1]), wt[-layers[i]]])        
        t += size
            
    return w