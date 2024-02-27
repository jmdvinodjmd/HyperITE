import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.utils.parametrizations import spectral_norm
import math

from src.models.hn_utils import *

'''
Please refer to our review of hypernetworks -- first review paper for hypernets.
This paper discusses different weight generation strategies as implemented below.
https://arxiv.org/abs/2306.06955

@article{chauhan2023brief,
  title={A Brief Review of Hypernetworks in Deep Learning},
  author={Chauhan, Vinod Kumar and Zhou, Jiandong and Lu, Ping and Molaei, Soheila and Clifton, David A},
  journal={arXiv preprint arXiv:2306.06955},
  year={2023}
}

'''

###########################################
## Def. of different types of hypernetworks
###########################################

class HyperNetwork(nn.Module):
    '''
        This hypernet generates target weights in one go.
        Inputs:
            target_layers       : list of number of neurons in different layers of the target network
            target_in           : input size for the target network, 
            hidden_layers       : list of number of neurons in different layers of the hypernetwork, 
            activations         : activations for different layers of the network
            num_embs            : number of embeddings which is equal to number of target networks
            emb_dim             : size of embedding
            dropout_rate        : droput rate for the hypernetwork
            spect_norm          : boolean to indicate use of spectral norm to stabilise hypernetworks
        Outputs:
            x                   : total weights for one of the target network having architecture as given in 'target_layers'

    '''
    def __init__(self, target_layers, target_in, hidden_layers=[100, 100], activations=[nn.ReLU, nn.ReLU, nn.ReLU],
                num_embs=2, emb_dim=16, dropout_rate=0.0, spect_norm=0):
        super(HyperNetwork, self).__init__()

        self.out_size = calc_params(target_in, target_layers)
        self.emb_dim = emb_dim
        self.dropout_rate = dropout_rate

        # create embedding
        # self.embedding_list = create_embeddings(num_embs, emb_dim)
        self.embedding_list = nn.Embedding(num_embs, emb_dim)

        # create network layers
        self.layers = nn.ModuleList()
        # adding hidden layers
        for i in range(len(hidden_layers)):
            if i == 0:
                if spect_norm:
                    self.layers.append(spectral_norm(nn.Linear(emb_dim, hidden_layers[0])))
                else:
                    self.layers.append(nn.Linear(emb_dim, hidden_layers[0]))
                self.layers.append(activations[0]())
                self.layers.append(nn.Dropout(dropout_rate))
            else:
                if spect_norm:
                    self.layers.append(spectral_norm(nn.Linear(hidden_layers[i-1], hidden_layers[i])))
                else:
                    self.layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
                self.layers.append(activations[i]())
                self.layers.append(nn.Dropout(dropout_rate))

        # adding the output layer to predict weights
        if spect_norm:
            self.layers.append(spectral_norm(nn.Linear(hidden_layers[-1], self.out_size)))
        else:
            self.layers.append(nn.Linear(hidden_layers[-1], self.out_size))
        
        # # initialise weights
        # self.apply(utils.init_weights)

        print('Created a hypernetwork with {:d} parameters for a target network with {:d} parameters.'.format(sum(p.numel() for p in self.parameters()), self.out_size))

    def forward(self, id, device='gpu'):
        # get the embedding first
        x = self.embedding_list(torch.tensor(id, device=device))
        # x = self.embedding_list[id]

        for layer in self.layers:
            x = layer(x)

        return x


class HyperNetworkLayerwise(nn.Module):
    '''
        This hypernet generates target weights in layerwise manner.
        Inputs:
            target_layers       : list of number of neurons in different layers of the target network
            target_in           : input size for the target network, 
            hidden_layers       : list of number of neurons in different layers of the hypernetwork, 
            activations         : activations for different layers of the network
            num_embs            : number of embeddings which is equal to number of target networks
            emb_dim             : size of embedding
            dropout_rate        : droput rate for the hypernetwork
            spect_norm          : boolean to indicate use of spectral norm to stabilise hypernetworks
        Outputs:
            x                   : total weights for one of the target network having architecture as given in 'target_layers'

    '''
    def __init__(self, target_layers, target_in, hidden_layers=[100, 100], activations=[nn.ReLU, nn.ReLU, nn.ReLU],
                num_embs=2, emb_dim=16, dropout_rate=0.0, spect_norm=0):
        super(HyperNetworkLayerwise, self).__init__()
        # calculate total number of required params
        self.out_size = calc_params(target_in, target_layers)
        # calculate max number of parameters among target layers
        layer_weights = calc_layer_param_max(target_in, target_layers)
        self.target_layers = target_layers
        self.emb_dim = emb_dim
        self.dropout_rate = dropout_rate

        # create embedding
        # self.embedding_list = create_embeddings(num_embs * len(target_layers), emb_dim)
        self.embedding_list = nn.Embedding(num_embs * len(target_layers), emb_dim)

        # create hypernetwork layers
        self.layers = nn.ModuleList()
        # adding hidden layers
        for i in range(len(hidden_layers)):
            if i == 0:
                if spect_norm:
                    self.layers.append(spectral_norm(nn.Linear(emb_dim, hidden_layers[0])))
                else:
                    self.layers.append(nn.Linear(emb_dim, hidden_layers[0]))
                self.layers.append(activations[0]())
                self.layers.append(nn.Dropout(dropout_rate))
            else:
                if spect_norm:
                    self.layers.append(spectral_norm(nn.Linear(hidden_layers[i-1], hidden_layers[i])))
                else:
                    self.layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
                self.layers.append(activations[i]())
                self.layers.append(nn.Dropout(dropout_rate))

        # adding the output layer of HN to predict weights equal to max among layers
        if spect_norm:
            self.layers.append(spectral_norm(nn.Linear(hidden_layers[-1], layer_weights)))
        else:
            self.layers.append(nn.Linear(hidden_layers[-1], layer_weights))

        # # initialise weights
        # self.apply(utils.init_weights)

        print('Created a hypernetwork with {:d} parameters for a target network with {:d} parameters.'.format(sum(p.numel() for p in self.parameters()), self.out_size))

    def forward(self, id, device='gpu'):
        id = torch.tensor([id]).to(device)

        weights = torch.Tensor([]).to(device)
        for i in range(len(self.target_layers)):
            # get the embedding first
            x = self.embedding_list(id * len(self.target_layers) + i).squeeze()
            # x = self.embedding_list[id * len(self.target_layers) + i]
            # generate weights for target layers one at time
            for layer in self.layers:
                x = layer(x)
            weights = torch.cat((weights, x))

        return weights


class HyperNetworkChunking(nn.Module):
    '''
        This hypernet generates target weights in small chunks.
        Inputs:
            target_layers       : list of number of neurons in different layers of the target network
            target_in           : input size for the target network, 
            hidden_layers       : list of number of neurons in different layers of the hypernetwork, 
            activations         : activations for different layers of the network
            num_embs            : number of embeddings which is equal to number of target networks
            emb_dim             : size of embedding
            dropout_rate        : droput rate for the hypernetwork
            spect_norm          : boolean to indicate use of spectral norm to stabilise hypernetworks
        Outputs:
            x                   : total weights for one of the target network having architecture as given in 'target_layers'

    '''
    def __init__(self, target_layers, target_in, num_chunks=1, hidden_layers=[100, 100], activations=[nn.ReLU, nn.ReLU, nn.ReLU],
                num_embs=2, emb_dim=16, dropout_rate=0.0, spect_norm=0):
        super(HyperNetworkChunking, self).__init__()

        self.out_size = calc_params(target_in, target_layers)
        self.emb_dim = emb_dim
        self.dropout_rate = dropout_rate
        self.num_chunks = num_chunks

        # create embedding
        self.embedding_list = nn.Embedding(num_embs * num_chunks, emb_dim)
        # self.embedding_list = create_embeddings(num_embs * num_chunks, emb_dim)

        # create network layers
        self.layers = nn.ModuleList()
        # adding hidden layers
        for i in range(len(hidden_layers)):
            if i == 0:
                if spect_norm:
                    self.layers.append(spectral_norm(nn.Linear(emb_dim, hidden_layers[0])))
                else:
                    self.layers.append(nn.Linear(emb_dim, hidden_layers[0]))
                self.layers.append(activations[0]())
                self.layers.append(nn.Dropout(dropout_rate))
            else:
                if spect_norm:
                    self.layers.append(spectral_norm(nn.Linear(hidden_layers[i-1], hidden_layers[i])))
                else:
                    self.layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
                self.layers.append(activations[i]())
                self.layers.append(nn.Dropout(dropout_rate))

        # adding the output layer to predict weights
        head_output = math.ceil(self.out_size / float(num_chunks))

        if spect_norm:
            self.layers.append(spectral_norm(nn.Linear(hidden_layers[-1], head_output)))
        else:
            self.layers.append(nn.Linear(hidden_layers[-1], head_output))

        # # initialise weights
        # self.apply(utils.init_weights)

        print('Created a hypernetwork with {:d} parameters for a target network with {:d} parameters.'.format(sum(p.numel() for p in self.parameters()), self.out_size))

    def forward(self, id, device='gpu'):
        id = torch.tensor([id]).to(device)

        weights = torch.Tensor([]).to(device)
        for i in range(self.num_chunks):
            # get the embedding first
            x = self.embedding_list(id * self.num_chunks + i).squeeze()
            # x = self.embedding_list[id * self.num_chunks + i]
            # generate weights for each chunk one at time
            for layer in self.layers:
                x = layer(x)
            weights = torch.cat((weights, x))

        return weights


class HyperNetworkSplitHead(nn.Module):
    '''
        This hypernet generates target weights using multiple heads (2 here) and can complement others.
        Inputs:
            target_layers       : list of number of neurons in different layers of the target network
            target_in           : input size for the target network, 
            hidden_layers       : list of number of neurons in different layers of the hypernetwork, 
            activations         : activations for different layers of the network
            num_embs            : number of embeddings which is equal to number of target networks
            emb_dim             : size of embedding
            dropout_rate        : droput rate for the hypernetwork
            spect_norm          : boolean to indicate use of spectral norm to stabilise hypernetworks
        Outputs:
            x                   : total weights for one of the target network having architecture as given in 'target_layers'

    '''
    def __init__(self, target_layers, target_in, hidden_layers=[100, 100], activations=[nn.ReLU, nn.ReLU, nn.ReLU],
                num_embs=2, emb_dim=16, dropout_rate=0.0, spect_norm=0, num_heads=2):

        super(HyperNetworkSplitHead, self).__init__()
        self.num_heads = num_heads
        output_size = calc_params(target_in, target_layers)
        self.out_size = math.ceil(output_size/num_heads)
        self.emb_dim = emb_dim
        self.dropout_rate = dropout_rate

        # create embedding
        # self.embedding_list = create_embeddings(num_embs, emb_dim)
        self.embedding_list = nn.Embedding(num_embs, emb_dim)

        # create network layers
        self.layers = nn.ModuleList()
        # adding hidden layers
        for i in range(len(hidden_layers)-1):
            if i == 0:
                if spect_norm:
                    self.layers.append(spectral_norm(nn.Linear(emb_dim, hidden_layers[0])))
                else:
                    self.layers.append(nn.Linear(emb_dim, hidden_layers[0]))
                self.layers.append(activations[0]())
                self.layers.append(nn.Dropout(dropout_rate))
            else:
                if spect_norm:
                    self.layers.append(spectral_norm(nn.Linear(hidden_layers[i-1], hidden_layers[i])))
                else:
                    self.layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
                self.layers.append(activations[i]())
                self.layers.append(nn.Dropout(dropout_rate))

        # create heads with output layers
        # head 1
        self.head1 = nn.ModuleList()
        if spect_norm:
            self.head1.append(spectral_norm(nn.Linear(hidden_layers[-2], int(hidden_layers[-1]/num_heads))))
        else:
            self.head1.append(nn.Linear(hidden_layers[-2], int(hidden_layers[-1]/num_heads)))
        self.head1.append(activations[-1]())
        self.head1.append(nn.Dropout(dropout_rate))
        if spect_norm:
            self.head1.append(spectral_norm(nn.Linear(int(hidden_layers[-1]/num_heads), self.out_size)))
        else:
            self.head1.append(nn.Linear(int(hidden_layers[-1]/num_heads), self.out_size))
        
        # head 2
        self.head2 = nn.ModuleList()
        if spect_norm:
            self.head2.append(spectral_norm(nn.Linear(hidden_layers[-2], int(hidden_layers[-1]/num_heads))))
        else:
            self.head2.append(nn.Linear(hidden_layers[-2], int(hidden_layers[-1]/num_heads)))
        self.head2.append(activations[-1]())
        self.head2.append(nn.Dropout(dropout_rate))
        if spect_norm:
            self.head2.append(spectral_norm(nn.Linear(int(hidden_layers[-1]/num_heads), self.out_size)))
        else:
            self.head2.append(nn.Linear(int(hidden_layers[-1]/num_heads), self.out_size))

        print('Created a hypernetwork with {:d} parameters for a target network with {:d} parameters.'.format(sum(p.numel() for p in self.parameters()), output_size))


    def forward(self, id, device='gpu'):
        # get the embedding first
        x = self.embedding_list(torch.tensor(id, device=device))
        # x = self.embedding_list[id]

        for layer in self.layers:
            x = layer(x)
        
        w1 = x
        for layer in self.head1:
            w1 = layer(w1)
        
        w2 = x
        for layer in self.head2:
            w2 = layer(w2)  

        return torch.cat((w1, w2), 0)


###########################################

class HyperNLearner(nn.Module):
    '''
        This is a generic hypernet which can generate weights for N learners simultaneously,
        using the specified weight generation strategy.
        Inputs:
            N               : number of target networks to generate
            hypernet        : type of hypernet, i.e., weight generation strategy
            args            : contains information for training
            input_size      : input size
            activations     : activations for the target network 
            target_layers   : list of neurons for the network
            emb_dim         : size of embeddings to represent target networks
            hn_drop_rate    : dropout rate to be used in the hypernetwork
            spect_norm      : boolean to denote if spectral norm to use or not
        Outputs:
            forward()       : generates outputs through the target network generated from hypernets
            predict()       : generates outputs through the target network generated from hypernets

    '''
    def __init__(self, N, hypernet, args, input_size, activations, 
                    target_layers, emb_dim, hn_drop_rate, spect_norm):
        
        super(HyperNLearner, self).__init__()
        self.target_layers = target_layers
        self.activations = activations
        self.dropout_rate = args.drop_rate
        self.N = N
        # store trained weights for N learners
        self.weights = [0]*self.N
        # create a hypernetwork
        if hypernet == 'layerwise':
            self.hypernetwork = HyperNetworkLayerwise(target_layers=target_layers, target_in=input_size, num_embs=self.N, 
                emb_dim=emb_dim, dropout_rate=hn_drop_rate, spect_norm=spect_norm)
        elif hypernet == 'chunking':
            self.hypernetwork = HyperNetworkChunking(num_chunks=args.num_chunks, target_layers=target_layers, target_in=input_size, 
                num_embs=self.N, emb_dim=emb_dim, dropout_rate=hn_drop_rate, spect_norm=spect_norm)
        elif hypernet == 'all':
            self.hypernetwork = HyperNetwork(target_layers=target_layers, target_in=input_size, num_embs=self.N, 
                emb_dim=emb_dim, dropout_rate=hn_drop_rate, spect_norm=spect_norm)
        elif hypernet == 'split':
            self.hypernetwork = HyperNetworkSplitHead(target_layers=target_layers, target_in=input_size, num_embs=self.N, 
                emb_dim=emb_dim, dropout_rate=hn_drop_rate, spect_norm=spect_norm)
        
    def forward(self, X, mask):
        device = next(self.parameters()).device
        outputs = []
        # create functional MLP (learners) with weights from hypernet
        for i in range(self.N):
            # use hypernetwork to generate weights
            self.weights[i] = self.hypernetwork(id=i, device=device)
            # create functional MLP and pass data to get output
            outputs.append(MLPFunctional(X[mask[i],:], self.weights[i], in_size=X.shape[-1], layers=self.target_layers, 
                                activations=self.activations[i], dropout_rate=self.dropout_rate))
            
        return outputs
    
    def predict(self, X, device):
        # X = torch.tensor(X).to(device).float()
        outputs = []
        # create functional MLP (learners) from weights learned during training
        for i in range(self.N):
            # create functional MLP and pass data to get output
            outputs.append(MLPFunctional(X, self.weights[i], in_size=X.shape[-1], layers=self.target_layers, 
                                activations=self.activations[i], dropout_rate=self.dropout_rate))                             

        return outputs

##############################
### Helper functions #########
##############################
def create_embeddings(num_embs, emb_dim):
    '''
        This function is used to create embeddings of given dimensions.
    '''
    embedding_list = nn.ParameterList()
    for i in range(num_embs):
        embedding_list.append(Parameter(torch.fmod(torch.randn(emb_dim).cuda(), 2)))

    return embedding_list

def calc_params(in_size, target_layers):
    '''
    This function calculates number of parameters required in the target network
    Inputs:
        in_size         : input size for the network
        target_layers   : number of neurons in different layers of the network

    Outputs:
        total           : total number of weights required the network.
    '''
    total = 0
    for i in range(len(target_layers)):
        if i==0:
            total += target_layers[i] + target_layers[i] * in_size
        else:
            total += target_layers[i] + target_layers[i] * target_layers[i-1]

    return total

def calc_layer_param_max(target_in, target_layers):
    '''
        This function calculates max number of params required among different target layers.
        Inputs:
            target_in       : input size for the network
            target_layers   : number of neurons in different layers of the network

        Outputs:
            max_w           : max of number of weights required by different layers.
    '''
    max_w = 0
    for i in range(len(target_layers)):
        if i==0:
            temp = target_in * target_layers[i] + target_layers[i]
        else:
            temp = target_layers[i-1] * target_layers[i] + target_layers[i]

        if max_w < temp:
            max_w = temp

    return max_w

##############################
