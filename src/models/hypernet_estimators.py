'''
Here, we define all the hyper-learners.
@author:
    Vinod Kumar Chauhan, University of Oxford, UK
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from src.models.hn_utils import MLPFunctional
from src.models.hypernetworks import *
from src import utils
from src.training_loop import train


############################
### meta-learners
############################
class HyperRALearner():
    '''
        Hypernet trained version of RALearner:
            - uses hypernet to generate two networks for the first step
            - second step uses standard MLP but can be generated with hypernet
        Inputs:
            target_layers       : list of neurons in layers of one network of RALearner
            activations         : activation corresponding to target_layers
            args                : training related arguments
            input_size          : input size
            device              : cpu or gpu
    '''
    def __init__(self, target_layers, activations, args, input_size, device):
        super(HyperRALearner, self).__init__()
        self.args = args
        self.target_layers = target_layers
        self.activations = activations
        self.input_size = input_size
        self.device = device
        self.first_hn = HyperNLearner(2, args.hypernet1, args, input_size, [activations[0][0],activations[0][0]], 
                    target_layers, args.emb_dim1, args.hn_drop_rate1, args.spect_norm1).to(device)
        if args.dataset == 'twins':
            acts = utils.copy_activations(activations)
            acts[0][0][2] = None
            acts[1][0][2] = None
        else:
            acts = activations
        self.net_te = utils.MLP_Model(target_layers, acts[1][0], input_size, dropout_rate=args.hn_drop_rate2).to(device)
    
    def create_net(self):
        net = HyperNLearner(2, self.args.hypernet1, self.args, self.input_size, [self.activations[0][0],self.activations[0][0]], 
                    self.target_layers, self.args.emb_dim1, self.args.hn_drop_rate1, self.args.spect_norm1).to(self.device)
        return net

    def fit(self, X, y, t, logger, wandb, device):
        '''
            function to fit HyperRALearner
            Inputs:
                X       : input data (features)
                y       : target
                t       : treatment
                logger  : logger
                wandb   : wandb
                device  : cpu or gpu
            Outputs:
                no direct return but updates treatment network after training
        '''
        # first step
        mu_0, mu_1 = self.first_step(X, y, t, logger, wandb, device)
        t = torch.tensor(t).to(device).float()
        y = torch.tensor(y).to(device).float()

        pseudo_outcome = utils.calc_pseudo_outcome_ral(mu_0, mu_1, t, y).detach().cpu().numpy()

        # second step
        self.net_te = self.second_step(X, pseudo_outcome, t, logger, wandb)

        return

    def predict(self, X, device):
        '''
            To predict treatment effect
            Inputs:
                X       : input data
            Outputs:
                out     : treatment effect
        '''
        self.net_te.eval()
        X = torch.tensor(X).to(device).float()
        [out] = self.net_te(X)
        return out
    
    def second_step(self, X, y, t, logger, wandb):
        '''
            process second step of RALearner
        '''
        self.args.lr1 = self.args.lr2
        t = t.squeeze().detach().cpu().numpy()
        X_train, X_val, y_train, y_val, t_train, t_val = train_test_split(X, y, t, test_size=self.args.val_size, random_state=42, stratify=t)

        return train(self.net_te, X_train, X_val, y_train, y_val, t_train, t_val, self.args, logger, wandb, name='te', f_loss='mse')
    
    def first_step(self, X, y, t, logger, wandb, device):
        '''
            process first step of RALearner
        '''
        if self.args.nfold==1:
            X_train, X_val, y_train, y_val, t_train, t_val = train_test_split(X, y, t, test_size=self.args.val_size, random_state=42, stratify=t)

            self.first_hn = train(self.first_hn, X_train, X_val, y_train, y_val, t_train, t_val, self.args, logger, wandb, name='po', f_loss='mse')

            X = torch.tensor(X).to(device).float()
            mask = torch.ones(X.shape[0], dtype=torch.bool, device=device)
            [po0, po1] = self.first_hn(X, [mask, mask])
            
        else:
            skf = StratifiedKFold(n_splits=self.args.nfold, shuffle=True, random_state=42)

            po0 = torch.zeros((X.shape[0],1)).to(device)
            po1 = torch.zeros((X.shape[0],1)).to(device)

            for i, (train_index, test_index) in enumerate(skf.split(X, t.squeeze())):
                logger.info('nFold-------------->>>>>>>>>>>>>>>>:'+str(i+1) + ' of '+str(self.args.nfold))
                # model = deepcopy(self.first_hn)
                model = self.create_net()
                
                model = train(model, X[train_index,:], X[test_index,:], y[train_index], y[test_index], t[train_index], t[test_index], self.args, logger, wandb, name='po', f_loss='mse')

                X_ = torch.tensor(X[test_index,:]).to(device).float()
                mask = torch.ones(X_.shape[0], dtype=torch.bool, device=device)
                [po0_, po1_] = model(X_, [mask, mask])
                
                po0[test_index] = po0_
                po1[test_index] = po1_

        return po0, po1


class HyperDRLearnerPartial():
    '''
        Hypernet trained version of DRLearner:
            - uses hypernet to generate three networks for the first step
            - second step uses standard MLP but can be generated with hypernet
            - it is called partial as it uses hypernet for the first step only
        Inputs:
            target_layers       : list of neurons in layers of one network of RALearner
            activations         : activation corresponding to target_layers
            args                : training related arguments
            input_size          : input size
            device              : cpu or gpu
    '''
    def __init__(self, target_layers, activations, args, input_size, device):
        super(HyperDRLearnerPartial, self).__init__()
        self.args = args
        self.target_layers = target_layers
        self.activations = activations
        self.input_size = input_size
        self.device = device
        self.first_hn = HyperNLearner(3, args.hypernet1, args, input_size, [activations[0][0],activations[0][0],activations[0][1]], 
                    target_layers, args.emb_dim1, args.hn_drop_rate1, args.spect_norm1).to(device)
        if args.dataset == 'twins':
            acts = utils.copy_activations(activations)
            acts[0][0][2] = None
            acts[1][0][2] = None
        else:
            acts = activations
        self.net_te = utils.MLP_Model(target_layers, acts[1][0], input_size, dropout_rate=args.hn_drop_rate2).to(device)
    
    def create_net(self):
        net = HyperNLearner(3, self.args.hypernet1, self.args, self.input_size, [self.activations[0][0],self.activations[0][0],self.activations[0][1]], 
                    self.target_layers, self.args.emb_dim1, self.args.hn_drop_rate1, self.args.spect_norm1).to(self.device)
        return net

    def fit(self, X, y, t, logger, wandb, device):
        '''
            function to fit HyperDRLearner
            Inputs:
                X       : input data (features)
                y       : target
                t       : treatment
                logger  : logger
                wandb   : wandb
                device  : cpu or gpu
            Outputs:
                no direct return but updates treatment network after training
        '''
        # first step
        mu_0, mu_1, p = self.first_step(X, y, t, logger, wandb, device)
        t = torch.tensor(t).to(device).float()
        y = torch.tensor(y).to(device).float()
        pseudo_outcome = utils.calc_pseudo_outcome_drl(mu_0, mu_1, t, y, p).detach().cpu().numpy()
    
        self.net_te = self.second_step(X, pseudo_outcome, t, logger, wandb)

        return
    
    def second_step(self, X, y, t, logger, wandb):
        '''
            process second step
        '''
        self.args.lr1 = self.args.lr2
        t = t.squeeze().detach().cpu().numpy()
        X_train, X_val, y_train, y_val, t_train, t_val = train_test_split(X, y, t, test_size=self.args.val_size, random_state=42, stratify=t)

        return train(self.net_te, X_train, X_val, y_train, y_val, t_train, t_val, self.args, logger, wandb, name='te', f_loss='mse')

    def predict(self, X, device):
        '''
            To predict treatment effect
            Inputs:
                X       : input data
            Outputs:
                out     : treatment effect
        '''
        self.net_te.eval()
        X = torch.tensor(X).to(device).float()
        [out] = self.net_te(X)
        return out

    def first_step(self, X, y, t, logger, wandb, device):
        '''
            process first step
        '''
        if self.args.nfold==1:
            X_train, X_val, y_train, y_val, t_train, t_val = train_test_split(X, y, t, test_size=self.args.val_size, random_state=42, stratify=t)

            self.first_hn = train(self.first_hn, X_train, X_val, y_train, y_val, t_train, t_val, self.args, logger, wandb, name='po', f_loss='mse')

            X = torch.tensor(X).to(device).float()
            mask = torch.ones(X.shape[0], dtype=torch.bool, device=device)
            [po0, po1, prop] = self.first_hn(X, [mask, mask, mask])
            # clip propensity scores
            # prop = torch.clamp(prop, min=0.01, max=0.99)
            
        else:
            skf = StratifiedKFold(n_splits=self.args.nfold, shuffle=True, random_state=42)

            po0 = torch.zeros((X.shape[0],1)).to(device)
            po1 = torch.zeros((X.shape[0],1)).to(device)
            prop = torch.zeros((X.shape[0],1)).to(device)

            for i, (train_index, test_index) in enumerate(skf.split(X, t.squeeze())):
                logger.info('nFold-------------->>>>>>>>>>>>>>>>:'+str(i+1) + ' of '+str(self.args.nfold))
                # model = deepcopy(self.first_hn)
                model = self.create_net()
                
                model = train(model, X[train_index,:], X[test_index,:], y[train_index], y[test_index], t[train_index], t[test_index], self.args, logger, wandb, name='po', f_loss='mse')

                X_ = torch.tensor(X[test_index,:]).to(device).float()
                mask = torch.ones(X_.shape[0], dtype=torch.bool, device=device)
                [po0_, po1_, prop_] = model(X_, [mask, mask, mask])
                
                po0[test_index] = po0_
                po1[test_index] = po1_
                prop[test_index] = prop_
            
            # clip propensity scores
            # prop = torch.clamp(prop, min=0.01, max=0.99)

        return po0, po1, prop


class HyperSLearner():
    '''
        Hypernet trained version of SLearner
        Inputs:
            target_layers       : list of neurons in layers of one network of RALearner
            activations         : activation corresponding to target_layers
            args                : training related arguments
            input_size          : input size
            device              : cpu or gpu
    '''
    def __init__(self, target_layers, activations, args, input_size, device):
        super(HyperSLearner, self).__init__()
        self.args = args
        self.device = device
        N = 1
        self.net = HyperNLearner(N, args.hypernet1, args, input_size, [activations[0]],
                    target_layers, args.emb_dim1, args.hn_drop_rate1, args.spect_norm1).to(device)

    def fit(self, X, y, t, logger, wandb, device):
        '''
            function to fit HyperSLearner
            Inputs:
                X       : input data (features)
                y       : target
                t       : treatment
                logger  : logger
                wandb   : wandb
                device  : cpu or gpu
            Outputs:
                no direct return but updates network after training
        '''
        X_train, X_val, y_train, y_val, t_train, t_val = train_test_split(X, y, t, test_size=self.args.val_size, random_state=42, stratify=t.squeeze())
        self.net = train(self.net, X_train, X_val, y_train, y_val, t_train, t_val, self.args, logger, wandb, name='po', f_loss='mse')
       
        return

    def predict(self, X, device):
        '''
            To predict treatment effect
            Inputs:
                X       : input data
            Outputs:
                out     : treatment effect
        '''
        self.net.eval()
        X = np.hstack((X, np.zeros((X.shape[0], 1), dtype=X.dtype)))
        X = torch.tensor(X).to(device).float()
        mask = torch.ones(X.shape[0], dtype=torch.bool, device=X.device)
        [out0] = self.net(X, [mask])
        X[:,-1:] = 1
        [out1] = self.net(X, [mask])
        
        return out1-out0


class HyperTLearner():
    '''
        Hypernet trained version of TLearner:
        - hypernet generates both neworks
        Inputs:
            target_layers       : list of neurons in layers of one network of RALearner
            activations         : activation corresponding to target_layers
            args                : training related arguments
            input_size          : input size
            device              : cpu or gpu
    '''
    def __init__(self, target_layers, activations, args, input_size, device):
        super(HyperTLearner, self).__init__()
        self.args = args
        self.device = device
        N = 2
        self.net = HyperNLearner(N, args.hypernet1, args, input_size, [activations[0], activations[0]],
                    target_layers, args.emb_dim1, args.hn_drop_rate1, args.spect_norm1).to(device)

    def fit(self, X, y, t, logger, wandb, device):
        '''
            function to fit learner
            Inputs:
                X       : input data (features)
                y       : target
                t       : treatment
                logger  : logger
                wandb   : wandb
                device  : cpu or gpu
            Outputs:
                no direct return but updates network after training
        '''
        X_train, X_val, y_train, y_val, t_train, t_val = train_test_split(X, y, t, test_size=self.args.val_size, random_state=42, stratify=t.squeeze())
        self.net = train(self.net, X_train, X_val, y_train, y_val, t_train, t_val, self.args, logger, wandb, name='po', f_loss='mse')
       
        return

    def predict(self, X, device):
        '''
            To predict treatment effect
            Inputs:
                X       : input data
            Outputs:
                out     : treatment effect
        '''
        self.net.eval()
        X = torch.tensor(X).to(device).float()
        mask = torch.ones(X.shape[0], dtype=torch.bool, device=X.device)
        [out0, out1] = self.net(X, [mask, mask])

        return out1-out0
    
############################
### NN-based learners
############################
class HyperMitNet():
    '''
        a wrapper function for HyperMitNetModel
        Inputs:
            target_layers       : list of neurons in layers of one network of RALearner
            activations         : activation corresponding to target_layers
            args                : training related arguments
            input_size          : input size
            device              : cpu or gpu
    '''
    def __init__(self, target_layers, activations, args, input, device):
        super(HyperMitNet, self).__init__()
        self.args = args
        self.device = device
        self.net = HyperMitNetModel(args, input, activations[0], target_layers).to(device)

    def fit(self, X, y, t, logger, wandb, device):
        '''
            function to fit learner
            Inputs:
                X       : input data (features)
                y       : target
                t       : treatment
                logger  : logger
                wandb   : wandb
                device  : cpu or gpu
            Outputs:
                no direct return but updates network after training
        '''
        X_train, X_val, y_train, y_val, t_train, t_val = train_test_split(X, y, t, test_size=self.args.val_size, random_state=42, stratify=t.squeeze())
        self.net = train(self.net, X_train, X_val, y_train, y_val, t_train, t_val, self.args, logger, wandb, name='po', f_loss='mse')
       
        return

    def predict(self, X, device):
        '''
            To predict treatment effect
            Inputs:
                X       : input data
            Outputs:
                out     : treatment effect
        '''
        self.net.eval()
        X = torch.tensor(X).to(device).float()
        [out0, out1, _] = self.net(X)
        return out1-out0

class HyperMitNetModel(nn.Module):
    '''
        Implementation of HyperMitNet using hypernetwork.
        - shared layers are generated through a standard MLP to avoid complexityy
        - three task specific heads are generated through hypernet
        - input/output details are same as class HyperMitNet
    '''
    def __init__(self, args, input_size, activations, target_layers):

        super(HyperMitNetModel, self).__init__()
        self.activations = activations
        self.binary = args.binary
        self.target_layers = target_layers
        self.dropout_rate = args.drop_rate
        
        # create common trunk
        self.phi = nn.ModuleList([
            nn.Linear(input_size, target_layers[0]),
            activations[0](),
            nn.Dropout(self.dropout_rate)
        ])

        # create a hypernetwork
        if args.hypernet1 == 'layerwise':
            self.hypernetwork = HyperNetworkLayerwise(target_layers=target_layers[1:], target_in=target_layers[0], 
                num_embs=3, emb_dim=args.emb_dim1, dropout_rate=args.hn_drop_rate1, spect_norm=args.spect_norm1)
        elif args.hypernet1 == 'chunking':
            self.hypernetwork = HyperNetworkChunking(num_chunks=args.num_chunks, target_layers=target_layers[1:], target_in=target_layers[0], 
                num_embs=3, emb_dim=args.emb_dim1, dropout_rate=args.hn_drop_rate1, spect_norm=args.spect_norm1)
        elif args.hypernet1 == 'all':
            self.hypernetwork = HyperNetwork(target_layers=target_layers[1:], target_in=target_layers[0], 
                num_embs=3, emb_dim=args.emb_dim1, dropout_rate=args.hn_drop_rate1, spect_norm=args.spect_norm1)
        elif args.hypernet1 == 'split':
            self.hypernetwork = HyperNetworkSplitHead(target_layers=target_layers[1:], target_in=target_layers[0], 
                num_embs=3, emb_dim=args.emb_dim1, dropout_rate=args.hn_drop_rate1, spect_norm=args.spect_norm1)
        
        # # initialise weights
        # self.apply(utils.init_weights)

        self.bce_loss = nn.BCELoss()
        self.bce_l = nn.BCELoss(reduction='none')

    def forward(self, X, alpha=1.0):
        out = X
        for layer in self.phi:
            out = layer(out)
        
        # use hypernetwork to generate weights for outcome heads
        weights0 = self.hypernetwork(id=0, device=out.device)  # weights for control head
        weights1 = self.hypernetwork(id=1, device=out.device)  # weights for treatment head
        weights_prop = self.hypernetwork(id=2, device=out.device)
        
        # create functional MLP with weights from hypernet
        out0 = MLPFunctional(out, weights0, in_size=out.shape[-1], layers=self.target_layers[1:], 
                                activations=self.activations[1:], dropout_rate=self.dropout_rate)
        out1 = MLPFunctional(out, weights1, in_size=out.shape[-1], layers=self.target_layers[1:],
                                activations=self.activations[1:], dropout_rate=self.dropout_rate)
        
        # gradient reversal
        prop = utils.GradientReversalLayer.apply(out, alpha)
        prop = MLPFunctional(prop, weights_prop, in_size=out.shape[-1], layers=self.target_layers[1:],
                                activations=[F.relu, F.sigmoid], dropout_rate=self.dropout_rate)

        return out0, out1, prop
    
    def loss(self, out0, out1, prop, y, t, beta=1.0):
        if self.binary:
            loss_outcome = torch.mean((1 - t)*self.bce_l(out0, y) + t*self.bce_l(out1, y))
        else:
            loss_outcome = torch.mean((1 - t)*torch.square(y - out0)) + torch.mean(t*torch.square(y - out1))

        loss_domain = self.bce_loss(prop, t)

        return loss_outcome + beta * loss_domain


class HyperTARNet():
    '''
        a wrapper function for HyperTARNetModel
        Inputs:
            target_layers       : list of neurons in layers of one network of RALearner
            activations         : activation corresponding to target_layers
            args                : training related arguments
            input_size          : input size
            device              : cpu or gpu
    '''
    def __init__(self, target_layers, activations, args, input, device):
        super(HyperTARNet, self).__init__()
        self.args = args
        self.device = device
        self.net = HyperTARNetModel(args, input, activations[0], target_layers).to(device)

    def fit(self, X, y, t, logger, wandb, device):
        '''
            function to fit learner
            Inputs:
                X       : input data (features)
                y       : target
                t       : treatment
                logger  : logger
                wandb   : wandb
                device  : cpu or gpu
            Outputs:
                no direct return but updates network after training
        '''
        X_train, X_val, y_train, y_val, t_train, t_val = train_test_split(X, y, t, test_size=self.args.val_size, random_state=42, stratify=t.squeeze())
        self.net = train(self.net, X_train, X_val, y_train, y_val, t_train, t_val, self.args, logger, wandb, name='po', f_loss='mse')
       
        return

    def predict(self, X, device):
        '''
            To predict treatment effect
            Inputs:
                X       : input data
            Outputs:
                out     : treatment effect
        '''
        self.net.eval()
        X = torch.tensor(X).to(device).float()
        [out0, out1, _] = self.net(X)
        return out1-out0

class HyperTARNetModel(nn.Module):
    '''
        Implementation of HyperTARNet using hypernetwork.
        - shared layers are generated through a standard MLP to avoid complexityy
        - two task specific heads are generated through hypernet
        - input/output details are same as class HyperTARNet
    '''
    def __init__(self, args, input_size, activations, target_layers):

        super(HyperTARNetModel, self).__init__()
        self.activations = activations
        # self.binary = args.binary
        self.target_layers = target_layers
        self.dropout_rate = args.drop_rate
        
        # create common trunk
        self.phi = nn.ModuleList([
            nn.Linear(input_size, target_layers[0]),
            activations[0](),
            nn.Dropout(self.dropout_rate)
        ])

        # create a hypernetwork
        if args.hypernet1 == 'layerwise':
            self.hypernetwork = HyperNetworkLayerwise(target_layers=target_layers[1:], target_in=target_layers[0], 
                num_embs=2, emb_dim=args.emb_dim1, dropout_rate=args.hn_drop_rate1, spect_norm=args.spect_norm1)
        elif args.hypernet1 == 'chunking':
            self.hypernetwork = HyperNetworkChunking(num_chunks=args.num_chunks, target_layers=target_layers[1:], target_in=target_layers[0], 
                num_embs=2, emb_dim=args.emb_dim1, dropout_rate=args.hn_drop_rate1, spect_norm=args.spect_norm1)
        elif args.hypernet1 == 'all':
            self.hypernetwork = HyperNetwork(target_layers=target_layers[1:], target_in=target_layers[0], 
                num_embs=2, emb_dim=args.emb_dim1, dropout_rate=args.hn_drop_rate1, spect_norm=args.spect_norm1)
        elif args.hypernet1 == 'split':
            self.hypernetwork = HyperNetworkSplitHead(target_layers=target_layers[1:], target_in=target_layers[0], 
                num_embs=2, emb_dim=args.emb_dim1, dropout_rate=args.hn_drop_rate1, spect_norm=args.spect_norm1)
        
        # # initialise weights
        # self.apply(utils.init_weights)

    def forward(self, X):
        out = X
        for layer in self.phi:
            out = layer(out)

        # use hypernetwork to generate weights for outcome heads
        weights0 = self.hypernetwork(id=0, device=out.device)  # weights for control head
        weights1 = self.hypernetwork(id=1, device=out.device)  # weights for treatment head
        
        # create functional MLP with weights from hypernet
        out0 = MLPFunctional(out, weights0, in_size=out.shape[-1], layers=self.target_layers[1:], 
                                activations=self.activations[1:], dropout_rate=self.dropout_rate)
        out1 = MLPFunctional(out, weights1, in_size=out.shape[-1], layers=self.target_layers[1:],
                                activations=self.activations[1:], dropout_rate=self.dropout_rate)

        return out0, out1, out
    

class HyperSNet():
    '''
        a wrapper function for HyperSNetModel
        Inputs:
            target_layers       : list of neurons in layers of one network of RALearner
            activations         : activation corresponding to target_layers
            args                : training related arguments
            input_size          : input size
            device              : cpu or gpu
    '''
    def __init__(self, target_layers, activations, args, input, device):
        super(HyperSNet, self).__init__()
        self.args = args
        self.device = device
        self.net = HyperSNetModel(args, input, dropout_rate=args.hn_drop_rate1).to(device)

    def fit(self, X, y, t, logger, wandb, device):
        '''
            function to fit learner
            Inputs:
                X       : input data (features)
                y       : target
                t       : treatment
                logger  : logger
                wandb   : wandb
                device  : cpu or gpu
            Outputs:
                no direct return but updates network after training
        '''
        X_train, X_val, y_train, y_val, t_train, t_val = train_test_split(X, y, t, test_size=self.args.val_size, random_state=42, stratify=t.squeeze())
        self.net = train(self.net, X_train, X_val, y_train, y_val, t_train, t_val, self.args, logger, wandb, name='po', f_loss='mse')
       
        return

    def predict(self, X, device):
        '''
            To predict treatment effect
            Inputs:
                X       : input data
            Outputs:
                out     : treatment effect
        '''
        self.net.eval()
        X = torch.tensor(X).to(device).float()
        [out0, out1, _] = self.net(X)
        return out1-out0
    
class HyperSNetModel(nn.Module):
    '''
        Implementation of HyperSNet using hypernetwork.
        - shared layers are generated through a standard MLP to avoid complexityy
        - three task specific heads for two PO functions and one propensity score are generated through hypernet
        - input/output details are same as class HyperSNet
    '''
    def __init__(self, args, input, dropout_rate):
        super(HyperSNetModel, self).__init__()
        self.dataset = args.dataset
        self.binary = args.binary
        self.dropout_rate = dropout_rate
        # self.grad_rev = grad_rev
        self.with_prop = True
        self.ortho_reg_type = "abs"
        self.phi_o = nn.ModuleList([
            nn.Linear(input, 50),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ])
        self.phi_mu0 = nn.ModuleList([
            nn.Linear(input, 50),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ])
        self.phi_mu1 = nn.ModuleList([
            nn.Linear(input, 50),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ])
        self.phi_w = nn.ModuleList([
            nn.Linear(input, 50),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ])
        self.phi_c = nn.ModuleList([
            nn.Linear(input, 50),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ])
        
        # create a hypernetwork
        if args.hypernet1 == 'layerwise':
            self.hypernetwork = HyperNetworkLayerwise(target_layers=[100, 1], target_in=150,
                num_embs=3, emb_dim=args.emb_dim1, dropout_rate=args.hn_drop_rate1, spect_norm=args.spect_norm1)
        elif args.hypernet1 == 'chunking':
            self.hypernetwork = HyperNetworkChunking(num_chunks=args.num_chunks, target_layers=[100, 1], target_in=150,
                num_embs=3, emb_dim=args.emb_dim1, dropout_rate=args.hn_drop_rate1, spect_norm=args.spect_norm1)
        elif args.hypernet1 == 'all':
            self.hypernetwork = HyperNetwork(target_layers=[100, 1], target_in=150,
                num_embs=3, emb_dim=args.emb_dim1, dropout_rate=args.hn_drop_rate1, spect_norm=args.spect_norm1)
        elif args.hypernet1 == 'split':
            self.hypernetwork = HyperNetworkSplitHead(target_layers=[100, 1], target_in=150,
                num_embs=3, emb_dim=args.emb_dim1, dropout_rate=args.hn_drop_rate1, spect_norm=args.spect_norm1)

        self.bce_loss = nn.BCELoss()

        # initialise weights
        # self.apply(utils.init_weights)
    
    def forward(self, X, alpha=1.0):
        out_o = X
        for layer in self.phi_o:
            out_o = layer(out_o)
        
        out_0 = X
        for layer in self.phi_mu0:
            out_0 = layer(out_0)
        
        out_1 = X
        for layer in self.phi_mu1:
            out_1 = layer(out_1)

        out_w = X
        for layer in self.phi_w:
            out_w = layer(out_w)

        out_c = X
        for layer in self.phi_c:
            out_c = layer(out_c)

        out0 = torch.cat((out_o, out_0, out_c), dim=1)
        out1 = torch.cat((out_o, out_1, out_c), dim=1)

        # gradient reversal
        out_c = utils.GradientReversalLayer.apply(out_c, alpha)
        prop = torch.cat((out_w, out_c), dim=1)

        # use hypernetwork to generate weights for outcome heads
        weights0 = self.hypernetwork(id=0, device=out_o.device)  # weights for \mu_0
        weights1 = self.hypernetwork(id=1, device=out_o.device)  # weights for \mu_0
        weightsp = self.hypernetwork(id=2, device=out_o.device)  # weights for \pi
        
        # create functional MLP with weights from hypernet
        out0 = MLPFunctional(out0, weights0, in_size=out0.shape[-1], layers=[100, 1], 
                                activations=[F.relu, F.sigmoid if self.binary else None], dropout_rate=self.dropout_rate)
        out1 = MLPFunctional(out1, weights1, in_size=out1.shape[-1], layers=[100, 1],
                                activations=[F.relu, F.sigmoid if self.binary else None], dropout_rate=self.dropout_rate)
        prop = MLPFunctional(prop, weightsp, in_size=prop.shape[-1], layers=[100, 1],
                                activations=[F.relu, F.sigmoid], dropout_rate=self.dropout_rate)
    
        return out0, out1, prop

    def loss(self, out0, out1, prop, y, t, X, alpha_o=1.0, alpha_d=1.0, ortho_reg_factor=0.01):
        # calculate standard loss
        if self.binary:
            loss_outcome = torch.mean((1 - t)*self.bce_loss(out0, y) + t*self.bce_loss(out1, y))
            # loss_outcome = -torch.mean((1 - t)*(out0.log()*y + (1-y)*(1-out0).log())) -torch.mean(t*(out1.log()*y + (1-y)*(1-out1).log()))
        else:
            loss_outcome = torch.mean((1 - t)*torch.square(y - out0)) + torch.mean(t*torch.square(y - out1))
        loss_domain = self.bce_loss(prop, t)

        if self.dataset == 'IHDP':
            loss = alpha_o * loss_outcome + alpha_d * loss_domain
        else:
            loss = alpha_o * loss_outcome + alpha_d * loss_domain + ortho_reg_factor * self._ortho_reg()

        return loss
    
    # https://github.com/AliciaCurth/CATENets/blob/21782d97ab34b3b40e7c03d69904bef790e2d55e/catenets/models/torch/snet.py#L399
    def _ortho_reg(self):
        def _get_abs_rowsum(mat):
            return torch.sum(torch.abs(mat), dim=0)
        
        def _get_cos_reg(params_0, params_1, normalize=False):
            if normalize:
                params_0 = params_0 / torch.linalg.norm(params_0, dim=0)
                params_1 = params_1 / torch.linalg.norm(params_1, dim=0)
            
            x_min = min(params_0.shape[0], params_1.shape[0])
            y_min = min(params_0.shape[1], params_1.shape[1])

            return (torch.linalg.norm(params_0[:x_min,:y_min] * params_1[:x_min,:y_min], "fro")**2 )

        reps_o_params = self.phi_o[0].weight
        reps_mu0_params = self.phi_mu0[0].weight
        reps_mu1_params = self.phi_mu1[0].weight

        if self.with_prop: # with propensity
            reps_c_params = self.phi_c[0].weight
            reps_prop_params = self.phi_w[0].weight
        
        # define ortho-reg function
        if self.ortho_reg_type == "abs":
            col_o = _get_abs_rowsum(reps_o_params)
            col_mu0 = _get_abs_rowsum(reps_mu0_params)
            col_mu1 = _get_abs_rowsum(reps_mu1_params)
            
            if self.with_prop:
                col_c = _get_abs_rowsum(reps_c_params)
                col_w = _get_abs_rowsum(reps_prop_params)
            
                return torch.sum(
                        col_c * col_o
                        + col_c * col_w
                        + col_c * col_mu1
                        + col_c * col_mu0
                        + col_w * col_o
                        + col_mu0 * col_o
                        + col_o * col_mu1
                        + col_mu0 * col_mu1
                        + col_mu0 * col_w
                        + col_w * col_mu1
                    )
            else:
                return torch.sum(
                 +col_mu0 * col_o + col_o * col_mu1 + col_mu0 * col_mu1
                )
    
        elif self.ortho_reg_type == "fro":
            if self.with_prop:
                return (_get_cos_reg(reps_c_params, reps_o_params)
                    + _get_cos_reg(reps_c_params, reps_mu0_params)
                    + _get_cos_reg(reps_c_params, reps_mu1_params)
                    + _get_cos_reg(reps_c_params, reps_prop_params)
                    + _get_cos_reg(reps_o_params, reps_mu0_params)
                    + _get_cos_reg(reps_o_params, reps_mu1_params)
                    + _get_cos_reg(reps_o_params, reps_prop_params)
                    + _get_cos_reg(reps_mu0_params, reps_mu1_params)
                    + _get_cos_reg(reps_mu0_params, reps_prop_params)
                    + _get_cos_reg(reps_mu1_params, reps_prop_params))
            else:
                return (+_get_cos_reg(reps_o_params, reps_mu0_params)
                    + _get_cos_reg(reps_o_params, reps_mu1_params)
                    + _get_cos_reg(reps_mu0_params, reps_mu1_params))
        else:
            raise ValueError(f"Invalid orth_reg_typ {self.ortho_reg_type}")
