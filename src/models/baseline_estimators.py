'''
@author:
    Vinod Kumar Chauhan
    Institute of Biomedical Engineering
    University of Oxford, UK.
'''
import torch
import torch.nn as nn

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from copy import deepcopy

from src.training_loop import train
from src import utils


############################
### meta-learners
############################
class RALearner():
    '''
        RALearner:
            - trains two networks in first step and one in the second for PO functions and treatment effect, respectively.
        Inputs:
            layers              : list of neurons in layers of one network
            activations         : activation corresponding to layers
            args                : training related arguments
            input               : input size
            device              : cpu or gpu
    '''
    def __init__(self, layers, activations, args, input, device):
        super(RALearner, self).__init__()
        self.args = args    
        self.net_po0 = utils.MLP_Model(layers, activations[0], input, dropout_rate=args.hn_drop_rate1).to(device)
        self.net_po1 = utils.MLP_Model(layers, activations[0], input, dropout_rate=args.hn_drop_rate1).to(device)
        if args.dataset == 'twins':
            acts = utils.copy_activations2(activations)
            acts[0][2] = None
        else:
            acts = activations
        self.net_te = utils.MLP_Model(layers, acts[0], input, dropout_rate=args.hn_drop_rate2).to(device)

    def fit(self, X, y, t, logger, wandb, device):
        '''
            function to fit the learner
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
        mu_0, mu_1 = self.first_step(X, y, t, logger, wandb, device)
        t = torch.tensor(t).to(device).float()
        y = torch.tensor(y).to(device).float()
        
        pseudo_outcome = utils.calc_pseudo_outcome_ral(mu_0, mu_1, t, y).detach().cpu().numpy()
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
    
    def first_step(self, X, y, t, logger, wandb, device):
        '''
            process first step of the learner
        '''
        if self.args.nfold==1:
            X_train, X_val, y_train, y_val, t_train, t_val = train_test_split(X, y, t, test_size=self.args.val_size, random_state=42, stratify=t)

            self.net_po0 = train(self.net_po0, X_train, X_val, y_train, y_val, t_train, t_val, self.args, logger, wandb, name='po0', f_loss='mse')
            self.net_po1 = train(self.net_po1, X_train, X_val, y_train, y_val, t_train, t_val, self.args, logger, wandb, name='po1', f_loss='mse')

            X = torch.tensor(X).to(device).float()
            [po0] = self.net_po0(X)
            [po1] = self.net_po1(X)
        else:
            skf = StratifiedKFold(n_splits=self.args.nfold, shuffle=True, random_state=42)

            po0 = torch.zeros((X.shape[0],1)).to(device)
            po1 = torch.zeros((X.shape[0],1)).to(device)

            for i, (train_index, test_index) in enumerate(skf.split(X, t.squeeze())):
                logger.info('nFold-------------->>>>>>>>>>>>>>>>:'+str(i+1) + ' of '+str(self.args.nfold))
                net_po0 = deepcopy(self.net_po0)
                net_po1 = deepcopy(self.net_po1)

                net_po0 = train(net_po0, X[train_index,:], X[test_index,:], y[train_index], y[test_index], t[train_index], t[test_index], self.args, logger, wandb, name='po0', f_loss='mse')
                net_po1 = train(net_po1, X[train_index,:], X[test_index,:], y[train_index], y[test_index], t[train_index], t[test_index], self.args, logger, wandb, name='po1', f_loss='mse')

                X_ = torch.tensor(X[test_index,:]).to(device).float()
                [po0_] = net_po0(X_)
                [po1_] = net_po1(X_)

                po0[test_index] = po0_
                po1[test_index] = po1_

        return po0, po1

    def second_step(self, X, y, t, logger, wandb):
        '''
            process second step of the learner
        '''
        self.args.lr1 = self.args.lr2
        t = t.squeeze().detach().cpu().numpy()
        X_train, X_val, y_train, y_val, t_train, t_val = train_test_split(X, y, t, test_size=self.args.val_size, random_state=42, stratify=t)

        return train(self.net_te, X_train, X_val, y_train, y_val, t_train, t_val, self.args, logger, wandb, name='te', f_loss='mse')

    
class DRLearner():
    '''
        DRLearner:
            - trains three networks in first step and one in the second for PO functions and propensity score,
              and treatment effect, respectively.
        Inputs:
            layers              : list of neurons in layers of one network
            activations         : activation corresponding to layers
            args                : training related arguments
            input               : input size
            device              : cpu or gpu
    '''
    def __init__(self, layers, activations, args, input, device):
        super(DRLearner, self).__init__()
        self.args = args
        self.net_po0 = utils.MLP_Model(layers, activations[0], input, dropout_rate=args.hn_drop_rate1).to(device)
        self.net_po1 = utils.MLP_Model(layers, activations[0], input, dropout_rate=args.hn_drop_rate1).to(device)
        self.net_prop = utils.MLP_Model(layers, activations[1], input, dropout_rate=args.hn_drop_rate1).to(device)
        if args.dataset == 'twins':
            acts = utils.copy_activations2(activations)
            acts[0][2] = None
        else:
            acts = activations
        self.net_te = utils.MLP_Model(layers, acts[0], input, dropout_rate=args.hn_drop_rate2).to(device)

    def fit(self, X, y, t, logger, wandb, device):
        '''
            function to fit the learner
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
        mu_0, mu_1, p = self.first_step(X, y, t, logger, wandb, device)
        t = torch.tensor(t).to(device).float()
        y = torch.tensor(y).to(device).float()

        pseudo_outcome = utils.calc_pseudo_outcome_drl(mu_0, mu_1, t, y, p).detach().cpu().numpy()

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
    
    def first_step(self, X, y, t, logger, wandb, device):
        if self.args.nfold==1:
            X_train, X_val, y_train, y_val, t_train, t_val = train_test_split(X, y, t, test_size=self.args.val_size, random_state=42, stratify=t)

            self.net_po0 = train(self.net_po0, X_train, X_val, y_train, y_val, t_train, t_val, self.args, logger, wandb, name='po0', f_loss='mse')
            self.net_po1 = train(self.net_po1, X_train, X_val, y_train, y_val, t_train, t_val, self.args, logger, wandb, name='po1', f_loss='mse')
            self.net_prop = train(self.net_prop, X_train, X_val, y_train, y_val, t_train, t_val, self.args, logger, wandb, name='p', f_loss='bce')

            X = torch.tensor(X).to(device).float()
            [po0] = self.net_po0(X)
            [po1] = self.net_po1(X)
            [prop] = self.net_prop(X)
            # clip propensity scores
            # prop = torch.clamp(prop, min=0.01, max=0.99)

        else:
            skf = StratifiedKFold(n_splits=self.args.nfold, shuffle=True, random_state=42)

            po0 = torch.zeros((X.shape[0],1)).to(device)
            po1 = torch.zeros((X.shape[0],1)).to(device)
            prop = torch.zeros((X.shape[0],1)).to(device)

            for i, (train_index, test_index) in enumerate(skf.split(X, t.squeeze())):
                logger.info('nFold-------------->>>>>>>>>>>>>>>>:'+str(i+1) + ' of '+str(self.args.nfold))
                net_po0 = deepcopy(self.net_po0)
                net_po1 = deepcopy(self.net_po1)
                net_prop = deepcopy(self.net_prop)

                net_po0 = train(net_po0, X[train_index,:], X[test_index,:], y[train_index], y[test_index], t[train_index], t[test_index], self.args, logger, wandb, name='po0', f_loss='mse')
                net_po1 = train(net_po1, X[train_index,:], X[test_index,:], y[train_index], y[test_index], t[train_index], t[test_index], self.args, logger, wandb, name='po1', f_loss='mse')
                net_prop = train(net_prop, X[train_index,:], X[test_index,:], y[train_index], y[test_index], t[train_index], t[test_index], self.args, logger, wandb, name='p', f_loss='bce')

                X_ = torch.tensor(X[test_index,:]).to(device).float()
                [po0_] = net_po0(X_)
                [po1_] = net_po1(X_)
                [prop_] = net_prop(X_)

                po0[test_index] = po0_
                po1[test_index] = po1_
                prop[test_index] = prop_
            
            # clip propensity scores
            # prop = torch.clamp(prop, min=0.01, max=0.99)

        return po0, po1, prop

    def second_step(self, X, y, t, logger, wandb):
        self.args.lr1 = self.args.lr2
        t = t.squeeze().detach().cpu().numpy()
        X_train, X_val, y_train, y_val, t_train, t_val = train_test_split(X, y, t, test_size=self.args.val_size, random_state=42, stratify=t)

        return train(self.net_te, X_train, X_val, y_train, y_val, t_train, t_val, self.args, logger, wandb, name='te', f_loss='mse')


class SLearner():
    '''
        SLearner:
            - trains one network
        Inputs:
            layers              : list of neurons in layers of one network
            activations         : activation corresponding to layers
            args                : training related arguments
            input               : input size
            device              : cpu or gpu
    '''
    def __init__(self, layers, activations, args, input, device):
        super(SLearner, self).__init__()
        self.args = args    
        self.net_po = utils.MLP_Model(layers, activations[0], input, dropout_rate=args.hn_drop_rate1).to(device)

    def fit(self, X, y, t, logger, wandb, device):
        '''
            function to fit the learner
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
        X_train, X_val, y_train, y_val, t_train, t_val = train_test_split(X, y, t, test_size=self.args.val_size, random_state=42, stratify=t.squeeze())
        self.net_po = train(self.net_po, X_train, X_val, y_train, y_val, t_train, t_val, self.args, logger, wandb, name='po', f_loss='mse')

        return

    def predict(self, X, device):
        '''
            To predict treatment effect
            Inputs:
                X       : input data
            Outputs:
                out     : treatment effect
        '''
        self.net_po.eval()
        X = np.hstack((X, np.zeros((X.shape[0], 1), dtype=X.dtype)))
        X = torch.tensor(X).to(device).float()
        [out0] = self.net_po(X)
        X[:,-1:] = 1
        [out1] = self.net_po(X)
        
        return out1-out0
    

class TLearner():
    '''
        TLearner:
            - trains two networks.
        Inputs:
            layers              : list of neurons in layers of one network
            activations         : activation corresponding to layers
            args                : training related arguments
            input               : input size
            device              : cpu or gpu
    '''
    def __init__(self, layers, activations, args, input, device):
        super(TLearner, self).__init__()
        self.args = args
        self.net_po0 = utils.MLP_Model(layers, activations[0], input, dropout_rate=args.hn_drop_rate1).to(device)
        self.net_po1 = utils.MLP_Model(layers, activations[0], input, dropout_rate=args.hn_drop_rate1).to(device)

    def fit(self, X, y, t, logger, wandb, device):
        '''
            function to fit the learner
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
        X_train, X_val, y_train, y_val, t_train, t_val = train_test_split(X, y, t, test_size=self.args.val_size, random_state=42, stratify=t.squeeze())

        self.net_po0 = train(self.net_po0, X_train, X_val, y_train, y_val, t_train, t_val, self.args, logger, wandb, name='po0', f_loss='mse')
        self.net_po1 = train(self.net_po1, X_train, X_val, y_train, y_val, t_train, t_val, self.args, logger, wandb, name='po1', f_loss='mse')
       
        return

    def predict(self, X, device):
        '''
            To predict treatment effect
            Inputs:
                X       : input data
            Outputs:
                out     : treatment effect
        '''
        self.net_po0.eval()
        self.net_po1.eval()
        X = torch.tensor(X).to(device).float()
        [out0] = self.net_po0(X)
        [out1] = self.net_po1(X)
        return out1-out0
    
############################
### NN-based learners
############################
class MitNet():
    '''
        MitNet wrapper:
            - trains one network with three task specific heads for PO functions and propensity score.
        Inputs:
            layers              : list of neurons in layers of one network
            activations         : activation corresponding to layers
            args                : training related arguments
            input               : input size
            device              : cpu or gpu
    '''
    def __init__(self, layers, activations, args, input, device):
        super(MitNet, self).__init__()
        self.args = args
        self.net = MitNetModel(args, layers, activations[0], input, dropout_rate=args.hn_drop_rate1).to(device)

    def fit(self, X, y, t, logger, wandb, device):
        '''
            function to fit the learner
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
    
class MitNetModel(nn.Module):
    '''
        MitNet:
            - trains one network with three task specific heads for PO functions and propensity score.
        Inputs:
            layers              : list of neurons in layers of one network
            activations         : activation corresponding to layers
            args                : training related arguments
            input               : input size
            device              : cpu or gpu
    '''
    def __init__(self, args, net_layers, activations, input_size, dropout_rate=0.0):
        super(MitNetModel, self).__init__()
        self.binary = args.binary
        self.phi = nn.ModuleList([
            nn.Linear(input_size, net_layers[0]),
            activations[0](),
            nn.Dropout(dropout_rate)
        ])
        self.linears0 = nn.ModuleList([
            nn.Linear(net_layers[0], net_layers[1]),
            activations[1](),
            nn.Dropout(dropout_rate),
            nn.Linear(net_layers[1], net_layers[2])
        ])
        self.linears1 = nn.ModuleList([
            nn.Linear(net_layers[0], net_layers[1]),
            activations[1](),
            nn.Dropout(dropout_rate),
            nn.Linear(net_layers[1], net_layers[2])
        ])
        if activations[2] is not None:
            self.linears0.append(activations[2]())
            self.linears1.append(activations[2]())
        
        # # initialise weights
        # self.apply(utils.init_weights)

        self.prop_unit = nn.ModuleList([
            nn.Linear(net_layers[0], net_layers[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(net_layers[1], 1),
            nn.Sigmoid()
        ])
        # if self.binary:
        #     self.bin_out = nn.Sigmoid()

        self.bce_loss = nn.BCELoss()
        self.bce_l = nn.BCELoss(reduction='none')
    
    def forward(self, X, alpha=1.0):
        out = X
        for layer in self.phi:
            out = layer(out)

        out0 = out
        for layer in self.linears0:
            out0 = layer(out0)
        
        out1 = out
        for layer in self.linears1:
            out1 = layer(out1)

        # if self.grad_rev:
        prop = utils.GradientReversalLayer.apply(out, alpha)
        for layer in self.prop_unit:
            prop = layer(prop)

        return out0, out1, prop
    
    def loss(self, out0, out1, prop, y, t, beta=1.0):
        if self.binary:
            loss_outcome = torch.mean((1 - t)*self.bce_l(out0, y) + t*self.bce_l(out1, y))
        else:
            loss_outcome = torch.mean((1 - t)*torch.square(y - out0)) + torch.mean(t*torch.square(y - out1))

        loss_domain = self.bce_loss(prop, t)

        return loss_outcome + beta * loss_domain


class TARNet():
    '''
        TARNet wrapper:
            - trains one network with two task specific heads for PO functions.
        Inputs:
            layers              : list of neurons in layers of one network
            activations         : activation corresponding to layers
            args                : training related arguments
            input               : input size
            device              : cpu or gpu
    '''
    def __init__(self, layers, activations, args, input, device):
        super(TARNet, self).__init__()
        self.args = args
        self.net = TARNetModel(layers, activations[0], input, dropout_rate=args.hn_drop_rate1).to(device)

    def fit(self, X, y, t, logger, wandb, device):
        '''
            function to fit the learner
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
    
class TARNetModel(nn.Module):
    '''
        TARNet:
            - trains one network with two task specific heads for PO functions.
        Inputs:
            layers              : list of neurons in layers of one network
            activations         : activation corresponding to layers
            args                : training related arguments
            input               : input size
            device              : cpu or gpu
    '''
    def __init__(self, net_layers, activations, input_size, dropout_rate=0.0):
        super(TARNetModel, self).__init__()
        self.phi = nn.ModuleList([
            nn.Linear(input_size, net_layers[0]),
            activations[0](),
            nn.Dropout(dropout_rate)
        ])
        self.linears0 = nn.ModuleList([
            nn.Linear(net_layers[0], net_layers[1]),
            activations[1](),
            nn.Dropout(dropout_rate),
            nn.Linear(net_layers[1], net_layers[2])
        ])
        self.linears1 = nn.ModuleList([
            nn.Linear(net_layers[0], net_layers[1]),
            activations[1](),
            nn.Dropout(dropout_rate),
            nn.Linear(net_layers[1], net_layers[2])
        ])
        if activations[2] is not None:
            self.linears0.append(activations[2]())
            self.linears1.append(activations[2]())
    
    def forward(self, X):
        out = X
        for layer in self.phi:
            out = layer(out)

        out0 = out
        for layer in self.linears0:
            out0 = layer(out0)
        
        out1 = out
        for layer in self.linears1:
            out1 = layer(out1)

        return out0, out1, out


class SNet():
    '''
        SNet wrapper.
        Inputs:
            layers              : list of neurons in layers of one network
            activations         : activation corresponding to layers
            args                : training related arguments
            input               : input size
            device              : cpu or gpu
    '''
    def __init__(self, layers, activations, args, input, device):
        super(SNet, self).__init__()
        self.args = args
        self.net = SNetModel(args, input, dropout_rate=args.hn_drop_rate1).to(device)

    def fit(self, X, y, t, logger, wandb, device):
        '''
            function to fit the learner
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

class SNetModel(nn.Module):
    '''
        SNet.
        Inputs:
            layers              : list of neurons in layers of one network
            activations         : activation corresponding to layers
            args                : training related arguments
            input               : input size
            device              : cpu or gpu
    
    @inproceedings{curth2021nonparametric,
        title={Nonparametric Estimation of Heterogeneous Treatment Effects: From Theory to Learning Algorithms},
        author={Curth, Alicia and van der Schaar, Mihaela},
        year={2021},
        booktitle={Proceedings of the 24th International Conference on Artificial Intelligence and Statistics (AISTATS)},
        organization={PMLR}
    }
    - phi_c and phi_w has 100, and phi_O, phi0 and phi1 has 50 units.
    - layers: 3 and 2
    - l2_reg = 0.0001 and ortho_reg_factor=0.0 for IHDP otherwise 0.01
    - disc_factor = 0
    '''
    def __init__(self, args, input, dropout_rate):
        super(SNetModel, self).__init__()
        self.dataset = args.dataset
        self.binary = args.binary
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
        self.linears0 = nn.ModuleList([
            nn.Linear(50 + 50 + 50, 100),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(100, 1)
        ])
        self.linears1 = nn.ModuleList([
            nn.Linear(50 + 50 + 50, 100),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(100, 1)
        ])
        self.prop_unit = nn.ModuleList([
            nn.Linear(50 + 50, 100),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(100, 1),
            nn.Sigmoid()
        ])
        if self.binary:
            self.bin_out = nn.Sigmoid()

        self.bce_loss = nn.BCELoss()
        self.bce_l = nn.BCELoss(reduction='none')

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
        for layer in self.linears0:
            out0 = layer(out0)
        
        out1 = torch.cat((out_o, out_1, out_c), dim=1)
        for layer in self.linears1:
            out1 = layer(out1)
        
        if self.binary:
            out0 = self.bin_out(out0)
            out1 = self.bin_out(out1)
        
        # if self.grad_rev:
        out_c = utils.GradientReversalLayer.apply(out_c, alpha)
        prop = torch.cat((out_w, out_c), dim=1)
        for layer in self.prop_unit:
            prop = layer(prop)

        return out0, out1, prop

    def loss(self, out0, out1, prop, y, t, X, alpha_o=1.0, alpha_d=1.0, ortho_reg_factor=0.01):
        # calculate standard loss
        if self.binary:
            loss_outcome = torch.mean((1 - t)*self.bce_l(out0, y) + t*self.bce_l(out1, y))
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


class FlexTENet():
    '''
        FlexTENet wrapper.
        Inputs:
            layers              : list of neurons in layers of one network
            activations         : activation corresponding to layers
            args                : training related arguments
            input               : input size
            device              : cpu or gpu
    '''
    def __init__(self, args, input, device):
        super(FlexTENet, self).__init__()
        self.args = args
        self.net = FlexTENetModel(args, input).to(device)

    def fit(self, X, y, t, logger, wandb, device):
        '''
            function to fit the learner
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
        [out0, out1] = self.net(X)
        return out1-out0
    
class FlexTENetModel(nn.Module):
    '''
        FlexTENet wrapper.
        Inputs:
            layers              : list of neurons in layers of one network
            activations         : activation corresponding to layers
            args                : training related arguments
            input               : input size
            device              : cpu or gpu

        Curth et al (2021) On Inductive Biases for Heterogeneous Treatment Effect Estimation, NeurIPS.
        @article{curth2021inductive,
            title={On Inductive Biases for Heterogeneous Treatment Effect Estimation},
            author={Curth, Alicia and van der Schaar, Mihaela},
            booktitle={Proceedings of the Thirty-Fifth Conference on Neural Information Processing Systems},
            year={2021}
        }
        neurons = [100, 100, 50, 50, 1] for shared and outcomes
            λ_1 = 0.0001, λ_2 = 100λ_1 (to induce a substantial difference) and λ_o = 0.1
            - l2-regularization: all weights of shared layer are used in regularization but only layer 3 onwards are used for private layers.
            - we report RMSE normalized by standard deviation of the observed factual training data.....
    '''
    def __init__(self, args, input):
        super(FlexTENetModel, self).__init__()
        self.dataset = args.dataset
        self.binary = args.binary
        self.normalize_ortho = False

        # designing tao_o first
        self.mu0_r0 = nn.ModuleList([
            nn.Linear(input, 50),
            nn.ReLU()
        ])
        self.mu0_h1 = nn.ModuleList([
            nn.Linear(50+50, 50),
            nn.ReLU()
        ])
        self.mu0_out = nn.Linear(50+50, 1)

        # designing tao_1 next
        self.mu1_r0 = nn.ModuleList([
            nn.Linear(input, 50),
            nn.ReLU()
        ])
        self.mu1_h1 = nn.ModuleList([
            nn.Linear(50+50, 50),
            nn.ReLU()
        ])
        self.mu1_out = nn.Linear(50+50, 1)

        # designing shared next
        self.mus_r0 = nn.ModuleList([
            nn.Linear(input, 50),
            nn.ReLU()
        ])
    
        self.mus_h1 = nn.ModuleList([
            nn.Linear(50, 50),
            nn.ReLU()
        ])
        self.mus_out = nn.Linear(50, 1)

        if self.binary:
            self.bin_out = nn.Sigmoid()
        
        self.bce_l = nn.BCELoss(reduction='none')
    
    def forward(self, X):
        # first layer
        out_0 = X
        for layer in self.mu0_r0:
            out_0 = layer(out_0)
        out_1 = X
        for layer in self.mu1_r0:
            out_1 = layer(out_1)
        out_s = X
        for layer in self.mus_r0:
            out_s = layer(out_s)
        
        # fourth layer
        out_0 = torch.cat((out_s, out_0), dim=1)
        for layer in self.mu0_h1:
            out_0 = layer(out_0)
        out_1 = torch.cat((out_s, out_1), dim=1)
        for layer in self.mu1_h1:
            out_1 = layer(out_1)
        for layer in self.mus_h1:
            out_s = layer(out_s)
        
        # output layer
        out_0 = torch.cat((out_s, out_0), dim=1)
        out_0 = self.mu0_out(out_0)
        out_1 = torch.cat((out_s, out_1), dim=1)
        out_1 = self.mu1_out(out_1)
        out_s = self.mus_out(out_s)

        out_0 += out_s
        out_1 += out_s

        if self.binary:
            out_0 = self.bin_out(out_0)
            out_1 = self.bin_out(out_1)

        return out_0, out_1

    def loss(self, out0, out1, y, t, lambda1=0.0001, lambada2=0.01, ortho_reg_factor=0.1):
        # calculate standard loss
        if self.binary:
            loss_outcome = torch.mean((1 - t)*self.bce_l(out0, y) + t*self.bce_l(out1, y))
        else:
            loss_outcome = torch.mean((1 - t)*torch.square(y - out0)) + torch.mean(t*torch.square(y - out1))

        loss = loss_outcome + self.l2_regularizer(lambda1, lambada2) + ortho_reg_factor * self._ortho_reg()

        return loss
    
    # https://github.com/AliciaCurth/CATENets/blob/21782d97ab34b3b40e7c03d69904bef790e2d55e/catenets/models/torch/snet.py#L399
    def _ortho_reg(self):
        def _get_cos_reg(params_0, params_1, normalize=False):
            if normalize:
                params_0 = params_0 / torch.linalg.norm(params_0, dim=0)
                params_1 = params_1 / torch.linalg.norm(params_1, dim=0)
            
            x_min = min(params_0.shape[0], params_1.shape[0])
            y_min = min(params_0.shape[1], params_1.shape[1])

            return (torch.linalg.norm(params_0[:x_min,:y_min] * params_1[:x_min,:y_min], "fro")**2)

        def _apply_reg_layer(layer_p0, layer_p1, layer_s, full):
            _ortho_body = 0
            if full:
                _ortho_body = _get_cos_reg(
                    layer_p0.weight, layer_p1.weight, self.normalize_ortho
                )
            _ortho_body += torch.sum(
                _get_cos_reg(layer_s.weight, layer_p0.weight, self.normalize_ortho)
                + _get_cos_reg(layer_s.weight, layer_p1.weight, self.normalize_ortho)
            )
            return _ortho_body

        ortho_body = 0
        ortho_body += _apply_reg_layer(self.mu0_r0[0], self.mu1_r0[0], self.mus_r0[0], full=True)
        ortho_body += _apply_reg_layer(self.mu0_h1[0], self.mu1_h1[0], self.mus_h1[0], full=False)
        ortho_body += _apply_reg_layer(self.mu0_out, self.mu1_out, self.mus_out, full=False)

        return ortho_body
    
    def l2_regularizer(self, lambda1, lambda2):
        regularizer_s = (self.mus_r0[0].weight**2).sum() + (self.mus_h1[0].weight**2).sum() + (self.mus_out.weight**2).sum()
                        
        regularizer_p = (self.mu0_h1[0].weight**2).sum() + (self.mu0_out.weight**2).sum()+ (self.mu1_h1[0].weight**2).sum()+ (self.mu1_out.weight**2).sum()
                       
        return 0.5 * (lambda1*regularizer_s +lambda2*regularizer_p)
    
###################################################
