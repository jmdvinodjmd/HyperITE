'''
    Common traning and evaluation of the models.
'''
import wandb
import os
import sys
import numpy as np

import torch
import torch.nn as nn

from src import utils


def train(model, X_train, X_val, y_train, y_val, t_train, t_val, args, logger, wandb, name, f_loss, w_train=None, w_val=None):
    # best_model = deepcopy(model)
    best_model = model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experimentID = model.__class__.__name__ + '-' + args.dataset  #int(SystemRandom().random()*100000)
    # checkpoint
    ckpt_path = os.path.join('./results/checkpoints/', str(experimentID) + '.ckpt')
    logger.info("Experiment " + str(experimentID))

    logger.info('args:\n')
    logger.info(args)
    logger.info(sys.argv)

    ######################################################
    ############### Prepare model and data ###############
    train_loader, val_loader = get_loader_from(X_train, X_val, y_train, y_val, t_train, t_val, device, args.batch_size, w_train=w_train, w_val=w_val)
    
    opt = torch.optim.Adam(model.parameters(), lr=args.lr1, weight_decay=args.weight_decay)

    # loss function
    if w_train is None:
        bce_loss = nn.BCELoss(reduction='none') 
        mse_loss = nn.MSELoss() 
    else:
        bce_loss = utils.weighted_binary_cross_entropy 
        mse_loss = utils.weighted_mse_loss

    # print model architecture and track gradients using wandb
    logger.info(model)
    wandb.watch(model)

    ############### TRAINING LOOP ###############    
    early_stopping = utils.EarlyStopping(patience=args.patience, path=ckpt_path, verbose=False, logger=logger)

    for epoch in range(1, args.niters+1):
        model.train()
        train_loss = 0

        for data_list in train_loader:
            opt.zero_grad()

            # forward pass
            loss = get_loss(args, model, data_list, bce_loss, mse_loss, name, device, epoch)

            # backward pass
            loss.backward()

            if args.clip_val_tag:
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=args.clip_value)
            elif args.clip_norm_tag:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_value)
                
            opt.step()
            train_loss += loss.item()
        
        logger.info('Train Loss:'+name +': {:.6f}'.format(train_loss/len(train_loader)))
        wandb.log({'Train Loss-'+name: train_loss/len(train_loader)})

        model.eval()
        test_loss = test(args, device, 'Val', val_loader, model, bce_loss, mse_loss, name, logger, wandb, epoch)
        logger.info('Val Loss:'+name +': {:.6f}'.format(test_loss/len(val_loader)))
        wandb.log({'Val Loss-'+name: test_loss/len(val_loader)})
        logger.info('Epoch: {}\n'.format(epoch))

        early_stopping(test_loss, model)
        if early_stopping.early_stop:
            logger.info("Early stopping....")
            break

    # load the best model from early stopping
    best_model.load_state_dict(torch.load(ckpt_path))

    return best_model

def test(args, device, setting, test_loader, model, bce_loss, mse_loss, name, logger, wandb, epoch):
    '''
    calculate loss for evaluation.
    '''
    test_loss = 0
    # with torch.no_grad():
    for data_list in test_loader:
        loss = get_loss(args, model, data_list, bce_loss, mse_loss, name, device, epoch)        
        test_loss += loss.item()
    
    return test_loss

def get_loss(args, model, data_list, bce_loss, mse_loss, name, device, epoch):
    '''
        calculate loss for a given learner.
    '''
    X, y, t = data_list[0], data_list[1], data_list[2]
    mask_no = torch.ones(X.shape[0], dtype=torch.bool, device=X.device)
    mask1 = t.squeeze()==1

    if args.model == 'TARNet' or args.model == 'HyperTARNet':
        [out0, out1, _] = model(X)
        if args.binary:
            loss = torch.mean((1 - t)*bce_loss(out0, y) + t*bce_loss(out1, y))
        else:
            loss = torch.mean((1 - t)*torch.square(y - out0))
            loss += torch.mean(t*torch.square(y - out1))
    elif args.model == 'MitNet' or args.model == 'HyperMitNet':
        grad_rev_multip = 1
        epoch_div = 300
        grl_alpha = grad_rev_multip * (2.0 / (1. + np.exp(-10.0 * (epoch/epoch_div if epoch<epoch_div else 1.0))) - 1)
        out0, out1, prop = model(X, grl_alpha)
        loss = model.loss(out0, out1, prop, y, t)
    elif args.model == 'SNet' or args.model == 'HyperSNet':
        grad_rev_multip = 1
        epoch_div = 300
        grl_alpha = grad_rev_multip * (2.0 / (1. + np.exp(-10.0 * (epoch/epoch_div if epoch<epoch_div else 1.0))) - 1)
        out0, out1, prop = model(X, grl_alpha)
        loss = model.loss(out0, out1, prop, y, t, X)
    elif args.model == 'FlexTENet':
        [out0, out1] = model(X)
        loss = model.loss(out0, out1, y, t)
    elif args.model == 'SLearner' or args.model == 'HyperSLearner':
        X = torch.hstack((X, t))
        [out] = model(X, [mask_no])
        if args.binary:
            loss = torch.mean(bce_loss(out, y))
        else:
            loss = mse_loss(out, y)
    elif args.model == 'TLearner' or args.model == 'DRLearner' or args.model == 'RALearner':
        if name == 'po0':
            [out] = model(X[~mask1,:], [~mask1])
            if args.binary:
                loss = torch.mean(bce_loss(out, y[~mask1]))
            else:
                loss = mse_loss(out, y[~mask1])
        elif name == 'po1':
            [out] = model(X[mask1,:], [mask1])
            if args.binary:
                loss = torch.mean(bce_loss(out, y[mask1]))
            else:
                loss = mse_loss(out, y[mask1])
        elif name == 'p':
            [out] = model(X, [mask_no])
            loss = torch.mean(bce_loss(out, t))
        if name == 'te' or name == 'te0' or name == 'te1':
            [out] = model(X, [mask_no])
            if args.binary and args.dataset!='twins':
                loss = torch.mean(bce_loss(out, y))
            else:
                loss = mse_loss(out, y)
    elif args.model == 'HyperTLearner':
        [out0, out1] = model(X, [mask_no, mask_no])
        if args.binary:
            loss = torch.mean((1 - t)*bce_loss(out0, y) + t*bce_loss(out1, y))
        else:
            loss = torch.mean((1 - t)*torch.square(y - out0))
            loss += torch.mean(t*torch.square(y - out1))
    elif args.model == 'HyperDRLearner' or args.model == 'HyperRALearner':
        if name == 'te':
            [out] = model(X, [mask_no])
            if args.binary and args.dataset!='twins':
                loss = torch.mean(bce_loss(out, y))
            else:
                loss = mse_loss(out, y)
        elif name == 'p':
            [out] = model(X, [mask_no])
            loss = torch.mean(bce_loss(out, t))
        elif name == 'po':
            [out0, out1] = model(X, [mask_no, mask_no])
            if args.binary:
                loss = torch.mean((1 - t)*bce_loss(out0, y) + t*bce_loss(out1, y))
            else:
                loss = torch.mean((1 - t)*torch.square(y - out0))
                loss += torch.mean(t*torch.square(y - out1))
    elif args.model == 'HyperDRLearnerFull' or args.model == 'HyperDRLearnerPartial':
        if name == 'te':
            [out] = model(X, [mask_no])
            if args.binary and args.dataset!='twins':
                loss = torch.mean(bce_loss(out, y))
            else:
                loss = mse_loss(out, y)
        elif name == 'po':
            [out0, out1, prop] = model(X, [mask_no, mask_no, mask_no])
            if args.binary:
                loss = torch.mean((1 - t)*bce_loss(out0, y) + t*bce_loss(out1, y))
            else:
                loss = torch.mean((1 - t)*torch.square(y - out0))
                loss += torch.mean(t*torch.square(y - out1))
            loss += torch.mean(bce_loss(prop, t))
    else:
        raise('wrong estimator selected')
    
    return loss

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, setting, data, labels, t):
        'Initialization'
        print(setting, data.shape, labels.shape)
        self.labels = labels
        self.data = data
        self.t = t
        
  def __len__(self):
        return self.labels.shape[0]

  def __getitem__(self, index):
        x = self.data[index,:]
        y = self.labels[index]
        t = self.t[index]

        return [x, y, t]


class Datasetw(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, setting, data, labels, t, weights):
        'Initialization'
        print(setting, data.shape, labels.shape)
        self.labels = labels
        self.data = data
        self.t = t
        self.weights = weights
        
  def __len__(self):
        return self.labels.shape[0]

  def __getitem__(self, index):
        x = self.data[index,:]
        y = self.labels[index]
        t = self.t[index]
        w = self.weights[index]

        return [x, y, t, w]


def get_loader_from(X_train, X_val, y_train, y_val, t_train, t_val, device, batch, w_train=None, w_val=None, test_size=0.20):
    # convert to Tensor
    X_train = torch.Tensor(X_train).to(device)
    y_train = torch.Tensor(y_train).to(device)
    t_train = torch.Tensor(t_train).to(device)

    X_val = torch.Tensor(X_val).to(device)
    y_val = torch.Tensor(y_val).to(device)
    t_val = torch.Tensor(t_val).to(device)

    if w_train is not None:
        w_train = torch.Tensor(w_train).to(device)
        w_val = torch.Tensor(w_val).to(device)

    if w_train is None:
        training_set = Dataset('Train', X_train, y_train, t_train)
        val_set = Dataset('Val', X_val, y_val, t_val)
    else:
        training_set = Datasetw('Train', X_train, y_train, t_train, w_train)
        val_set = Datasetw('Val', X_val, y_val, t_val, w_val)

    train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch, shuffle=False)
    
    return train_loader, val_loader