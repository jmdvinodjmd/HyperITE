import argparse, logging
import os
import numpy as np
import torch
from torch import nn


def init_args():
    parser = argparse.ArgumentParser('ITE')
    parser.add_argument('--niters', type=int, default=5000)
    parser.add_argument('--print', type=int, default=1)
    parser.add_argument('--nfold', type=int, default=1) 
    parser.add_argument('--clip-value', type=float, default=1)
    parser.add_argument('--clip-val-tag', type=int, default=0)
    parser.add_argument('--clip-norm-tag', type=int, default=0) 
    parser.add_argument('--binary', type=int, default=0)
    parser.add_argument('--ortho-reg',  type=float, default=0, help="ortho_regularisation factor.")
    parser.add_argument('--lr1',  type=float, default=1e-04, help="learning rate for plug-in approach or for the first stage")
    parser.add_argument('--lr2',  type=float, default=1e-04, help="learning rate.")
    parser.add_argument('--weight-decay',  type=float, default=1e-04, help="weight_decay")
    parser.add_argument('-b', '--batch-size', type=int, default=1024)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--model', type=str, default='SLearner', help="SLearner, TLearner, ")
    parser.add_argument('--exp-name', type=str, default='', help="SLearner, TLearner, ")
    parser.add_argument('--project', type=str, default='ITE-test', help="SLearner, TLearner, ")
    parser.add_argument('--dataset', type=str, default='IHDP', help="IHDP, syn3000, syn10000")
    parser.add_argument('--save', type=str, default='results/', help="Path for save checkpoints")
    parser.add_argument('--load', type=str, default=None, help="ID of the experiment to load for evaluation. If None, run a new experiment.")
    parser.add_argument('-r', '--random-seed', type=int, default=2022, help="Random_seed")

    parser.add_argument('--spect-norm1', type=int, default=1)
    parser.add_argument('--spect-norm2', type=int, default=1)
    parser.add_argument('--emb-dim1', type=int, default=16)
    parser.add_argument('--emb-dim2', type=int, default=1)
    parser.add_argument('--num-chunks', type=int, default=10)
    parser.add_argument('--drop-rate',  type=float, default=0.0, help="dropout rate")
    parser.add_argument('--hn-drop-rate1',  type=float, default=0.0, help="dropout rate in HN1.")
    parser.add_argument('--hn-drop-rate2',  type=float, default=0.0, help="dropout rate in HN2.")
    parser.add_argument('--hypernet1', type=str, default='all', help="layerwise, chunking, all")
    parser.add_argument('--hypernet2', type=str, default='all', help="layerwise, chunking, all")

    parser.add_argument('--val-size',  type=float, default=0.30, help="")
    parser.add_argument('--data-size',  type=int, default=0, help="for scale experiments.")

    # for twins dataset
    parser.add_argument('--p',  type=float, default=0.1, help="probability of selecting twin")
    parser.add_argument('--n',  type=str, default=500, help="for scale experiments.")

    args = parser.parse_args()

    return args

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def save_checkpoint(state, save, epoch):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, 'checkpt-%04d.pth' % epoch)
    torch.save(state, filename)

def get_logger(logpath, filepath, package_files=[],
               displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if (logger.hasHandlers()):
        logger.handlers.clear()
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode='w')
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)

    for f in package_files:
        logger.info(f)
        with open(f, 'r') as package_f:
            logger.info(package_f.read())

    return logger

def calc_metrics(args, te_pred, mu0, mu1, mode, wandb, logger):
    te_true=(mu1-mu0).squeeze()
    ate_pred= torch.mean(te_pred) # taking mean of absolute for ATE
    ate_pred = ate_pred.detach().cpu().numpy()
    te_pred = te_pred.detach().cpu().numpy()

    pehe = np.mean(np.square((te_true - te_pred)))
    sqrt_pehe = np.sqrt(pehe)

    logger.info("PEHE:"+'-'+mode +': '+ str(sqrt_pehe) +" ATE:"+'-'+mode +': '+ str(ate_pred))
    # wandb.log({"PEHE"+'_'+mode: sqrt_pehe})
    # wandb.log({"ATE"+'_'+mode: ate_pred})
    
    return sqrt_pehe, ate_pred


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
        https://github.com/Bjarten/early-stopping-pytorch/blob/master/MNIST_Early_Stopping_example.ipynb
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print, logger=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_auc_max = np.Inf
        self.delta = delta
        self.path = path
        # self.trace_func = trace_func
        self.logger = logger

    def __call__(self, val_auc, model):

        score = - val_auc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_auc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_auc, model)
            self.counter = 0

    def save_checkpoint(self, val_auc, model):
        '''Saves model when validation acu increase.'''
        if self.verbose:
            self.logger.info(f'Validation auc increased ({self.val_auc_max:.6f} --> {val_auc:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_auc_max = val_auc

###############################################################################
###### Hypernet utilities ##############
###############################################################################
def weighted_mse_loss(input, target, weight=None):
    if weight is None:
        return torch.mean((input - target) ** 2)
    else:
        return torch.mean(weight * (input - target) ** 2)

def weighted_binary_cross_entropy(pred, y, weight=None):
    loss = torch.nn.BCELoss(reduction='none')
    if weight is None:
        return torch.mean(loss(pred, y))
    else:
        return torch.mean(weight * loss(pred, y))

############################################################
class MLP_Model(nn.Module):
    '''
    This class is used to create a standard MLP using given layers, activations, input_size and dropout_rate.
    '''
    def __init__(self, net_layers, activations, input_size, dropout_rate=0.0):
        super(MLP_Model, self).__init__()
        self.mlp = nn.ModuleList()
        # add hidden layers
        for i in range(len(net_layers)-1):
            if i == 0:
                self.mlp.append(nn.Linear(input_size, net_layers[0]))
                self.mlp.append(activations[0]())
                self.mlp.append(nn.Dropout(dropout_rate))
            else:
                self.mlp.append(nn.Linear(net_layers[i-1], net_layers[i]))
                self.mlp.append(activations[i]())
                self.mlp.append(nn.Dropout(dropout_rate))
        
        # adding the output layer 
        self.mlp.append(nn.Linear(net_layers[len(net_layers)-2], net_layers[len(net_layers)-1]))
        if activations[len(net_layers)-1] is not None:
            self.mlp.append(activations[len(net_layers)-1]())
    
    def forward(self, X, mask=None):
        out = X
        for layer in self.mlp:
            out = layer(out)
        return [out]

######### pseudo-outcome caclculation starts ##############
def calc_pseudo_outcome_drl(mu_0, mu_1, w, y, p):
    '''
    calculate pseudo-outcome for DR-Learner
    '''
    if p is None:
        p = torch.full(y.shape, 0.5)
    # print(w)
    EPS = torch.tensor(1e-7).to(p.device)
    w_1 = w / (p + EPS)
    w_0 = (1 - w) / (EPS + 1 - p)

    return (w_1 - w_0) * y + ((1 - w_1) * mu_1 - (1 - w_0) * mu_0)

def calc_pseudo_outcome_ral(mu_0, mu_1, w, y):
    '''
    calculate pseudo-outcome for RA-Learner
    '''
    return w * (y - mu_0) + (1 - w) * (mu_1 - y)

######### pseudo-outcome caclculation ends ############## 

################################
class GradientReversalLayer(torch.autograd.Function):
    '''
    Layer for gradient reversal
    '''
    @staticmethod
    def forward(ctx, x, alpha):
        # store the context for backpropagation
        ctx.alpha = alpha
        # no-op for forward pass
        return x

    @staticmethod
    def backward(ctx, grad_output):
        output = -ctx.alpha * grad_output
        return output, None
###############################


def copy_activations(activations):
    acts = []
    for i in range(2):
        a = []
        for j in range(2):
            a.append(activations[i][j].copy())
        acts.append(a)
    
    return acts

def copy_activations2(activations):
    acts = []
    for i in range(len(activations)):
        acts.append(activations[i].copy())

    return acts