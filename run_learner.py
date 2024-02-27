'''
    Script to run experiments for one learner at a time
      For example:
        python run_learner.py --model HyperTLearner --dataset $dataset --n $n --p $p --project $project --emb-dim1 8 --hn-drop-rate1 0.05 --random-seed $seed --exp-name 8
    By Vinod Kumar Chauhan, University of Oxford UK.
'''
import wandb
import os
from tqdm import tqdm
import sys
import numpy as np
import random
from sklearn.model_selection import train_test_split

import torch

import src.utils as utils
from src.data.load_dataset import get_data_loaders, load_data
from src.load_model_data import get_meta_model, get_dataset

# COMMON SETUP #
args = utils.init_args()
# selecting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# setting seed for reproducibility
torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)
random.seed(args.random_seed)

# load data
file_train, file_test, input_size, repetitions, binary = get_dataset(args.dataset, p=args.p, tr_size=args.n)
args.binary = binary

utils.makedirs(args.save)
experimentID = args.load
if experimentID is None:
    experimentID = args.model + '-' + args.dataset
utils.makedirs('./results/checkpoints/')
utils.makedirs('./results/figures/')
utils.makedirs('./results/raw/')

# set logger
log_path = os.path.join("./results/logs/" + "exp_" + str(experimentID) + ".log")
utils.makedirs("./results/logs/")
logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__), displaying=True)
logger.info("Experiment " + str(experimentID))

for i in tqdm([0]): #range(repetitions)
    logger.info('rep:' + str(i))
    # initilising wandb
    # wandb.init(project=args.project, entity="xyz", reinit=True)
    wandb.init(mode="disabled")
    logger.info('\n\nRun..................>>>>>>>>' + args.model + '-' + args.dataset + str(args.data_size) + '-' + str(i) + '-'+ str(args.random_seed))
    logger.info('args:\n')
    logger.info(args)
    logger.info(sys.argv)
    wandb.run.name = args.model + '-' + args.dataset + str(args.data_size)+ str(args.emb_dim1)#+ str(args.hypernet1) + args.exp_name#+ '-'+ str(args.random_seed) #+ '-NV'+ str(clip_norm_tag)+str(clip_val_tag)+str(clip_value)
    wandb.config = vars(args)

    X_full, X_test, y_full, y_test, t_full, t_test, mu0_full, mu0_test, mu1_full, mu1_test = load_data(file_train, file_test, i)

    # experiments to study effect of scale/dataset size
    if args.data_size:
        logger.info('\nRunning scale experiments : ' + str(args.data_size))
        X_full, y_full, t_full, mu0_full, mu1_full = X_full[:args.data_size,:], y_full[:args.data_size], t_full[:args.data_size], mu0_full[:args.data_size], mu1_full[:args.data_size]

    # create model
    model = get_meta_model(args, input_size, device)
    logger.info(model)

    # train model
    model.fit(X_full, y_full, t_full, logger, wandb, device)

    # evaluate model
    te = model.predict(X_test, device)
    pehe, ate = utils.calc_metrics(args, te, mu0_test, mu1_test, 'out', wandb, logger)
    wandb.log({"PEHE"+'-out': pehe})

    te = model.predict(X_full, device)
    pehe, ate = utils.calc_metrics(args, te, mu0_full, mu1_full, 'in ', wandb, logger)
    wandb.log({"PEHE"+'-in': pehe})

    # validatition
    X_train, X_val, y_train, y_val, t_train, t_val, mu0_train, mu0_val, mu1_train, mu1_val = train_test_split(X_full, y_full, t_full, mu0_full, mu1_full, test_size=args.val_size, random_state=42, stratify=t_full.squeeze())
    te = model.predict(X_val, device)
    pehe, ate = utils.calc_metrics(args, te, mu0_val, mu1_val, 'val ', wandb, logger)
    wandb.log({"PEHE"+'-val': pehe})

logger.info('args:\n')
logger.info(args)
logger.info('...........Experiment ended.............')
#########################################################
