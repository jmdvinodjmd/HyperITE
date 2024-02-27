from torch import nn

from src.models.hypernet_estimators import *
from src.models.baseline_estimators import *


def get_meta_model(args, input_size, device):
    layers = [100, 100, 1]
    target_layers = layers
    activations = [[nn.ReLU, nn.ReLU, nn.Sigmoid if args.binary else None],
                   [nn.ReLU, nn.ReLU, nn.Sigmoid]]
    hn_target_activations = [[F.relu, F.relu, F.sigmoid if args.binary else None],
                       [F.relu, F.relu, F.sigmoid]]

    if args.model == "SLearner":
        model = SLearner(layers, activations, args, input_size, device)
    elif args.model == "HyperSLearner":
         model = HyperSLearner(target_layers, hn_target_activations, args, input_size, device)
    elif args.model == "TLearner":
        model = TLearner(layers, activations, args, input_size-1, device)
    elif args.model == "HyperTLearner":
         model = HyperTLearner(target_layers, hn_target_activations, args, input_size-1, device)
    elif args.model == "TARNet":
        model = TARNet(layers, activations, args, input_size-1, device)
    elif args.model == "MitNet":
        model = MitNet(layers, activations, args, input_size-1, device)
    elif args.model == "HyperTARNet":
         hn_target_activations = [[nn.ReLU, F.relu, F.sigmoid if args.binary else None],
                       [nn.ReLU, F.relu, F.sigmoid]]
         model = HyperTARNet(target_layers, hn_target_activations, args, input_size-1, device)
    elif args.model == "HyperMitNet":
         hn_target_activations = [[nn.ReLU, F.relu, F.sigmoid if args.binary else None],
                       [nn.ReLU, F.relu, F.sigmoid]]
         model = HyperMitNet(target_layers, hn_target_activations, args, input_size-1, device)
    elif args.model == "SNet":
        model = SNet(layers, activations, args, input_size-1, device)
    elif args.model == "HyperSNet":
         model = HyperSNet(target_layers, hn_target_activations, args, input_size-1, device)
    elif args.model == "DRLearner":
        model = DRLearner(layers, activations, args, input_size-1, device)
    elif args.model == "HyperDRLearnerPartial":
        model = HyperDRLearnerPartial(target_layers, [hn_target_activations, activations], args, input_size-1, device)    
    elif args.model == "RALearner":
        model = RALearner(layers, activations, args, input_size-1, device)
    elif args.model == "HyperRALearner":
        model = HyperRALearner(target_layers, [hn_target_activations, activations], args, input_size-1, device)
    elif args.model == "FlexTENet":
        args.weight_decay = 0.0
        model = FlexTENet(args, input = input_size -1, device=device)
    else:
        raise Exception('incorrect method selected...', args.model)    

    return model


def get_dataset(dataset, p, tr_size):
    binary = 0
    if dataset =="IHDP":
        input_size = 25+1
        file_train = '../SNet+/data/ihdp_npci_1-100.train.npz'
        file_test =  '../SNet+/data/ihdp_npci_1-100.test.npz'
        repetitions = 100
    elif dataset =="acic":
        input_size = 55+1
        file_train = '../SNet+/data/acic2016-train.npz'
        file_test = '../SNet+/data/acic2016-test.npz'
        repetitions = 1
    elif dataset =="twins":
        input_size = 39+1
        file_train = '../SNet+/data/twins/twins-t-'+str(p)+'-tr-'+str(tr_size)+'-train.npz'
        file_test = '../SNet+/data/twins/twins-t-'+str(p)+'-tr-'+str(tr_size)+'-test.npz'
        repetitions = 1
        binary = 1
    else:
        raise Exception('please select correct dataset - ' + dataset)
    
    return file_train, file_test, input_size, repetitions, binary