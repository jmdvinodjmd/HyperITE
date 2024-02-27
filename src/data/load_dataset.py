import torch

import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np


class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, setting, data, labels, mu0, mu1, ycf):
        'Initialization'
        print(setting, data.shape, labels.shape)
        self.setting = setting
        self.labels = labels
        self.data = data
        self.mu0 = mu0
        self.mu1 = mu1
        self.ycf = ycf
        
  def __len__(self):
        'Denotes the total number of samples'
        return self.labels.shape[0]

  def __getitem__(self, index):
        x = self.data[index,:]
        y = self.labels[index]
        mu0 = self.mu0[index]
        mu1 = self.mu1[index]
        if self.ycf is not None:
            cf = self.ycf[index]
        else:
            cf = -1

        return x, y, cf, mu0, mu1

# load ith simulation of data
def load_data(train_file, test_file, i):
    with open(train_file, 'rb') as trf, open(test_file, 'rb') as tef:
        train_data = np.load(trf)
        test_data = np.load(tef)

        y_train = train_data['yf'][:,i]
        y_test = test_data['yf'][:,i]
        # ycf_train = train_data['ycf'][:,i]
        # ycf_test = test_data['ycf'][:,i]
        t_train = train_data['t'][:,i]
        t_test = test_data['t'][:,i]
        X_train = train_data['x'][:,:,i]
        X_test = test_data['x'][:,:,i]
        mu0_train = train_data['mu0'][:,i]
        mu0_test = test_data['mu0'][:,i]
        mu1_train = train_data['mu1'][:,i]
        mu1_test = test_data['mu1'][:,i]

        t_train = t_train.reshape(-1,1)
        t_test = t_test.reshape(-1,1)
        y_train = y_train.reshape(-1,1)
        y_test = y_test.reshape(-1,1)
        # ycf_train = ycf_train.reshape(-1,1)
        # ycf_test = ycf_test.reshape(-1,1)

    # return X_train, X_test, y_train, y_test, ycf_train, ycf_test, t_train, t_test, mu0_train, mu0_test, mu1_train, mu1_test
    return X_train, X_test, y_train, y_test, t_train, t_test, mu0_train, mu0_test, mu1_train, mu1_test

def get_data_loaders(file_train, file_test, device, batch, i=0, test_size= 0.20):
    X_full, X_test, y_full, y_test, t_full, t_test, mu0_full, mu0_test, mu1_full, mu1_test = load_data(file_train, file_test, i)

    #concatenate t so we can use it as input
    X_full = np.concatenate([X_full, t_full], 1)
    X_test = np.concatenate([X_test, t_test], 1)
    
    # X_train, X_val, y_train, y_val, ycf_train, ycf_val, mu0_train, mu0_val, mu1_train, mu1_val = train_test_split(X_full, y_full, ycf_full, mu0_full, mu1_full, test_size=0.30, random_state=42)
    X_train, X_val, y_train, y_val, mu0_train, mu0_val, mu1_train, mu1_val = train_test_split(X_full, y_full, mu0_full, mu1_full, test_size=test_size, random_state=42, stratify=X_full[:,-1:].squeeze())

    # finding the nearest counterfactual in train data
    y_train_cf = y_train

    # convert to Tensor
    X_full = torch.Tensor(X_full).to(device)
    y_full = torch.Tensor(y_full).to(device)
    mu0_full = torch.Tensor(mu0_full).to(device)
    mu1_full = torch.Tensor(mu1_full).to(device)

    X_train = torch.Tensor(X_train).to(device)
    y_train_cf = torch.Tensor(y_train_cf).to(device)
    y_train = torch.Tensor(y_train).to(device)
    mu0_train = torch.Tensor(mu0_train).to(device)
    mu1_train = torch.Tensor(mu1_train).to(device)

    X_val = torch.Tensor(X_val).to(device)
    y_val = torch.Tensor(y_val).to(device)
    mu0_val = torch.Tensor(mu0_val).to(device)
    mu1_val = torch.Tensor(mu1_val).to(device)

    X_test = torch.Tensor(X_test).to(device)
    y_test = torch.Tensor(y_test).to(device)
    mu0_test = torch.Tensor(mu0_test).to(device)
    mu1_test = torch.Tensor(mu1_test).to(device)

    normalization_factor = y_train.std()

    trainingfull_set = Dataset('Train-full', X_full, y_full, mu0_full, mu1_full, None)
    training_set = Dataset('Train', X_train, y_train, mu0_train, mu1_train, y_train_cf)
    val_set = Dataset('Val', X_val, y_val, mu0_val, mu1_val, None)
    test_set = Dataset('Test', X_test, y_test, mu0_test, mu1_test, None)

    trainfull_loader = torch.utils.data.DataLoader(trainingfull_set, batch_size=batch, shuffle=True)
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch, shuffle=True) #, collate_fn=collate_batch
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch, shuffle=False)
    
    return trainfull_loader, train_loader, val_loader, test_loader, normalization_factor