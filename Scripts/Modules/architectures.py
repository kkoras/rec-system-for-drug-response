import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
import os

from sklearn import metrics
from scipy.stats import pearsonr

import torch
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch import nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
import os

from sklearn import metrics
from scipy.stats import pearsonr

import torch
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch import nn

# Network definitions
# Linear model
class LinearMatrixFactorizationWithFeatures(torch.nn.Module):
    def __init__(self, drug_input_dim, cell_line_input_dim, output_dim, 
                 out_activation_func=None,
                 drug_bias=True,
                 cell_line_bias=True):
        super(LinearMatrixFactorizationWithFeatures, self).__init__()
        self.drug_linear = torch.nn.Linear(drug_input_dim, output_dim, bias=drug_bias)
        self.cell_line_linear = torch.nn.Linear(cell_line_input_dim, output_dim, bias=cell_line_bias)
        self.out_activation = out_activation_func
        
    def forward(self, drug_features, cell_line_features):
        drug_outputs = self.drug_linear(drug_features)
        cell_line_outputs = self.cell_line_linear(cell_line_features)
        
        final_outputs = torch.sum(torch.mul(drug_outputs, cell_line_outputs), dim=1).view(-1, 1)
        if self.out_activation:
            return self.out_activation(final_outputs)
        return final_outputs
    

# Deep autoencoder with one hidden layer
class DeepAutoencoderOneHiddenLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, code_dim, activation_func=nn.ReLU, 
                 code_activation=True, dropout=False, dropout_rate=0.5):
        super(DeepAutoencoderOneHiddenLayer, self).__init__()
        # Establish encoder
        modules = []
        modules.append(nn.Linear(input_dim, hidden_dim))
        modules.append(activation_func())
        if dropout:
            modules.append(nn.Dropout(dropout_rate))
        modules.append(nn.Linear(hidden_dim, code_dim))
        if code_activation:
            modules.append(activation_func())
        self.encoder = nn.Sequential(*modules)
        # Establish decoder
        modules = []
        modules.append(nn.Linear(code_dim, hidden_dim))
        modules.append(activation_func())
        if dropout:
            modules.append(nn.Dropout(dropout_rate))
        modules.append(nn.Linear(hidden_dim, input_dim))
        self.decoder = nn.Sequential(*modules)
        
    def forward(self, x):
        x = self.encoder(x)
        code = x
        x = self.decoder(x)
        return code, x
    
# Deep autoencoder with two hidden layers
class DeepAutoencoderTwoHiddenLayers(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, code_dim, activation_func=nn.ReLU,
                 code_activation=True, dropout=False, dropout_rate=0.5):
        super(DeepAutoencoderTwoHiddenLayers, self).__init__()
        # Establish encoder
        modules = []
        modules.append(nn.Linear(input_dim, hidden_dim1))
        modules.append(activation_func())
        if dropout:
            modules.append(nn.Dropout(dropout_rate))
        modules.append(nn.Linear(hidden_dim1, hidden_dim2))
        modules.append(activation_func())
        if dropout:
            modules.append(nn.Dropout(dropout_rate))
        modules.append(nn.Linear(hidden_dim2, code_dim))
        if code_activation:
            modules.append(activation_func())
        self.encoder = nn.Sequential(*modules)
        
        # Establish decoder
        modules = []
        modules.append(nn.Linear(code_dim, hidden_dim2))
        modules.append(activation_func())
        if dropout:
            modules.append(nn.Dropout(dropout_rate))
        modules.append(nn.Linear(hidden_dim2, hidden_dim1))
        modules.append(activation_func())
        if dropout:
            modules.append(nn.Dropout(dropout_rate))
        modules.append(nn.Linear(hidden_dim1, input_dim))
        self.decoder = nn.Sequential(*modules)
        
    def forward(self, x):
        x = self.encoder(x)
        code = x
        x = self.decoder(x)
        return code, x
    
# Deep autoencoder with three hidden layers
class DeepAutoencoderThreeHiddenLayers(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, code_dim, activation_func=nn.ReLU,
                 code_activation=True, dropout=False, dropout_rate=0.5):
        super(DeepAutoencoderThreeHiddenLayers, self).__init__()
        # Establish encoder
        modules = []
        modules.append(nn.Linear(input_dim, hidden_dim1))
        modules.append(activation_func())
        if dropout:
            modules.append(nn.Dropout(dropout_rate))
        modules.append(nn.Linear(hidden_dim1, hidden_dim2))
        modules.append(activation_func())
        if dropout:
            modules.append(nn.Dropout(dropout_rate))
        modules.append(nn.Linear(hidden_dim2, hidden_dim3))
        modules.append(activation_func())
        if dropout:
            modules.append(nn.Dropout(dropout_rate))
        modules.append(nn.Linear(hidden_dim3, code_dim))
        if code_activation:
            modules.append(activation_func())
        self.encoder = nn.Sequential(*modules)
        
        # Establish decoder
        modules = []
        modules.append(nn.Linear(code_dim, hidden_dim3))
        modules.append(activation_func())
        if dropout:
            modules.append(nn.Dropout(dropout_rate))
        modules.append(nn.Linear(hidden_dim3, hidden_dim2))
        modules.append(activation_func())
        if dropout:
            modules.append(nn.Dropout(dropout_rate))
        modules.append(nn.Linear(hidden_dim2, hidden_dim1))
        modules.append(activation_func())
        if dropout:
            modules.append(nn.Dropout(dropout_rate))
        modules.append(nn.Linear(hidden_dim1, input_dim))
        self.decoder = nn.Sequential(*modules)
        
    def forward(self, x):
        x = self.encoder(x)
        code = x
        x = self.decoder(x)
        return code, x

# Deep autoencoder with one hidden layer and batch normalization
class DeepAutoencoderOneHiddenLayerBatchNorm(nn.Module):
    def __init__(self, input_dim, hidden_dim, code_dim, activation_func=nn.ReLU, 
                 code_activation=True, dropout=False, dropout_rate=0.5, batch_norm=False):
        super(DeepAutoencoderOneHiddenLayerBatchNorm, self).__init__()
        # Establish encoder
        modules = []
        modules.append(nn.Linear(input_dim, hidden_dim))
        if batch_norm:
            modules.append(nn.BatchNorm1d(num_features=hidden_dim))
        modules.append(activation_func())
        if dropout:
            modules.append(nn.Dropout(dropout_rate))
        modules.append(nn.Linear(hidden_dim, code_dim))
        if code_activation:
            modules.append(activation_func())
        self.encoder = nn.Sequential(*modules)
        
        # Establish decoder
        modules = []
        modules.append(nn.Linear(code_dim, hidden_dim))
        if batch_norm:
            modules.append(nn.BatchNorm1d(num_features=hidden_dim))
        modules.append(activation_func())
        if dropout:
            modules.append(nn.Dropout(dropout_rate))
        modules.append(nn.Linear(hidden_dim, input_dim))
        self.decoder = nn.Sequential(*modules)
        
    def forward(self, x):
        x = self.encoder(x)
        code = x
        x = self.decoder(x)
        return code, x
    
# Rec system with incorporated autoencoders
class RecSystemWithAutoencoders(torch.nn.Module):
    def __init__(self, 
                 drug_autoencoder,
                 cell_line_autoencoder,
                 out_activation=None):
        
        super(RecSystemWithAutoencoders, self).__init__()
        self.drug_autoencoder = drug_autoencoder
        self.cell_line_autoencoder = cell_line_autoencoder
        self.out_activation = out_activation
        
    def forward(self, drug_features, cell_line_features):
        drug_code, drug_reconstruction = self.drug_autoencoder(drug_features)
        cell_line_code, cell_line_reconstruction = self.cell_line_autoencoder(cell_line_features)
        
        final_outputs = torch.sum(torch.mul(drug_code, cell_line_code), dim=1).view(-1, 1)
        if self.out_activation:
            return self.out_activation(final_outputs), drug_reconstruction, cell_line_reconstruction
        return final_outputs, drug_reconstruction, cell_line_reconstruction
    
        
        
    
class ForwardNetworkOneHiddenLayer(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim1, activation_func=nn.ReLU,
                out_activation=None, dropout_rate=0.0):
        super(ForwardNetworkOneHiddenLayer, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            activation_func(),
            nn.Linear(hidden_dim1, 1))
        self.out_activation = out_activation
    
    def forward(self, x):
        if self.out_activation:
            return self.out_activation(self.layers(x))
        else:
            return self.layers(x)
        
class ForwardLinearRegression(torch.nn.Module):
    def __init__(self, input_dim, out_activation=None):
        super(ForwardLinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.out_activation = out_activation
        
    def forward(self, x):
        if self.out_activation:
            return self.out_activation(self.linear(x))
        return self.linear(x)

class ForwardNetworkTwoHiddenLayers(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, activation_func=nn.ReLU,
                out_activation=None, dropout_rate=0.0):
        super(ForwardNetworkTwoHiddenLayers, self).__init__()
        
        self.layers = nn.Sequential(
             nn.Linear(input_dim, hidden_dim1),
             activation_func(),
             nn.Dropout(dropout_rate),
             nn.Linear(hidden_dim1, hidden_dim2),
             activation_func(),
             nn.Linear(hidden_dim2, 1))
        
        self.out_activation = out_activation
        
    
    def forward(self, x):
        if self.out_activation:
            return self.out_activation(self.layers(x))
        else:
            return self.layers(x)

class RecSystemCodeConcatenation(torch.nn.Module):
    def __init__(self, drug_autoencoder, cell_line_autoencoder, 
                 forward_network, 
                 code_interactions=False):
        super(RecSystemCodeConcatenation, self).__init__()
        self.drug_autoencoder = drug_autoencoder
        self.cell_line_autoencoder = cell_line_autoencoder
        self.forward_network = forward_network
        self.code_interactions = code_interactions
        
    def forward(self, drug_features, cell_line_features):
        drug_code, drug_reconstruction = self.drug_autoencoder(drug_features)
        cell_line_code, cell_line_reconstruction = self.cell_line_autoencoder(cell_line_features)
                
        if self.code_interactions:
            drug_code_t = drug_code.view(drug_code.shape[0], drug_code.shape[1], 1)
            cell_line_code_t = cell_line_code.view(cell_line_code.shape[0], 1, cell_line_code.shape[1])
            x = torch.bmm(drug_code_t, cell_line_code_t)
            x = x.view(cell_line_code.shape[0], x.shape[1] * x.shape[2])
            x = torch.cat((drug_code, cell_line_code, x), axis=1)
            return self.forward_network(x), drug_reconstruction, cell_line_reconstruction

        else:
            # Concatenate codes without interactions
            x = torch.cat((drug_code, cell_line_code), axis=1)
            return self.forward_network(x), drug_reconstruction, cell_line_reconstruction