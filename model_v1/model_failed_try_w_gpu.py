# -*- coding: utf-8 -*-
"""
Class definition for the `LSTMModel` nn.

Created on Fri May  7 15:38:46 2021

@author: alialp(crazy_boi_93@gmail.com)
edited(2021_05_08): yusuf mert
edited(2021_05_10): Connor W. Colombo (colombo@cmu.edu)
"""
from __future__ import annotations  # Activate postponed annotations (for using classes as return type in their own methods)

from typing import List, Any, Tuple, Union
import attr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


@attr.s(frozen=True, cmp=True, slots=True, auto_attribs=True)
class CNNHyperparams:
    """
    Collection of tunable hyperparameters for the CNNs inside the `LSTMModel` class.

    General Structure:
    Convolutions(Conv->MaxPool->Activation->BN->Dropout) + Linears(LinLayer->Activation->BN->Dropout)
    """
    # Enter a list for each element [in_channels,out_channels,kernel_size,stride]
    conv_layers: List[Union[int, List[int]]] = attr.Factory(list)
    # Enter a list for each element [kernel_size,stride] or False if no pool layer:
    MaxPoolLayers: List[Union[bool, int, List[int]]] = attr.Factory(list)
    # Enter "sigmoid","tanh", "linear" or "relu", everything else="linear":
    conv_layer_activations: List[str] = attr.Factory(list)
    # Enter False or num_features for BN, length should be equal to CNN_AE_layers:
    conv_BN_layers: List[Union[bool, int, List[int]]] = attr.Factory(list)
    # Enter False or propability of dropout:
    conv_Dropout_layers: List[Union[bool, float]] = attr.Factory(list)
    # Enter tuple (input_size,output_size):
    linear_layers: List[Tuple[int, int]] = attr.Factory(list)
    # Enter "sigmoid","tanh", "identity" or "relu":
    linear_activations: List[str] = attr.Factory(list)
    linear_BN_layers: List[Union[bool, int, List[int]]] = attr.Factory(list)
    linear_Dropout_layers: List[Union[bool, float]] = attr.Factory(list)

    @property
    def properties(self) -> List[List[Any]]:
        return [
            self.conv_layers,
            self.MaxPoolLayers,
            self.conv_layer_activations,
            self.conv_BN_layers,
            self.conv_Dropout_layers,
            self.linear_layers,
            self.linear_activations,
            self.linear_BN_layers,
            self.linear_Dropout_layers
        ]


@attr.s(frozen=True, cmp=True, slots=True, auto_attribs=True)
class LSTMHyperparams:
    """
    Collection of tunable hyperparameters for the LSTM inside the `LSTMModel` class.
    """
    hidden_size: int
    nr_of_layers: int
    dropout: float

    @property
    def properties(self) -> List[Union[float, int]]:
        return [
            self.hidden_size,
            self.nr_of_layers,
            self.dropout
        ]


@attr.s(frozen=True, cmp=True, slots=True, auto_attribs=True)
class ModelHyperparams:
    """Collection of tunable hyperparameters for the `LSTMModel` class."""
    CNN_AE: CNNHyperparams
    CNN_MIC: CNNHyperparams
    CNN_FORCE: CNNHyperparams
    LSTM: LSTMHyperparams

    @property
    def properties(self) -> List[List[Any]]:
        return [
            self.CNN_AE.properties,
            self.CNN_MIC.properties,
            self.CNN_FORCE.properties,
            self.LSTM.properties
        ]


class LSTMModel(nn.Module):
    def __init__(self, CNN_AE_properties, CNN_MIC_properties, CNN_FORCE_properties, LSTM_properties, ):
        super(LSTMModel, self).__init__()
        # %% Convolution Layer for the ACOUSTIC EMISSION
        self.CNN_AE_conv_layers = CNN_AE_properties[0]
        self.CNN_AE_conv_MaxPoolLayers = CNN_AE_properties[1]
        self.CNN_AE_conv_layer_activations = CNN_AE_properties[2]
        self.CNN_AE_conv_BN_layers = CNN_AE_properties[3]
        self.CNN_AE_conv_Dropout_layers = CNN_AE_properties[4]
        self.CNN_AE_linear_layers = CNN_AE_properties[5]
        self.CNN_AE_linear_activations = CNN_AE_properties[6]
        self.CNN_AE_linear_BN_layers = CNN_AE_properties[7]
        self.CNN_AE_linear_Dropout_layers = CNN_AE_properties[8]
        self.CNN_AE_conv_modules: List[nn.Module] = []
        self.CNN_AE_lin_modules: List[nn.Module] = []


        self.CNN_MIC_conv_layers = CNN_MIC_properties[0]
        self.CNN_MIC_conv_MaxPoolLayers = CNN_MIC_properties[1]
        self.CNN_MIC_conv_layer_activations = CNN_MIC_properties[2]
        self.CNN_MIC_conv_BN_layers = CNN_MIC_properties[3]
        self.CNN_MIC_conv_Dropout_layers = CNN_MIC_properties[4]
        self.CNN_MIC_linear_layers = CNN_MIC_properties[5]
        self.CNN_MIC_linear_activations = CNN_MIC_properties[6]
        self.CNN_MIC_linear_BN_layers = CNN_MIC_properties[7]
        self.CNN_MIC_linear_Dropout_layers = CNN_MIC_properties[8]
        self.CNN_MIC_conv_modules: List[nn.Module] = []
        self.CNN_MIC_lin_modules: List[nn.Module] = []

        self.CNN_FORCE_conv_layers = CNN_FORCE_properties[0]
        self.CNN_FORCE_conv_MaxPoolLayers = CNN_FORCE_properties[1]
        self.CNN_FORCE_conv_layer_activations = CNN_FORCE_properties[2]
        self.CNN_FORCE_conv_BN_layers = CNN_FORCE_properties[3]
        self.CNN_FORCE_conv_Dropout_layers = CNN_FORCE_properties[4]
        self.CNN_FORCE_linear_layers = CNN_FORCE_properties[5]
        self.CNN_FORCE_linear_activations = CNN_FORCE_properties[6]
        self.CNN_FORCE_linear_BN_layers = CNN_FORCE_properties[7]
        self.CNN_FORCE_linear_Dropout_layers = CNN_FORCE_properties[8]
        self.CNN_FORCE_conv_modules: List[nn.Module] = []
        self.CNN_FORCE_lin_modules: List[nn.Module] = []

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print("Device:", device)

        for index, i in enumerate(self.CNN_AE_conv_layers):
            self.CNN_AE_conv_modules.append(
                nn.Conv1d(i[0], i[1], i[2], stride=i[3]))
            if self.CNN_AE_conv_MaxPoolLayers[index] != False:
                self.CNN_AE_conv_modules.append(nn.MaxPool1d(kernel_size=self.CNN_AE_conv_MaxPoolLayers[index][0],
                                                             stride=self.CNN_AE_conv_MaxPoolLayers[index][
                    0]).to(device))  # ?? What this value should be

            if self.CNN_AE_conv_layer_activations[index] == "sigmoid":
                self.CNN_AE_conv_modules.append(nn.Sigmoid().to(device))
            elif self.CNN_AE_conv_layer_activations[index] == "relu":
                self.CNN_AE_conv_modules.append(nn.ReLU().to(device))
            elif self.CNN_AE_conv_layer_activations[index] == "tanh":
                self.CNN_AE_conv_modules.append(nn.Tanh().to(device))
            else:
                self.CNN_AE_conv_modules.append(nn.Identity().to(device))

            if self.CNN_AE_conv_BN_layers[index] != False:
                self.CNN_AE_conv_modules.append(
                    nn.BatchNorm1d(self.CNN_AE_conv_BN_layers[index]).to(device))

            if self.CNN_AE_conv_Dropout_layers != False:
                self.CNN_AE_conv_modules.append(nn.Dropout(
                    self.CNN_AE_conv_Dropout_layers[index]).to(device))

        for index, i in enumerate(self.CNN_AE_linear_layers):
            self.CNN_AE_lin_modules.append(nn.Linear(i[0], i[1]).to(device))

            if self.CNN_AE_linear_activations[index] == "sigmoid":
                self.CNN_AE_lin_modules.append(nn.Sigmoid().to(device))
            elif self.CNN_AE_linear_activations[index] == "relu":
                self.CNN_AE_lin_modules.append(nn.ReLU().to(device))
            elif self.CNN_AE_linear_activations[index] == "tanh":
                self.CNN_AE_lin_modules.append(nn.Tanh().to(device))
            else:
                self.CNN_AE_lin_modules.append(nn.Identity().to(device))

            if self.CNN_AE_linear_BN_layers[index] != False:
                self.CNN_AE_lin_modules.append(
                    nn.BatchNorm1d(self.CNN_AE_linear_BN_layers[index]).to(device))

            if self.CNN_AE_linear_Dropout_layers != False:
                self.CNN_AE_lin_modules.append(nn.Dropout(
                    self.CNN_AE_linear_Dropout_layers[index]).to(device))

        self.CNN_AE_conv = nn.Sequential(*self.CNN_AE_conv_modules).double()
        self.CNN_AE_conv.to(device)
        self.CNN_AE_lin = nn.Sequential(*self.CNN_AE_lin_modules).double()
        self.CNN_AE_lin.to(device)

        for index, i in enumerate(self.CNN_MIC_conv_layers):
            self.CNN_MIC_conv_modules.append(
                nn.Conv1d(i[0], i[1], i[2], stride=i[3]))
            if self.CNN_MIC_conv_MaxPoolLayers[index] != False:
                self.CNN_MIC_conv_modules.append(nn.MaxPool1d(kernel_size=self.CNN_MIC_conv_MaxPoolLayers[index][0],
                                                              stride=self.CNN_MIC_conv_MaxPoolLayers[index][
                    0]).to(device))  # ?? What this value should be

            if self.CNN_MIC_conv_layer_activations[index] == "sigmoid":
                self.CNN_MIC_conv_modules.append(nn.Sigmoid().to(device))
            elif self.CNN_MIC_conv_layer_activations[index] == "relu":
                self.CNN_MIC_conv_modules.append(nn.ReLU().to(device))
            elif self.CNN_MIC_conv_layer_activations[index] == "tanh":
                self.CNN_MIC_conv_modules.append(nn.Tanh().to(device))
            else:
                self.CNN_MIC_conv_modules.append(nn.Identity().to(device))

            if self.CNN_MIC_conv_BN_layers[index] != False:
                self.CNN_MIC_conv_modules.append(
                    nn.BatchNorm1d(self.CNN_MIC_conv_BN_layers[index]).to(device))

            if self.CNN_MIC_conv_Dropout_layers != False:
                self.CNN_MIC_conv_modules.append(nn.Dropout(
                    self.CNN_MIC_conv_Dropout_layers[index]).to(device))

        for index, i in enumerate(self.CNN_MIC_linear_layers):
            self.CNN_MIC_lin_modules.append(nn.Linear(i[0], i[1]).to(device))

            if self.CNN_MIC_linear_activations[index] == "sigmoid":
                self.CNN_MIC_lin_modules.append(nn.Sigmoid().to(device))
            elif self.CNN_MIC_linear_activations[index] == "relu":
                self.CNN_MIC_lin_modules.append(nn.ReLU().to(device))
            elif self.CNN_MIC_linear_activations[index] == "tanh":
                self.CNN_MIC_lin_modules.append(nn.Tanh().to(device))
            else:
                self.CNN_MIC_lin_modules.append(nn.Identity().to(device))

            if self.CNN_MIC_linear_BN_layers[index] != False:
                self.CNN_MIC_lin_modules.append(
                    nn.BatchNorm1d(self.CNN_MIC_linear_BN_layers[index]).to(device))

            if self.CNN_MIC_linear_Dropout_layers != False:
                self.CNN_MIC_lin_modules.append(nn.Dropout(
                    self.CNN_MIC_linear_Dropout_layers[index]).to(device))

        self.CNN_MIC_conv = nn.Sequential(*self.CNN_MIC_conv_modules).double()
        self.CNN_MIC_conv.to(device)
        self.CNN_MIC_lin = nn.Sequential(*self.CNN_MIC_lin_modules).double()
        self.CNN_MIC_lin.to(device)

        # %% CONVOLUTIONAL LAYER FOR FORCES
        for index, i in enumerate(self.CNN_FORCE_conv_layers):
            self.CNN_FORCE_conv_modules.append(
                nn.Conv1d(i[0], i[1], i[2], stride=i[3]))
            if self.CNN_FORCE_conv_MaxPoolLayers[index] != False:
                self.CNN_FORCE_conv_modules.append(nn.MaxPool1d(kernel_size=self.CNN_FORCE_conv_MaxPoolLayers[index][0],
                                                                stride=self.CNN_FORCE_conv_MaxPoolLayers[index][
                    0]).to(device))  # ?? What this value should be

            if self.CNN_FORCE_conv_layer_activations[index] == "sigmoid":
                self.CNN_FORCE_conv_modules.append(nn.Sigmoid().to(device))
            elif self.CNN_FORCE_conv_layer_activations[index] == "relu":
                self.CNN_FORCE_conv_modules.append(nn.ReLU().to(device))
            elif self.CNN_FORCE_conv_layer_activations[index] == "tanh":
                self.CNN_FORCE_conv_modules.append(nn.Tanh().to(device))
            else:
                self.CNN_FORCE_conv_modules.append(nn.Identity().to(device))

            if self.CNN_FORCE_conv_BN_layers[index] != False:
                self.CNN_FORCE_conv_modules.append(
                    nn.BatchNorm1d(self.CNN_FORCE_conv_BN_layers[index]).to(device))

            if self.CNN_FORCE_conv_Dropout_layers != False:
                self.CNN_FORCE_conv_modules.append(nn.Dropout(
                    self.CNN_FORCE_conv_Dropout_layers[index]).to(device))

        for index, i in enumerate(self.CNN_FORCE_linear_layers):
            self.CNN_FORCE_lin_modules.append(nn.Linear(i[0], i[1]).to(device))

            if self.CNN_FORCE_linear_activations[index] == "sigmoid":
                self.CNN_FORCE_lin_modules.append(nn.Sigmoid().to(device))
            elif self.CNN_FORCE_linear_activations[index] == "relu":
                self.CNN_FORCE_lin_modules.append(nn.ReLU().to(device))
            elif self.CNN_FORCE_linear_activations[index] == "tanh":
                self.CNN_FORCE_lin_modules.append(nn.Tanh().to(device))
            else:
                self.CNN_FORCE_lin_modules.append(nn.Identity().to(device))

            if self.CNN_FORCE_linear_BN_layers[index] != False:
                self.CNN_FORCE_lin_modules.append(
                    nn.BatchNorm1d(self.CNN_FORCE_linear_BN_layers[index]).to(device))

            if self.CNN_FORCE_linear_Dropout_layers != False:
                self.CNN_FORCE_lin_modules.append(nn.Dropout(
                    self.CNN_FORCE_linear_Dropout_layers[index]).to(device))

        self.CNN_Forces_conv = nn.Sequential(
            *self.CNN_FORCE_conv_modules).double()
        self.CNN_Forces_conv.to(device)
        self.CNN_Forces_lin = nn.Sequential(
            *self.CNN_FORCE_lin_modules).double()
        self.CNN_Forces_lin.to(device)

        self.input_size = self.CNN_FORCE_linear_layers[-1][1] + self.CNN_MIC_linear_layers[-1][1] \
            + self.CNN_AE_linear_layers[-1][1]

        self.hidden_size = LSTM_properties[0]
        self.num_layers = LSTM_properties[1]
        self.dropout = LSTM_properties[2]

        self.lstmcell1 = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=False,
                                 dropout=self.dropout).to(device)


        # the final output should be 17 data points.
        self.Dense1 = nn.Linear(self.hidden_size, 180).to(device)
        self.Dense1Act = nn.Tanh().to(device)
        # self.Dense2=nn.Linear(34,17) #It generates 17 data points

    @classmethod
    def from_hyperparams(cls, hyperparams: ModelHyperparams) -> LSTMModel:
        return cls(*hyperparams.properties)

    def forward(self, AE, Mic, Forces, tillChannelN):
        tillChannelN=tillChannelN.item()
        #  There will be sensor values from 1 to n and m number of features will be produced let`s say for LSTM, Therefore n x m
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print("Device:", device)
        AE_2d_Features = torch.empty(
            1, tillChannelN, self.CNN_AE_linear_layers[-1][1])
        AE_2d_Features.to(device)
        Mic_2d_Features = torch.empty(
            1, tillChannelN, self.CNN_MIC_linear_layers[-1][1])
        Mic_2d_Features.to(device)
        Forces_2d_Features = torch.empty(
            1, tillChannelN, self.CNN_FORCE_linear_layers[-1][1])
        Forces_2d_Features.to(device)
        LSTMFeature = torch.empty(1, tillChannelN, self.input_size)
        LSTMFeature.to(device)

        AE = AE.unsqueeze(dim=0)
        AE.to(device)
        Mic = Mic.unsqueeze(dim=0)
        Mic.to(device)

        for i in range(tillChannelN):
            # if ChannelN=20 , tensor dim = 20 X C_AE_Feature
            inp1 = torch.tensor(AE[0:1, i:i + 1, :].double())
            inp1.to(device)
            self.CNN_AE_conv.to(device)
            output = self.CNN_AE_conv(inp1)
            total_dim = output.size(1)*output.size(2)
            output = output.view(-1, total_dim)
            output = self.CNN_AE_lin(output)
            AE_2d_Features[0, i:i + 1, :] = output

            # if ChannelN=20 , tensor dim = 20 X C_MIC_Feature
            output = self.CNN_MIC_conv(Mic[0:1, i:i + 1, :])
            total_dim = output.size(1) * output.size(2)
            output = output.view(-1, total_dim)
            output = self.CNN_MIC_lin(output)
            Mic_2d_Features[0, i:i + 1, :] = output

            # for an input of size (N, C_in, L_in)
            output = self.CNN_Forces_conv(Forces[:, i, :].unsqueeze(dim=0))
            total_dim = output.size(1) * output.size(2)
            output = output.view(-1, total_dim)
            output = self.CNN_Forces_lin(output)
            Forces_2d_Features[0, i:i + 1, :] = output

            LSTMFeature[0:1, i:i + 1, :] = torch.cat((AE_2d_Features[0:1, i:i + 1, :], Mic_2d_Features[0:1, i:i + 1, :],
                                                      Forces_2d_Features[0:1, i:i + 1, :]), 2)


        # adds a 0-th dimension of size 1
        LSTMFeature = LSTMFeature.squeeze(dim=0)
        # adds a 0-th dimension of size 1
        LSTMFeature = LSTMFeature.unsqueeze(dim=1)
        # in a different source it says that Input must be 3 dimensional (Sequence len, batch, input dimensions)

        # print(LSTMFeature.shape)

        LSTMoutput, LSTMhidden = self.lstmcell1(LSTMFeature)  #

        # Output is output, h_n, c_n where : output (Seq_len,batch, num_directions*hidden_size),
        # if input is [batch_size, sequence_length, input_dim],LSTMFeature.size()=[1, 30, 100]
        # Output[0].size()=[1,30,300] output[1][0].size=[2,30,300], output[1][1].size=[2,30,300]
        # h_n(num_layer*num_directions,batch,hidden_size)

        OutputForceY = F.tanh(self.Dense1(LSTMoutput[-1, :, :]))

        return OutputForceY


def load_preconfigured_model() -> LSTMModel:
    """
    Creates a fully configured instance of `LSTMModel`.
    """

    hyperparams = ModelHyperparams(
        CNN_AE=CNNHyperparams(
            # Enter a list for each element [in_channels,out_channels,kernel_size,stride]:
            conv_layers=[[1, 3, 61, 20],
                         [3, 5, 23, 10],
                         [5, 8, 4, 2]],
            # Enter a list for each element [kernel_size,stride] or False if no pool layer:
            MaxPoolLayers=[False, False, [4, 2]],
            # Enter "sigmoid","tanh", "linear" or "relu", everything else="linear":
            conv_layer_activations=["relu", "relu", "relu"],
            # Enter False or num_features for BN, length should be equal to CNN_AE_layers:
            conv_BN_layers=[3, 5, False],
            # Enter False or propability of dropout:
            conv_Dropout_layers=[False, False, False],
            # Enter tuple (input_size,output_size):
            linear_layers=[(184, 40)],
            # Enter "sigmoid","tanh", "identity" or "relu":
            linear_activations=["identity"],
            linear_BN_layers=[False],
            linear_Dropout_layers=[False]
        ),

        CNN_MIC=CNNHyperparams(
            # Enter a list for each element [in_channels,out_channels,kernel_size,stride]:
            conv_layers=[[1, 3, 46, 10],
                         [3, 5, 14, 5],
                         [5, 8, 4, 2]],
            # Enter a list for each element [kernel_size,stride] or False if no pool layer:
            MaxPoolLayers=[False, False, [4, 2]],
            # Enter "sigmoid","tanh", "linear" or "relu", everything else="linear":
            conv_layer_activations=["relu", "relu", "relu"],
            # Enter False or num_features for BN, length should be equal to CNN_MIC_layers:
            conv_BN_layers=[3, 5, False],
            # Enter False or propability of dropout
            conv_Dropout_layers=[False, False, False],
            # Enter tuple (input_size,output_size):
            linear_layers=[(176, 40)],
            # Enter "sigmoid","tanh", "identity" or "relu":
            linear_activations=["identity"],
            linear_BN_layers=[False],
            linear_Dropout_layers=[False]
        ),

        CNN_FORCE=CNNHyperparams(
            # Enter a list for each element [in_channels,out_channels,kernel_size,stride]:
            conv_layers=[[3, 6, 7, 2],
                         [6, 15, 3, 1],
                         [15, 24, 4,
                          2]],
            # Enter a list for each element [kernel_size,stride] or False if no pool layer:
            MaxPoolLayers=[False, False, [4, 2]],
            # Enter "sigmoid","tanh", "linear" or "relu", everything else="linear":
            conv_layer_activations=["relu", "relu", "relu"],
            # Enter False or num_features for BN, length should be equal to CNN_FORCE_layers:
            conv_BN_layers=[6, 15, False],
            # Enter False or propability of dropout:
            conv_Dropout_layers=[False, False, False],
            # Enter tuple (input_size,output_size):
            linear_layers=[(240, 20)],
            # Enter "sigmoid","tanh", "identity" or "relu":
            linear_activations=["identity"],
            linear_BN_layers=[False],
            linear_Dropout_layers=[False]
        ),

        LSTM=LSTMHyperparams(
            hidden_size=300,
            nr_of_layers=2,
            dropout=0.2
        )
    )

    model = LSTMModel.from_hyperparams(hyperparams)


    return model
