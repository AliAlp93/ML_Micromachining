# -*- coding: utf-8 -*-
"""
Created on Fri May  7 15:38:46 2021

@author: alialp(crazy_boi_93@gmail.com)
edited(2021_05_08): yusuf mert
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import glob
from mat4py import loadmat
import matplotlib.pyplot as plt


class LSTMModel(nn.Module):
    def __init__(self, CNN_AE_properties, CNN_MIC_properties, CNN_FORCE_properties, LSTM_properties):
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
        self.CNN_AE_conv_modules = []
        self.CNN_AE_lin_modules = []

        self.CNN_MIC_conv_layers = CNN_MIC_properties[0]
        self.CNN_MIC_conv_MaxPoolLayers = CNN_MIC_properties[1]
        self.CNN_MIC_conv_layer_activations = CNN_MIC_properties[2]
        self.CNN_MIC_conv_BN_layers = CNN_MIC_properties[3]
        self.CNN_MIC_conv_Dropout_layers = CNN_MIC_properties[4]
        self.CNN_MIC_linear_layers = CNN_MIC_properties[5]
        self.CNN_MIC_linear_activations = CNN_MIC_properties[6]
        self.CNN_MIC_linear_BN_layers = CNN_MIC_properties[7]
        self.CNN_MIC_linear_Dropout_layers = CNN_MIC_properties[8]
        self.CNN_MIC_conv_modules = []
        self.CNN_MIC_lin_modules = []

        self.CNN_FORCE_conv_layers = CNN_FORCE_properties[0]
        self.CNN_FORCE_conv_MaxPoolLayers = CNN_FORCE_properties[1]
        self.CNN_FORCE_conv_layer_activations = CNN_FORCE_properties[2]
        self.CNN_FORCE_conv_BN_layers = CNN_FORCE_properties[3]
        self.CNN_FORCE_conv_Dropout_layers = CNN_FORCE_properties[4]
        self.CNN_FORCE_linear_layers = CNN_FORCE_properties[5]
        self.CNN_FORCE_linear_activations = CNN_FORCE_properties[6]
        self.CNN_FORCE_linear_BN_layers = CNN_FORCE_properties[7]
        self.CNN_FORCE_linear_Dropout_layers = CNN_FORCE_properties[8]
        self.CNN_FORCE_conv_modules = []
        self.CNN_FORCE_lin_modules = []

        for index, i in enumerate(self.CNN_AE_conv_layers):
            self.CNN_AE_conv_modules.append(nn.Conv1d(i[0], i[1], i[2], stride=i[3]))
            if self.CNN_AE_conv_MaxPoolLayers[index] != False:
                self.CNN_AE_conv_modules.append(nn.MaxPool1d(kernel_size=self.CNN_AE_conv_MaxPoolLayers[index][0],
                                                        stride=self.CNN_AE_conv_MaxPoolLayers[index][
                                                            0]))  # ?? What this value should be

            if self.CNN_AE_conv_layer_activations[index] == "sigmoid":
                self.CNN_AE_conv_modules.append(nn.Sigmoid())
            elif self.CNN_AE_conv_layer_activations[index] == "relu":
                self.CNN_AE_conv_modules.append(nn.ReLU())
            elif self.CNN_AE_conv_layer_activations[index] == "tanh":
                self.CNN_AE_conv_modules.append(nn.Tanh())
            else:
                self.CNN_AE_conv_modules.append(nn.Identity())

            if self.CNN_AE_conv_BN_layers[index] != False:
                self.CNN_AE_conv_modules.append(nn.BatchNorm1d(self.CNN_AE_conv_BN_layers[index]))

            if self.CNN_AE_conv_Dropout_layers != False:
                self.CNN_AE_conv_modules.append(nn.Dropout(self.CNN_AE_conv_Dropout_layers[index]))

        for index, i in enumerate(self.CNN_AE_linear_layers):
            self.CNN_AE_lin_modules.append(nn.Linear(i[0], i[1]))

            if self.CNN_AE_linear_activations[index] == "sigmoid":
                self.CNN_AE_lin_modules.append(nn.Sigmoid())
            elif self.CNN_AE_linear_activations[index] == "relu":
                self.CNN_AE_lin_modules.append(nn.ReLU())
            elif self.CNN_AE_linear_activations[index] == "tanh":
                self.CNN_AE_lin_modules.append(nn.Tanh())
            else:
                self.CNN_AE_lin_modules.append(nn.Identity())

            if self.CNN_AE_linear_BN_layers[index] != False:
                self.CNN_AE_lin_modules.append(nn.BatchNorm1d(self.CNN_AE_linear_BN_layers[index]))

            if self.CNN_AE_linear_Dropout_layers != False:
                self.CNN_AE_lin_modules.append(nn.Dropout(self.CNN_AE_linear_Dropout_layers[index]))

        self.CNN_AE_conv = nn.Sequential(*self.CNN_AE_conv_modules)
        self.CNN_AE_lin = nn.Sequential(*self.CNN_AE_lin_modules)

        for index, i in enumerate(self.CNN_MIC_conv_layers):
            self.CNN_MIC_conv_modules.append(nn.Conv1d(i[0], i[1], i[2], stride=i[3]))
            if self.CNN_MIC_conv_MaxPoolLayers[index] != False:
                self.CNN_MIC_conv_modules.append(nn.MaxPool1d(kernel_size=self.CNN_MIC_conv_MaxPoolLayers[index][0],
                                                         stride=self.CNN_MIC_conv_MaxPoolLayers[index][
                                                             0]))  # ?? What this value should be

            if self.CNN_MIC_conv_layer_activations[index] == "sigmoid":
                self.CNN_MIC_conv_modules.append(nn.Sigmoid())
            elif self.CNN_MIC_conv_layer_activations[index] == "relu":
                self.CNN_MIC_conv_modules.append(nn.ReLU())
            elif self.CNN_MIC_conv_layer_activations[index] == "tanh":
                self.CNN_MIC_conv_modules.append(nn.Tanh())
            else:
                self.CNN_MIC_conv_modules.append(nn.Identity())

            if self.CNN_MIC_conv_BN_layers[index] != False:
                self.CNN_MIC_conv_modules.append(nn.BatchNorm1d(self.CNN_MIC_conv_BN_layers[index]))

            if self.CNN_MIC_conv_Dropout_layers != False:
                self.CNN_MIC_conv_modules.append(nn.Dropout(self.CNN_MIC_conv_Dropout_layers[index]))

        for index, i in enumerate(self.CNN_MIC_linear_layers):
            self.CNN_MIC_lin_modules.append(nn.Linear(i[0], i[1]))

            if self.CNN_MIC_linear_activations[index] == "sigmoid":
                self.CNN_MIC_lin_modules.append(nn.Sigmoid())
            elif self.CNN_MIC_linear_activations[index] == "relu":
                self.CNN_MIC_lin_modules.append(nn.ReLU())
            elif self.CNN_MIC_linear_activations[index] == "tanh":
                self.CNN_MIC_lin_modules.append(nn.Tanh())
            else:
                self.CNN_MIC_lin_modules.append(nn.Identity())

            if self.CNN_MIC_linear_BN_layers[index] != False:
                self.CNN_MIC_lin_modules.append(nn.BatchNorm1d(self.CNN_MIC_linear_BN_layers[index]))

            if self.CNN_MIC_linear_Dropout_layers != False:
                self.CNN_MIC_lin_modules.append(nn.Dropout(self.CNN_MIC_linear_Dropout_layers[index]))

        self.CNN_MIC_conv = nn.Sequential(*self.CNN_MIC_conv_modules)
        self.CNN_MIC_lin = nn.Sequential(*self.CNN_MIC_lin_modules)

        # %% CONVOLUTIONAL LAYER FOR FORCES
        for index, i in enumerate(self.CNN_FORCE_conv_layers):
            self.CNN_FORCE_conv_modules.append(nn.Conv1d(i[0], i[1], i[2], stride=i[3]))
            if self.CNN_FORCE_conv_MaxPoolLayers[index] != False:
                self.CNN_FORCE_conv_modules.append(nn.MaxPool1d(kernel_size=self.CNN_FORCE_conv_MaxPoolLayers[index][0],
                                                           stride=self.CNN_FORCE_conv_MaxPoolLayers[index][
                                                               0]))  # ?? What this value should be

            if self.CNN_FORCE_conv_layer_activations[index] == "sigmoid":
                self.CNN_FORCE_conv_modules.append(nn.Sigmoid())
            elif self.CNN_FORCE_conv_layer_activations[index] == "relu":
                self.CNN_FORCE_conv_modules.append(nn.ReLU())
            elif self.CNN_FORCE_conv_layer_activations[index] == "tanh":
                self.CNN_FORCE_conv_modules.append(nn.Tanh())
            else:
                self.CNN_FORCE_conv_modules.append(nn.Identity())

            if self.CNN_FORCE_conv_BN_layers[index] != False:
                self.CNN_FORCE_conv_modules.append(nn.BatchNorm1d(self.CNN_FORCE_conv_BN_layers[index]))

            if self.CNN_FORCE_conv_Dropout_layers != False:
                self.CNN_FORCE_conv_modules.append(nn.Dropout(self.CNN_FORCE_conv_Dropout_layers[index]))

        for index, i in enumerate(self.CNN_FORCE_linear_layers):
            self.CNN_FORCE_lin_modules.append(nn.Linear(i[0], i[1]))

            if self.CNN_FORCE_linear_activations[index] == "sigmoid":
                self.CNN_FORCE_lin_modules.append(nn.Sigmoid())
            elif self.CNN_FORCE_linear_activations[index] == "relu":
                self.CNN_FORCE_lin_modules.append(nn.ReLU())
            elif self.CNN_FORCE_linear_activations[index] == "tanh":
                self.CNN_FORCE_lin_modules.append(nn.Tanh())
            else:
                self.CNN_FORCE_lin_modules.append(nn.Identity())

            if self.CNN_FORCE_linear_BN_layers[index] != False:
                self.CNN_FORCE_lin_modules.append(nn.BatchNorm1d(self.CNN_FORCE_linear_BN_layers[index]))

            if self.CNN_FORCE_linear_Dropout_layers != False:
                self.CNN_FORCE_lin_modules.append(nn.Dropout(self.CNN_FORCE_linear_Dropout_layers[index]))


        self.CNN_Forces_conv = nn.Sequential(*self.CNN_FORCE_conv_modules)
        self.CNN_Forces_lin = nn.Sequential(*self.CNN_FORCE_lin_modules)



        self.input_size = self.CNN_FORCE_linear_layers[-1][1] + self.CNN_MIC_linear_layers[-1][1] \
                          + self.CNN_AE_linear_layers[-1][1]

        self.hidden_size = LSTM_properties[0]
        self.num_layers = LSTM_properties[1]
        self.dropout = LSTM_properties[2]

        self.lstmcell1 = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=False,
                                 dropout=self.dropout)

        self.Dense1 = nn.Linear(self.hidden_size, 125)  # the final output should be 17 data points.
        self.Dense1Act = nn.Tanh()
        # self.Dense2=nn.Linear(34,17) #It generates 17 data points

    # %%
    # lstm_input_size is determined by the number of features we get from self.features()
    # self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden_size,num_layers=1, batch_first=True)
    # self.classifier = nn.Linear(32, 2)

    # self.optimizer = optim.Adam(self.parameters())
    # self.criterion = nn.NLLLoss()

    def forward(self, AE, Mic, Forces, tillTimeT):

        #  There will be sensor values from 1 to n and m number of features will be produced let`s say for LSTM, Therefore n x m 

        AE_2d_Features = torch.empty(1,tillTimeT, self.CNN_AE_linear_layers[-1][1])
        Mic_2d_Features = torch.empty(1,tillTimeT, self.CNN_MIC_linear_layers[-1][1])
        Forces_2d_Features = torch.empty(1,tillTimeT, self.CNN_FORCE_linear_layers[-1][1])
        LSTMFeature = torch.empty(1,tillTimeT, self.input_size)

        AE = AE.unsqueeze(dim=0)
        Mic = Mic.unsqueeze(dim=0)

        for i in range(tillTimeT):
            output = self.CNN_AE_conv(AE[0:1, i:i + 1, :])  # if T=20 , tensor dim = 20 X C_AE_Feature
            total_dim= output.size(1)*output.size(2)
            output = output.view(-1, total_dim)
            output = self.CNN_AE_lin(output)
            AE_2d_Features[0, i:i + 1, :] = output


            output = self.CNN_MIC_conv(Mic[0:1, i:i + 1, :])  # if T=20 , tensor dim = 20 X C_MIC_Feature
            total_dim = output.size(1) * output.size(2)
            output = output.view(-1, total_dim)
            output = self.CNN_MIC_lin(output)
            Mic_2d_Features[0, i:i + 1, :] = output

            output = self.CNN_Forces_conv(Forces[:, i, :].unsqueeze(dim=0))  # for an input of size (N, C_in, L_in)
            total_dim = output.size(1) * output.size(2)
            output = output.view(-1, total_dim)
            output = self.CNN_Forces_lin(output)
            Forces_2d_Features[0, i:i + 1, :] = output


            LSTMFeature[0:1, i:i + 1, :] = torch.cat((AE_2d_Features[0:1, i:i + 1, :], Mic_2d_Features[0:1, i:i + 1, :],
                                                      Forces_2d_Features[0:1, i:i + 1, :]), 2)

        LSTMFeature = LSTMFeature.squeeze(dim=0)  # adds a 0-th dimension of size 1
        LSTMFeature = LSTMFeature.unsqueeze(dim=1)  # adds a 0-th dimension of size 1
        # in a different source it says that Input must be 3 dimensional (Sequence len, batch, input dimensions)

        # print(LSTMFeature.shape)

        LSTMoutput, LSTMhidden = self.lstmcell1(LSTMFeature)  #

        # Output is output, h_n, c_n where : output (Seq_len,batch, num_directions*hidden_size),
        # if input is [batch_size, sequence_length, input_dim],LSTMFeature.size()=[1, 30, 100]
        # Output[0].size()=[1,30,300] output[1][0].size=[2,30,300], output[1][1].size=[2,30,300]
        # h_n(num_layer*num_directions,batch,hidden_size)

        OutputForceY = F.tanh(self.Dense1(LSTMoutput[-1, :, :]))

        return OutputForceY


def main():

    # General Structure: Convolutions(Conv->MaxPool->Activation->BN->Dropout) + Linears(LinLayer->Activation->BN->Dropout)
    CNN_AE_conv_layers = [[1, 3, 61, 20],
                          [3, 5, 23, 10],
                          [5, 8, 4, 2]]  # Enter a list for each element [in_channels,out_channels,kernel_size,stride]
    CNN_AE_MaxPoolLayers = [False, False,
                            [4, 2]]  # Enter a list for each element [kernel_size,stride] or False if no pool layer
    CNN_AE_conv_layer_activations = ["relu",
                                     "relu",
                                     "relu"]  # Enter "sigmoid","tanh", "linear" or "relu", everything else="linear"
    CNN_AE_conv_BN_layers = [3, 5, False]  # Enter False or num_features for BN, length should be equal to CNN_AE_layers
    CNN_AE_conv_Dropout_layers = [False, False, False]  # Enter False or propability of dropout
    CNN_AE_linear_layers = [[184, 40]]  # Enter tuple (input_size,output_size)
    CNN_AE_linear_activations = ["identity"]  # Enter "sigmoid","tanh", "identity" or "relu"
    CNN_AE_linear_BN_layers = [False]
    CNN_AE_linear_Dropout_layers = [False]

    CNN_AE_properties = [CNN_AE_conv_layers,
                         CNN_AE_MaxPoolLayers,
                         CNN_AE_conv_layer_activations,
                         CNN_AE_conv_BN_layers,
                         CNN_AE_conv_Dropout_layers,
                         CNN_AE_linear_layers,
                         CNN_AE_linear_activations,
                         CNN_AE_linear_BN_layers,
                         CNN_AE_linear_Dropout_layers]  #

    CNN_MIC_conv_layers = [[1, 3, 46, 10],
                           [3, 5, 14, 5],
                           [5, 8, 4, 2]]  # Enter a list for each element [in_channels,out_channels,kernel_size,stride]
    CNN_MIC_MaxPoolLayers = [False, False,
                             [4, 2]]  # Enter a list for each element [kernel_size,stride] or False if no pool layer
    CNN_MIC_conv_layer_activations = ["relu",
                                      "relu",
                                      "relu"]  # Enter "sigmoid","tanh", "linear" or "relu", everything else="linear"
    CNN_MIC_conv_BN_layers = [3, 5,
                              False]  # Enter False or num_features for BN, length should be equal to CNN_MIC_layers
    CNN_MIC_conv_Dropout_layers = [False, False, False]  # Enter False or propability of dropout
    CNN_MIC_linear_layers = [[176, 40]]  # Enter tuple (input_size,output_size)
    CNN_MIC_linear_activations = ["identity"]  # Enter "sigmoid","tanh", "identity" or "relu"
    CNN_MIC_linear_BN_layers = [False]
    CNN_MIC_linear_Dropout_layers = [False]

    CNN_MIC_properties = [CNN_MIC_conv_layers,
                          CNN_MIC_MaxPoolLayers,
                          CNN_MIC_conv_layer_activations,
                          CNN_MIC_conv_BN_layers,
                          CNN_MIC_conv_Dropout_layers,
                          CNN_MIC_linear_layers,
                          CNN_MIC_linear_activations,
                          CNN_MIC_linear_BN_layers,
                          CNN_MIC_linear_Dropout_layers]

    CNN_FORCE_conv_layers = [[3, 6, 7, 2],
                             [6, 15, 3, 1],
                             [15, 24, 4,
                              2]]  # Enter a list for each element [in_channels,out_channels,kernel_size,stride]
    CNN_FORCE_MaxPoolLayers = [False, False,
                               [4, 2]]  # Enter a list for each element [kernel_size,stride] or False if no pool layer
    CNN_FORCE_conv_layer_activations = ["relu",
                                        "relu",
                                        "relu"]  # Enter "sigmoid","tanh", "linear" or "relu", everything else="linear"
    CNN_FORCE_conv_BN_layers = [6, 15,
                                False]  # Enter False or num_features for BN, length should be equal to CNN_FORCE_layers
    CNN_FORCE_conv_Dropout_layers = [False, False, False]  # Enter False or propability of dropout
    CNN_FORCE_linear_layers = [[168, 20]]  # Enter tuple (input_size,output_size)
    CNN_FORCE_linear_activations = ["identity"]  # Enter "sigmoid","tanh", "identity" or "relu"
    CNN_FORCE_linear_BN_layers = [False]
    CNN_FORCE_linear_Dropout_layers = [False]

    CNN_FORCE_properties = [CNN_FORCE_conv_layers,
                            CNN_FORCE_MaxPoolLayers,
                            CNN_FORCE_conv_layer_activations,
                            CNN_FORCE_conv_BN_layers,
                            CNN_FORCE_conv_Dropout_layers,
                            CNN_FORCE_linear_layers,
                            CNN_FORCE_linear_activations,
                            CNN_FORCE_linear_BN_layers,
                            CNN_FORCE_linear_Dropout_layers]

    lstm_hidden_size = 300
    lstm_nr_of_layers = 2
    dropout = 0.2
    num_epochs = 3
    lr = 0.001

    lstm_properties = [lstm_hidden_size, lstm_nr_of_layers, dropout]

    model = LSTMModel(CNN_AE_properties, CNN_MIC_properties, CNN_FORCE_properties, lstm_properties)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    signalAE = []
    signalMic = []
    signalFx = []
    signalFy = []
    signalFz = []

    path = path = './LSTM/'

    # or
    # path = './'

    # Extract the list of filenames
    files = glob.glob(path + '*', recursive=True)

    # Loop to print the filenames
    for filename in files:
        data = loadmat(filename)

        signalAE.append(data["LSTMinput"][0][0]["AE_FFT"][0])
        signalMic.append(data["LSTMinput"][0][0]["Mic_FFT"][0])
        signalFx.append(data["LSTMinput"][0][0]["statFx"][0][:125])
        signalFy.append(data["LSTMinput"][0][0]["statFy"][0][:125])
        signalFz.append(data["LSTMinput"][0][0]["statFz"][0][:125])

        print(filename)

    signalAE_T = torch.tensor(signalAE)
    signalMic_T = torch.tensor(signalMic)
    signalFx_T = torch.tensor(signalFx)
    signalFy_T = torch.tensor(signalFy)
    signalFz_T = torch.tensor(signalFz)

    signalForces = torch.stack((signalFx_T, signalFy_T, signalFz_T))


    for epoch in range(num_epochs):
        start = time.time() #record run TIME

        for i in range(5):

            tillTime = 30 + i * 5
            outputs = model.forward(signalAE_T, signalMic_T, signalForces, tillTime)
            # OutputForceY.size() torch.Size([1, 125])
            loss = criterion(outputs, signalFy_T[tillTime + 5, :])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("loss is calculated as:", loss.item())


        end = time.time()
        print("for this training time passed:", end - start)

    plt.figure()
    plt.plot(range(125), outputs.squeeze(dim=0).detach().numpy())
    plt.ylabel('Predicted')
    plt.show()

    plt.figure()
    plt.plot(range(125), signalFy[54])
    plt.ylabel('Real')
    plt.show()



if __name__ == "__main__":
    main()
