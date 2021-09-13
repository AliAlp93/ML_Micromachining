# -*- coding: utf-8 -*-
"""
Created on Fri May  7 15:38:46 2021

@author: alialp
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LSTMModel(nn.Module):
    def __init__(self, cnnAE_Feature, cnnMIC_Feature,  cnnFORCE_Feature, hidden_size, num_layers, dropout):
        super(LSTMModel, self).__init__()
#%% Convolution Layer for the ACOUSTIC EMISSION
        self.C_FORCE_Feature=cnnFORCE_Feature
        self.C_MIC_Feature=cnnMIC_Feature
        self.C_AE_Feature=cnnAE_Feature 
        #start with 37501 points
        
        self.CNN_AE_features = nn.Sequential(
                   
            nn.Conv1d(1, 3, 61, stride=20),   #1873 points in 3 channels
            nn.BatchNorm1d(3), 
            nn.ReLU(),
            nn.Conv1d(3, 5, 23 , stride=10), # 186 points in 5 channels
            nn.ReLU(),
            nn.BatchNorm1d(5),
       
               
            nn.Conv1d(5, 8, 4 , stride=2), #92 points in 8 channels 
            nn.ReLU(), 
       
            nn.AvgPool1d(kernel_size=4,stride=2), # 45 points averaged!!

        # self.Dense1=nn.Linear(12*18,120)  # Number of channels x Points in the channels if no Avgpool is applied. 
        # self.Dense1Act=nn.ReLU()

            # nn.Linear(45,self.C_AE_Feature),  # Number of channels x Points in the channels if no Avgpool is applied. 
            # nn.ReLU()
        )
        
        self.cnnAE_dense=nn.Linear(360,self.C_AE_Feature)  # Number of channels x Points in the channels if no Avgpool is applied. 
        
#%% Convolutional Layer For Microphone

        self.CNN_Mic_features = nn.Sequential(
            #start with 9376 points
            nn.Conv1d(1, 3, 46, stride=10),   #934 points in 3 channels
            nn.BatchNorm1d(3), #?? What this value should be
            nn.ReLU(),
            nn.Conv1d(3, 5, 14 , stride=5),# 186 points in 5 channels
            nn.ReLU(),
            nn.BatchNorm1d(5),
       
               
            nn.Conv1d(5, 8, 3, stride=2), #92 points in 8 channels
            nn.ReLU(), 
       
            nn.AvgPool1d(kernel_size=4,stride=2), # 45 points averaged!!

        # self.Dense1=nn.Linear(12*18,120)  # Number of channels x Points in the channels if no Avgpool is applied. 
        # self.Dense1Act=nn.ReLU()
        # 45 x 8 = 360

        )
        self.cnnMIC_dense=nn.Linear(360,self.C_MIC_Feature)  # Number of channels x Points in the channels if no Avgpool is applied. 
#%% CONVOLUTIONAL LAYER FOR FORCES
        self.CNN_Forces_features = nn.Sequential(
            #start with 125 points
            nn.Conv1d(3, 6, 7, stride=2),   #60 points in 3 channels
            nn.BatchNorm1d(6), #?? What this value should be
            nn.ReLU(),
            nn.Conv1d(6, 15, 3 , stride=1), # 58 points in 5 channels
            nn.ReLU(),
            nn.BatchNorm1d(15),
       
               
            nn.Conv1d(15, 24, 4 , stride=2), #28 points in 8 channels 
            nn.ReLU(), 
       
            nn.AvgPool1d(kernel_size=4,stride=2), # 13 points averaged!!

            # THIS MIGHT BE MORE APPROPRIATE compared to Average pooling here but how to implement it ???
            # self.Dense1=nn.Linear(12*18,120)  # Number of channels x Points in the channels if no Avgpool is applied. 
            # self.Dense1Act=nn.ReLU()
            #output=output.view(-1,12*18) #out = out.view(out.size(0), -1) CHANNEL X NUMBER OF POINTS IN EACH CHANNEL IS  FED TO DENSE LAYER
            
            
            # nn.Linear(21,self.C_FORCE_Feature)  # Number of channels x Points in the channels if no Avgpool is applied. 
            # nn.ReLU()
        )
        
        self.cnnForce_dense=nn.Linear(24*13,self.C_FORCE_Feature)  # Number of channels x Points in the channels if no Avgpool is applied. 

        self.input_size=self.C_FORCE_Feature+self.C_MIC_Feature+self.C_AE_Feature
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.dropout=dropout
        
        self.lstmcell1=nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=False, dropout= self.dropout)
        
        self.Dense1=nn.Linear(self.hidden_size,125)  #the final output should be 17 data points.
        self.Dense1Act=nn.Sigmoid()
        # self.Dense2=nn.Linear(34,17) #It generates 17 data points
        pass
#%%
        # lstm_input_size is determined by the number of features we get from self.features()
        # self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden_size,num_layers=1, batch_first=True)
        # self.classifier = nn.Linear(32, 2)

        # self.optimizer = optim.Adam(self.parameters())
        # self.criterion = nn.NLLLoss()

    def forward(self, AE,Mic,Forces,tillTimeT):

        #  There will be sensor values from 1 to n and m number of features will be produced let`s say for LSTM, Therefore n x m 
        
        AE_2d_Features=torch.empty(1,tillTimeT,self.C_AE_Feature)
        Mic_2d_Features=torch.empty(1,tillTimeT,self.C_MIC_Feature)       
        Forces_2d_Features=torch.empty(1,tillTimeT,self.C_FORCE_Feature)  
        LSTMFeature=torch.empty(1,tillTimeT,self.input_size) 
        
        AE=AE.unsqueeze(dim=0)
        Mic=Mic.unsqueeze(dim=0)
        #Forces=Forces.unsqueeze(dim=0)
        
        for i in range(tillTimeT):
            output=self.CNN_AE_features(AE[0:1,i:i+1,:]) # if T=20 , tensor dim = 20 X C_AE_Feature
            output=output.view(-1,360)
            output=F.relu(self.cnnAE_dense(output))
            AE_2d_Features[0:1,i:i+1,:]=output
            
            output=self.CNN_Mic_features(Mic[0:1,i:i+1,:])  # if T=20 , tensor dim = 20 X C_MIC_Feature  
            output=output.view(-1,360)
            output=F.relu(self.cnnMIC_dense(output))
            Mic_2d_Features[0:1,i:i+1,:]=output
            
            
            output=self.CNN_Forces_features(Forces[:,i,:].unsqueeze(dim=0)) # for an input of size (N, C_in, L_in)
            output=output.view(-1,24*13)
            output=F.relu(self.cnnForce_dense(output))
            Forces_2d_Features[0:1,i:i+1,:]=output
            
            
            
            LSTMFeature[0:1,i:i+1,:]=torch.cat(( AE_2d_Features[0:1,i:i+1,:], Mic_2d_Features[0:1,i:i+1,:], Forces_2d_Features[0:1,i:i+1,:]), 2)

                
        # LSTMFeature=torch.cat((AE_2d_Features, Mic_2d_Features, Forces_2d_Features), 1)   That might work too.
        #print(LSTMFeature.shape)
        LSTMFeature = LSTMFeature.squeeze(dim=0) # adds a 0-th dimension of size 1
        LSTMFeature = LSTMFeature.unsqueeze(dim=1) # adds a 0-th dimension of size 1
        #in a different source it says that Input must be 3 dimensional (Sequence len, batch, input dimensions)

        #print(LSTMFeature.shape)
        
        LSTMoutput,LSTMhidden =self.lstmcell1(LSTMFeature)  # 
        
        #Output is output, h_n, c_n where : output (Seq_len,batch, num_directions*hidden_size), 
        #if input is [batch_size, sequence_length, input_dim],LSTMFeature.size()=[1, 30, 100]
        # Output[0].size()=[1,30,300] output[1][0].size=[2,30,300], output[1][1].size=[2,30,300] 
        #h_n(num_layer*num_directions,batch,hidden_size)
        
        OutputForceY=F.tanh(self.Dense1(LSTMoutput[-1,:,:]))

        
        return OutputForceY