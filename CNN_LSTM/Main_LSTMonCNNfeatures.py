import os
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from mat4py import loadmat
import matplotlib.pyplot as plt

# from dataset import FlowDataset
# from CNN_AE import CNNfeatureFromAE
# from CNN_MIC import CNNfeatureFromMic
# from CNN_FORCE  import CNNfeatureFromForce


from CNN_LSTM_MODEL import LSTMModel

import time


#%% Load .Mat File
# data = loadmat('LSTM//channel1Brass.mat')

#%% Iterate in the file, but channel1brass has to be changed to channel01brass for it to work properly

signalAE=[]
signalMic=[]
signalFx=[]
signalFy=[]
signalFz=[]


# Importing the glob library
import glob 
 
# Path to the directory
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

#%% Create Tensors!!
signalAE_T=torch.tensor(signalAE)
signalMic_T=torch.tensor(signalMic)
signalFx_T=torch.tensor(signalFx)
signalFy_T=torch.tensor(signalFy)
signalFz_T=torch.tensor(signalFz)

signalForces=torch.stack((signalFx_T,signalFy_T,signalFz_T))


#%%
def main():
    # check if cuda available
#%%
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu' #% it selects CPU check it out!!!
#%%
    # define dataset and dataloader
    # train_dataset = FlowDataset(mode='train')
    # test_dataset = FlowDataset(mode='test')
    # train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, num_workers=0)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    
#%% Example of data loading from Lalit`s code

    # train_data = np.load('processed_data_1120.npy', allow_pickle = True)
    # labels = np.load('processed_labels_1120.npy', allow_pickle = True)

    # split = 0.8

    # split_idx = int(train_data.shape[0]*split)

    # train_x = train_data[: split_idx]
    # train_y = labels[: split_idx]

    # val_x = train_data[split_idx:]
    # val_y = labels[split_idx:]

    # trainset = Dataset(train_x, train_y)
    # valset = Dataset(val_x, val_y)

    # batch_size = 32
    # train_loader = data.DataLoader(dataset = trainset, batch_size = batch_size, shuffle = True)
    # val_loader = data.DataLoader(dataset = valset, batch_size = batch_size, shuffle = False)





#%%
    # hyper-parameters for LSTM
    num_epochs = 3
    lr = 0.001
    input_size = 17 # do not change input size
    hidden_size = 128
    num_layers = 2
    dropout = 0.2
    
    #Hyper-parameters for CNN
    lr_AE = 0.0002 # AE learning rate
    lr_MIC = 0.0003 # Micropohne learning rate
    lr_Force = 0.0003 # Forces learning rate
    FeatureForAll=30;
    beta1 = 0.5 # beta1 value for Adam optimizer
    
    # CNN_AE = CNNfeatureFromAE(Feature_dim=FeatureForAll).to(device)
    # CNN_MIC = CNNfeatureFromMic(Feature_dim=FeatureForAll).to(device)
    # CNN_FORCE = CNNfeatureFromForce(Feature_dim=FeatureForAll).to(device)
    
    model = LSTMModel(cnnAE_Feature=40, cnnMIC_Feature=40,  cnnFORCE_Feature=20, hidden_size=300, num_layers=2, dropout=0.2)
    
    
    
    
    # model = FlowLSTM(
    #     input_size=input_size, 
    #     hidden_size=hidden_size, 
    #     num_layers=num_layers, 
    #     dropout=dropout
    # ).to(device)
    
    
    
#%%
    # define your LSTM loss function here
    # loss_func = ?
    
    criterion=nn.MSELoss()
    LossInEpoch=np.zeros(num_epochs)
    
    # define optimizer for lstm model
    optim = Adam(model.parameters(), lr=lr)
    
    # define optimizer for CNN feature models.
    
    # optim_AE = Adam(dis.parameters(), lr=lr_AE,betas=(beta1, 0.999))
    # optim_Mic = Adam(gen.parameters(), lr=lr_MIC,betas=(beta1, 0.999))
    # optim_Force = Adam(gen.parameters(), lr=lr_Force,betas=(beta1, 0.999))
    
    for epoch in range(num_epochs):
        start = time.time() #record run TIME

        # for n_batch, (in_batch, label) in enumerate(train_loader):
        #     in_batch, label = in_batch.to(device), label.to(device)  
        
        for i in range(5):
            
            # ForceX,ForceY,ForceZ,FFT_AE,FFT_MIC = in_batch.ForceX, in_batch.ForceY, in_batch.ForceZ,in_batch.FFTAE,in_batch.FFTMIC
           
            # AE_features=CNN_AE(FFT_AE)
            # Mic_features=CNN_MIC(FFT_MIC)            
            # # ForceX_features=CNN_FORCE(ForceX)           
            # # ForceY_features=CNN_FORCE(ForceY)           
            # # ForceZ_features=CNN_FORCE(ForceZ)       
            # #Vector 1 x  FeatureForAll is generated for all signals, same amount of features(30) so 1x30 vector size
            
            # Force_features = CNN_FORCE(signalForces)
            
            
            
            # LSTMinput=torch.cat((AE_features, Mic_features, Force_features), 0)
            
            # All features from CNN is concatenated to from a 5by30 vector for LSTM 
            tillTime=30+i*5
            outputs = model.forward(signalAE_T,signalMic_T,signalForces,tillTime)
            #OutputForceY.size() torch.Size([1, 125])
            loss = criterion(outputs,signalFy_T[tillTime+5,:])
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            print("loss is calculated as:",loss.item())
            # print loss while training
        
            # if (n_batch + 1) % 81 == 0:
            #     print("Epoch: [{}/{}], Batch: {}, Loss: {}".format(
            #         epoch, num_epochs, n_batch, loss.item()))
            #     LossInEpoch[epoch] =  loss.item()
            #     print("loss is saved in a list as:",LossInEpoch[epoch])

            
        end=time.time()
        print("for this training time passed:", end-start)
    
    #%%
    
    plt.figure()
    plt.plot( range(125),outputs.squeeze(dim=0).detach().numpy())
    plt.ylabel('Predicted')
    plt.show()
    
    plt.figure()
    plt.plot( range(125),signalFy[54])
    plt.ylabel('Real')
    plt.show()
    
    
    #%%    
    PATH = './LSTM.pt'
    model = torch.load(PATH)
    ##model.load_state_dict(torch.load(PATH,device))
    #%%
    # test trained LSTM model
    # l1_err, l2_err = 0, 0
    # l1_loss = nn.L1Loss()
    # l2_loss = nn.MSELoss()
    # model.eval()
    # with torch.no_grad():
    #     for n_batch, (in_batch, label) in enumerate(test_loader):
    #         in_batch, label = in_batch.to(device), label.to(device)
            
    #         pred = model.test(in_batch.to(device))

    #         l1_err += l1_loss(pred.to(device), label).item()
    #         l2_err += l2_loss(pred.to(device), label).item()

    # print("Test L1 error:", l1_err)
    # print("Test L2 error:", l2_err)


    # visualize the prediction comparing to the ground truth

    #%% Try Any Number from the test set of 10000 data
    # testID=60
    # if device is 'cpu':
    #     predforprint = pred.detach().numpy()[testID,:,:]
    #     labelforprint = label.detach().numpy()[testID,:,:]
    # else:
    #     predforprint = pred.detach().cpu().numpy()[testID,:,:]
    #     labelforprint = label.detach().cpu().numpy()[testID,:,:]

    # r = []
    # num_points = 17
    # interval = 1./num_points
    # x = int(num_points/2)
    # for j in range(-x,x+1):
    #     r.append(interval*j)

    # from matplotlib import pyplot as plt
    # plt.figure()
    # for i in range(1, len(predforprint)):
    #     c = (i/(num_points+1), 1-i/(num_points+1), 0.5)
    #     plt.plot(predforprint[i], r, label='t = %s' %(i), c=c)
    # plt.xlabel('velocity [m/s]')
    # plt.ylabel('r [m]')
    # plt.legend(bbox_to_anchor=(1,1),fontsize='x-small')
    # plt.show()

    # plt.figure()
    # for i in range(1, len(labelforprint)):
    #     c = (i/(num_points+1), 1-i/(num_points+1), 0.5)
    #     plt.plot(labelforprint[i], r, label='t = %s' %(i), c=c)
    # plt.xlabel('True velocity [m/s]')
    # plt.ylabel('r [m]')
    # plt.legend(bbox_to_anchor=(1,1),fontsize='x-small')
    # plt.show()


    # plt.figure()
    # plt.plot( range(num_epochs),LossInEpoch)
    # plt.xlabel('Number of Epoch')
    # plt.ylabel('Loss MSELoss')
    # plt.legend(bbox_to_anchor=(1,1),fontsize='x-small')
    # plt.show()
    
        
    # PATH = './LSTM.pt'
    
    # torch.save(model, PATH)
    
#%%    
if __name__ == "__main__":
    main()

