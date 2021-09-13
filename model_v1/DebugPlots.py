# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 15:48:47 2021

@author: alial
"""
import matplotlib.pyplot as plt  # type: ignore
from data import MachiningDataset
import numpy as np
from torch.utils.data import DataLoader
from model import LSTMModel

    
    
def DebugPlots(dataset,ExperimentNumber,ChannelNumber):
    
    ae_FFT=np.array(dataset.examples[ExperimentNumber].stack.signalAE.numpy()) 
    mic_FFT=dataset.examples[ExperimentNumber].stack.signalMic.numpy()
    
    Fx=dataset.examples[ExperimentNumber].stack.signalFx.numpy()
    Fy=dataset.examples[ExperimentNumber].stack.signalFy.numpy()
    Fz=dataset.examples[ExperimentNumber].stack.signalFz.numpy()
    
    Sa=dataset.examples[ExperimentNumber].Roughness.numpy()
       
     
    ExperimentInfo=  str(dataset.examples[ExperimentNumber].spindle_speed) +"k "+str(dataset.examples[ExperimentNumber].feed_rate) +"mm "+str(dataset.examples[ExperimentNumber].depth) +"um "
    print(ExperimentInfo)
    
    fig, ax = plt.subplots()
    Ae_lenth=len(ae_FFT[ChannelNumber,:])
    ax.plot(np.linspace(0, Ae_lenth-1, Ae_lenth), ae_FFT[ChannelNumber,:] ,linewidth=3.0) #Correct the dimensions!!!  
     
    ax.set(xlabel='FFT bins', ylabel=' amplitude ',
           title=ExperimentInfo+'for AE')
        
    ax.grid()
     
    # fig.savefig("test.png")
    plt.show()
     
     
    fig, ax = plt.subplots()
    Mic_lenth=len(mic_FFT[ChannelNumber,:])
    ax.plot(np.linspace(0, Mic_lenth-1, Mic_lenth), mic_FFT[ChannelNumber,:] ,linewidth=3.0) #Correct the dimensions!!!  
     
    ax.set(xlabel='FFT bins', ylabel=' amplitude ',
           title=ExperimentInfo+'for MIC')
        
    ax.grid() 
    # fig.savefig("test.png")
    plt.show()
    
    
     
    fig, ax = plt.subplots()
    F_length=len(Fy[ChannelNumber,:])
    ax.plot(np.linspace(0, F_length-1, F_length), Fy[ChannelNumber,:] ,linewidth=3.0) #Correct the dimensions!!!  
     
    ax.set(xlabel='angle bins=2', ylabel=' Force(N) ',
           title=ExperimentInfo+'for Fy')
        
    ax.grid() 
    # fig.savefig("test.png")
    plt.show()
     
    fig, ax = plt.subplots()
    
    ax.plot(np.linspace(0, len(Sa)-1, len(Sa)), Sa ,linewidth=3.0) #Correct the dimensions!!!  
     
    ax.set(xlabel='Channel Numbers', ylabel=' Force(N) ',
           title=str(ExperimentInfo)+'for Sa')
        
    ax.grid() 
    # fig.savefig("test.png")
    plt.show()
     
     
    return     

def ChannelPeaktoPeakForce(dataset, ExperimentNumber):
    
    Fx=dataset.examples[ExperimentNumber].stack.signalFx.numpy()
    Fy=dataset.examples[ExperimentNumber].stack.signalFy.numpy()
    Fz=dataset.examples[ExperimentNumber].stack.signalFz.numpy()
    
    ExperimentInfo=  str(dataset.examples[ExperimentNumber].spindle_speed) +"k "+str(dataset.examples[ExperimentNumber].feed_rate) +"mm "+str(dataset.examples[ExperimentNumber].depth) +"um "
    print(ExperimentInfo)
    
    FxPtP = np.empty((len(Fx), 3))
    FyPtP=np.empty((len(Fx), 3))
    FzPtP=np.empty((len(Fx), 3))
    
    for i in range(56):
    
        FxPtP[i,0]=max(Fx[i,:])  #Fx[i,:].argmax() if index number is needed
        FxPtP[i,1]=min(Fx[i,:])
        FxPtP[i,2]=FxPtP[i,0]-FxPtP[i,1]
    
        FyPtP[i,0]=max(Fy[i,:])
        FyPtP[i,1]=min(Fy[i,:])
        FyPtP[i,2]=FyPtP[i,0]-FyPtP[i,1]    
        
        FzPtP[i,0]=max(Fz[i,:])
        FzPtP[i,1]=min(Fz[i,:])
        FzPtP[i,2]=FzPtP[i,0]-FzPtP[i,1]
    
    
    
    fig, ax = plt.subplots()
    
    F_length=len(FyPtP)
    ax.plot(np.linspace(0, F_length-1, F_length), FyPtP[:,2] ,linewidth=3.0) #Correct the dimensions!!!  
     
    ax.set(xlabel='channel Number', ylabel=' Force(N) ',
           title= 'Fy peak to peak for'+str(ExperimentInfo))
        
    ax.grid() 
    # fig.savefig("test.png")
    plt.show()
       
    
    
    
    return 



def AllPeaktoPeakForce(dataset,range1,range2):
    
    
    ExperimentNames= []
    
    fig, ax = plt.subplots()

    for ExperimentNumber in range(range1,range2): #len(dataset.examples)
    
        Fx=dataset.examples[ExperimentNumber].stack.signalFx.numpy()
        Fy=dataset.examples[ExperimentNumber].stack.signalFy.numpy()
        Fz=dataset.examples[ExperimentNumber].stack.signalFz.numpy()
    
        ExperimentInfo=  str(dataset.examples[ExperimentNumber].spindle_speed) +"k "+str(dataset.examples[ExperimentNumber].feed_rate) +"mm "+str(dataset.examples[ExperimentNumber].depth) +"um "
        print(ExperimentInfo)
    
        FxPtP = np.empty((len(Fx), 3))
        FyPtP=np.empty((len(Fx), 3))
        FzPtP=np.empty((len(Fx), 3))
    
        for i in range(56):
        
            FxPtP[i,0]=max(Fx[i,:])  #Fx[i,:].argmax() if index number is needed
            FxPtP[i,1]=min(Fx[i,:])
            FxPtP[i,2]=FxPtP[i,0]-FxPtP[i,1]
            
            FyPtP[i,0]=max(Fy[i,:])
            FyPtP[i,1]=min(Fy[i,:])
            FyPtP[i,2]=FyPtP[i,0]-FyPtP[i,1]    
            
            FzPtP[i,0]=max(Fz[i,:])
            FzPtP[i,1]=min(Fz[i,:])
            FzPtP[i,2]=FzPtP[i,0]-FzPtP[i,1]
    
    
        F_length=len(FyPtP)
   
        ax.plot(np.linspace(0, F_length-1, F_length), FyPtP[:,2] ,linewidth=3.0) #Correct the dimensions!!!  
        
        ax.set(xlabel='channel Number', ylabel=' Force(N) ',
        title= 'Fy peak to peak for all')
        
        
        # ax.legend(['A simple line'])
        ExperimentNames.append(ExperimentInfo)
        
        
    ax.grid()
    plt.legend(ExperimentNames);
        # fig.savefig("test.png")
    plt.show()    
    
    
    
    
    return 

def PlotResults(dataset,model):
    
    train_loader = DataLoader(
    dataset,
    batch_size=1,
    )

    
    
    for batch_idx, ex in enumerate(train_loader):
        stack = ex['stack']
        Surface_Rough = ex['Roughness']
        num_in_batch = len(ex['spindle_speed'])
    
        for j in range(num_in_batch):
                total_num_channels = stack['signalAE'][j].shape[0]
                channel_nums = [*range(total_num_channels)]

                # tillChannelN = 45
                
                plt.figure()

                RealSurfaceData=Surface_Rough.detach().numpy()

                plt.plot(np.linspace(0, 55, RealSurfaceData.size), RealSurfaceData.transpose(),linewidth=3.0) #Correct the dimensions!!!
                
                # sns.set_theme()
                for tillChannelN in range(4, 50, 4):

                    outputs = model.forward(
                        stack['signalAE'][j], stack['signalMic'][j], stack['signalForces'][j],
                        tillChannelN)
                    
            
                    y_pred = outputs.squeeze(dim=0).detach().numpy()
                
                    
                    
                    plt.plot(np.linspace(tillChannelN+1, tillChannelN+7, 7),y_pred)
                    
                plt.show()


                plt.figure()
        
                y_real = stack['signalFy'][j][tillChannelN+7, :].numpy()
                
                plt.plot(np.linspace(0, 360, y_real.size), y_real)
                plt.show()     
    
    
    
    return 

# ax.legend(['A simple line'])










