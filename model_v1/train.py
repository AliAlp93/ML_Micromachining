"""
Training methods for `LSTMModel`.

@author(2021_05_10): Connor W. Colombo (colombo@cmu.edu)
"""
from typing import Optional, List, Any, Dict, Tuple, Union
import attr
import random

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore

from model import LSTMModel
from data import MachiningDataset
from logger import logger
from opts import get_opts

opts = get_opts()


@attr.s(frozen=True, cmp=True, slots=True, auto_attribs=True)
class TrainHyperparams:
    """Collection of tunable hyperparameters for the training process."""
    # Number of training epochs:
    num_epochs: int
    # Adam learning rate:
    lr: float
    # Batch size:
    batch_size: int
    # Proportion of total dataset which should be used for validation:
    validation_split: float
    # Proportion of total dataset which should be used for final testing:
    test_split: float
    # Which channel number to start forecasting at:
    channel_forecast_start: int
    # How many channels to step by when increasing forecast distance:
    channel_forecast_step: int


def train(
    model: LSTMModel,
    dataset: MachiningDataset,
    random_seed: int = 42,
    hyperparams: Optional[TrainHyperparams] = None
) -> LSTMModel:
    """
    Trains the given `LSTMModel` `model` on the given `data` and returns the trained model.
    """
    if hyperparams is None:
        hyperparams = TrainHyperparams(
            num_epochs=opts.num_epochs,
            lr=opts.lr,
            batch_size=opts.batch_size,
            validation_split=opts.validation_split,
            test_split=opts.test_split,
            channel_forecast_start=opts.channel_forecast_start,
            channel_forecast_step=opts.channel_forecast_step
        )

    dataset_size = len(dataset)

    if hyperparams.batch_size > dataset_size:
        raise ValueError(
            f"Batch size ({hyperparams.batch_size}) can't be greater than "
            f"total number of examples in dataset ({dataset_size})."
        )

    if (hyperparams.validation_split + hyperparams.test_split) >= 1.0:
        raise ValueError(
            f"Validation split ({hyperparams.validation_split*100:3.1f}%) "
            f"and test split ({hyperparams.test_split*100:3.1f}%) "
            f"can't add up to >= 100% since there has to be samples left over "
            f"for training (ideally >50% for training)."
        )
    if (hyperparams.validation_split + hyperparams.test_split) > 0.5:
        logger.warning(
            f"Validation split ({hyperparams.validation_split*100:3.1f}%) "
            f"and test split ({hyperparams.test_split*100:3.1f}%) "
            f"currently add up to {(hyperparams.validation_split + hyperparams.test_split)*100:3.1f}%. "
            f"Ideally, they should add up to less than 50% so most of the data will be used for training. "
        )

    # Creating data indices for training and validation splits:
    indices = [*range(dataset_size)]

    splits = [
        int(hyperparams.validation_split * dataset_size),
        int((hyperparams.validation_split+hyperparams.test_split) * dataset_size)
    ]
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    val_indices = indices[:splits[0]]
    test_indices = indices[splits[0]:splits[1]]
    train_indices = indices[splits[1]:]

    # if len(val_indices) < 2 or len(test_indices) < 2 or len(train_indices) < 2:
    #     raise ValueError(
    #         "Dataset is partitioned poorly. Testing, Validation, and Training "
    #         "should each have at least two examples (ideally a lot more) but "
    #         f"instead dataset contains {len(dataset)} `DataExample`s split into: "
    #         f"{(1-hyperparams.validation_split-hyperparams.test_split)*100:3.1f}% = {len(train_indices)} for training, "
    #         f"{hyperparams.validation_split*100:3.1f}% = {len(val_indices)} for validation, "
    #         f"and {hyperparams.test_split*100:3.1f}% = {len(test_indices)} for final testing."
    #     )

    logger.verbose(  # type: ignore
        f"Dataset contains {len(dataset)} `DataExample`s split into: "
        f"{(1-hyperparams.validation_split-hyperparams.test_split)*100:3.1f}% = {len(train_indices)} for training, "
        f"{hyperparams.validation_split*100:3.1f}% = {len(val_indices)} for validation, "
        f"and {hyperparams.test_split*100:3.1f}% = {len(test_indices)} for final testing."
    )

    # Creating PT data samplers and loaders:
    train_loader = DataLoader(
        dataset,
        batch_size=hyperparams.batch_size,
        sampler=SubsetRandomSampler(train_indices)
    )
    val_loader = DataLoader(
        dataset,
        batch_size=hyperparams.batch_size,
        sampler=SubsetRandomSampler(val_indices)
    )
    test_loader = DataLoader(
        dataset,
        batch_size=hyperparams.batch_size,
        sampler=SubsetRandomSampler(test_indices)
    )

    # Setup:
    LossInEpoch=np.zeros(hyperparams.num_epochs)
    LossInEpoch_val=np.zeros(hyperparams.num_epochs)
    criterion = nn.MSELoss()
    beta1=0.7
    optimizer = optim.Adam(model.parameters(), lr=hyperparams.lr,betas=(beta1, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    logger.info("Beginning training . . .")
    
    stackForPlot=[] # Empthy List
    RoughnessForPlot=[]
    for batch_idx, ex in enumerate(train_loader):
        stack = ex['stack']
        Surface_Rough=ex['Roughness']
        stackForPlot.append(stack)
        RoughnessForPlot.append(Surface_Rough)

    
    
    for epoch in range(hyperparams.num_epochs):
        epoch_start_time = time.time()  # record run TIME
        running_loss=0
        numberoflossitem=0
        # Run over training data:
        for batch_idx, ex in enumerate(train_loader):
            stack = ex['stack']
            Surface_Rough=ex['Roughness']
            num_in_batch = len(ex['spindle_speed'])
            # for j in range(num_in_batch):
            total_num_channels = stack['signalAE'].shape[1]
            channel_nums = [*range(total_num_channels)]

            start = hyperparams.channel_forecast_start
            step = hyperparams.channel_forecast_step
            stop = total_num_channels - step  # leave one more step to forecast
            
            for tillChannelN in channel_nums[start:stop:1]:
                outputs = model.forward(
                    stack['signalAE'], stack['signalMic'], stack['signalForces'],
                    tillChannelN,step,hyperparams.batch_size
                )
                # OutputForceY.size() torch.Size([1, 125])
                loss = criterion(
                    outputs.double().squeeze(dim=2),
                    Surface_Rough[ :,tillChannelN-step :tillChannelN ]
                )
                
                # print('loss Calculated!! as:', loss)
                running_loss += loss.item()
                numberoflossitem+=1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
        print('SurfaceRoughness Real Value:',Surface_Rough[ :,tillChannelN-step :tillChannelN ])
        print('Predictions: ',outputs.double().squeeze(dim=2).detach().numpy())
        scheduler.step()
        
        running_loss_val=0
        numberoflossitem_val=0
        
        for batch_idx, ex in enumerate(val_loader):
            stack = ex['stack']
            Surface_Rough=ex['Roughness']

            num_in_batch = len(ex['spindle_speed'])
            # for j in range(num_in_batch):
            total_num_channels = stack['signalAE'].shape[1]
            channel_nums = [*range(total_num_channels)]

            start = hyperparams.channel_forecast_start
            step = hyperparams.channel_forecast_step
            stop = total_num_channels - step  # leave one more step to forecast

            for tillChannelN in channel_nums[start:stop:1]:
                outputs = model.forward(
                    stack['signalAE'], stack['signalMic'], stack['signalForces'],
                    tillChannelN,step,hyperparams.batch_size
                )
                
                loss = criterion(
                    outputs.double().squeeze(dim=2),
                    Surface_Rough[:, tillChannelN-step  :tillChannelN ]
                )
                
                # # OutputForceY.size() torch.Size([1, 125])
                # loss = criterion(
                #     outputs.double().flatten(),
                #     stack['signalFy'][j][tillChannelN + step, :]
                # )
                
                running_loss_val += loss.item()
                numberoflossitem_val+=1
                    
        # ! TODO: Test against validation set
        LossInEpoch[epoch] =  running_loss/numberoflossitem
        LossInEpoch_val[epoch] =  running_loss_val/numberoflossitem_val

        epoch_end_time = time.time()
        logger.verbose(  # type: ignore
            f"Epoch [{epoch}/{hyperparams.num_epochs-1}]: {(epoch_end_time-epoch_start_time):5.1f}s \t "
            f"— Running Training Loss: {LossInEpoch[epoch]} \t"
            f"— Validation Loss: {LossInEpoch_val[epoch]} \t"
        )




    
    plt.figure()
    plt.plot( range(hyperparams.num_epochs),LossInEpoch)
    plt.plot( range(hyperparams.num_epochs),LossInEpoch_val)

    plt.xlabel('Number of Epoch')
    plt.ylabel('Loss MSELoss')
    plt.legend(f'Loss in Training',f'Loss in Validation')
    plt.show()

                       
    # ! TODO: Plot loss against train and validation sets over training
    
    # plt.figure()
    # plt.plot( range(hyperparams.num_epochs),LossInEpoch)
    # plt.xlabel('Number of Epoch')
    # plt.ylabel('Loss MSELoss')
    # plt.legend(bbox_to_anchor=(1,1),fontsize='x-small')
    # plt.show()

    # ! TODO: Test against test set holdout after training

    # Plot Results for end of validation:
    # for i in range(11):
        
        # j=random.randrange(11)
        # tillChannelN=random.randrange(20,40)
        # SignalAE=stackForPlot[j]['signalAE'].squeeze(dim=1)
        # SignalMIC=stackForPlot[j]['signalMic'].squeeze(dim=1)
        # SignalForce=stackForPlot[j]['signalForces'].squeeze(dim=0)
        
    for batch_idx, ex in enumerate(train_loader):
        stack = ex['stack']
        Surface_Rough = ex['Roughness']
        num_in_batch = len(ex['spindle_speed'])
    
    # for j in range(num_in_batch):
        total_num_channels = stack['signalAE'].shape[1]
        channel_nums = [*range(total_num_channels)]

        # tillChannelN = 45
        

        fig, (ax1, ax2) = plt.subplots(2, 1)
        # make a little extra space between the subplots
        fig.subplots_adjust(hspace=0.5)
        
        RealSurfaceData=Surface_Rough.detach().numpy()

        ax1.plot(np.linspace(0, 55, RealSurfaceData.shape[1]), RealSurfaceData[0,:].transpose() , linewidth=3.0) #Correct the dimensions!!!
        ax2.plot(np.linspace(0, 55, RealSurfaceData.shape[1]), RealSurfaceData[1,:].transpose() , linewidth=3.0) #Correct the dimensions!!!

        # sns.set_theme()
        for tillChannelN in range(8, 52, 4):

            outputs = model.forward(
                    stack['signalAE'], stack['signalMic'], stack['signalForces'],
                    tillChannelN,step,hyperparams.batch_size
                )
            
    
            y_pred = outputs.double().squeeze(dim=2).detach().numpy()

        
            
            
            ax1.plot(np.linspace(tillChannelN-step, tillChannelN, 7),y_pred[0,:])
            ax2.plot(np.linspace(tillChannelN-step, tillChannelN, 7),y_pred[1,:])
            
        plt.show()


            # plt.figure()
    
            # y_real = stack['signalFy'][j][tillChannelN+7, :].numpy()
            
            # plt.plot(np.linspace(0, 360, y_real.size), y_real)
            # plt.show()        
        
        # outputs = model.forward(
        #     SignalAE,SignalMIC , SignalForce,
        #     tillChannelN
        # )
                    
        
        
        
        
        # y_real = stackForPlot['signalFy'][j][tillChannelN+step, :].numpy()
        
        # y_real_current = stackForPlot['signalFy'][j][tillChannelN, :].numpy()
        
        
        
        # y_real_current = stackForPlot['signalFy'][j][tillChannelN, :].numpy()
        # y_real_10previous = stackForPlot['signalFy'][j][tillChannelN-hyperparams.channel_forecast_step, :].numpy()

        # plt.plot(np.linspace(0, 360, y_pred.size), y_pred,linewidth=3.0)
        # plt.plot(np.linspace(0, 360, y_real.size), y_real)
        # plt.plot(np.linspace(0, 360, y_real.size), y_real_current,linewidth=3.0)
        # plt.plot(np.linspace(0, 360, y_real.size), y_real_10previous)
        # plt.xticks(range(361)[::60])
        # plt.gca().margins(0)

        # plt.xlabel('Bit Rotation Angle [deg]')
        # plt.ylabel('Normalized Force')
        # plt.legend((f'Prediction (+{hyperparams.channel_forecast_step})', f'Real (+{hyperparams.channel_forecast_step})', f'Current Ch ({tillChannelN})', f'(-{hyperparams.channel_forecast_step}) Previous'))

        # plt.title(f'LSTM Force Prediction for {j} data Channel {tillChannelN+step} Training')
        
        
    # for i in range(11):
        
    #     j=random.randrange(11)
    #     tillChannelN=random.randrange(20,40)
        
    #     outputs = model.forward(
    #         stackForPlot['signalAE'][j], stackForPlot['signalMic'][j], stackForPlot['signalForces'][j],
    #         tillChannelN
    #     )
                    
    #     sns.set_theme()
    #     plt.figure()
    
    #     y_pred = outputs.squeeze(dim=0).detach().numpy()
        
    #     # y_real = stackForPlot['signalFy'][j][tillChannelN+step, :].numpy()
        
    #     y_real_current = stackForPlot['signalFy'][j][tillChannelN, :].numpy()
    #     y_real_10previous = stackForPlot['signalFy'][j][tillChannelN-hyperparams.channel_forecast_step, :].numpy()

    #     plt.plot(np.linspace(0, 360, y_pred.size), y_pred,linewidth=3.0)
    #     plt.plot(np.linspace(0, 360, y_real.size), y_real)
    #     plt.plot(np.linspace(0, 360, y_real.size), y_real_current,linewidth=3.0)
    #     plt.plot(np.linspace(0, 360, y_real.size), y_real_10previous)
    #     plt.xticks(range(361)[::60])
    #     plt.gca().margins(0)

    #     plt.xlabel('Bit Rotation Angle [deg]')
    #     plt.ylabel('Normalized Force')
    #     plt.legend((f'Prediction (+{hyperparams.channel_forecast_step})', f'Real (+{hyperparams.channel_forecast_step})', f'Current Ch ({tillChannelN})', f'(-{hyperparams.channel_forecast_step}) Previous'))

    #     plt.title(f'LSTM Force Prediction for {j} data Channel {tillChannelN+step} Training')
    #     plt.show()

    # for i in range(8):
        
    #     j=random.randrange(5)
    #     tillChannelN=random.randrange(20,45)
        
    #     outputs = model.forward(
    #         stack['signalAE'][j], stack['signalMic'][j], stack['signalForces'][j],
    #         tillChannelN
    #     )
                    
    #     sns.set_theme()
    #     plt.figure()
    
    #     y_pred = outputs.squeeze(dim=0).detach().numpy()
    #     y_real = stack['signalFy'][j][tillChannelN+step, :].numpy()
        
    #     y_real_current = stack['signalFy'][j][tillChannelN, :].numpy()
    #     y_real_10previous = stack['signalFy'][j][tillChannelN-hyperparams.channel_forecast_step, :].numpy()

    #     plt.plot(np.linspace(0, 360, y_pred.size), y_pred,linewidth=3.0)
    #     plt.plot(np.linspace(0, 360, y_real.size), y_real)
    #     plt.plot(np.linspace(0, 360, y_real.size), y_real_current,linewidth=3.0)
    #     plt.plot(np.linspace(0, 360, y_real.size), y_real_10previous)
    #     plt.xticks(range(361)[::60])
    #     plt.gca().margins(0)

    #     plt.xlabel('Bit Rotation Angle [deg]')
    #     plt.ylabel('Normalized Force')
    #     plt.legend((f'Prediction (+{hyperparams.channel_forecast_step})', f'Real (+{hyperparams.channel_forecast_step})', f'Current Ch ({tillChannelN})', f'(-{hyperparams.channel_forecast_step}) Previous'))

    #     plt.title(f'LSTM Force Prediction for {j} data Channel {tillChannelN+step} Validation')
    #     plt.show()
        
    return model
