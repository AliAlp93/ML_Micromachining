"""
Training methods for `LSTMModel`.

@author(2021_05_10): Connor W. Colombo (colombo@cmu.edu)
"""
from typing import Optional, List, Any, Dict, Tuple, Union
import attr

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
            f"Validation split ({hyperparams.validation_split * 100:3.1f}%) "
            f"and test split ({hyperparams.test_split * 100:3.1f}%) "
            f"can't add up to >= 100% since there has to be samples left over "
            f"for training (ideally >50% for training)."
        )
    if (hyperparams.validation_split + hyperparams.test_split) > 0.5:
        logger.warning(
            f"Validation split ({hyperparams.validation_split * 100:3.1f}%) "
            f"and test split ({hyperparams.test_split * 100:3.1f}%) "
            f"currently add up to {(hyperparams.validation_split + hyperparams.test_split) * 100:3.1f}%. "
            f"Ideally, they should add up to less than 50% so most of the data will be used for training. "
        )

    # Creating data indices for training and validation splits:
    indices = [*range(dataset_size)]

    splits = [
        int(hyperparams.validation_split * dataset_size),
        int((hyperparams.validation_split + hyperparams.test_split) * dataset_size)
    ]
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    val_indices = indices[:splits[0]]
    test_indices = indices[splits[0]:splits[1]]
    train_indices = indices[splits[1]:]

    if len(val_indices) < 2 or len(test_indices) < 2 or len(train_indices) < 2:
        raise ValueError(
            "Dataset is partitioned poorly. Testing, Validation, and Training "
            "should each have at least two examples (ideally a lot more) but "
            f"instead dataset contains {len(dataset)} `DataExample`s split into: "
            f"{(1 - hyperparams.validation_split - hyperparams.test_split) * 100:3.1f}% = {len(train_indices)} for training, "
            f"{hyperparams.validation_split * 100:3.1f}% = {len(val_indices)} for validation, "
            f"and {hyperparams.test_split * 100:3.1f}% = {len(test_indices)} for final testing."
        )

    logger.verbose(  # type: ignore
        f"Dataset contains {len(dataset)} `DataExample`s split into: "
        f"{(1 - hyperparams.validation_split - hyperparams.test_split) * 100:3.1f}% = {len(train_indices)} for training, "
        f"{hyperparams.validation_split * 100:3.1f}% = {len(val_indices)} for validation, "
        f"and {hyperparams.test_split * 100:3.1f}% = {len(test_indices)} for final testing."
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
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)
    # model=model.to(device)
    # Setup:
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparams.lr)

    logger.info("Beginning training . . .")

    for epoch in range(hyperparams.num_epochs):
        epoch_start_time = time.time()  # record run TIME

        # Run over training data:
        for batch_idx, ex in enumerate(train_loader):
            stack = ex['stack']
            num_in_batch = len(ex['spindle_speed'])
            for j in range(num_in_batch):
                total_num_channels = stack['signalAE'][j].shape[0]
                channel_nums = [*range(total_num_channels)]

                start = hyperparams.channel_forecast_start
                step = hyperparams.channel_forecast_step
                stop = total_num_channels - step  # leave one more step to forecast

                for tillChannelN in channel_nums[start:stop:step]:
                    input1 = stack['signalAE'][j]
                    input1.to(device)
                    input2 = stack['signalMic'][j].cuda()
                    input2.to(device)
                    input3 = stack['signalForces'][j].cuda()
                    input3.to(device)
                    input4 = torch.tensor(tillChannelN).cuda()
                    input4.to(device)
                    model.to(device)
                    outputs = model.forward(input1, input2, input3, input4)
                    # OutputForceY.size() torch.Size([1, 125])
                    loss = criterion(
                        outputs.double().flatten().to(device),
                        stack['signalFy'][j][tillChannelN + step, :].to(device)
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        # ! TODO: Test against validation set

        epoch_end_time = time.time()
        logger.verbose(  # type: ignore
            f"Epoch [{epoch}/{hyperparams.num_epochs - 1}]: {(epoch_end_time - epoch_start_time):5.1f}s \t "
            f"— Training Loss: {loss.item()} \t"
            f"— Validation Loss: NaN \t"
        )

    # ! TODO: Plot loss against train and validation sets over training

    # ! TODO: Test against test set holdout after training

    # Plot Results for end of validation:
    sns.set_theme()
    plt.figure()

    y_pred = outputs.squeeze(dim=0).detach().numpy()
    y_real = stack['signalFy'][j][tillChannelN + step, :].numpy()
    plt.plot(np.linspace(0, 360, y_pred.size), y_pred)
    plt.plot(np.linspace(0, 360, y_real.size), y_real)

    plt.xlabel('Bit Rotation Angle [deg]')
    plt.ylabel('Normalized Force')
    plt.legend(('Prediction', 'Real'))
    plt.title(f'LSTM Force Prediction for Channel {tillChannelN + step}')
    plt.show()

    return model
