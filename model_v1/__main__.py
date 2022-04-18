"""
Highlevel project script.
"""
import torch
import ulid

from loader import load_all_mats
from model import LSTMModel, load_preconfigured_model
from train import TrainHyperparams, train
from logger import logger
# import DebugPlots as DP 
from DebugPlots import DebugPlots
from DebugPlots import ChannelPeaktoPeakForce
from DebugPlots import AllPeaktoPeakForce
from DebugPlots import PlotResults

#%%

logger.setLevel('VERBOSE')

#%%
# Load all data:
Sa,_, dataset = load_all_mats()
i=3
# for i in range(14):
DebugPlots(dataset,0,5)

#%% Plots   


 
# DebugPlots(dataset,2,1) # DebugPlots(dataset,ExperimentNumber,ChannelNumber, plotAE='true',plotMIC='true',plotFY='true',plotSA='true')
# ChannelPeaktoPeakForce(dataset,2)

# AllPeaktoPeakForce(dataset,7,14)



#%%
# Load pre-configured model:
# model = load_preconfigured_model()

# Train the model:
# trained_model = train(model, dataset)
# 
# Save the final model:
# torch.save(trained_model, f'model_v2_trained_{ulid.new()}.pth')
#%% Load A specific Model and Run to get plots


# model = torch.load( f'model_v2_trained_01F8B7ME8RT4SYKC3F4GAX90ZS.pth')
# model.eval()

# PlotResults(dataset,model)
