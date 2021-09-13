"""
Highlevel project script.
"""
import torch
import ulid

from loader import load_all_mats
from model import LSTMModel, load_preconfigured_model
from train import TrainHyperparams, train
from logger import logger

logger.setLevel('VERBOSE')

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("Device:",device)

# Load all data:
_, dataset = load_all_mats()

# Load pre-configured model:
model = load_preconfigured_model()
model.to(device)
# Train the model:
trained_model = train(model, dataset)

# Save the final model:
torch.save(trained_model, f'model_v2_trained_{ulid.new()}.pth')
