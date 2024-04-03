import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from prediction_network import Encoder, Decoder, CustomDataset

import os

from tqdm import tqdm

from utility import retrieve_trajectories_from_experiment

exp_name = "StochasticMatrix4RoomsCorridorG01T55Oracle005"
belief_dim = 45
action_dim = 4
observation_dim = 45
hidden_dim = 64

# Initialize encoder and decoder 
encoder = Encoder(belief_dim, action_dim, observation_dim, hidden_dim)
decoder = Decoder(belief_dim, observation_dim, hidden_dim)

# Retrieve trajectories
sequences = retrieve_trajectories_from_experiment(exp_name=exp_name)

# Create dataset
dataset = CustomDataset(sequences)

# Load the dataset
batch_size = 256
shuffle = True
num_workers = 16
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)

# Training loop
num_epochs = 300

for epoch in range(num_epochs):
    print(f"Start of epoch #{epoch}")
    encoder.train()
    decoder.train()
    epoch_loss = 0.0
    with tqdm(total=len(dataloader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
        for batch in dataloader:  # Assuming you have a DataLoader for your training set
            optimizer.zero_grad()

            # Forward pass
            updated_belief = encoder(batch['belief'], batch['action'], batch['observation'])
            #print(updated_belief)
            predicted_observation = decoder(updated_belief)
            #print(f"Predicted observation is: {predicted_observation}")

            # Compute loss
            #print(f"Batch observation is: {batch['next_observation']}")
            loss = criterion(predicted_observation, batch['next_observation'])

            # Backpropagation
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.update(1)

    # Optionally, evaluate the model on the validation set after each epoch
    with torch.no_grad():
        encoder.eval()
        decoder.eval()
        # Validation loop...
        
    # Print training loss for monitoring
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save models
# File paths for saving encoder and decoder
exp_folder = os.path.join('Results', exp_name)
encoder_path = os.path.join(exp_folder, 'encoder_model.pth')
decoder_path = os.path.join(exp_folder, 'decoder_model.pth')

# Save encoder model
torch.save(encoder.state_dict(), encoder_path)

# Save decoder model
torch.save(decoder.state_dict(), decoder_path)
