import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
'''
class Encoder(nn.Module):
    def __init__(self, belief_dim, obs_dim, action_dim, latent_dim):
        super(Encoder, self).__init__()
        self.belief_embedding = nn.Linear(belief_dim, 64)
        self.obs_embedding = nn.Embedding(obs_dim, 32)
        self.action_embedding = nn.Embedding(action_dim, 32)
        self.fc = nn.Linear(128, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, belief, observation, action):
        belief_emb = self.relu(self.belief_embedding(belief))
        obs_emb = self.obs_embedding(observation)
        action_emb = self.action_embedding(action)
        x = torch.cat((belief_emb, obs_emb, action_emb), dim=1)
        x = self.relu(self.fc(x))
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
'''
class Encoder(nn.Module):
    def __init__(self, belief_dim, action_dim, observation_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.belief_dim = belief_dim
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.observation_embedding = nn.Embedding(observation_dim, hidden_dim)

        self.fc1 = nn.Linear(belief_dim + action_dim + hidden_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, belief_dim)

    def forward(self, old_belief, action, observation):
        # One-hot encode the actions
        embedded_action = nn.functional.one_hot(action, num_classes=self.action_dim).float()
        
        # Embed the observations
        embedded_observation = self.observation_embedding(observation)

        ## Check the shapes
        #print("Old belief shape:", old_belief.shape)
        #print("Embedded action shape:", embedded_action.shape)
        #print("Embedded observation shape:", embedded_observation.shape)

        # Concatenate old belief with embeddings
        concatenated = torch.cat((old_belief, embedded_action, embedded_observation), dim=1)

        # Pass through hidden layers
        out = torch.relu(self.fc1(concatenated))
        out = torch.relu(self.fc2(out))

        # Output layer with softmax activation
        updated_belief_logits = self.fc3(out)
        updated_belief = torch.softmax(updated_belief_logits, dim=1)
        
        return updated_belief

class Decoder(nn.Module):
    def __init__(self, belief_dim, observation_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(belief_dim, 128)
        self.fc2 = nn.Linear(128, observation_dim)

    def forward(self, updated_belief):
        # Pass through hidden layers
        out = torch.relu(self.fc1(updated_belief))
        out = self.fc2(out)
        
        return out

class CustomDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        beliefs = torch.tensor(sequence['belief'], dtype=torch.float32)
        actions = torch.tensor(sequence['action'], dtype=torch.long)  # Assuming actions are discrete
        observations = torch.tensor(sequence['observation'], dtype=torch.long)  # Assuming observations are discrete
        next_observations = torch.tensor(sequence['next_observation'], dtype=torch.long)  # Assuming observations are discrete
        return {'belief': beliefs, 'action': actions, 'observation': observations, 'next_observation': next_observations}

