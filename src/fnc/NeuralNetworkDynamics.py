"""
Neural Network Dynamics Model for Vehicle Prediction
Learns state transitions: x_{k+1} = f(x_k, u_k)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class VehicleDynamicsNN(nn.Module):
    """Neural Network for learning vehicle dynamics"""
    def __init__(self, state_dim=6, input_dim=2, hidden_dim=128):
        super(VehicleDynamicsNN, self).__init__()
        
        self.state_dim = state_dim
        self.input_dim = input_dim
        
        # Network architecture
        self.network = nn.Sequential(
            nn.Linear(state_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # Residual learning (predict change in state)
        self.residual = True
        
    def forward(self, state, action):
        """
        Predict next state given current state and action
        Args:
            state: (batch, state_dim) - [vx, vy, wz, epsi, s, ey]
            action: (batch, input_dim) - [steering, acceleration]
        Returns:
            next_state: (batch, state_dim)
        """
        x = torch.cat([state, action], dim=-1)
        delta = self.network(x)
        
        if self.residual:
            return state + delta
        else:
            return delta


class TrajectoryDataset(Dataset):
    """Dataset for storing trajectory data"""
    def __init__(self, states, actions, next_states):
        self.states = torch.FloatTensor(states)
        self.actions = torch.FloatTensor(actions)
        self.next_states = torch.FloatTensor(next_states)
        
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.next_states[idx]


class NNDynamicsPredictor:
    """Neural Network-based dynamics predictor for LMPC"""
    def __init__(self, state_dim=6, input_dim=2, hidden_dim=128, 
                 learning_rate=1e-3, device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = VehicleDynamicsNN(state_dim, input_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        self.state_dim = state_dim
        self.input_dim = input_dim
        
        # Storage for training data
        self.all_states = []
        self.all_actions = []
        self.all_next_states = []
        
        # Normalization statistics
        self.state_mean = None
        self.state_std = None
        self.action_mean = None
        self.action_std = None
        
    def add_trajectory(self, states, actions):
        """Add trajectory data for training"""
        for t in range(len(states) - 1):
            self.all_states.append(states[t])
            self.all_actions.append(actions[t])
            self.all_next_states.append(states[t + 1])
    
    def compute_normalization(self):
        """Compute mean and std for normalization"""
        if len(self.all_states) == 0:
            return
        
        states = np.array(self.all_states)
        actions = np.array(self.all_actions)
        
        self.state_mean = states.mean(axis=0)
        self.state_std = states.std(axis=0) + 1e-6
        self.action_mean = actions.mean(axis=0)
        self.action_std = actions.std(axis=0) + 1e-6
    
    def train(self, epochs=100, batch_size=64, verbose=True):
        """Train the neural network on collected data"""
        if len(self.all_states) == 0:
            print("No training data available!")
            return
        
        # Compute normalization
        self.compute_normalization()
        
        # Create dataset
        states = np.array(self.all_states)
        actions = np.array(self.all_actions)
        next_states = np.array(self.all_next_states)
        
        dataset = TrajectoryDataset(states, actions, next_states)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_states, batch_actions, batch_next_states in dataloader:
                batch_states = batch_states.to(self.device)
                batch_actions = batch_actions.to(self.device)
                batch_next_states = batch_next_states.to(self.device)
                
                # Forward pass
                pred_next_states = self.model(batch_states, batch_actions)
                loss = self.criterion(pred_next_states, batch_next_states)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if verbose and (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    def predict(self, state, action):
        """Predict next state given current state and action"""
        self.model.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
            
            pred = self.model(state_tensor, action_tensor)
            return pred.cpu().numpy().squeeze()
    
    def save_model(self, filepath):
        """Save model to file"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'state_mean': self.state_mean,
            'state_std': self.state_std,
            'action_mean': self.action_mean,
            'action_std': self.action_std,
        }, filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model from file"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.state_mean = checkpoint['state_mean']
        self.state_std = checkpoint['state_std']
        self.action_mean = checkpoint['action_mean']
        self.action_std = checkpoint['action_std']
        print(f"✓ Model loaded from {filepath}")
