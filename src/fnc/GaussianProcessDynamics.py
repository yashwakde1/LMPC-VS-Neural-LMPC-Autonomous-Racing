"""
Gaussian Process Dynamics Predictor for Vehicle Dynamics
Provides uncertainty-aware predictions for LMPC
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
import pickle
import os

class GPDynamicsPredictor:
    """
    Gaussian Process-based dynamics predictor
    Unlike Neural Networks, this gives uncertainty estimates!
    """
    def __init__(self, state_dim=6, input_dim=2, noise_level=0.1):
        self.state_dim = state_dim
        self.input_dim = input_dim
        
        # Storage for training data
        self.all_states = []
        self.all_actions = []
        self.all_next_states = []
        
        # Normalization parameters
        self.X_mean = None
        self.X_std = None
        
        # Create one GP for each state dimension
        self.gp_models = []
        
        for i in range(state_dim):
            # Kernel: combines RBF (smooth) + noise
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=noise_level)
            
            gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=10,
                alpha=1e-5,
                normalize_y=True
            )
            self.gp_models.append(gp)
        
        print(f"✓ Initialized {state_dim} Gaussian Process models")
    
    def add_trajectory(self, states, actions):
        """Add trajectory data for training"""
        for t in range(len(states) - 1):
            self.all_states.append(states[t])
            self.all_actions.append(actions[t])
            self.all_next_states.append(states[t + 1])
    
    def train(self, verbose=True):
        """Train all Gaussian Process models"""
        if len(self.all_states) == 0:
            print("⚠ No training data available!")
            return
        
        # Prepare data
        states = np.array(self.all_states)
        actions = np.array(self.all_actions)
        next_states = np.array(self.all_next_states)
        
        # Input: [state, action]
        X = np.hstack([states, actions])
        
        # Normalize inputs for better convergence
        if self.X_mean is None:
            self.X_mean = X.mean(axis=0)
            self.X_std = X.std(axis=0) + 1e-8
        
        X_normalized = (X - self.X_mean) / self.X_std
        
        if verbose:
            print(f"\nTraining Gaussian Processes on {X.shape[0]} samples...")
            print(f"Input dimension: {X.shape[1]} (state={self.state_dim}, action={self.input_dim})")
        
        state_names = ['vx', 'vy', 'wz', 'epsi', 's', 'ey']
        
        # Train one GP for each state dimension
        for i, gp in enumerate(self.gp_models):
            y = next_states[:, i]
            
            if verbose:
                print(f"  Training GP for {state_names[i]}...", end=' ')
            
            try:
                gp.fit(X_normalized, y)
                
                if verbose:
                    score = gp.score(X_normalized, y)
                    print(f"R² = {score:.4f} ✓")
            except Exception as e:
                if verbose:
                    print(f"Warning: {str(e)[:50]}... (continuing)")
        
        if verbose:
            print("✓ All GPs trained!\n")
    
    def predict(self, state, action, return_std=True):
        """Predict next state with uncertainty"""
        # Prepare input
        X = np.hstack([state, action]).reshape(1, -1)
        
        # Normalize using training statistics
        if self.X_mean is not None:
            X = (X - self.X_mean) / self.X_std
        
        means = []
        stds = []
        
        # Get prediction from each GP
        for gp in self.gp_models:
            try:
                if return_std:
                    mean, std = gp.predict(X, return_std=True)
                    means.append(mean[0])
                    stds.append(std[0])
                else:
                    mean = gp.predict(X)
                    means.append(mean[0])
            except Exception as e:
                means.append(0.0)
                if return_std:
                    stds.append(1.0)
        
        mean_pred = np.array(means)
        
        if return_std:
            std_pred = np.array(stds)
            return mean_pred, std_pred
        else:
            return mean_pred
    
    def predict_trajectory(self, initial_state, actions, return_std=True):
        """Predict trajectory over multiple time steps"""
        states = [initial_state]
        stds_list = [np.zeros(self.state_dim)]
        
        current_state = initial_state
        
        for action in actions:
            if return_std:
                next_state, std = self.predict(current_state, action, return_std=True)
                stds_list.append(std)
            else:
                next_state = self.predict(current_state, action, return_std=False)
            
            states.append(next_state)
            current_state = next_state
        
        if return_std:
            return np.array(states), np.array(stds_list)
        else:
            return np.array(states)
    
    def get_uncertainty_map(self, test_states, test_actions):
        """Get uncertainty for a set of state-action pairs"""
        uncertainties = []
        
        for state, action in zip(test_states, test_actions):
            _, std = self.predict(state, action, return_std=True)
            total_uncertainty = np.mean(std)
            uncertainties.append(total_uncertainty)
        
        return np.array(uncertainties)
    
    def save_model(self, filepath):
        """Save GP models"""
        data = {
            'gp_models': self.gp_models,
            'state_dim': self.state_dim,
            'input_dim': self.input_dim,
            'X_mean': self.X_mean,
            'X_std': self.X_std,
            'all_states': self.all_states,
            'all_actions': self.all_actions,
            'all_next_states': self.all_next_states
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✓ GP models saved to {filepath}")
    
    def load_model(self, filepath):
        """Load GP models"""
        if not os.path.exists(filepath):
            print(f"⚠ Model file not found: {filepath}")
            return
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.gp_models = data['gp_models']
        self.state_dim = data['state_dim']
        self.input_dim = data['input_dim']
        self.X_mean = data.get('X_mean', None)
        self.X_std = data.get('X_std', None)
        self.all_states = data.get('all_states', [])
        self.all_actions = data.get('all_actions', [])
        self.all_next_states = data.get('all_next_states', [])
        
        print(f"✓ GP models loaded from {filepath}")


# Test code
if __name__ == "__main__":
    print("="*60)
    print("Testing Gaussian Process Dynamics Predictor")
    print("="*60 + "\n")
    
    gp_predictor = GPDynamicsPredictor(state_dim=6, input_dim=2, noise_level=0.01)
    
    print("Generating dummy training data...")
    for traj in range(5):
        T = 50
        states = np.random.randn(T, 6) * 0.5
        actions = np.random.randn(T, 2) * 0.3
        gp_predictor.add_trajectory(states, actions)
    
    print("\nTraining Gaussian Processes...")
    gp_predictor.train(verbose=True)
    
    print("Testing prediction with uncertainty...")
    test_state = np.array([0.5, 0.1, 0.0, 0.0, 1.0, 0.0])
    test_action = np.array([0.1, 0.5])
    
    mean, std = gp_predictor.predict(test_state, test_action, return_std=True)
    
    print(f"\nInput state:  {test_state}")
    print(f"Input action: {test_action}")
    print(f"\nPredicted next state: {mean}")
    print(f"Uncertainty (std):    {std}")
    
    gp_predictor.save_model("test_gp_model.pkl")
    
    print("\n" + "="*60)
    print("Test complete!")
    print("="*60)