"""
Gaussian Process Dynamics Predictor for Vehicle Dynamics
Provides uncertainty-aware predictions using Bayesian methods
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
import pickle
import os

class GPDynamicsPredictor:
    """
    Gaussian Process-based dynamics predictor
    Provides predictions AND uncertainty estimates
    """
    def __init__(self, state_dim=6, input_dim=2, noise_level=0.1):
        self.state_dim = state_dim
        self.input_dim = input_dim
        
        # Storage
        self.all_states = []
        self.all_actions = []
        self.all_next_states = []
        
        # Normalization
        self.X_mean = None
        self.X_std = None
        
        # One GP per state dimension
        self.gp_models = []
        
        for i in range(state_dim):
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
        """Add trajectory data"""
        for t in range(len(states) - 1):
            self.all_states.append(states[t])
            self.all_actions.append(actions[t])
            self.all_next_states.append(states[t + 1])
    
    def train(self, verbose=True):
        """Train all GPs"""
        if len(self.all_states) == 0:
            print("⚠ No training data!")
            return
        
        states = np.array(self.all_states)
        actions = np.array(self.all_actions)
        next_states = np.array(self.all_next_states)
        
        X = np.hstack([states, actions])
        
        # Normalize
        if self.X_mean is None:
            self.X_mean = X.mean(axis=0)
            self.X_std = X.std(axis=0) + 1e-8
        
        X_norm = (X - self.X_mean) / self.X_std
        
        if verbose:
            print(f"\nTraining Gaussian Processes on {X.shape[0]} samples...")
        
        state_names = ['vx', 'vy', 'wz', 'epsi', 's', 'ey']
        
        for i, gp in enumerate(self.gp_models):
            y = next_states[:, i]
            
            if verbose:
                print(f"  Training GP for {state_names[i]}...", end=' ')
            
            try:
                gp.fit(X_norm, y)
                if verbose:
                    score = gp.score(X_norm, y)
                    print(f"R² = {score:.4f} ✓")
            except Exception as e:
                if verbose:
                    print(f"Warning: {str(e)[:30]}...")
        
        if verbose:
            print("✓ All GPs trained!\n")
    
    def predict(self, state, action, return_std=True):
        """Predict with uncertainty"""
        X = np.hstack([state, action]).reshape(1, -1)
        
        if self.X_mean is not None:
            X = (X - self.X_mean) / self.X_std
        
        means = []
        stds = []
        
        for gp in self.gp_models:
            try:
                if return_std:
                    mean, std = gp.predict(X, return_std=True)
                    means.append(mean[0])
                    stds.append(std[0])
                else:
                    mean = gp.predict(X)
                    means.append(mean[0])
            except:
                means.append(0.0)
                if return_std:
                    stds.append(1.0)
        
        mean_pred = np.array(means)
        
        if return_std:
            std_pred = np.array(stds)
            return mean_pred, std_pred
        else:
            return mean_pred
    
    def save_model(self, filepath):
        """Save GP models"""
        data = {
            'gp_models': self.gp_models,
            'state_dim': self.state_dim,
            'input_dim': self.input_dim,
            'X_mean': self.X_mean,
            'X_std': self.X_std
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✓ GP saved to {filepath}")
    
    def load_model(self, filepath):
        """Load GP models"""
        if not os.path.exists(filepath):
            print(f"⚠ File not found: {filepath}")
            return
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.gp_models = data['gp_models']
        self.state_dim = data['state_dim']
        self.input_dim = data['input_dim']
        self.X_mean = data.get('X_mean', None)
        self.X_std = data.get('X_std', None)
        
        print(f"✓ GP loaded from {filepath}")