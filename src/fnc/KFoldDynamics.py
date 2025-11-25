"""
K-Fold Cross-Validation Dynamics Predictor
Ensemble learning using multiple Ridge regression models
"""

import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
import pickle
import os

class KFoldDynamicsPredictor:
    """
    K-Fold ensemble predictor
    Trains K models on different data splits, averages predictions
    """
    def __init__(self, state_dim=6, input_dim=2, n_folds=5):
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.n_folds = n_folds
        
        # Storage for training data
        self.all_states = []
        self.all_actions = []
        self.all_next_states = []
        
        # Ensemble of models
        self.fold_models = []  # List of [model1, model2, ..., model6] for each fold
        self.fold_scores = []
        
        # Normalization
        self.X_mean = None
        self.X_std = None
        
        print(f"✓ Initialized K-Fold predictor (K={n_folds})")
    
    def add_trajectory(self, states, actions):
        """Add trajectory data for training"""
        for t in range(len(states) - 1):
            self.all_states.append(states[t])
            self.all_actions.append(actions[t])
            self.all_next_states.append(states[t + 1])
    
    def train(self, verbose=True):
        """
        Train using K-Fold Cross-Validation
        Creates ensemble of K×6 models
        """
        if len(self.all_states) == 0:
            print("⚠ No training data!")
            return
        
        states = np.array(self.all_states)
        actions = np.array(self.all_actions)
        next_states = np.array(self.all_next_states)
        
        # Input: [state, action]
        X = np.hstack([states, actions])
        
        # Normalize
        if self.X_mean is None:
            self.X_mean = X.mean(axis=0)
            self.X_std = X.std(axis=0) + 1e-8
        
        X_norm = (X - self.X_mean) / self.X_std
        
        if verbose:
            print(f"\nK-Fold Cross-Validation Training ({self.n_folds} folds)")
            print(f"Total samples: {X.shape[0]}")
        
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        self.fold_models = []
        self.fold_scores = []
        
        state_names = ['vx', 'vy', 'wz', 'epsi', 's', 'ey']
        
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_norm)):
            if verbose:
                print(f"\n  Fold {fold_idx + 1}/{self.n_folds}:")
            
            X_train, X_val = X_norm[train_idx], X_norm[val_idx]
            y_train, y_val = next_states[train_idx], next_states[val_idx]
            
            # Train one model per output dimension
            models_this_fold = []
            scores_this_fold = []
            
            for state_idx in range(self.state_dim):
                # Ridge regression with small regularization
                model = Ridge(alpha=0.1)
                
                # Train
                model.fit(X_train, y_train[:, state_idx])
                
                # Validate
                score = model.score(X_val, y_val[:, state_idx])
                scores_this_fold.append(score)
                models_this_fold.append(model)
                
                if verbose:
                    print(f"    {state_names[state_idx]}: R²={score:.4f}")
            
            self.fold_models.append(models_this_fold)
            avg_score = np.mean(scores_this_fold)
            self.fold_scores.append(avg_score)
            
            if verbose:
                print(f"    Average: R²={avg_score:.4f}")
        
        if verbose:
            overall_avg = np.mean(self.fold_scores)
            print(f"\n✓ K-Fold training complete!")
            print(f"  Overall average R²: {overall_avg:.4f}\n")
    
    def predict(self, state, action, return_std=True):
        """
        Predict using ensemble (average across all K folds)
        Uncertainty from disagreement between folds
        """
        X = np.hstack([state, action]).reshape(1, -1)
        
        # Normalize
        if self.X_mean is not None:
            X = (X - self.X_mean) / self.X_std
        
        # Get predictions from all folds
        all_predictions = []
        
        for fold_models in self.fold_models:
            fold_pred = []
            for state_idx, model in enumerate(fold_models):
                pred = model.predict(X)[0]
                fold_pred.append(pred)
            all_predictions.append(fold_pred)
        
        all_predictions = np.array(all_predictions)  # Shape: (n_folds, state_dim)
        
        # Average across folds
        mean_pred = np.mean(all_predictions, axis=0)
        
        if return_std:
            # Uncertainty = standard deviation across folds
            std_pred = np.std(all_predictions, axis=0)
            return mean_pred, std_pred
        else:
            return mean_pred
    
    def save_model(self, filepath):
        """Save all fold models"""
        data = {
            'fold_models': self.fold_models,
            'fold_scores': self.fold_scores,
            'state_dim': self.state_dim,
            'input_dim': self.input_dim,
            'n_folds': self.n_folds,
            'X_mean': self.X_mean,
            'X_std': self.X_std
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✓ K-Fold models saved to {filepath}")
    
    def load_model(self, filepath):
        """Load fold models"""
        if not os.path.exists(filepath):
            print(f"⚠ File not found: {filepath}")
            return
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.fold_models = data['fold_models']
        self.fold_scores = data['fold_scores']
        self.state_dim = data['state_dim']
        self.input_dim = data['input_dim']
        self.n_folds = data['n_folds']
        self.X_mean = data.get('X_mean', None)
        self.X_std = data.get('X_std', None)
        
        print(f"✓ K-Fold models loaded from {filepath}")


# Test
if __name__ == "__main__":
    print("="*60)
    print("Testing K-Fold Dynamics Predictor")
    print("="*60 + "\n")
    
    kfold = KFoldDynamicsPredictor(n_folds=5)
    
    # Dummy data
    for traj in range(5):
        T = 100
        states = np.random.randn(T, 6) * 0.5
        actions = np.random.randn(T, 2) * 0.3
        kfold.add_trajectory(states, actions)
    
    kfold.train(verbose=True)
    
    # Test
    test_state = np.array([0.5, 0.1, 0.0, 0.0, 1.0, 0.0])
    test_action = np.array([0.1, 0.5])
    
    mean, std = kfold.predict(test_state, test_action, return_std=True)
    
    print(f"\nTest prediction:")
    print(f"  Mean: {mean}")
    print(f"  Std:  {std}")
    
    print("\n" + "="*60)