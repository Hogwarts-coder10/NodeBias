import numpy as np
import random
import itertools
from copy import deepcopy

class RandomizedSearchCV:
    def __init__(self, model, param_distributions, n_iter=10, random_state=None):
        """
        model: An instantiated GlassBoxML model
        param_distributions: A dictionary of hyperparameters to test 
        n_iter: Number of random combinations to actually test (Saves the CPU!)
        """
        self._base_model = model
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state
        
        self.best_params_ = None
        self.best_score_ = -float('inf')
        self.best_model_ = None

    def fit(self, X, y):
        if self.random_state is not None:
            random.seed(self.random_state)
            
        # 1. Extract keys and values
        keys = list(self.param_distributions.keys())
        values = list(self.param_distributions.values())
        
        # 2. Generate the total universe of combinations
        all_combinations = list(itertools.product(*values))
        total_possible = len(all_combinations)
        
        # 3. The CPU Saver: Only sample 'n_iter' combinations
        # (Make sure we don't ask for more iterations than actually exist)
        actual_iters = min(self.n_iter, total_possible)
        sampled_combinations = random.sample(all_combinations, actual_iters)
        
        print(f"RandomizedSearchCV: Testing {actual_iters} random combinations out of {total_possible} total...")

        # 4. Quick 80/20 train/validation split
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # 5. The Lean Loop
        for combo in sampled_combinations:
            params = dict(zip(keys, combo))
            
            # Clone a fresh model
            current_model = deepcopy(self._base_model)
            
            # Inject the parameters
            for key, value in params.items():
                setattr(current_model, key, value)

            # Train and Score
            current_model.fit(X_train, y_train)
            predictions = current_model.predict(X_val)
            score = np.sum(predictions == y_val) / len(y_val)

            # Update the champion
            if score > self.best_score_:
                self.best_score_ = score
                self.best_params_ = params
                self.best_model_ = deepcopy(current_model)

        print(f"-> Random Search Complete! Best Validation Score: {self.best_score_ * 100:.2f}%")
        print(f"-> Winning Parameters: {self.best_params_}")
        
        # 6. Final Polish: Retrain the winner on all the data
        self.best_model_.fit(X, y)
        
        return self.best_model_