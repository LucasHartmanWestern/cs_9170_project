import os
import time

import numpy as np
import torch
from agents.ffnn_agent2 import FFNNAgent
from sklearn.model_selection import RandomizedSearchCV


class FFNN_CV:
    def __init__(
        self,
        param_distributions,
        n_iter,
        cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        seed=42,
        device='cpu'
    ):
        self.seed = seed
        self.device = torch.device('cuda' if torch.cuda.is_available() else device)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision('high')


        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs

        self.search_results = RandomizedSearchCV(
            estimator=FFNNAgent(),
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            random_state=seed,
            n_jobs=n_jobs
        )
        

    def get_params(self, x_train, y_train):#Should be numpy
        self.search_results.fit(x_train, y_train)
        best_params = self.search_results.best_params_
        print("Optimal parameters found: ", self.search_results.best_params_)
        # Ensure required keys are present (CV already sets many of these)
        best_params.setdefault("input_size", self.pca_components)
        best_params.setdefault("output_size", 2)
        best_params.setdefault("type", "classification")
        best_params.setdefault("classes", [0, 1])
        best_params.setdefault("device", self.device)
        best_params.setdefault("seed", self.seed)
        if isinstance(best_params.get("hidden_sizes"), tuple):
            best_params["hidden_sizes"] = list(best_params["hidden_sizes"])

        return best_params