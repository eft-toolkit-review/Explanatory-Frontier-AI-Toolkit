import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

__all__ = ['MetricsTop', 'set_seed']


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MetricsTop:
    def getMetrics(self, dataset_name):
        return {'CEO_TRUST': self.evaluate_regression}[dataset_name.upper()]

    @staticmethod
    def evaluate_regression(y_pred, y_true):
        """Compute MAE/Corr on all samples and Acc/F1 on nonzero labels."""
        y_pred = y_pred.detach().cpu().numpy().reshape(-1)
        y_true = y_true.detach().cpu().numpy().reshape(-1)

        nonzero = y_true != 0
        binary_pred = y_pred[nonzero] > 0
        binary_true = y_true[nonzero] > 0

        return {
            'Acc': round(accuracy_score(binary_true, binary_pred), 4),
            'F1_score': round(f1_score(binary_true, binary_pred, average='weighted'), 4),
            'MAE': round(np.mean(np.abs(y_pred - y_true)).astype(np.float64), 4),
            'Corr': round(np.corrcoef(y_pred, y_true)[0, 1], 4),
        }
