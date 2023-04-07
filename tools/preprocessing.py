import numpy as np 
from sklearn.base import BaseEstimator, TransformerMixin
from tools.utils import to_time_series_dataset

class ZNormalization(BaseEstimator,TransformerMixin): #Tested Valid

    def __init__(self) -> None:
        super().__init__()

    def fit(self,X:np.ndarray)->None:
        """Compute aveage and std for each sequence

        Args:
            X (np.ndarray): object array of sequences, each sequence is of shape (N,)
        """
        fit_mean = []
        fit_std = []
        for ts in X: 
            fit_mean.append(np.mean(ts))
            fit_std.append(np.std(ts))
        self.mean_ = np.array(fit_mean,dtype=float)
        self.std_ = np.array(fit_std,dtype=float)
    
    def transform(self,X:np.ndarray)->np.ndarray: 
        """Normalized a set of sequences

        Args:
            X (np.ndarray): object array of sequences, each sequence is of shape (N,)

        Returns:
            np.ndarray: object array of normalized sequences, each sequence is of shape (N,1)
        """
        t_lst = []
        for i,ts in enumerate(X): 
            t_lst.append((ts-self.mean_[i])/self.std_[i])
        return to_time_series_dataset(t_lst)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)




