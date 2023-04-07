from typing import Callable
import numpy as np
from joblib import Parallel,delayed

def cdist(measure:Callable, dataset1:np.ndarray,dataset2=None,diagonal=False,njobs=1)->np.ndarray: 
    """Compute cross distance matrix

    Args:
        measure (Callable): Distance measure to use
        dataset1 (np.ndarray): formated sequence dataset.
        dataset2 (np.ndarray, optional): formated sequence dataset. If None Dataset 1 is duplicated. Defaults to None. 
        diagonal (bool, optional): True to compute diagonal element. Defaults to False.
        njobs (int, optional): number of core to use. Defaults to 1.

    Returns:
        np.ndarray: cross distance matrix
    """
    m = len(dataset1)
    if dataset2 is None: 
        cdist_array = np.zeros((m,m))
        idx1 = np.triu_indices(m,1-int(diagonal))
        cdist_array[idx1] = Parallel(n_jobs=njobs,prefer='threads')(
            delayed(measure.distance)(dataset1[i],dataset1[j]) for i in range(m) for j in range(i+1-int(diagonal),m)
        )
        idx2 = np.tril_indices(m,-1)
        cdist_array[idx2] = cdist_array.T[idx2]
        return cdist_array
    else: 
        cdist_array = Parallel(n_jobs=njobs,prefer='threads')(
            delayed(measure.distance)(s1,s2) for s1 in dataset1 for s2 in dataset2
        )
        return np.array(cdist_array).reshape(m,-1)


def to_time_series_dataset(data : list) -> np.ndarray: 
    """Transform a list of sequences to a format dataset

    Args:
        data (list): list of sequences [ts_1,ts_2,....] where ts_i is a np.ndarray of shape (N_i,) or (N_i,1)

    Returns:
        np.ndarray: object array of sequences of shape (N,1).
    """
    tdata = [to_time_series(ts) for ts in data]
    sizes = [ts.shape[0] for ts in data]
    if (len(tdata)>1) and not (np.all(np.array(sizes) == sizes[0])) :
        return np.array(tdata, dtype = object)
    else: 
        return np.array(tdata,dtype = float)

def to_time_series(ts : np.ndarray) -> np.ndarray: 
    if ts.ndim == 1: 
        return ts.reshape(-1,1).astype(float)
    else: 
        return ts.astype(float)
