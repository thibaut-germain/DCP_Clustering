import numpy as np
from itertools import groupby
from joblib import Parallel,delayed

from tools.base_dtw import cost_matrix,warping_path,single_valence_vector,single_warping_vector


class DTW(object): 

    def __init__(self,radius=-1) -> None:
        """Initialization

        Args:
            radius (int, optional): Sakoe-Chiba radius, if set to -1, computes dtw distance without radius. Defaults to -1.
        """
        self.radius = radius

    def distance(self,x:np.ndarray,y:np.ndarray)->float: 
        """Compute DTW distance 

        Args:
            x (np.ndarray): sequence, shape(N_x,1), must be float type. 
            y (np.ndarray): sequence, shape(N_y,1), must be float type. 

        Returns:
            float: DTW distance between x and y.
        """
        cm = cost_matrix(x,y,self.radius)
        return cm[-1,-1]

    def warping_path(self,x:np.ndarray,y:np.ndarray)->np.ndarray:
        """_summary_

        Args:
            x (np.ndarray): sequence, shape(N_x,1), must be float type. 
            y (np.ndarray): sequence, shape(N_y,1), must be float type. 

        Returns:
            np.ndarray: warping path, shape(n_step,2) where n_step is the number of step in the warping path, the first column for the first sequence and the second for the second. 
        """
        cm = cost_matrix(x,y,self.radius)
        return warping_path(cm)

    def warping_path_and_distance(self,x:np.ndarray,y:np.ndarray)->np.ndarray:
        """_summary_

        Args:
            x (np.ndarray): sequence, shape(N_x,1), must be float type. 
            y (np.ndarray): sequence, shape(N_y,1), must be float type. 

        Returns:
            np.ndarray,float: warping path and distance
        """
        cm = cost_matrix(x,y,self.radius)
        return warping_path(cm),cm[-1,-1]

class BSDBA(object): 

    def __init__(self,metric:DTW,max_iter=30,batch_size=10,initial_step_size=.05,final_step_size=.005,tol=1e-5,verbose=False,njobs=1) -> None:
        """Initialization

        Args:
            metric (DTW): DTW object, measure of distance. 
            max_iter (int, optional): Maximum number of gradient descent step. Defaults to 30.
            batch_size (int, optional): Batch size for the stochastic gradient descent. Defaults to 10.
            initial_step_size (float, optional): Initial learning rate. Defaults to .05.
            final_step_size (float, optional): Final learning rate. Defaults to .005.
            tol (float, optional): Stopping criterion. Defaults to 1e-5.
            verbose (bool, optional): Display status. Defaults to False.
            njobs (int, optional): Tjread number for parallel computing. Defaults to 1.
        """
        self.metric = metric
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.initial_step_size = initial_step_size
        self.final_step_size = final_step_size
        self.tol = tol 
        self.verbose = verbose
        self.njobs = njobs

    def transform(self,X:np.ndarray,init_barycenter:np.ndarray)->np.ndarray:
        """Compute barycenter

        Args:
            X (np.ndarray): Time-series dataset
            init_barycenter (np.ndarray): Initial barycenter, shape(N,1)

        Returns:
            np.ndarray: Final barycenter
        """
        b_size = init_barycenter.shape[0]
        cost_prev, cost = np.inf,np.inf
        eta = self.initial_step_size
        n_sample = X.shape[0]
        barycenter = init_barycenter.copy()
        for it in range(self.max_iter): 
            batches = self.create_batches_selection(n_sample, self.batch_size)
            n_batches = len(batches)
            for batch in batches: 
                paths, cost = self._paths_cost_barycenter(X[batch],barycenter)
                if self.verbose:
                    print(cost)
                    print("[BSDBA] epoch %d, cost: %.3f" % (it + 1, cost))
                subgradient = self._subgradient_step(X[batch], paths,barycenter,b_size)
                barycenter -= eta*subgradient
                if it == 0: 
                    eta -= (self.initial_step_size-self.final_step_size)/n_batches 
                if (cost_prev - cost) < self.tol: 
                    break
                else: 
                    cost_prev = cost
        return barycenter

    def create_batches_selection(self,n_sample:int,batch_size:int)->list:
        """Create list of batches index

        Args:
            n_sample (int): Total number of samples
            batch_size (int): number of samples per batch

        Returns:
            list: List of batches
        """
        ite = groupby(np.arange(n_sample), key= lambda x: x//batch_size)
        r_idx = np.random.permutation(np.arange(n_sample))
        lst = []
        for i,t_idx in ite: 
            lst.append(r_idx[np.array(list(t_idx),dtype=int)])
        return lst

    def _paths_cost_barycenter(self,X:np.ndarray,barycenter:np.ndarray)->list: 
        """Compute path and cost to the initial barycenter

        Args:
            X (np.ndarray): Time-series dataset
            init_barycenter (np.ndarray): Initial barycenter, shape(N,1)

        Returns:
            list: List of paths, Average alignement cost
        """
        paths_cost_set = Parallel(self.njobs)(delayed(self.metric.warping_path_and_distance)(barycenter,X_i.astype(float)) for X_i in X)
        paths, cost = list(zip(*paths_cost_set))
        cost = np.sum(np.array(cost,dtype=float)**2)/X.shape[0]
        return paths,cost

    def _subgradient_step(self,X:np.ndarray,paths:list,barycenter:np.ndarray,barycenter_size:int)->np.ndarray:
        """Compute one subgradient step

        Args:
            X (np.ndarray): time-series dataset
            paths (list): list of warping path associated to the time-series dataset. 
            barycenter (np.ndarray): initial barycenter, shape(N,1)
            barycenter_size (int): Barycenter size

        Returns:
            np.ndarray: subgradient
        """
        valence_vectors = Parallel(self.njobs)(delayed(single_valence_vector)(path,barycenter_size) for path in paths)
        valence_vectors = np.array(valence_vectors)
        valence_vetor = np.sum(valence_vectors,axis=0)*barycenter

        warping_vectors = Parallel(self.njobs)(delayed(single_warping_vector)(path,ts.astype(float),barycenter_size) for path,ts in zip(paths,X))
        warping_vectors = np.array(warping_vectors)
        warping_vector = np.sum(warping_vectors,axis=0)

        return 2*(valence_vetor-warping_vector)/X.shape[0]