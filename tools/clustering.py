import numpy as np
from scipy.signal import resample
from sklearn.base import BaseEstimator, ClusterMixin
import sys

from tools.utils import cdist,to_time_series_dataset
from tools.dtw import DTW, BSDBA

class Kmeans(BaseEstimator,ClusterMixin): #tested valid

    def __init__(self,ncluster : int, measure : callable, barycenter : callable, centroid_size : int ,n_iteration = 10 ,njobs = 1,verbose = False): 
        """Initialiation

        Args:
            ncluster (int): number of cluster
            measure (callable): distance measure
            barycenter (callable): Barycenter method
            centroid_size (int): centroid size.
            n_iteration (int, optional): maximum number of iteration for a averaging step. Defaults to 10.
            njobs (int, optional): nuber of core used. Defaults to 1.
            verbose (bool, optional): True to dusplay state. Defaults to False.
        """
        super().__init__()
        self.ncluster = ncluster
        self.measure = measure
        self.barycenter = barycenter 
        self.n_iteration = n_iteration
        self.centroid_size = centroid_size
        self.njobs = njobs
        self.verbose = verbose

    def _initialization(self,X:np.ndarray)->None:
        """Kmean++ initialization

        Args: 
            X (np.ndarray): formated preprocessed dataset
        """
        self.nsamples_ = len(X)
        dist_to_cluster = np.full((self.nsamples_,self.ncluster), np.inf)
        cluster_id = [np.random.randint(0,self.nsamples_)]
        while len(cluster_id) < self.ncluster: 
            temp_centroid = [X[cluster_id[-1]]]
            dist_to_cluster[:,len(cluster_id)-1] = cdist(self.measure,X,to_time_series_dataset(temp_centroid),njobs=self.njobs).flatten()
            min_dist = np.min(dist_to_cluster, axis=1)
            prob = min_dist**2/np.sum(min_dist**2)
            temp_id = np.random.choice(np.arange(self.nsamples_),p=prob)
            while temp_id in cluster_id:
                temp_id = np.random.choice(np.arange(self.nsamples_),p=prob)
            cluster_id.append(int(temp_id))
        init_centroid = [resample(ts.reshape(-1), self.centroid_size) for ts in X[cluster_id]]
        self.centroid_ = to_time_series_dataset(init_centroid)

        return self

    def _transform(self,X:np.ndarray)->np.ndarray: # Tested Valid
        """Kmean++ initialization

        Args: 
            X (np.ndarray): formated preprocessed dataset

        Returns: 
            np.ndarray: cross-distance array
        """
        return cdist(self.measure,X,self.centroid_,njobs=self.njobs)

    def _assign(self,X:np.ndarray,udapte_fitting = True)->np.ndarray: #Tested Valid
        """assigning method

        Args: 
            X (np.ndarray): formated preprocessed dataset
            update_fitting (optional,bool): Differentiate the case the function is  used for learning or predicting. Default True: learning.

        Returns: 
            np.ndarray: assigned labels
        """
        distances = self._transform(X)
        labels = np.argmin(distances,axis=1)
        if udapte_fitting: 
            self.distances_ = distances
            self.labels_ = labels
            self._check_no_empty_cluster()
            self.inertia_.append(self._compute_inertia())
        else:
            return labels
    
    def _update_centroid(self,X:np.ndarray)->None: #tested Valid
        """updating centroids

        Args: 
            X (np.ndarray): formated preprocessed dataset
        """
        for k,centroid in enumerate(self.centroid_): 
            self.centroid_[k]= self.barycenter.transform(X[self.labels_ == k],centroid)

    def fit(self,X:np.ndarray)->None:
        """fitting metohd

        Args: 
            X (np.ndarray): formated preprocessed dataset
        """
        if not sys.warnoptions:
            import warnings
            warnings.simplefilter("ignore")

        #initialisation
        success = False
        max_attempt = 10
        n_attempt = 0

        while (not success) and (n_attempt < max_attempt):
            try: 
                self._one_try_fit(X)
                success = True
            except EmptyClusterError:
                if self.verbose: 
                    print(f'Resumed because of an empty cluster. Attempt {n_attempt}/{max_attempt}')
                n_attempt +=1

        if not success: 
            raise EmptyClusterError

        return self
    
    def _one_try_fit(self,X:np.ndarray)->None:  # tested valid
        """learning loop

        Args: 
            X (np.ndarray): formated preprocessed dataset
        """
        #Initialization
        self.inertia_ = []
        self._initialization(X)

        #Iteration
        for i in range(self.n_iteration):
            self._assign(X)
            if self.verbose: 
                print(f"Iteration : {i+1}/{self.n_iteration} -- Inertia: {self.inertia_[-1]}")
            self._update_centroid(X)
        self._assign(X)
        return self    

    def predict(self,X:np.ndarray)->np.ndarray: #Tested Valid
        """predicting method

        Args: 
            X (np.ndarray): formated preprocessed dataset

        Returns: 
            np.ndarray: predicted labels
        """
        return self._assign(X,False)
    
    def fit_predict(self,X:np.ndarray)->np.ndarray: #Tested Valid
        """fitting and predicting method

        Args: 
            X (np.ndarray): formated preprocessed dataset

        Returns: 
            np.ndarray: predicted labels
        """
        self.fit(X)
        return self.labels_

    def _check_no_empty_cluster(self): #Tested Valid
        reshape_label = np.zeros((self.nsamples_,self.ncluster))
        reshape_label[np.arange(self.nsamples_),self.labels_] = 1
        if np.any(np.sum(reshape_label, axis = 0) == 0): 
            raise EmptyClusterError

    def _compute_inertia(self)->float: #tested Valid
        return np.sum(self.distances_[np.arange(self.nsamples_),self.labels_]**2)/self.nsamples_
            

class EmptyClusterError(Exception): #Tested Valid
    def __init__(self, message=""):
        super().__init__()
        self.message = message

    def __str__(self):
        if len(self.message) > 0:
            suffix = " (%s)" % self.message
        else:
            suffix = ""
        return "Cluster assignments lead to at least one empty cluster" + suffix


class KmeanDTW(Kmeans): #tested valid

    def __init__(self, ncluster: int, centroid_size: int, radius=-1, n_iteration=10, quantile_threshold =None ,njobs=1, verbose=False):
        """Initialiation

        Args:
            ncluster (int): number of cluster
            centroid_size (int): centroid size.
            radius (int,optional): radius for sakoe chiba constraint. If -1 no constraint applied. Defaults -1.
            n_iteration (int, optional): maximum number of iteration for a averaging step. Defaults to 10.
            quantile_threshold (float, optionnal): quantile threshold, must be between 0 and 1. Default None.  
            njobs (int, optional): nuber of core used. Defaults to 1.
            verbose (bool, optional): True to dusplay state. Defaults to False.
        """
        #set dtw measure
        self.radius = radius
        self.dtw = DTW(radius=radius)

        #set dtw barycenter method
        self.bar_dtw = BSDBA(self.dtw,max_iter=30, batch_size=10, initial_step_size=0.05, final_step_size=0.005, tol=1e-05, verbose=False, njobs=njobs)
        self.quantile_threshold = quantile_threshold

        #set Kmeans algorithm
        super().__init__(ncluster, self.dtw, self.bar_dtw, centroid_size, n_iteration, njobs, verbose)


    def fit(self,X): #tested valid
        super().fit(X)

        #ordering cluster in average length increasing order
        cluster_avg_lengths = []
        for i in range(self.ncluster): 
            cluster_avg_lengths.append(np.mean([len(ts) for ts in X[self.labels_ == i]]))
        self.cluster_avg_lengths_= np.array(cluster_avg_lengths)
        self.len_sort_ = np.argsort(self.cluster_avg_lengths_)
        self.len_inverse_sort_ = np.argsort(self.len_sort_)
        self.cluster_avg_lengths_ = self.cluster_avg_lengths_[self.len_sort_]
        self.centroid_ = self.centroid_[self.len_sort_]
        self.distances_ = self.distances_[:,self.len_sort_]
        self.labels_ = self.len_inverse_sort_[self.labels_]

        #compute distance threshold if needed
        if self.quantile_threshold is not None: 
            threshold_dists = []
            for i in range(self.ncluster): 
                distance_label_set = self.distances_[self.labels_ == i,i]
                threshold_dists.append(np.quantile(distance_label_set,self.quantile_threshold))

            self.threshold_dists_ = np.array(threshold_dists).astype(float)
            thresholds = self.threshold_dists_[self.labels_]
            idx = self.distances_[np.arange(self.labels_.shape[0]),self.labels_]>thresholds
            self.refined_labels_ = self.labels_
            self.refined_labels_[idx] =-1

        return self


    def predict(self,X:np.ndarray)->np.ndarray:
        distances = self._transform(X)
        labels = np.argmin(distances,axis=1) 
        if self.quantile_threshold is not None:
            distances = distances[np.arange(labels.shape[0]),labels]
            thresholds = self.threshold_dists_[labels]
            idx = distances>thresholds
            labels[idx] =-1
            return labels
        else: 
            return labels






