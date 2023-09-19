import numpy as np 
from tools.pipeline import Pipeline
import pickle

def execute_experiment(
        X:np.ndarray,
        sampfreq:int, prominence:float, wlen:float, cycle_minimum_duration:float,cycle_maximum_duration:float,training_size_per_interval:int,interval:int,
        in_ncluster: int, in_centroid_duration: int, out_ncluster: int, out_centroid_duration: int,
        down_sampfreq=None,
        radius=-1,n_iteration=10, quantile_threshold = None, njobs=1, verbose=False,
        ) -> None:
    
    pipe = Pipeline(sampfreq, prominence,wlen,cycle_minimum_duration,cycle_maximum_duration,training_size_per_interval,interval,
        in_ncluster, in_centroid_duration, out_ncluster, out_centroid_duration,
        down_sampfreq,
        radius,n_iteration, quantile_threshold,njobs,verbose,
        )
    pipe.fit(X)
    return pickle.dumps(pipe)

def load_pipe(pickled_exp): 
    pipe = pickle.loads(pickled_exp)
    return pipe

def plot_medoid(pickled_exp):
    pipe = load_pipe(pickled_exp)
    return pipe.plot_medoid().to_json()