import numpy as np 
import pandas as pd
import plotly.graph_objects as go 
from  plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import json

from tools.segmentation import PhaseSeg
from tools.preprocessing import ZNormalization
from tools.clustering import KmeanDTW
from tools.utils import to_time_series_dataset

class Pipeline(object): 

    def __init__(self,
        sampfreq:int, prominence:float, wlen:float, cycle_minimum_duration:float,cycle_maximum_duration:float,training_size_per_interval:int,interval:int,
        in_ncluster: int, in_centroid_duration: int, out_ncluster: int, out_centroid_duration: int,
        down_sampfreq=None,
        radius=-1,n_iteration=10, quantile_threshold = None, njobs=1, verbose=False,
        ) -> None:

        #Segmentation initialization
        self.sampfreq = sampfreq
        self.down_sampfreq = down_sampfreq
        self.prominence = prominence
        self.wlen = wlen 
        self.cycle_minimum_duration = cycle_minimum_duration
        self.cycle_maximum_duration = cycle_maximum_duration
        self.training_size_per_interval = training_size_per_interval
        self.interval = interval

        #clustering
        self.in_ncluster = in_ncluster
        self.in_centroid_duration = in_centroid_duration
        self.out_ncluster = out_ncluster
        self.out_centroid_duration = out_centroid_duration
        self.radius = radius
        self.n_iteration = n_iteration
        self.quantile_threshold = quantile_threshold
        self.njobs = njobs
        self.verbose = verbose


    def _segmentation(self,X:np.ndarray)->None:
        seg_lst = []
        for ts in X: 
            phs = PhaseSeg(self.freq_,self.prominence,self.wlen,self.cycle_minimum_duration,self.cycle_maximum_duration,self.training_size_per_interval,self.interval)
            phs.fit(ts)
            seg_lst.append(phs)
        self.seg_lst = seg_lst

    def _get_training_set(self,kind,processed = True): 
        lst = []
        for phs in self.seg_lst: 
            lst.append(phs.get_sequences(kind,phs.training_selection_))
        arr = np.concatenate(lst)
        if processed: 
            zn = ZNormalization()
            training_set = to_time_series_dataset(zn.fit_transform(arr))
        else: 
            training_set = to_time_series_dataset(arr)
        return training_set

    def _predict(self): 
        
        #inhalation
        pred = []
        json_pred = []
        zn = ZNormalization()
        for phs in self.seg_lst:
            #inhalation
            in_pred_set = phs.get_sequences('inhalation',phs.valid_mask_)
            in_pred_set = to_time_series_dataset(zn.fit_transform(in_pred_set))
            in_valid_labels = self.in_kdtw_.predict(in_pred_set)
            in_labels = np.full_like(phs.valid_mask_,-1,dtype=int)
            in_labels[phs.valid_mask_] = in_valid_labels

            #exhalation
            out_pred_set = phs.get_sequences('exhalation',phs.valid_mask_)
            out_pred_set = to_time_series_dataset(zn.fit_transform(out_pred_set))
            out_valid_labels = self.out_kdtw_.predict(out_pred_set)
            out_labels = np.full_like(phs.valid_mask_,-1,dtype=int)
            out_labels[phs.valid_mask_] = out_valid_labels

            #setting outliers to -1 
            in_labels = np.where(out_labels == -1 , -1, in_labels)
            out_labels = np.where(in_labels == -1, -1, out_labels)
            labels = np.vstack((in_labels,out_labels)).T
            pred.append(labels)

            #append json pred. 
            tdata = np.vstack([phs.insp_start_[:-1],in_labels,phs.exp_start_,out_labels]).T
            tdf = pd.DataFrame(tdata,columns=["in_start_index","in_cluster", "out_start_index", "out_cluster"])
            json_pred.append(tdf.to_json())

        self.predictions_ = np.array(pred,dtype=object)
        self.json_predictions_ = json.dumps(json_pred)

        return self

    def fit(self, X:np.ndarray)->None: 

        #Defined the downsamplig factor and the associated final frequency. 
        if self.down_sampfreq is not None: 
            self.freq_ratio_ = self.sampfreq // self.down_sampfreq
            self.freq_ = self.sampfreq/self.freq_ratio_
        else: 
            self.freq_ratio_ = 1
            self.freq_ = self.sampfreq

        X_ = np.array([ts[::self.freq_ratio_] for ts in X],dtype=object)

        #1. Segmentation step
        self._segmentation(X_)

        #2. Clustering Inhalation step
        if self.verbose: 
            print(' ### Inhalation ###')
        self.in_kdtw_ = KmeanDTW(self.in_ncluster,int(self.in_centroid_duration*self.freq_),int(self.radius*self.freq_),self.n_iteration,self.quantile_threshold,self.njobs,self.verbose)
        in_train_set = self._get_training_set('inhalation',processed=True)
        self.in_kdtw_.fit(in_train_set)

        #3. Clustering Exhalation step
        if self.verbose:
            print('\n ### Exhalation ###')
        self.out_kdtw_ = KmeanDTW(self.out_ncluster,int(self.out_centroid_duration*self.freq_),int(self.radius*self.freq_),self.n_iteration,self.quantile_threshold,self.njobs,self.verbose)
        out_train_set = self._get_training_set('exhalation',processed=True)
        self.out_kdtw_.fit(out_train_set)

        #4. Prediction
        self._predict()

        return self


    def plot_intertia(self): 
        fig = make_subplots(1,2,subplot_titles=['Inhalation Inertia Convergence', 'Exhalation Inertia Convergence'])
        fig.add_trace(
            go.Scatter(y = self.in_kdtw_.inertia_),
            row =1,
            col =1
        )
        fig.add_trace(
            go.Scatter(y = self.out_kdtw_.inertia_),
            row =1,
            col =2
        )
        fig.update_layout(
            xaxis = dict(title = 'Iteration'),
            yaxis = dict(title = 'Inertia'),
            xaxis2 = dict(title = 'Iteration'),
            yaxis2 = dict(title = 'Inertia'),

            showlegend=False
        )

        return fig

    def plot_medoid(self,in_palette = 'autumn',out_palette='winter'): 

        X_in = self._get_training_set(kind='inhalation', processed=False)
        X_out = self._get_training_set(kind='exhalation',processed=False)
        n_samples = X_in.shape[0]

        #get colors
        in_cmap = plt.cm.get_cmap(in_palette,self.in_ncluster)
        in_colors = np.array([in_cmap(i) for i in range(self.in_ncluster)])
        in_colors = [f"rgb({int(x[0])},{int(x[1])},{int(x[2])})" for x in in_colors*256]
        out_cmap = plt.cm.get_cmap(out_palette,self.out_ncluster)
        out_colors = np.array([out_cmap(i) for i in range(self.out_ncluster)])
        out_colors = [f"rgb({int(x[0])},{int(x[1])},{int(x[2])})" for x in out_colors*256]

        in_labels = self.in_kdtw_.labels_
        out_labels = self.out_kdtw_.labels_
        in_distances = self.in_kdtw_.distances_[np.arange(n_samples),self.in_kdtw_.labels_]
        out_distances = self.out_kdtw_.distances_[np.arange(n_samples),self.out_kdtw_.labels_]
        distance = np.sqrt(in_distances**2+out_distances**2)

        idx = np.zeros((self.in_ncluster,self.out_ncluster),dtype=int)
        for i in range(self.in_ncluster): 
            for j in range(self.out_ncluster): 
                try:
                    mask, = np.where((in_labels == i)&(out_labels==j))
                    t_idx = np.argmin(distance[mask])
                    idx[i,j] = mask[t_idx]
                except:
                    idx[i,j] = -1

        
        fig = make_subplots(rows = self.in_ncluster,cols = self.out_ncluster,shared_xaxes='all', shared_yaxes='all', x_title = "time (s)", subplot_titles = [f'<b>{i}</b>' for i in  np.arange(self.out_ncluster)], horizontal_spacing=0.02, vertical_spacing=0.02)

        for i in range(self.in_ncluster): 
            for j in range(self.out_ncluster): 
                if idx[i,j] !=  -1:
                    ts_in = X_in[idx[i,j]].reshape(-1)
                    ts_out = X_out[idx[i,j]].reshape(-1)
                    ts_in = np.hstack((ts_in,ts_out[0]))
                    fig.add_trace(
                        go.Scatter(x = np.arange(ts_in.shape[0])/self.freq_,y=ts_in, mode = "lines",marker =dict(color = in_colors[i])),
                        row = i+1,
                        col =j+1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=np.arange(ts_in.shape[0]-1,ts_in.shape[0]-1+ts_out.shape[0])/self.freq_,y=ts_out,mode = "lines",marker = dict(color = out_colors[j])),
                        row = i+1,
                        col =j+1
                    )


        for i, color in enumerate(in_colors):
            fig['layout'][f'yaxis{self.in_ncluster*i+1}']['title'] = f'<b>{chr(65+i)}</b>'
            fig['layout'][f'yaxis{self.in_ncluster*i+1}']['title']['font'] = dict(size =40,color = color)
        for i,color in enumerate(out_colors):
            fig['layout']['annotations'][i]['font'] = dict(size =40,color = color)
        fig.update_layout(
                hovermode = 'closest',
                width = self.out_ncluster*200+100,
                height = self.in_ncluster*150+100,
            )
        fig.update_layout(
            showlegend = False
        )
        
        return fig
    
    
    def get_centroid_representer_(self): 

        X_in = self._get_training_set(kind='inhalation', processed=False)
        X_out = self._get_training_set(kind='exhalation',processed=False)
        n_samples = X_in.shape[0]

        #get colors
        in_labels = self.in_kdtw_.labels_
        out_labels = self.out_kdtw_.labels_
        in_distances = self.in_kdtw_.distances_[np.arange(n_samples),self.in_kdtw_.labels_]
        out_distances = self.out_kdtw_.distances_[np.arange(n_samples),self.out_kdtw_.labels_]
        distance = np.sqrt(in_distances**2+out_distances**2)

        idx = np.zeros((self.in_ncluster,self.out_ncluster),dtype=int)
        for i in range(self.in_ncluster): 
            for j in range(self.out_ncluster): 
                try:
                    mask, = np.where((in_labels == i)&(out_labels==j))
                    t_idx = np.argmin(distance[mask])
                    idx[i,j] = mask[t_idx]
                except:
                    idx[i,j] = -1
        dct = {}
        json_dct = {}
        for i in range(self.in_ncluster): 
            for j in range(self.out_ncluster): 
                if idx[i,j] !=  -1:
                    ts_in = X_in[idx[i,j]]
                    ts_out = X_out[idx[i,j]]
                    dct[f"{i}-{j}"] = [ts_in,ts_out]
                    json_dct[f"{i}-{j}"] = [ts_in.tolist(),ts_out.tolist()]
        
        self.centroid_representer_ = dct
        self.json_centroid_representer = json.dumps(json_dct)

        return self.centroid_representer_

    
    @property
    def get_json_experiment_(self): 
        exp = dict()
        exp["prediction"] = self.json_predictions_
        exp["in_centroid"] = json.dumps(self.in_kdtw_.centroid_.tolist())
        exp["out_centroid"] = json.dumps(self.out_kdtw_.centroid_.tolist())
        self.get_centroid_representer_()
        exp["representer"] = self.json_centroid_representer
        return json.dumps(exp)

