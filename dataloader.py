import numpy as np
import pandas as pd

class RandomClassDataloader:
    def __init__(self,X,z,  ratio_non_drifting=0.333,drifting_classes=1,non_drifting_classes=1):
        self.X = X
        self.z = z
        self.z_clazz, self.z_count = np.unique(z, return_counts=True)
        self.ratio_non_drifting,self.drifting_classes,self.non_drifting_classes = ratio_non_drifting,drifting_classes,non_drifting_classes
        self.ratio()
    def ratio(self,ratio_non_drifting=None,drifting_classes=None,non_drifting_classes=None):
        if ratio_non_drifting is not None:
            self.ratio_non_drifting = ratio_non_drifting
        if drifting_classes is not None:
            self.drifting_classes = drifting_classes
        if non_drifting_classes is not None:
            self.non_drifting_classes = non_drifting_classes
        
        drifting_classes,non_drifting_classes = self.drifting_classes,self.non_drifting_classes

        prop = np.zeros(shape=(3,self.z_clazz.shape[0]))
        
        sel = np.random.choice(prop.shape[1],size=2*drifting_classes+non_drifting_classes, p = self.z_count / self.z_count.sum(), replace=False)

        prop[0,sel[:drifting_classes]] = self.z_count[sel[:drifting_classes]] 
        prop[1,sel[drifting_classes:-drifting_classes]] = self.z_count[sel[drifting_classes:-drifting_classes]] 
        prop[2,sel[-drifting_classes:]] = self.z_count[sel[-drifting_classes:]] 
        
        prop = prop / prop.sum(axis=1)[:,None]

        self.prop = prop
        
        
    def take(self, size, create_testset=False):
        number_of_drifting = int((1-self.ratio_non_drifting)*size/2)
        number_of_non_drifting = size-2*number_of_drifting
        number_of_non_drifting_before = number_of_non_drifting//2
        number_of_non_drifting_after = number_of_non_drifting-number_of_non_drifting_before
        if np.random.random() > 0.5:
            number_of_non_drifting_before, number_of_non_drifting_after = number_of_non_drifting_after, number_of_non_drifting_before
        
        sel = [
            np.random.choice(range(self.prop.shape[1]),p=self.prop[0], size=number_of_drifting, replace=True),
            np.random.choice(range(self.prop.shape[1]),p=self.prop[1], size=number_of_non_drifting, replace=True),
            np.random.choice(range(self.prop.shape[1]),p=self.prop[2], size=number_of_drifting, replace=True)]
        
        typ = [-np.ones(shape=number_of_drifting),
               -0.25*np.ones(shape=number_of_non_drifting_before),
               0.25*np.ones(shape=number_of_non_drifting_after),
               np.ones(shape=number_of_drifting)]
        perm = np.random.permutation(size)
        sel,typ = np.hstack(sel)[perm], np.hstack(typ)[perm]

        X = np.empty( shape=(size, self.X.shape[1]) )
        cls,cls_c = np.unique(sel,return_counts=True)
        for tp,n in zip(cls,cls_c):
            X[sel==tp] = self.X[np.random.choice(np.where(self.z==tp)[0],size=n,replace=True)]

        observation_state = (typ < 0).astype(int)
        drift_state = (np.abs(typ) > 0.5).astype(int)
        z = (2*observation_state-1)*drift_state
        
        if create_testset:
            raise ValueError()
            return (X,observation_state,drift_state,z), ()

        return X,observation_state,drift_state,z

class NoImageNetLoader(RandomClassDataloader):
    def __init__(self,**kwds):
        df = np.load("data/NINCO_embedding.npz")
        super().__init__(df["X"],df["z"], **kwds)
class FashionMNISTLoader(RandomClassDataloader):
    def __init__(self,**kwds):
        df = np.load("data/FashionMNIST.npz")
        super().__init__(df["X"].reshape(df["y"].shape[0],-1),df["y"], **kwds)

class FishHeadLoader:
    def __init__(self,ratio_non_drifting=0.333):
        self.df = np.load("data/fish_head_embedding.npz")
        self.q = ratio_non_drifting
    def ratio(self, ratio_non_drifting=None):
         if ratio_non_drifting is not None: 
             self.q = ratio_non_drifting
    def take(self, size, create_testset=False):
        embedding,z = self.df["X"], self.df["z"]-1
        q = 1-self.q
        sel = np.hstack( [np.random.choice(np.where(z==i)[0], size=int( size *(q/2 if i != 0 else 1-q)), replace=True) for i in np.unique(z)] ).flatten()
        train_observation_state = np.zeros(sel.shape[0],dtype=int)
        train_observation_state[z[sel]==1] = 1
        train_observation_state[np.random.choice(np.where(z[sel]==0)[0],size=(z[sel]==0).sum()//2,replace=False)] = 1

        if create_testset:
            nsel = np.ones(embedding.shape[0], dtype=bool)
            nsel[sel] = False
            return (embedding[sel],train_observation_state,np.abs(z[sel]),z[sel]), (embedding[nsel],z[nsel])

        return embedding[sel],train_observation_state,np.abs(z[sel]),z[sel]
    
class SyntheticTimeSeriesLoader:
    def __init__(self, ratio_non_drifting=0.5, n_features=2):
        self.q = ratio_non_drifting
        self.n_features = n_features
        
    def ratio(self, ratio_non_drifting=None):
        if ratio_non_drifting is not None:
            self.q = ratio_non_drifting
            
    def take(self, size, create_testset=False):
        # 1. Simulate time steps
        t = np.arange(size)
        
        # 2. Observation State (os): 0 = before drift, 1 = after drift
        # We trigger the concept drift exactly halfway through the data stream.
        os_flag = (t >= size // 2).astype(int)
        
        # 3. Ground Truth Subpopulations (z): 0 = stationary, 1 = drifting subpopulation
        z = np.random.choice([0, 1], size=size, p=[self.q, 1 - self.q])
        
        # 4. Drift State (ds): Ground truth for the localizer
        ds = z.copy() 
        
        X = np.zeros((size, self.n_features))
        
        # 5. Generate the feature data based on populations and time
        for i in range(size):
            if z[i] == 0:
                # Stationary Subpopulation: Mean remains constant
                X[i] = np.random.normal(loc=-2.0, scale=1.0, size=self.n_features)
            else:
                # Drifting Subpopulation: Mean shifts after halfway mark
                if os_flag[i] == 0:
                    X[i] = np.random.normal(loc=2.0, scale=1.0, size=self.n_features)
                else:
                    X[i] = np.random.normal(loc=6.0, scale=1.0, size=self.n_features)
                    
        if create_testset:
            raise NotImplementedError("create_testset not currently supported for synthetic series.")
            
        return X, os_flag, ds, z

    def plot(self, X, os_flag, ds):

        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(14, 6))
        time_steps = np.arange(X.shape[0])
        
        # Plot stationary population
        stat_mask = (ds == 0)
        plt.scatter(time_steps[stat_mask], X[stat_mask, 0], 
                    alpha=0.5, label='Stationary Population (Stays ~ -2.0)', color='blue', s=15)
        
        # Plot drifting population
        drift_mask = (ds == 1)
        plt.scatter(time_steps[drift_mask], X[drift_mask, 0], 
                    alpha=0.5, label='Drifting Population (Shifts 2.0 -> 6.0)', color='red', s=15)
        
        # Find the exact moment the drift happens (when os_flag switches to 1)
        drift_idx = np.where(os_flag == 1)[0][0]
        plt.axvline(x=drift_idx, color='black', linestyle='--', linewidth=2, label=f'Drift Trigger (Time={drift_idx})')
        
        plt.title('Synthetic Concept Drift Over Time (Feature 1)')
        plt.xlabel('Time Step')
        plt.ylabel('Feature 1 Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

